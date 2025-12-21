import sys
from ai_chat_bot.clients import GeminiClient
from ai_chat_bot.clients.base import BaseClient
from ai_chat_bot.config import get_settings
from ai_chat_bot.utils.exceptions import (
    AuthenticationError,
    APIConnectionError,
    APIError,
    ChatbotError,
    ConfigurationError,
    RateLimitError,
)
from ai_chat_bot.models import Conversation
from ai_chat_bot.utils import (
    Display,
    TokenEstimator,
    CostTracker,
    ContextManager,
    estimate_cost,
    MODEL_CONTEXT_LIMITS,
    RetryConfig,
    retry_with_backoff,
)


class ChatBot:
    """Main chatbot class with token management.

    Features:
    - Streaming responses with real-time display
    - Token estimation before sending (predict costs)
    - Cost tracking per session
    - Context window management
    """

    def __init__(self) -> None:
        self.display = Display()
        self.conversation = Conversation()
        self.client: BaseClient | None = None
        self.running = False

        # Token management
        self.token_estimator = TokenEstimator()
        self.cost_tracker = CostTracker(
            warning_threshold=0.10,  # Warn at $0.10
            critical_threshold=1.00,  # Critical at $1.00
        )
        # Context manager - will be initialized with model's context limit
        self.context_manager: ContextManager | None = None

        # Retry configuration for API calls
        self.retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
        )
        
    def _init_client(self)->bool:
        try:
            self.client = GeminiClient()

            # Initialize context manager with model's limit
            model = self.client.model_name
            context_limit = MODEL_CONTEXT_LIMITS.get(model, 128_000)
            self.context_manager = ContextManager(
                max_context_tokens=context_limit,
                estimator=self.token_estimator
            )
            return True
        except ConfigurationError as e:
            self.display.error(
                f"Configuration error: {e.message}\n\n"
                "Make sure GEMINI_API_KEY is set in your .env file."
            )
            return False
        except Exception as e:
            self.display.error(f"Failed to initialize: {e}")
            return False
            
    def handle_command(self,command:str)->bool:

        cmd = command.lower().strip()
        if cmd in ("/quit","/exit","/q"):
            return False
        if cmd in ("/help","/h","/?"):
            self.display.help()
            return True
        if cmd in ("/clear","/c"):
            self.conversation.clear()
            self.display.cleared()
            return True
        if cmd in ("/stats","/s"):
            self._show_session_stats()
            return True

        self.display.warning(f"Unknown command: {command}")
        self.display.info("Type /help for available commands")
        return True

    def _show_session_stats(self) -> None:
        """Display session statistics."""
        # Get cost tracker summary
        summary = self.cost_tracker.get_summary()

        self.display.info("Session Statistics:")
        self.display.info(f"  Requests: {summary['request_count']}")
        self.display.info(f"  Total Tokens: {summary['total_tokens']:,}")
        self.display.info(f"  Total Cost: ${summary['total_cost_usd']:.6f}")

        if summary['request_count'] > 0:
            self.display.info(
                f"  Avg Tokens/Request: {summary['avg_tokens_per_request']:.0f}"
            )

        # Show context usage if available
        if self.context_manager:
            ctx = self.context_manager.get_context_usage(self.conversation)
            self.display.info(
                f"  Context Usage: {ctx['usage_percent']:.1f}% "
                f"({ctx['estimated_tokens']:,}/{ctx['max_tokens']:,} tokens)"
            )
    
    def send_message(self,message:str)->None:
        
        if not self.client:
            self.display.error("Client not intialized")
            return 
        self.conversation.add_user_message(message)
        self.display.thinking()
        try:
            response = self.client.chat(self.conversation)
            self.conversation.add_assistant_message(response)
            
            self.display.assistant_message(response)
            self.display.stats(model=self.client.model_name)

        except AuthenticationError as e:
            self.display.error(f"Authentication failed: {e.message}")
            self.display.info("Check your GEMINI_API_KEY in .env file")
            # Remove the failed user message
            self.conversation.messages.pop()
            
        except RateLimitError as e:
            self.display.warning(f"Rate limited: {e.message}")
            if e.retry_after:
                self.display.info(f"Try again in {e.retry_after} seconds")
            # Remove the failed user message
            self.conversation.messages.pop()
            
        except APIConnectionError as e:
            self.display.error(f"Connection error: {e.message}")
            self.display.info("Check your internet connection")
            # Remove the failed user message
            self.conversation.messages.pop()
            
        except APIError as e:
            self.display.error(f"API error: {e.message}")
            # Remove the failed user message
            self.conversation.messages.pop()
            
            
            
    def run(self) -> None:
        """Run the interactive chat loop."""
        # Show welcome message
        self.display.welcome()
        
        # Initialize client
        if not self._init_client():
            return
        
        self.display.success("Connected to Gemini API!")
        self.display.info(f"Model: {self.client.model_name}")
        
        self.running = True
        
        try:
            while self.running:
                # Get user input
                user_input = self.display.prompt()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle commands (start with /)
                if user_input.startswith("/"):
                    self.running = self.handle_command(user_input)
                    continue
                
                # Send message to AI
                self.send_message(user_input)
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            pass
        
        finally:
            # Cleanup
            self.display.goodbye()
            if self.client:
                self.client.close()

    def _on_retry(self, attempt: int, exception: Exception, delay: float) -> None:
        """Callback when a retry is about to happen."""
        self.display.warning(
            f"Request failed: {exception}. Retrying in {delay:.1f}s... "
            f"(attempt {attempt}/{self.retry_config.max_retries})"
        )

    def _stream_response(self) -> str:
        """Stream response from API and return full text.

        This method is wrapped with retry logic.
        Returns the complete response text.
        """
        full_response = ""

        for chunk in self.client.chat_stream(self.conversation):
            self.display.assistant_stream_chunk(chunk)
            full_response += chunk

        return full_response

    def send_message_stream(self, message: str) -> None:
        """Send a message and stream the response with retry support.

        Args:
            message: The user's message
        """
        if not self.client:
            self.display.error("Client not initialized")
            return

        # =============================================
        # BEFORE SENDING: Token estimation & context check
        # =============================================

        # 1. Estimate tokens for this request (used for context check)
        _ = self.token_estimator.estimate_with_new_message(
            self.conversation, message
        )

        # 2. Check if we're approaching context limit
        if self.context_manager:
            model = self.client.model_name
            usage = self.context_manager.get_context_usage(self.conversation)

            # Warn if over 80% of context used
            if usage["usage_percent"] > 80:
                self.display.warning(
                    f"Context {usage['usage_percent']:.0f}% full "
                    f"({usage['estimated_tokens']:,}/{usage['max_tokens']:,} tokens)"
                )

        # Add user message to conversation
        self.conversation.add_user_message(message)

        try:
            # Start streaming display
            self.display.assistant_stream_start()

            # =============================================
            # STREAM WITH RETRY SUPPORT
            # =============================================
            # Wrap the streaming call with retry logic
            # On RateLimitError or APIConnectionError, it will:
            # 1. Wait with exponential backoff
            # 2. Retry the request
            full_response = retry_with_backoff(
                func=self._stream_response,
                config=self.retry_config,
                on_retry=self._on_retry,
            )

            # End streaming display
            self.display.assistant_stream_end()

            # Add complete response to conversation history
            self.conversation.add_assistant_message(full_response)

            # =============================================
            # AFTER RECEIVING: Cost tracking
            # =============================================

            model = self.client.model_name
            tokens = None
            cost = 0.0

            if self.client.last_stream_usage:
                usage = self.client.last_stream_usage
                tokens = usage.total_tokens

                # Calculate actual cost
                cost = estimate_cost(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    model=model
                )

                # Record in cost tracker and check for alerts
                alert = self.cost_tracker.record(usage, cost, model)
                if alert:
                    self.display.warning(alert)

            # Show stats
            self.display.stats(model=model, tokens=tokens, cost=cost)

        except AuthenticationError as e:
            # Auth errors are NOT retryable - fail immediately
            self.display.assistant_stream_end()
            self.display.error(f"Authentication failed: {e.message}")
            self.display.info("Check your GEMINI_API_KEY in .env file")
            self.conversation.messages.pop()

        except RateLimitError as e:
            # Rate limit after all retries exhausted
            self.display.assistant_stream_end()
            self.display.error(f"Rate limited (retries exhausted): {e.message}")
            if e.retry_after:
                self.display.info(f"Try again in {e.retry_after} seconds")
            self.conversation.messages.pop()

        except APIConnectionError as e:
            # Connection error after all retries exhausted
            self.display.assistant_stream_end()
            self.display.error(f"Connection failed (retries exhausted): {e.message}")
            self.display.info("Check your internet connection")
            self.conversation.messages.pop()

        except APIError as e:
            # Generic API errors are NOT retryable
            self.display.assistant_stream_end()
            self.display.error(f"API error: {e.message}")
            self.conversation.messages.pop()
    
    def run(self) -> None:
        """Run the interactive chat loop."""
        self.display.welcome()
        
        if not self._init_client():
            return
        
        self.display.success("Connected to Gemini API!")
        self.display.info(f"Model: {self.client.model_name}")
        self.display.info("Streaming enabled ✨")
        
        self.running = True
        
        try:
            while self.running:
                user_input = self.display.prompt()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    self.running = self.handle_command(user_input)
                    continue
                
                # Show user message and stream response
                # self.display.user_message(user_input)
                self.send_message_stream(user_input)
                
        except KeyboardInterrupt:
            pass
        
        finally:
            self.display.goodbye()
            if self.client:
                self.client.close()


def main() -> None:
    """Entry point for the CLI."""
    try:
        bot = ChatBot()
        bot.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


# This allows running: python -m ai_chatbot.main
if __name__ == "__main__":
    main()