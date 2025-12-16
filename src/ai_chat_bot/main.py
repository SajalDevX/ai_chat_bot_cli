import sys
from ai_chat_bot.clients import GeminiClient
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
from ai_chat_bot.utils import Display 



class ChatBot:
    
    def __init__(self)->None:
        self.display = Display()
        self.conversation = Conversation()
        self.client:GeminiClient|None=None
        self.running = False
        
    def _init_client(self)->bool:
        try:
            self.client = GeminiClient()
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
        
        self.display.warning(f"Unknown command: {command}")
        self.display.info("Type /help for available commands")
        return True
    
    def send_message(self,message:str)->None:
        
        if not self.client:
            self.display.error("Client not intialized")
            return 
        self.conversation.add_user_message(message)
        self.display.thinking()
        try:
            response = self.client.chat(self.conversation)
            self.conversation.add_model_message(response)
            
            self.display.assistant_message(response)
            self.display.stats(model=self.client.settings.gemini_model)

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
        self.display.info(f"Model: {self.client.settings.gemini_model}")
        
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

    def send_message_stream(self, message: str) -> None:
        """Send a message and stream the response.
        
        Args:
            message: The user's message
        """
        if not self.client:
            self.display.error("Client not initialized")
            return
        
        # Add user message to conversation
        self.conversation.add_user_message(message)
        
        try:
            # Start streaming display
            self.display.assistant_stream_start()
            
            # Collect full response for conversation history
            full_response = ""
            
            # Stream chunks from API
            for chunk in self.client.chat_stream(self.conversation):
                # Display chunk immediately
                self.display.assistant_stream_chunk(chunk)
                # Accumulate for history
                full_response += chunk
            
            # End streaming display
            self.display.assistant_stream_end()
            
            # Add complete response to conversation history
            self.conversation.add_model_message(full_response)
            
            # Show stats
            self.display.stats(model=self.client.settings.gemini_model)
            
        except AuthenticationError as e:
            self.display.assistant_stream_end()  # Clean up display
            self.display.error(f"Authentication failed: {e.message}")
            self.display.info("Check your GEMINI_API_KEY in .env file")
            self.conversation.messages.pop()
            
        except RateLimitError as e:
            self.display.assistant_stream_end()
            self.display.warning(f"Rate limited: {e.message}")
            if e.retry_after:
                self.display.info(f"Try again in {e.retry_after} seconds")
            self.conversation.messages.pop()
            
        except APIConnectionError as e:
            self.display.assistant_stream_end()
            self.display.error(f"Connection error: {e.message}")
            self.display.info("Check your internet connection")
            self.conversation.messages.pop()
            
        except APIError as e:
            self.display.assistant_stream_end()
            self.display.error(f"API error: {e.message}")
            self.conversation.messages.pop()
    
    def run(self) -> None:
        """Run the interactive chat loop."""
        self.display.welcome()
        
        if not self._init_client():
            return
        
        self.display.success("Connected to Gemini API!")
        self.display.info(f"Model: {self.client.settings.gemini_model}")
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