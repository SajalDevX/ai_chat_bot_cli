# src/ai_chat_bot/utils/tokens.py
"""Token estimation and budget management utilities.

This module provides:
- Token estimation before API calls (to predict costs)
- Token budgeting (to prevent runaway costs)
- Context window management (to handle long conversations)

Why estimate tokens?
- Know costs BEFORE sending requests
- Warn users about expensive operations
- Stay within context window limits
- Manage rate limits (tokens per minute)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ai_chat_bot.models import Conversation, TokenUsage


# =============================================================================
# SECTION 1: Token Estimation
# =============================================================================

class TokenEstimator:
    """Estimate token counts before making API calls.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is an approximation - actual counts vary by model and content.

    For more accurate estimation, you could use:
    - tiktoken library (OpenAI's tokenizer)
    - Model-specific tokenizers from HuggingFace

    But for cost estimation, ~15% accuracy is usually sufficient.

    Usage:
        estimator = TokenEstimator()
        tokens = estimator.estimate_text("Hello, world!")
        tokens = estimator.estimate_conversation(conversation)
    """

    # Average characters per token (empirically determined)
    # English text: ~4 chars/token
    # Code: ~3 chars/token (more symbols)
    # Other languages: varies (CJK can be 1-2 chars/token)
    CHARS_PER_TOKEN = 4

    # Overhead tokens per message (role, formatting)
    # Each message has: {"role": "user", "content": "..."}
    # The role and structure add ~4 tokens overhead
    MESSAGE_OVERHEAD = 4

    # Conversation overhead (start/end tokens)
    CONVERSATION_OVERHEAD = 3

    def estimate_text(self, text: str) -> int:
        """Estimate tokens in a text string.

        Args:
            text: The text to estimate

        Returns:
            Estimated token count

        Example:
            >>> estimator = TokenEstimator()
            >>> estimator.estimate_text("Hello, world!")
            4  # 13 chars / 4 ≈ 4 tokens
        """
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def estimate_message(self, content: str) -> int:
        """Estimate tokens for a single message including overhead.

        Args:
            content: The message content

        Returns:
            Estimated token count including message overhead
        """
        return self.estimate_text(content) + self.MESSAGE_OVERHEAD

    def estimate_conversation(self, conversation: Conversation) -> int:
        """Estimate total tokens in a conversation.

        Includes:
        - System prompt (if any)
        - All messages with overhead
        - Conversation overhead

        Args:
            conversation: The conversation to estimate

        Returns:
            Estimated total token count
        """
        total = self.CONVERSATION_OVERHEAD

        # System prompt
        if conversation.system_prompt:
            total += self.estimate_message(conversation.system_prompt)

        # All messages
        for msg in conversation.messages:
            total += self.estimate_message(msg.content)

        return total

    def estimate_with_new_message(
        self,
        conversation: Conversation,
        new_message: str
    ) -> int:
        """Estimate tokens if a new message were added.

        Useful for checking if adding a message would exceed limits.

        Args:
            conversation: Current conversation
            new_message: Message to be added

        Returns:
            Estimated total tokens after adding message
        """
        current = self.estimate_conversation(conversation)
        new_msg_tokens = self.estimate_message(new_message)
        return current + new_msg_tokens


# =============================================================================
# SECTION 2: Cost Estimation
# =============================================================================

# Pricing per 1 million tokens (as of 2024)
# Format: {model: {"input": price, "output": price}}
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

    # Google Gemini
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during preview
    "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0},  # Check current pricing

    # Groq (FREE!)
    "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},
    "mixtral-8x7b-32768": {"input": 0.0, "output": 0.0},
}

# Context window sizes (max tokens)
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-2.5-flash-lite": 1_000_000,
    "llama-3.3-70b-versatile": 128_000,
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str
) -> float:
    """Estimate cost for a request.

    Args:
        input_tokens: Number of input (prompt) tokens
        output_tokens: Number of output (completion) tokens
        model: Model name

    Returns:
        Estimated cost in USD

    Example:
        >>> estimate_cost(1000, 500, "gpt-4o")
        0.0075  # (1000/1M * 2.50) + (500/1M * 10.00)
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def estimate_request_cost(
    input_tokens: int,
    model: str,
    expected_output_tokens: int = 500
) -> dict:
    """Estimate cost range for a request before sending.

    Args:
        input_tokens: Estimated input tokens
        model: Model name
        expected_output_tokens: Expected output tokens (default 500)

    Returns:
        Dict with min, expected, and max cost estimates

    Example:
        >>> estimate_request_cost(1000, "gpt-4o")
        {"min": 0.003, "expected": 0.0075, "max": 0.015}
    """
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})

    input_cost = (input_tokens / 1_000_000) * pricing["input"]

    # Estimate output range: min 50, expected 500, max 2000 tokens
    min_output_cost = (50 / 1_000_000) * pricing["output"]
    expected_output_cost = (expected_output_tokens / 1_000_000) * pricing["output"]
    max_output_cost = (2000 / 1_000_000) * pricing["output"]

    return {
        "input_cost": input_cost,
        "min": input_cost + min_output_cost,
        "expected": input_cost + expected_output_cost,
        "max": input_cost + max_output_cost,
    }


# =============================================================================
# SECTION 3: Token Budget Management
# =============================================================================

@dataclass
class TokenBudget:
    """Track and enforce token usage limits.

    Prevents runaway costs by setting daily/monthly limits.

    Usage:
        budget = TokenBudget(daily_limit=100_000, cost_limit_usd=5.0)

        # Before each request:
        can_proceed, reason = budget.can_make_request(estimated_tokens)
        if not can_proceed:
            print(f"Budget exceeded: {reason}")
            return

        # After each request:
        budget.record_usage(response.usage, response.cost)
    """

    # Limits
    daily_limit: int = 100_000          # Max tokens per day
    monthly_limit: int = 2_000_000      # Max tokens per month
    cost_limit_usd: float = 10.0        # Max cost per month in USD

    # Current usage (tracked automatically)
    tokens_today: int = 0
    tokens_this_month: int = 0
    cost_this_month: float = 0.0

    # Timestamps for reset logic
    last_daily_reset: datetime = field(default_factory=datetime.now)
    last_monthly_reset: datetime = field(default_factory=datetime.now)

    def _maybe_reset(self) -> None:
        """Reset counters if day/month has changed."""
        now = datetime.now()

        # Daily reset: if we're on a new day
        if now.date() > self.last_daily_reset.date():
            self.tokens_today = 0
            self.last_daily_reset = now

        # Monthly reset: if we're in a new month
        if (now.year > self.last_monthly_reset.year or
            now.month > self.last_monthly_reset.month):
            self.tokens_this_month = 0
            self.cost_this_month = 0.0
            self.last_monthly_reset = now

    def can_make_request(self, estimated_tokens: int) -> tuple[bool, str]:
        """Check if a request is within budget.

        Args:
            estimated_tokens: Estimated tokens for the request

        Returns:
            Tuple of (allowed: bool, reason: str)

        Example:
            >>> budget = TokenBudget(daily_limit=1000)
            >>> budget.tokens_today = 900
            >>> budget.can_make_request(200)
            (False, "Would exceed daily limit (1000 tokens)")
        """
        self._maybe_reset()

        # Check daily limit
        if self.tokens_today + estimated_tokens > self.daily_limit:
            return False, f"Would exceed daily limit ({self.daily_limit:,} tokens)"

        # Check monthly limit
        if self.tokens_this_month + estimated_tokens > self.monthly_limit:
            return False, f"Would exceed monthly limit ({self.monthly_limit:,} tokens)"

        return True, "OK"

    def record_usage(self, usage: TokenUsage, cost: float = 0.0) -> None:
        """Record actual usage after a request.

        Args:
            usage: TokenUsage from the response
            cost: Calculated cost in USD
        """
        self._maybe_reset()

        self.tokens_today += usage.total_tokens
        self.tokens_this_month += usage.total_tokens
        self.cost_this_month += cost

    def get_remaining(self) -> dict:
        """Get remaining budget.

        Returns:
            Dict with remaining tokens and cost
        """
        self._maybe_reset()

        return {
            "daily_tokens_remaining": self.daily_limit - self.tokens_today,
            "monthly_tokens_remaining": self.monthly_limit - self.tokens_this_month,
            "cost_remaining_usd": self.cost_limit_usd - self.cost_this_month,
        }

    def get_usage_summary(self) -> dict:
        """Get usage summary.

        Returns:
            Dict with usage statistics
        """
        self._maybe_reset()

        return {
            "tokens_today": self.tokens_today,
            "tokens_this_month": self.tokens_this_month,
            "cost_this_month_usd": self.cost_this_month,
            "daily_usage_percent": (self.tokens_today / self.daily_limit) * 100,
            "monthly_usage_percent": (self.tokens_this_month / self.monthly_limit) * 100,
        }


# =============================================================================
# SECTION 4: Context Window Management
# =============================================================================

class ContextManager:
    """Manage conversation context to fit within model limits.

    When conversations get too long, we need to trim older messages
    while keeping the most recent context.

    Strategies:
    1. Keep system prompt always
    2. Keep most recent messages
    3. Optionally summarize old messages

    Usage:
        manager = ContextManager(max_tokens=128_000)
        trimmed = manager.trim_conversation(conversation, reserve_output=4000)
    """

    def __init__(
        self,
        max_context_tokens: int = 128_000,
        estimator: TokenEstimator | None = None
    ):
        """Initialize context manager.

        Args:
            max_context_tokens: Maximum context window size
            estimator: Token estimator (creates default if not provided)
        """
        self.max_context = max_context_tokens
        self.estimator = estimator or TokenEstimator()

    def fits_in_context(
        self,
        conversation: Conversation,
        reserve_output: int = 4000
    ) -> bool:
        """Check if conversation fits in context window.

        Args:
            conversation: The conversation to check
            reserve_output: Tokens to reserve for model output

        Returns:
            True if conversation fits, False otherwise
        """
        estimated = self.estimator.estimate_conversation(conversation)
        available = self.max_context - reserve_output
        return estimated <= available

    def trim_conversation(
        self,
        conversation: Conversation,
        reserve_output: int = 4000
    ) -> Conversation:
        """Trim conversation to fit context window.

        Keeps:
        - System prompt (always)
        - Most recent messages that fit

        Args:
            conversation: Conversation to trim
            reserve_output: Tokens to reserve for output

        Returns:
            New Conversation object with trimmed messages
        """
        available = self.max_context - reserve_output

        # Start with system prompt
        used_tokens = 0
        if conversation.system_prompt:
            used_tokens = self.estimator.estimate_message(conversation.system_prompt)

        available -= used_tokens

        # Add messages from most recent to oldest
        kept_messages = []

        for msg in reversed(conversation.messages):
            msg_tokens = self.estimator.estimate_message(msg.content)

            if msg_tokens <= available:
                kept_messages.insert(0, msg)  # Insert at beginning to maintain order
                available -= msg_tokens
            else:
                # Can't fit this message, stop
                break

        # Create new conversation with trimmed messages
        from ai_chat_bot.models import Conversation as Conv
        trimmed = Conv(
            system_prompt=conversation.system_prompt,
            messages=kept_messages
        )

        return trimmed

    def get_context_usage(self, conversation: Conversation) -> dict:
        """Get context window usage statistics.

        Args:
            conversation: The conversation to analyze

        Returns:
            Dict with usage statistics
        """
        estimated = self.estimator.estimate_conversation(conversation)

        return {
            "estimated_tokens": estimated,
            "max_tokens": self.max_context,
            "usage_percent": (estimated / self.max_context) * 100,
            "tokens_remaining": self.max_context - estimated,
        }


# =============================================================================
# SECTION 5: Cost Tracker (Session-level)
# =============================================================================

@dataclass
class CostTracker:
    """Track costs within a session with alerting.

    Useful for:
    - Showing running total to users
    - Warning when costs get high
    - Generating session summaries

    Usage:
        tracker = CostTracker(warning_threshold=1.0)

        # After each request:
        alert = tracker.record(response)
        if alert:
            print(alert)  # "Warning: Session cost is $1.23"
    """

    # Alert thresholds
    warning_threshold: float = 1.0   # Warn at $1
    critical_threshold: float = 5.0  # Critical at $5

    # Tracking
    session_cost: float = 0.0
    request_count: int = 0
    total_tokens: int = 0

    # History
    request_history: list = field(default_factory=list)

    def record(
        self,
        usage: TokenUsage,
        cost: float,
        model: str = ""
    ) -> str | None:
        """Record a request and return any alerts.

        Args:
            usage: Token usage from response
            cost: Calculated cost
            model: Model name

        Returns:
            Alert message if threshold exceeded, None otherwise
        """
        self.session_cost += cost
        self.request_count += 1
        self.total_tokens += usage.total_tokens

        self.request_history.append({
            "timestamp": datetime.now(),
            "model": model,
            "tokens": usage.total_tokens,
            "cost": cost,
        })

        # Check thresholds
        if self.session_cost >= self.critical_threshold:
            return f"CRITICAL: Session cost is ${self.session_cost:.4f}"
        elif self.session_cost >= self.warning_threshold:
            return f"Warning: Session cost is ${self.session_cost:.4f}"

        return None

    def get_summary(self) -> dict:
        """Get session summary.

        Returns:
            Dict with session statistics
        """
        avg_cost = self.session_cost / self.request_count if self.request_count > 0 else 0
        avg_tokens = self.total_tokens / self.request_count if self.request_count > 0 else 0

        return {
            "total_cost_usd": self.session_cost,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "avg_cost_per_request": avg_cost,
            "avg_tokens_per_request": avg_tokens,
        }

    def format_summary(self) -> str:
        """Get formatted summary string.

        Returns:
            Human-readable summary
        """
        s = self.get_summary()
        return (
            f"Session Summary:\n"
            f"  Requests: {s['request_count']}\n"
            f"  Total Tokens: {s['total_tokens']:,}\n"
            f"  Total Cost: ${s['total_cost_usd']:.4f}\n"
            f"  Avg Cost/Request: ${s['avg_cost_per_request']:.4f}"
        )
