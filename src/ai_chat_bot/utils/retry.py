# src/ai_chat_bot/utils/retry.py
"""Retry utilities with exponential backoff.

This module provides:
- Exponential backoff for transient failures
- Configurable retry strategies
- Jitter to prevent thundering herd

Why exponential backoff?
- Gives servers time to recover
- Prevents overwhelming a struggling service
- Required by most API rate limit policies
"""

import time
import random
from dataclasses import dataclass
from typing import Callable, TypeVar, Any
from functools import wraps

from ai_chat_bot.utils.exceptions import (
    APIError,
    APIConnectionError,
    RateLimitError,
)


# =============================================================================
# SECTION 1: Retry Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries (caps exponential growth)
        exponential_base: Multiplier for exponential backoff (usually 2)
        jitter: Add randomness to prevent thundering herd (0.0-1.0)

    Example:
        config = RetryConfig(max_retries=3, base_delay=1.0)
        # Delays: 1s, 2s, 4s (then gives up)
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # 10% randomness

    # Which exceptions should trigger a retry?
    retryable_exceptions: tuple = (
        RateLimitError,
        APIConnectionError,
    )

    # Which status codes should trigger a retry?
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


# Default configs for different scenarios
AGGRESSIVE_RETRY = RetryConfig(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
)

CONSERVATIVE_RETRY = RetryConfig(
    max_retries=2,
    base_delay=2.0,
    max_delay=10.0,
)

NO_RETRY = RetryConfig(max_retries=0)


# =============================================================================
# SECTION 2: Backoff Calculator
# =============================================================================

def calculate_backoff(
    attempt: int,
    config: RetryConfig,
    retry_after: float | None = None
) -> float:
    """Calculate delay before next retry.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        retry_after: Server-suggested delay (from Retry-After header)

    Returns:
        Delay in seconds

    Example:
        >>> config = RetryConfig(base_delay=1.0, exponential_base=2.0)
        >>> calculate_backoff(0, config)  # 1st retry
        1.0
        >>> calculate_backoff(1, config)  # 2nd retry
        2.0
        >>> calculate_backoff(2, config)  # 3rd retry
        4.0
    """
    # If server tells us how long to wait, respect it
    if retry_after is not None:
        return min(retry_after, config.max_delay)

    # Calculate exponential delay: base * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base ** attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter: random value between -jitter% and +jitter%
    if config.jitter > 0:
        jitter_range = delay * config.jitter
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)  # Never negative


# =============================================================================
# SECTION 3: Retry Executor
# =============================================================================

T = TypeVar('T')


def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Execute a function with retry logic.

    Args:
        func: The function to execute (no arguments)
        config: Retry configuration (uses default if None)
        on_retry: Callback called before each retry with (attempt, exception, delay)

    Returns:
        The result of func()

    Raises:
        The last exception if all retries fail

    Example:
        def make_api_call():
            return client.chat(conversation)

        result = retry_with_backoff(
            make_api_call,
            config=RetryConfig(max_retries=3),
            on_retry=lambda a, e, d: print(f"Retry {a} after {d}s: {e}")
        )
    """
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return func()

        except config.retryable_exceptions as e:
            last_exception = e

            # Check if we have retries left
            if attempt >= config.max_retries:
                raise  # No more retries, propagate exception

            # Get retry_after from RateLimitError if available
            retry_after = None
            if isinstance(e, RateLimitError) and e.retry_after:
                retry_after = e.retry_after

            # Calculate delay
            delay = calculate_backoff(attempt, config, retry_after)

            # Notify callback
            if on_retry:
                on_retry(attempt + 1, e, delay)

            # Wait before retry
            time.sleep(delay)

        except Exception:
            # Non-retryable exception, propagate immediately
            raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop completed without result or exception")


# =============================================================================
# SECTION 4: Decorator Version
# =============================================================================

def with_retry(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
):
    """Decorator to add retry logic to a function.

    Args:
        config: Retry configuration
        on_retry: Callback for retry events

    Example:
        @with_retry(config=RetryConfig(max_retries=3))
        def make_api_call(message: str):
            return client.chat(message)

        result = make_api_call("Hello")  # Automatically retries on failure
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config,
                on_retry=on_retry,
            )
        return wrapper
    return decorator
