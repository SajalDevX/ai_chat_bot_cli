# src/ai_chatbot/__init__.py
"""AI Chatbot - Multi-provider LLM client.

UPDATED: Version 1.1.0 with multi-provider support.
"""

__version__ = "1.1.0"
__author__ = "Airo"

# Models
from ai_chat_bot.models import (
    Message,
    Conversation,
    Role,
    ChatResponse,
    TokenUsage,
)

# Config
from ai_chat_bot.config import Settings, get_settings

# Clients
from ai_chat_bot.clients import (
    BaseClient,
    GeminiClient,
    GroqClient, 
)

# Utils
from ai_chat_bot.utils import Display

# Exceptions
from ai_chat_bot.utils.exceptions import (
    ChatbotError,
    ConfigurationError,
    APIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    ValidationError,
    AllProvidersFailedError,
)

__all__ = [
    "__version__",
    "Message",
    "Conversation",
    "Role",
    "ChatResponse",
    "TokenUsage",
    "Settings",
    "get_settings",
    "BaseClient",
    "GeminiClient",
    "GroqClient",
    "Display",
    "ChatbotError",
    "ConfigurationError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "APIConnectionError",
    "ValidationError",
    "AllProvidersFailedError",
]