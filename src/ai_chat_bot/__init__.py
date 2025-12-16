# src/ai_chatbot/__init__.py
"""AI Chatbot - A CLI chatbot using Google's Gemini API.

Quick Start:
    # Interactive mode
    from ai_chatbot.main import main
    main()
    
    # Programmatic use
    from ai_chatbot import GeminiClient, Conversation
    
    with GeminiClient() as client:
        conv = Conversation()
        conv.add_user_message("Hello!")
        response = client.chat(conv)
        print(response)
"""

__version__ = "1.0.0"
__author__ = "Airo"

# Main components
from ai_chat_bot.clients import GeminiClient
from ai_chat_bot.config import Settings, get_settings
from ai_chat_bot.models import Conversation, Message, Role
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
)

__all__ = [
    # Version
    "__version__",
    # Client
    "GeminiClient",
    # Config
    "Settings",
    "get_settings",
    # Models
    "Message",
    "Conversation",
    "Role",
    # Utils
    "Display",
    # Exceptions
    "ChatbotError",
    "ConfigurationError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "APIConnectionError",
    "ValidationError",
]