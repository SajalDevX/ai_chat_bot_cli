from abc import ABC, abstractmethod
from typing import Generator
from ai_chat_bot.config import Settings
from ai_chat_bot.models import Conversation, ChatResponse


class BaseClient(ABC):

    PROVIDER_NAME: str = "base"
    BASE_URL: str = ""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the current model name for this provider."""
        pass

    @abstractmethod
    def chat(self, conversation: Conversation) -> ChatResponse:
        """Send a chat request and get the full response."""
        pass
    
    @abstractmethod
    def chat_stream(self, conversation: Conversation) -> Generator[str, None, None]:
        """Send a chat request and get a streaming response."""
        pass
    
    @abstractmethod
    def _build_payload(self, conversation: Conversation) -> dict:
        """Convert conversation to provider's API format."""
        pass
    
    @abstractmethod
    def _parse_response(self, response_data: dict) -> ChatResponse:
        """Parse provider's response to ChatResponse."""
        pass
    
    @abstractmethod
    def _parse_stream(self, response_data: dict) -> ChatResponse:
        """Parse provider's response to ChatResponse."""
        pass
    
    def close(self) -> None:
        """Close HTTP client. Override if cleanup needed."""
        pass
    
    def __enter__(self) -> "BaseClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.PROVIDER_NAME})"    