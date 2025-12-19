
class ChatbotError(Exception):
    """Base exception for all chatbot errors.
    
    All our custom exceptions inherit from this.
    Allows catching all chatbot errors with one except clause.
    
    Attributes:
        message: Human-readable error description
    """
    def __init__(self,message:str,provider:str)->None:
        self.message = message
        self.provider = provider
        super().__init__(self.message)
            
    def __str__(self) -> str:
        if self.provider:
            return f"[{self.provider}] {self.message}"
        return self.message
    
class ConfigurationError(ChatbotError):
    """Raised when configuration is missing or invalid.
    
    Examples:
        - Missing API key
        - Invalid model name
    """
    pass


class APIError(ChatbotError):
    """Base class for API-related errors.
    
    Attributes:
        message: Error description
        status_code: HTTP status code (if available)
    """
    
    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None) -> None:
        self.status_code = status_code
        super().__init__(message, provider)


class AuthenticationError(APIError):
    """Raised when API authentication fails.
    
    Usually means invalid or expired API key.
    """
    
    def __init__(self, message: str = "Invalid API key", provider: str | None = None) -> None:
        super().__init__(message, status_code=401, provider=provider)


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded.
    
    Attributes:
        retry_after: Seconds to wait before retrying (if known)
    """
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        retry_after: int | None = None,
        provider: str | None = None
    ):
        super().__init__(message, status_code=429, provider=provider)
        self.retry_after = retry_after
        
class APIConnectionError(APIError):
    """Raised when unable to connect to the API.
    
    Could be network issues, timeout, or server down.
    """
    
    def __init__(self, message: str = "Failed to connect to API") -> None:
        super().__init__(message)


class ValidationError(ChatbotError):
    """Raised when input validation fails.
    
    Examples:
        - Empty message
        - Message too long
    """
    pass

class AllProvidersFailedError(ChatbotError):
    """All configured providers failed (NEW).
    
    Used when multi-provider fallback exhausts all options.
    """
    
    def __init__(self, errors: list[ChatbotError]):
        self.errors = errors
        
        provider_errors = []
        for e in errors:
            provider = e.provider or "unknown"
            provider_errors.append(f"{provider}: {e.message}")
        
        message = "All providers failed:\n  " + "\n  ".join(provider_errors)
        super().__init__(message)