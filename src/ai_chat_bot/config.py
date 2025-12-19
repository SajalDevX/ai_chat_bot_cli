from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    gemini_api_key: str = Field(
        default=None,
        description="Google Gemini API key",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key (starts with sk-ant-)"
    )
    
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use",
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Default Anthropic model"
    )
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key (free at console.groq.com)"
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default Groq model"
    )
    
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=16000,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation",
    )
    timeout: float = Field(
        default=60.0,
        ge=5.0,
        description="API request timeout in seconds",
    )

    def has_gemini(self) -> bool:
        """Check if Gemini is configured."""
        return self.gemini_api_key is not None
    
    def has_groq(self) -> bool:
        """Check if Groq is configured."""
        return self.has_groq is not None
    
    def has_anthropic(self) -> bool:
        """Check if Anthropic is configured."""
        return self.anthropic_api_key is not None
    
    def available_providers(self) -> list[str]:
        """List of configured providers."""
        providers = []
        if self.has_gemini():
            providers.append("gemini")
        if self.has_groq():
            providers.append("groq")
        if self.has_anthropic():
            providers.append("anthropic")
        return providers
    
    def get_primary_provider(self) -> str | None:
        """Get the first available provider."""
        providers = self.available_providers()
        return providers[0] if providers else None

@lru_cache
def get_settings() -> Settings:
    return Settings()