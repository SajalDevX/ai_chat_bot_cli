from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use",
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for response generation",
    )
    timeout: float = Field(
        default=30.0,
        description="API request timeout in seconds",
    )


def get_settings() -> Settings:
    return Settings()