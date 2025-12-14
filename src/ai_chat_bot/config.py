from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding ="utf-8",
        case_sensetive=False
    )

    
    gemini_api_key: str = Field(
        ..., 
        description="Google Gemini API key",
    )    
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use",
    )  
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use",
    )

def get_settings()-> Settings:
    return Settings()