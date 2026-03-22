from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import logging

class Settings(BaseSettings):
    # AWS Bedrock Config
    aws_access_key_id: str = Field(default="dummy")
    aws_secret_access_key: str = Field(default="dummy")
    aws_region: str = Field(default="eu-north-1")
    bedrock_model_id: str = Field(default="anthropic.claude-haiku-4-5-20251001-v1:0")

    # Google Gemini Config
    google_api_key: str = Field(default="dummy")
    gemini_model: str = Field(default="gemini-1.5-flash")
    
    # Model Config
    model_provider: str = Field(default="bedrock")
    model_temperature: float = Field(default=0.0)
    model_max_tokens: int = Field(default=1024)
    
    # System Performance
    request_timeout: int = Field(default=5)
    max_retries: int = Field(default=3)
    rate_limit_per_minute: int = Field(default=5)
    
    # System Modes
    fallback_mode: str = Field(default="graceful")
    response_mode: str = Field(default="structured")
    
    
    # News Config
    news_source: str = Field(default="google_rss")
    newsapi_key: str = Field(default="")
    news_fetch_limit: int = Field(default=5)
    news_language: str = Field(default="en")
    news_sort_by: str = Field(default="publishedAt")

    # Market Data Fallback APIs
    alphavantage_api_key: str = Field(default="")
    finnhub_api_key: str = Field(default="")
    
    # Vector DB Config
    vector_db_type: str = Field(default="faiss")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    
    # Logging
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
