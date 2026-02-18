"""Configuration management for RAG service."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "rag_documents"
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # LLM Configuration
    llm_provider: str = "groq"
    llm_api_key: Optional[str] = None
    llm_model: str = "llama-3-8b-8192"
    ollama_base_url: str = "http://localhost:11434"
    
    # Timeout Configuration (in seconds)
    llm_timeout: float = 2.0
    search_timeout: float = 0.8
    
    # RRF Configuration
    rrf_k: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
