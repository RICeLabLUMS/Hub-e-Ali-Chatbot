from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "hub-e-ali"

    # OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # API auth
    API_KEY: str = "change-me-in-production"

    # Local working dirs
    UPLOAD_DIR: str = "./tmp/rag_uploads"

    # Embedding device: auto | cuda | cpu
    EMBEDDING_DEVICE: str = "auto"

    # Retrieval tuning
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_K: int = 5

    @property
    def upload_dir_path(self) -> Path:
        p = Path(self.UPLOAD_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
