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

    # WordPress sync
    WORDPRESS_URL: str = ""
    WORDPRESS_USERNAME: str = ""
    WORDPRESS_APP_PASSWORD: str = ""
    WORDPRESS_CPT_ROUTES: str = ""              # comma-separated CPT slugs, e.g. "lectures,fatwas"
    WORDPRESS_SYNC_INTERVAL_MINUTES: int = 60
    WORDPRESS_STATE_FILE: str = "./tmp/wordpress_sync_state.json"
    WORDPRESS_SYNC_ENABLED: bool = False        # opt-in APScheduler hook in FastAPI lifespan
    # Some WAFs (Cloudflare, Wordfence, ModSecurity) reject requests with non-browser
    # User-Agent strings, returning a TCP reset (WinError 10054). Override only if
    # your WP site requires a specific UA.
    WORDPRESS_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    WORDPRESS_VERIFY_SSL: bool = True

    @property
    def upload_dir_path(self) -> Path:
        p = Path(self.UPLOAD_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
