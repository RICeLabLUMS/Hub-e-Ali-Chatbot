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

    # CORS - comma-separated list of allowed origins, or "*" for any (dev only).
    # Tighten in production to your front-end origin(s).
    CORS_ALLOWED_ORIGINS: str = "*"

    # Chat rate limit (per-IP, sliding 60s window). 0 disables.
    CHAT_RATE_LIMIT_PER_MINUTE: int = 20

    # Public base URL of THIS backend, used by the front-end widget to call
    # /chat. Leave empty to use same-origin relative URLs (works for the
    # default deployment where the API and UI live at the same domain).
    # Set e.g. "https://api.hubeali.com" if the chat widget is embedded on
    # another domain and needs to call this API cross-origin. Should NOT end
    # with a trailing slash (we trim defensively anyway).
    PUBLIC_BACKEND_URL: str = ""

    # Local working dirs
    UPLOAD_DIR: str = "./tmp/rag_uploads"

    # Embedding device: auto | cuda | cpu
    EMBEDDING_DEVICE: str = "auto"

    # Retrieval tuning
    # RETRIEVAL_TOP_K: dense+sparse hybrid candidates fetched from Qdrant.
    # RERANK_TOP_K:    candidates the cross-encoder ranks AND the count that
    #                  reaches the LLM as context. Wider = better recall but
    #                  more reranker compute (negligible on GPU) and more
    #                  tokens to the LLM (real cost).
    RETRIEVAL_TOP_K: int = 40
    RERANK_TOP_K: int = 8
    # bge-reranker-v2-m3 supports up to 8192 tokens. 512 (the previous default)
    # was too short: chunks of ~400-500 tokens + query of ~30-50 tokens hit the
    # cap and got truncated, degrading rerank quality. 1024 fits any chunk +
    # query comfortably with headroom.
    RERANKER_MAX_LENGTH: int = 1024

    # OCR tuning (Surya, applied to scanned PDF pages only)
    # Render multiplier passed to PyMuPDF before OCR. Higher = sharper input
    # for the recognizer but more memory + compute. 3 is the sweet spot for
    # printed Arabic/Urdu; bump to 4 on very low-DPI source scans.
    OCR_RENDER_SCALE: float = 3.0
    # Minimum confidence to keep an OCR'd line. Lower = more text (incl. noise);
    # higher = cleaner text but may drop borderline-legible lines.
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    # Use Surya's LayoutPredictor for model-based reading-order reconstruction.
    # Gives semantic region labels (Title / Text / List / Footnote / etc.) and
    # proper reading order across columns. Costs ~one extra model load and ~2x
    # OCR wall-time per scanned page. Falls back to bbox heuristics if layout
    # can't be loaded.
    OCR_USE_LAYOUT: bool = True

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
    # Discover PDFs linked from post/page/CPT bodies (e.g. static files served
    # outside /wp-content/uploads/) and ingest them as separate documents.
    WORDPRESS_INGEST_LINKED_PDFS: bool = True
    # Extra hosts to allow PDF downloads from. The WP site's own host is always
    # allowed; this lets you whitelist CDNs or related domains. Comma-separated.
    WORDPRESS_LINKED_PDF_HOSTS: str = ""

    @property
    def upload_dir_path(self) -> Path:
        p = Path(self.UPLOAD_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
