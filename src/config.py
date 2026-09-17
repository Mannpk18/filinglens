"""Central, env-driven configuration. Import `settings` everywhere instead of
reading os.environ directly, so every module shares one source of truth."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Database
    database_url: str = "postgresql://filinglens:filinglens@localhost:5432/filinglens"

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024  # bge-large output dim

    # SEC EDGAR
    sec_user_agent: str = "FilingLens research@example.com"
    edgar_base_url: str = "https://data.sec.gov"
    edgar_submissions_url: str = "https://data.sec.gov/submissions"
    edgar_full_text_search_url: str = "https://efts.sec.gov/LATEST/search-index"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Feature flags / retrieval tuning
    verifier_enabled: bool = True
    max_verify_retries: int = 2
    retrieval_top_k: int = 8
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75


settings = Settings()
