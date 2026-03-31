from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_DATA = PROJECT_ROOT / "data" / "reference_tickets.jsonl"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    reference_data_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("TICKET_SENTIMENT_REFERENCE_PATH", str(DEFAULT_REFERENCE_DATA))
        )
    )
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("TICKET_SENTIMENT_TOP_K", "5"))
    )
    transformer_model_name: str = field(
        default_factory=lambda: os.getenv(
            "TICKET_SENTIMENT_TRANSFORMER_MODEL",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "TICKET_SENTIMENT_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    enable_embedding_retrieval: bool = field(
        default_factory=lambda: _env_flag("TICKET_SENTIMENT_ENABLE_EMBEDDINGS", default=True)
    )
    enable_transformer_model: bool = field(
        default_factory=lambda: _env_flag("TICKET_SENTIMENT_ENABLE_TRANSFORMER", default=True)
    )
    enable_vader_model: bool = field(
        default_factory=lambda: _env_flag("TICKET_SENTIMENT_ENABLE_VADER", default=True)
    )
    local_files_only: bool = field(
        default_factory=lambda: _env_flag("TICKET_SENTIMENT_LOCAL_FILES_ONLY", default=False)
    )
    llm_provider: str = field(
        default_factory=lambda: os.getenv("TICKET_SENTIMENT_LLM_PROVIDER", "auto").lower()
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
        or os.getenv("TICKET_SENTIMENT_OPENAI_API_KEY")
    )
    openai_model: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL")
        or os.getenv("TICKET_SENTIMENT_OPENAI_MODEL")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "TICKET_SENTIMENT_OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv(
            "TICKET_SENTIMENT_OLLAMA_MODEL",
            "llama3.1:8b-instruct",
        )
    )
    llm_margin_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("TICKET_SENTIMENT_LLM_MARGIN_THRESHOLD", "0.14")
        )
    )
    always_use_llm: bool = field(
        default_factory=lambda: _env_flag("TICKET_SENTIMENT_ALWAYS_USE_LLM", default=False)
    )
    max_text_length: int = field(
        default_factory=lambda: int(os.getenv("TICKET_SENTIMENT_MAX_TEXT_LENGTH", "512"))
    )
