"""Application configuration using Pydantic BaseSettings.

Loads configuration from environment variables with sensible defaults
for local development and mock-mode operation.
"""

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central application settings.

    All values can be overridden via environment variables
    (case-insensitive, prefix-free).
    """

    # ── Application ───────────────────────────────────────────────
    app_name: str = "Brain MRI AI Detection System"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Networking / CORS ─────────────────────────────────────────
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:3000",
        ],
        description="Origins allowed for CORS requests (Streamlit, dev tools).",
    )

    # ── File Storage ──────────────────────────────────────────────
    upload_dir: Path = Field(
        default=Path("data/uploads"),
        description="Directory for uploaded scan files.",
    )
    max_file_size_mb: int = Field(
        default=500,
        description="Maximum upload size in megabytes.",
    )
    retention_days: int = Field(
        default=7,
        description="Days to retain uploaded scans before cleanup.",
    )

    # ── Model / Inference ─────────────────────────────────────────
    mock_inference: bool = Field(
        default=True,
        description="When True, return synthetic predictions without real models.",
    )
    tumour_model_path: Path = Field(
        default=Path("models/tumour_classifier.pth"),
        description="Path to the brain-tumour classification model weights.",
    )
    alzheimer_model_path: Path = Field(
        default=Path("models/alzheimer_classifier.pth"),
        description="Path to the Alzheimer classification model weights.",
    )

    # ── Database ──────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///data/audit.db",
        description="SQLAlchemy-style database URL for the audit log.",
    )

    # ── Security ──────────────────────────────────────────────────
    api_key: str = Field(
        default="dev-api-key-change-me",
        description="Shared API key for simple authentication (v1).",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per client.",
    )

    # ── Pydantic-settings config ──────────────────────────────────
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    # ── Computed helpers ──────────────────────────────────────────
    @property
    def max_file_size_bytes(self) -> int:
        """Return the maximum upload size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


# Module-level singleton so ``from core.config import settings`` works.
settings = Settings()
