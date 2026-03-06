"""Configuration management for CBZ processor."""

from __future__ import annotations

from pathlib import Path
from typing import final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@final
class AppConfig(BaseSettings):
    """Application configuration loaded from environment and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # vLLM endpoint configuration
    VLLM_ENDPOINT: str = Field(
        default="http://localhost:7997",
        description="vLLM server endpoint for embedding generation",
    )

    # vLLM model name
    VLLM_MODEL: str = Field(
        default="Qwen/Qwen3-VL-Embedding-2B",
        description="Model name for embedding generation",
    )

    # Batch sizes
    CBZ_CHUNK_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of CBZ files to process per chunk",
    )

    EMBEDDING_BATCH_SIZE: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Images per embedding batch",
    )

    QDRANT_BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Points per Qdrant insert batch",
    )

    # Worker pools
    WORKER_PARALLELISM: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Parallel CBZ file processors",
    )

    # Qdrant configuration
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Qdrant database host",
    )

    QDRANT_PORT: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="Qdrant database port",
    )

    QDRANT_COLLECTION: str = Field(
        default="cbz_embeddings",
        description="Qdrant collection name",
    )

    # Vector dimensions
    EMBEDDING_DIMENSION: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Embedding vector dimension",
    )

    # File paths
    CHECKPOINT_FILE: Path = Field(
        default=Path("data/checkpoint.json"),
        description="Path to checkpoint file for resume",
    )

    LOG_DIR: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )

    # Processing options
    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failing operations",
    )

    # Metadata extraction
    SUPPORTED_IMAGE_TYPES: tuple[str, ...] = Field(
        default=("png", "jpeg", "jpg", "gif", "webp"),
        description="Supported image extensions in CBZ",
    )

    def model_post_init(self, __context):
        """Validate configuration and create directories."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)


config = AppConfig()
