"""Pydantic data models for CBZ processing."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class ComicInfo(BaseModel):
    """Comic metadata extracted from ComicInfo.xml."""

    model_config = ConfigDict(extra="allow")

    title: str | None = Field(default=None, description="Comic title")
    series: str | None = Field(default=None, description="Series name")
    number: str | None = Field(default=None, description="Series number")
    writer: str | list[str] | None = Field(
        default=None,
        description="Comic writer(s)",
    )
    translator: str | list[str] | None = Field(
        default=None,
        description="Comic translator(s)",
    )
    genre: str | list[str] | None = Field(
        default=None,
        description="Comic genre(s)",
    )
    web: str | None = Field(default=None, description="Web reference")
    summary: str | None = Field(default=None, description="Comic summary")
    publisher: str | None = Field(default=None, description="Publisher name")
    month: int | None = Field(default=None, description="Publication month")
    year: int | None = Field(default=None, description="Publication year")
    language: str | None = Field(default=None, description="Language code")
    key_characters: str | list[str] | None = Field(
        default=None,
        description="Key characters",
    )


class ImageMetadata(BaseModel):
    """Metadata for a single image."""

    filename: str = Field(..., description="Image filename")
    width: int = Field(..., ge=1, description="Image width in pixels")
    height: int = Field(..., ge=1, description="Image height in pixels")
    resolution: Annotated[str, Field(description="Resolution as 'WxH' string")] = (
        ""
    )

    def model_post_init(self, __context):
        self.resolution = f"{self.width}x{self.height}"


class CBZProcessingResult(BaseModel):
    """Result of processing a single CBZ file."""

    source_path: str = Field(..., description="Path to CBZ file")
    hash: str = Field(..., description="File hash for integrity")
    comic_info: ComicInfo | None = Field(
        default=None,
        description="Extracted comic metadata",
    )
    images: list[ImageMetadata] = Field(
        default_factory=list,
        description="Extracted image metadata",
    )
    embeddings: list[list[float]] = Field(
        default_factory=list,
        description="Generated embeddings",
    )
    processed_at: datetime = Field(default_factory=datetime.now)


class QdrantPoint(BaseModel):
    """Point structure for Qdrant vector database."""

    id: str = Field(..., description="Unique point ID")
    vector: list[float] = Field(..., description="Embedding vector")
    payload: dict[str, Any] = Field(..., description="Metadata payload")
