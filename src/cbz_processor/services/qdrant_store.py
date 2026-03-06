"""Qdrant vector database integration."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from cbz_processor.config.config import config

if TYPE_CHECKING:
    from collections.abc import Iterator


class QdrantStore:
    """Qdrant vector database store for CBZ embeddings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "cbz_embeddings",
    ):
        """Initialize Qdrant client.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of collection
        """
        self.client = QdrantClient(host=host, port=port, prefer_grpc=False)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            self._create_collection()

    def _create_collection(self) -> None:
        """Create Qdrant collection with proper schema."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=config.EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE,
                on_disk=True,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                on_disk=False,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                memmap_threshold=20_000,
            ),
        )

    def upsert_points(
        self,
        points: list[models.PointStruct],
        batch_size: int = 100,
    ) -> int:
        """Upsert points to Qdrant in batches.

        Args:
            points: List of Qdrant points
            batch_size: Points per batch

        Returns:
            Number of points successfully inserted
        """
        success_count = 0

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]

            for attempt in range(3):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True,
                    )
                    success_count += len(batch)
                    break
                except (UnexpectedResponse, Exception) as e:
                    if attempt < 2:
                        time.sleep(2**attempt)
                    else:
                        raise

        return success_count

    def prepare_point(
        self,
        source_path: str,
        image_filename: str,
        resolution: tuple[int, int],
        comic_info: dict,
        embedding: list[float],
        point_id: int,
    ) -> models.PointStruct:
        """Create Qdrant point from image data.

        Args:
            source_path: Path to CBZ file
            image_filename: Image filename
            resolution: (width, height)
            comic_info: Comic metadata dict
            embedding: Embedding vector
            point_id: Unique point ID

        Returns:
            Qdrant PointStruct
        """
        payload = {
            "source_path": source_path,
            "image_filename": image_filename,
            "resolution": f"{resolution[0]}x{resolution[1]}",
            "comic_info": comic_info,
        }

        return models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        )

    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        """Search for similar embeddings.

        Args:
            embedding: Query embedding
            limit: Max results
            score_threshold: Minimum score filter

        Returns:
            List of scored points
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
            score_threshold=score_threshold,
        )

    def count_points(self) -> int:
        """Count total points in collection."""
        return self.client.count(collection_name=self.collection_name).count

    def get_point(self, point_id: int | str) -> models.PointStruct | None:
        """Get single point by ID."""
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
        )[0]
