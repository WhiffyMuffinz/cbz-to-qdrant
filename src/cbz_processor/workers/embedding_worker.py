"""Embedding batch processor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cbz_processor.config.config import config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import NoReturn

    from cbz_processor.models.data_models import CBZProcessingResult


class EmbeddingBatchProcessor:
    """Processes embeddings in batches using vLLM.

    Keeps embedding model loaded in memory.
    """

    def __init__(self):
        """Initialize embedding batch processor."""
        self.endpoint = config.VLLM_ENDPOINT
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        self.client = self._create_client()

    def _create_client(self):
        """Create vLLM embedding client."""
        from cbz_processor.services.embedding_service import EmbeddingClient

        return EmbeddingClient(
            endpoint=self.endpoint,
            max_retries=config.MAX_RETRIES,
        )

    def process_batch(
        self,
        results: list[dict],
    ) -> list[dict]:
        """Process embeddings for batch of CBZ results.

        Args:
            results: List of CBZ processing results (dicts)

        Returns:
            Results with embeddings populated
        """
        all_images: list[tuple[str, str, bytes, str]] = []
        result_map: dict[str, list[int]] = {}

        for idx, result in enumerate(results):
            if not result:
                continue

            for img in result.get("images", []):
                filename = img.get("filename")
                if not filename:
                    continue
                all_images.append((result["source_path"], filename, img.get("data", b""), filename))
                result_map.setdefault(result["source_path"], []).append(len(all_images) - 1)

        if not all_images:
            return results

        image_data_list = [img[2] for img in all_images]

        batch_embeddings = self.client.generate_embeddings_batch(image_data_list)

        embedding_idx = 0
        for idx, result in enumerate(results):
            if not result:
                continue
            if "embeddings" not in result:
                result["embeddings"] = []
            for img in result.get("images", []):
                if embedding_idx < len(batch_embeddings):
                    result["embeddings"].append(batch_embeddings[embedding_idx])
                embedding_idx += 1

        return results

    def is_available(self) -> bool:
        """Check if vLLM endpoint is available."""
        if not self.client:
            self.client = self._create_client()
        return self.client.test_connection()
