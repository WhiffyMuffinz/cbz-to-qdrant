"""Main pipeline orchestrator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from cbz_processor.config.config import config
from cbz_processor.storage.checkpoint import CheckpointManager
from cbz_processor.utils.file_discovery import discover_cbz_files
from cbz_processor.utils.logging_utils import (
    log_error,
    log_info,
    log_warning,
    setup_logging,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import NoReturn

    from cbz_processor.models.data_models import CBZProcessingResult


class Pipeline:
    """Main pipeline orchestrator for CBZ processing."""

    def __init__(
        self,
        root_path: str | Path,
        checkpoint_file: str | Path | None = None,
    ):
        """Initialize pipeline.

        Args:
            root_path: Root directory containing CBZ files
            checkpoint_file: Optional checkpoint file path
        """
        self.root_path = Path(root_path)
        self.checkpoint_file = Path(checkpoint_file or config.CHECKPOINT_FILE)

        self.logger = setup_logging()
        self.checkpoint = CheckpointManager(self.checkpoint_file)

        self.embeddings_pending: list[dict] = []
        self.total_start_time = time.time()

    def run(self) -> None:
        """Run the complete processing pipeline."""
        all_files = discover_cbz_files(self.root_path)
        remaining_files = self.checkpoint.get_remaining_files(all_files)

        log_info(self.logger, f"Found {len(remaining_files)} CBZ files to process")

        if not remaining_files:
            log_info(self.logger, "No files to process")
            return

        self.checkpoint.update_checkpoint(status="running")

        total_files = len(remaining_files)
        batch_count = 0

        with tqdm(
            total=total_files,
            desc="Processing CBZ files",
            unit="file",
        ) as pbar:
            for filepath, result in self._process_cbz_batch(remaining_files):
                if result:
                    self.embeddings_pending.append(result)
                    self._process_current_batch()
                    batch_count += 1

                pbar.update(1)
                pbar.set_postfix(
                    images=self.checkpoint.state["images_extracted"],
                    points=self.checkpoint.state["points_inserted"],
                )

        self.checkpoint.update_checkpoint(status="completed")
        elapsed = time.time() - self.total_start_time

        log_info(
            self.logger,
            "Pipeline completed",
            summary=self.checkpoint.get_summary(),
            elapsed_seconds=round(elapsed, 2),
            elapsed_minutes=round(elapsed / 60, 2),
        )

    def _process_cbz_batch(
        self,
        files: list[str],
    ) -> Iterator[tuple[str, dict | None]]:
        """Process CBZ files in batch."""
        from cbz_processor.workers.cbz_worker import CBZProcessor

        processor = CBZProcessor()

        for filepath in files:
            result = processor.process_cbz_file(filepath)
            yield filepath, result

    def _process_current_batch(self) -> None:
        """Process and insert current batch of embeddings."""
        results: list[dict] = self.embeddings_pending

        if not results:
            return

        embedding_processor = self._create_embedding_processor()

        if embedding_processor.is_available():
            processed_results = embedding_processor.process_batch(
                [  # type: ignore
                    result  # type: ignore
                    for result in results
                    if result  # type: ignore
                ],
            )
            results = processed_results

        qdrant_store = self._create_qdrant_store()
        points = []
        global_point_id = self.checkpoint.state.get("points_inserted", 0)

        for result in results:
            if not result:
                continue

            source_path = result.get("source_path", "")

            for idx, image in enumerate(result.get("images", [])):
                embedding = (
                    result.get("embeddings", [])[idx]
                    if idx < len(result.get("embeddings", []))
                    else None
                )

                if embedding:
                    point = qdrant_store.prepare_point(
                        source_path=source_path,
                        image_filename=image.get("filename", ""),
                        resolution=(
                            image.get("width", 0),
                            image.get("height", 0),
                        ),
                        comic_info=result.get("comic_info", {}),
                        embedding=embedding,
                        point_id=global_point_id,
                    )
                    points.append(point)
                    global_point_id += 1

        if points:
            inserted = qdrant_store.upsert_points(
                points,
                batch_size=config.QDRANT_BATCH_SIZE,
            )

            processed_files = [r.get("source_path", "") for r in results if r and r.get("source_path")]
            for cbz_file in processed_files:
                self.checkpoint.update_checkpoint(
                    cbz_file=cbz_file,
                    images_extracted=0,
                    embeddings_generated=0,
                    points_inserted=0,
                )

            self.checkpoint.update_checkpoint(
                cbz_file=None,
                images_extracted=sum(len(r.get("images", [])) for r in results if r),
                embeddings_generated=sum(
                    len(r.get("embeddings", [])) for r in results if r
                ),
                points_inserted=inserted,
            )

        self.embeddings_pending = []

    def _create_embedding_processor(self):
        """Create embedding batch processor."""
        from cbz_processor.workers.embedding_worker import (
            EmbeddingBatchProcessor,
        )

        return EmbeddingBatchProcessor()

    def _create_qdrant_store(self):
        """Create Qdrant store."""
        from cbz_processor.services.qdrant_store import QdrantStore

        return QdrantStore(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            collection_name=config.QDRANT_COLLECTION,
        )
