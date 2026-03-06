"""Main processing pipeline workers."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from cbz_processor.config.config import config
from cbz_processor.services.cbz_extractor import (
    extract_comic_info,
    extract_images,
    validate_cbz,
)
from cbz_processor.utils.hash_utils import compute_file_hash
from cbz_processor.utils.logging_utils import log_error, log_info, setup_logging

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import NoReturn

    from cbz_processor.models.data_models import CBZProcessingResult


class CBZProcessor:
    """Processes individual CBZ files."""

    def __init__(self):
        """Initialize CBZ processor."""
        self.logger = setup_logging()
        self.vllm_available = True

    def process_cbz_file(
        self,
        cbz_path: str | Path,
    ) -> CBZProcessingResult | None:
        """Process single CBZ file.

        Args:
            cbz_path: Path to CBZ file

        Returns:
            Processing result or None if failed
        """
        cbz_path = Path(cbz_path)
        if not validate_cbz(cbz_path):
            log_error(
                self.logger,
                f"Invalid CBZ file (corrupt or not ZIP): {cbz_path}",
            )
            return None

        try:
            file_hash = compute_file_hash(cbz_path)
            comic_info = extract_comic_info(cbz_path)

            images = []
            image_data_list = []

            images_data_list = []

            for filename, img_data, resolution in extract_images(cbz_path):
                image_bytes = img_data.read()
                images_data_list.append(image_bytes)
                images.append(
                    {
                        "filename": filename,
                        "width": resolution[0],
                        "height": resolution[1],
                        "data": image_bytes,
                    },
                )

            return {
                "source_path": str(cbz_path),
                "hash": file_hash,
                "comic_info": comic_info.model_dump() if comic_info else {},
                "images": images,
                "embeddings": [],
            }
        except Exception as e:
            log_error(
                self.logger,
                f"Error processing {cbz_path}: {e}",
                traceback=traceback.format_exc(),
            )
            return None


def process_cbz_batch(
    cbz_files: list[str],
    worker_count: int = config.WORKER_PARALLELISM,
) -> Iterator[tuple[str, dict | None]]:
    """Process multiple CBZ files in parallel.

    Args:
        cbz_files: List of CBZ file paths
        worker_count: Number of parallel workers

    Yields:
        Tuple of (filepath, result dict or None)
    """
    processor = CBZProcessor()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(processor.process_cbz_file, f): f
            for f in cbz_files
        }

        for future in as_completed(futures):
            filepath = futures[future]
            try:
                result = future.result()
                yield str(filepath), result
            except Exception as e:
                log_error(
                    processor.logger,
                    f"Worker error for {filepath}: {e}",
                )
                yield str(filepath), None
