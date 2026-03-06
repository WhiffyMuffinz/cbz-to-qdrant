"""Utility functions subpackage."""

from cbz_processor.utils.file_discovery import (
    discover_cbz_files,
    discover_cbz_files_generator,
    get_cbz_file_count,
)
from cbz_processor.utils.hash_utils import compute_file_hash, compute_file_hashes_batch
from cbz_processor.utils.logging_utils import (
    JSONFormatter,
    log_debug,
    log_error,
    log_info,
    log_warning,
    setup_logging,
)

__all__ = [
    "discover_cbz_files",
    "discover_cbz_files_generator",
    "get_cbz_file_count",
    "compute_file_hash",
    "compute_file_hashes_batch",
    "JSONFormatter",
    "log_debug",
    "log_error",
    "log_info",
    "log_warning",
    "setup_logging",
]
