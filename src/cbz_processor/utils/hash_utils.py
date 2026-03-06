"""Hash utilities for file integrity tracking."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(filepath: Path | str, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity tracking.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)

    Returns:
        Hexadecimal hash string
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    hash_func = hashlib.new(algorithm)
    buffer_size = 65536

    with open(filepath, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hash_func.update(data)

    return hash_func.hexdigest()


def compute_file_hashes_batch(
    filepaths: list[Path | str],
    algorithm: str = "sha256",
) -> dict[str, str]:
    """Compute hashes for multiple files.

    Args:
        filepaths: List of file paths
        algorithm: Hash algorithm

    Returns:
        Dictionary mapping filepath to hash
    """
    return {
        str(Path(p)): compute_file_hash(p, algorithm)
        for p in filepaths
    }
