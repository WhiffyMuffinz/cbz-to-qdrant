"""File discovery utilities for CBZ processing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def discover_cbz_files(
    root_path: Path | str,
    recursive: bool = True,
) -> list[str]:
    """Discover all CBZ files in the given directory path.

    Args:
        root_path: Root directory to search from
        recursive: Whether to search subdirectories

    Returns:
        List of absolute paths to CBZ files
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    cbz_files: list[str] = []

    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(".cbz"):
                    cbz_files.append(str(Path(dirpath) / filename))
    else:
        cbz_files = [
            str(f)
            for f in root.glob("*.cbz")
            if f.is_file()
        ]

    return sorted(cbz_files)


def discover_cbz_files_generator(
    root_path: Path | str,
) -> Iterator[str]:
    """Generator that yields CBZ files one at a time for memory efficiency.

    Args:
        root_path: Root directory to search from

    Yields:
        Absolute path to each CBZ file
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".cbz"):
                yield str(Path(dirpath) / filename)


def get_cbz_file_count(root_path: Path | str) -> int:
    """Get count of CBZ files without loading all paths into memory.

    Args:
        root_path: Root directory to search from

    Returns:
        Number of CBZ files
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    count = 0
    for _ in discover_cbz_files_generator(root):
        count += 1

    return count
