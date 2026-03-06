"""Checkpoint and state management for resume capability."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


class CheckpointManager:
    """Manages checkpoint state for resumable processing."""

    def __init__(self, checkpoint_file: Path | str = "data/checkpoint.json"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint JSON file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_checkpoint()

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load checkpoint from file or create new state."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception):
                pass
        return self._create_initial_state()

    def _create_initial_state(self) -> dict[str, Any]:
        """Create initial checkpoint state."""
        return {
            "started_at": None,
            "last_updated": None,
            "cbz_files_processed": [],
            "cbz_files_failed": [],
            "images_extracted": 0,
            "embeddings_generated": 0,
            "points_inserted": 0,
            "last_cbz_file": None,
            "current_batch": 0,
            "status": "pending",
        }

    def get_remaining_files(self, all_files: list[str]) -> list[str]:
        """Get list of files that haven't been processed yet.

        Args:
            all_files: List of all CBZ files to process

        Returns:
            List of unprocessed file paths
        """
        processed = set(self.state["cbz_files_processed"])
        return [f for f in all_files if f not in processed]

    def update_checkpoint(
        self,
        cbz_file: str | None = None,
        images_extracted: int = 0,
        embeddings_generated: int = 0,
        points_inserted: int = 0,
        status: str = "running",
        success: bool = True,
    ) -> None:
        """Update checkpoint state.

        Args:
            cbz_file: Path to recently processed CBZ file
            images_extracted: Number of new images extracted
            embeddings_generated: Number of new embeddings
            points_inserted: Number of new DB points
            status: Processing status
            success: Whether operation succeeded
        """
        if cbz_file and cbz_file not in self.state["cbz_files_processed"]:
            if success:
                self.state["cbz_files_processed"].append(cbz_file)
            else:
                self.state["cbz_files_failed"].append(cbz_file)

        self.state["images_extracted"] += images_extracted
        self.state["embeddings_generated"] += embeddings_generated
        self.state["points_inserted"] += points_inserted
        self.state["current_batch"] += 1
        self.state["last_updated"] = datetime.now().isoformat()
        self.state["status"] = status

        if cbz_file:
            self.state["last_cbz_file"] = cbz_file

        if self.state["started_at"] is None:
            self.state["started_at"] = datetime.now().isoformat()

        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Save checkpoint to file."""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_summary(self) -> dict[str, Any]:
        """Get processing summary."""
        return {
            "total_files": len(self.state["cbz_files_processed"])
            + len(self.get_remaining_files([])),
            "processed": len(self.state["cbz_files_processed"]),
            "failed": len(self.state["cbz_files_failed"]),
            "images": self.state["images_extracted"],
            "embeddings": self.state["embeddings_generated"],
            "points": self.state["points_inserted"],
            "status": self.state["status"],
        }

    def reset(self) -> None:
        """Reset checkpoint for fresh run."""
        self.state = self._create_initial_state()
        self._save_checkpoint()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
