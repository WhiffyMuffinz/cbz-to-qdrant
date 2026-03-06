"""CLI entry point for CBZ processor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cbz_processor.config.config import config
from cbz_processor.pipeline import Pipeline
from cbz_processor.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=" CBZ file processor with Qdrant vector storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "root_path",
        type=str,
        help="Root directory containing CBZ files",
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to checkpoint file for resume",
    )

    parser.add_argument(
        "--reset",
        "-r",
        action="store_true",
        help="Reset checkpoint and start fresh",
    )

    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show what would be processed without running",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        root_path = Path(args.root_path)
        if not root_path.exists():
            print(f"Error: Root path not found: {root_path}", file=sys.stderr)
            return 1

        if args.reset and Path(args.checkpoint or config.CHECKPOINT_FILE).exists():
            Path(args.checkpoint or config.CHECKPOINT_FILE).unlink()

        pipeline = Pipeline(
            root_path=root_path,
            checkpoint_file=args.checkpoint,
        )

        if args.dry_run:
            from cbz_processor.storage.checkpoint import CheckpointManager
            from cbz_processor.utils.file_discovery import discover_cbz_files

            all_files = discover_cbz_files(root_path)
            checkpoint_path = Path(args.checkpoint or config.CHECKPOINT_FILE)
            checkpoint = CheckpointManager(checkpoint_path)
            remaining = checkpoint.get_remaining_files(all_files)
            processed = len(all_files) - len(remaining)
            print(f"Found {len(all_files)} CBZ files")
            print(f"Processed: {processed}")
            print(f"Remaining: {len(remaining)}")
            return 0

        pipeline.run()
        return 0

    except KeyboardInterrupt:
        print("\n interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        logger = setup_logging()
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
