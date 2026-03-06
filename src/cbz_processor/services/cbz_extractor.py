"""CBZ file extraction utilities."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from cbz_processor.models.data_models import CBZProcessingResult, ComicInfo, ImageMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator


def is_image_file(filename: str) -> bool:
    """Check if a filename represents an image file."""
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    return Path(filename).suffix.lower() in image_extensions


def is_metadata_file(filename: str) -> bool:
    """Check if a filename represents a metadata file."""
    metadata_patterns = {"comicinfo.xml", "metainfo.xml", "metadata.xml"}
    return Path(filename).name.lower() in metadata_patterns


def extract_comic_info(zip_path: Path) -> ComicInfo | None:
    """Extract comic metadata from ComicInfo.xml inside a CBZ file.

    Args:
        zip_path: Path to CBZ file

    Returns:
        ComicInfo model or None if not found
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            comicinfo = None
            for info in zf.infolist():
                if info.filename.lower() == "comicinfo.xml":
                    comicinfo = info
                    break

            if not comicinfo:
                return None

            with zf.open(comicinfo) as f:
                return parse_comicinfo_xml(f.read())
    except (zipfile.BadZipFile, Exception):
        return None


def parse_comicinfo_xml(xml_content: bytes | str) -> ComicInfo:
    """Parse ComicInfo.xml content into ComicInfo model.

    Args:
        xml_content: XML content as bytes or string

    Returns:
        ComicInfo model with extracted metadata
    """
    import xml.etree.ElementTree as ET

    if isinstance(xml_content, bytes):
        xml_content = xml_content.decode("utf-8")

    try:
        root = ET.fromstring(xml_content)
        comic_info = {}

        for child in root:
            tag = child.tag.lower()
            value = child.text.strip() if child.text else None

            if value:
                if tag == "genre":
                    comic_info[tag] = [g.strip() for g in value.split(",")]
                elif tag in ("writer", "translator", "keycharacters"):
                    comic_info[tag] = [n.strip() for n in value.split(",")]
                else:
                    comic_info[tag] = value

        if "number" in comic_info:
            comic_info["number"] = comic_info["number"].zfill(3)

        return ComicInfo(**comic_info)
    except (ET.ParseError, Exception):
        return ComicInfo()


def extract_images(
    zip_path: Path,
    supported_types: tuple[str, ...] = ("png", "jpeg", "jpg", "gif", "webp"),
) -> Iterator[tuple[str, BytesIO, tuple[int, int]]]:
    """Extract images from CBZ file.

    Args:
        zip_path: Path to CBZ file
        supported_types: Supported image extensions

    Yields:
        Tuple of (filename, image_bytes, (width, height))
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if not is_image_file(info.filename):
                    continue

                try:
                    with zf.open(info) as f:
                        img_data = BytesIO(f.read())
                        with Image.open(img_data) as img:
                            width, height = img.size
                        img_data.seek(0)
                        yield info.filename, img_data, (width, height)
                except (OSError, Exception):
                    continue
    except (zipfile.BadZipFile, Exception):
        return


def validate_cbz(filepath: Path) -> bool:
    """Validate that a file is a valid CBZ archive.

    Args:
        filepath: Path to potential CBZ file

    Returns:
        True if valid CBZ, False otherwise
    """
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.testzip()
        return True
    except (zipfile.BadZipFile, Exception):
        return False
