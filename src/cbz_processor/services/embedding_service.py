"""Embedding generation service using vLLM."""

from __future__ import annotations

import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import NoReturn


class EmbeddingClient:
    """Client for generating embeddings via vLLM API."""

    def __init__(
        self,
        endpoint: str = "http://localhost:7997",
        model: str = "Qwen/Qwen3-VL-Embedding-2B",
        max_retries: int = 3,
    ):
        """Initialize embedding client.

        Args:
            endpoint: vLLM API endpoint URL
            model: Model name for embeddings
            max_retries: Maximum retry attempts
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.last_error: str | None = None

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    def _call_vllm_api(self, image_bytes: bytes, filename: str = "") -> list[float] | None:
        """Call vLLM embedding API endpoint.

        Args:
            image_bytes: Image data
            filename: Image filename for context

        Returns:
            Embedding vector or None on failure
        """
        image_b64 = self._encode_image_to_base64(image_bytes)
        
        # Use vision format with base64 image
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this image briefly"
                        }
                    ]
                }
            ]
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/v1/embeddings",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()

                result = response.json()
                return result["data"][0]["embedding"]
            except (requests.RequestException, KeyError, Exception) as e:
                self.last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                continue

        return None

    def generate_embedding(
        self,
        image_bytes: bytes,
        filename: str = "",
    ) -> list[float] | None:
        """Generate single embedding for image.

        Args:
            image_bytes: Image data
            filename: Image filename for context

        Returns:
            Embedding vector or None if failed
        """
        return self._call_vllm_api(image_bytes, filename)

    def generate_embeddings_batch(
        self,
        image_data_list: list[bytes],
    ) -> list[list[float]]:
        """Generate embeddings for batch of images asynchronously.

        Args:
            image_data_list: List of image data

        Returns:
            List of embedding vectors
        """
        embeddings: list[list[float]] = []

        if not image_data_list:
            return embeddings

        with ThreadPoolExecutor(max_workers=len(image_data_list)) as executor:
            future_to_idx = {
                executor.submit(self.generate_embedding, img): idx
                for idx, img in enumerate(image_data_list)
            }

            results: list[list[float] | None] = [None] * len(image_data_list)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embedding = future.result()
                    if embedding is not None:
                        results[idx] = embedding
                except Exception:
                    results[idx] = None

        embeddings = [e for e in results if e is not None]
        return embeddings

    def is_available(self) -> bool:
        """Check if vLLM endpoint is reachable."""
        try:
            response = requests.get(
                f"{self.endpoint}/v1/models",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def test_connection(self) -> bool:
        """Test vLLM connection with a simple input."""
        try:
            response = requests.post(
                f"{self.endpoint}/v1/embeddings",
                json={
                    "model": self.model,
                    "input": "test",
                },
                timeout=10,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
