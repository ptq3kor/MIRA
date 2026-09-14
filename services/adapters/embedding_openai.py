"""
Embedding adapter using OpenAI's embeddings API.

Requires OPENAI_API_KEY environment variable.

API reference: POST https://api.openai.com/v1/embeddings
  body: {"model": "text-embedding-3-small", "input": ["text1", "text2", ...]}
  response: {"data": [{"embedding": [...], "index": 0}, ...]}

OpenAI's docs cap batched inputs around 2048 items per request and by
total token count - BATCH_SIZE below is deliberately conservative for
short technician text, and safely under both limits.
"""

import os
import numpy as np

from services.adapters._http_retry import post_with_retry
from services.interfaces import EmbeddingClient

BATCH_SIZE = 100


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embedding client using text-embedding-3-small by default."""

    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]  # fail loudly if missing
        self.model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self._dim = None

    @property
    def dimension(self) -> int:
        """Return embedding dimension. Unknown until first embed_batch() call."""
        if self._dim is None:
            raise RuntimeError("dimension unknown until embed_batch() has been called once")
        return self._dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            List of L2-normalized embedding vectors (float lists), in same order.
        """
        all_vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i:i + BATCH_SIZE]
            result = post_with_retry(
                "https://api.openai.com/v1/embeddings",
                json={"model": self.model, "input": chunk},
                headers=headers,
            )
            # OpenAI does not guarantee response order matches input order in
            # theory, but does in practice; sort by "index" defensively.
            sorted_data = sorted(result["data"], key=lambda d: d["index"])
            all_vectors.extend(d["embedding"] for d in sorted_data)

        matrix = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms  # REQUIRED - see EmbeddingClient contract
        self._dim = normalized.shape[1]
        return normalized.tolist()