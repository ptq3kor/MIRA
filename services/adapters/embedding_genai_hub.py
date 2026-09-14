"""
Embedding adapter using SAP Generative AI Hub (via gen-ai-hub-sdk).

Requires the following environment variables:
- AICORE_AUTH_URL
- AICORE_CLIENT_ID
- AICORE_CLIENT_SECRET
- AICORE_BASE_URL
- AICORE_RESOURCE_GROUP
- AICORE_OPENAI_EMBEDDING_DEPLOYMENT_ID
- AICORE_OPENAI_EMBEDDING_MODEL

The gen-ai-hub-sdk handles authentication and token refresh automatically.
"""

import os
import numpy as np
from typing import List

from services.interfaces import EmbeddingClient

# Lazy import to avoid hard dependency if not using this backend
try:
    from gen_ai_hub.proxy.native.openai import OpenAI
    from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
    GENAI_HUB_AVAILABLE = True
except ImportError:
    GENAI_HUB_AVAILABLE = False


class GenAIHubEmbeddingClient(EmbeddingClient):
    """SAP Generative AI Hub embedding client using text-embedding-3-small."""

    def __init__(self):
        if not GENAI_HUB_AVAILABLE:
            raise RuntimeError(
                "gen-ai-hub-sdk not installed. Run: pip install gen-ai-hub-sdk"
            )
        
        # Initialize the proxy client which handles auth
        self.proxy_client = get_proxy_client('gen-ai-hub')
        self.openai_client = OpenAI(proxy_client=self.proxy_client)
        
        self.deployment_id = os.environ["AICORE_OPENAI_EMBEDDING_DEPLOYMENT_ID"]
        self.model = os.environ["AICORE_OPENAI_EMBEDDING_MODEL"]
        self._dim = None
        
        # Batch size for embedding requests
        self.batch_size = 100

    @property
    def dimension(self) -> int:
        """Return embedding dimension. Unknown until first embed_batch() call."""
        if self._dim is None:
            raise RuntimeError("dimension unknown until embed_batch() has been called once")
        return self._dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using SAP AI Core deployment.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            List of L2-normalized embedding vectors (float lists), in same order.
        """
        all_vectors: List[List[float]] = []
        
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            
            # Call the embeddings API with deployment_id
            response = self.openai_client.embeddings.create(
                input=chunk,
                deployment_id=self.deployment_id,
                model=self.model,
            )
            
            # Sort by index to maintain order (defensive)
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_vectors.extend(d.embedding for d in sorted_data)

        # L2-normalize (required by EmbeddingClient contract)
        matrix = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        self._dim = normalized.shape[1]
        return normalized.tolist()