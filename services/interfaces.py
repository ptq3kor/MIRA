"""
Abstract interfaces for MIRA backend components.

All business logic (retriever.py, synthesizer.py, api.py) must depend only on
these interfaces, never on concrete implementations. This enables swapping
backends (OpenAI -> SAP Generative AI Hub, FAISS -> HANA, SQLite -> HANA)
by changing only config.py and adding new adapter files.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class EmbeddingClient(ABC):
    """Interface for text embedding backends."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Return one L2-normalized dense vector per input text, in order.
        Normalization is mandatory so inner product == cosine similarity
        in every VectorStore implementation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension (e.g., 1536 for text-embedding-3-small)."""
        raise NotImplementedError


@dataclass
class VectorSearchResult:
    """A single search result from the vector store."""
    id: str
    text: str
    metadata: dict
    score: float


class VectorStore(ABC):
    """Interface for vector similarity search backends."""

    @abstractmethod
    def build(self, ids: list[str], vectors: list[list[float]],
              texts: list[str], metadatas: list[dict]) -> None:
        """Build the index from the full corpus."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int,
               filters: Optional[dict] = None) -> list[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: L2-normalized query embedding.
            top_k: Number of results to return.
            filters: Optional exact-match constraints, e.g. {"equipment": "12002267"}.
                    If filters match nothing, fall back to unfiltered search rather
                    than returning empty.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        raise NotImplementedError


@dataclass
class FeedbackEvent:
    """A single feedback event from a technician."""
    query_issue_text: str
    suggested_notification_id: Optional[str]
    action: str  # "ACCEPT" | "OVERRIDE" | "REJECT"
    technician_id: str  # SAP personnel/user ID - never a display name
    override_text: Optional[str] = None


class FeedbackStore(ABC):
    """Interface for feedback persistence."""

    @abstractmethod
    def log(self, event: FeedbackEvent) -> str:
        """Log a feedback event. Returns the feedback_id."""
        raise NotImplementedError

    @abstractmethod
    def recent(self, limit: int = 50) -> list[dict]:
        """Retrieve recent feedback events."""
        raise NotImplementedError


class LLMClient(ABC):
    """
    Text generation interface. Used exclusively by services/synthesizer.py
    to add an optional grounded summary ON TOP of retrieval results.
    Generation is additive, never a replacement for the traceable ranked list.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from system + user prompts."""
        raise NotImplementedError