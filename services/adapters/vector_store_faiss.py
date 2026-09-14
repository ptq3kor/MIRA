"""
FAISS vector store implementation.

Backend-agnostic - doesn't care which embedding backend produced the vectors,
only requires L2-normalized float vectors of consistent dimension.
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from services.interfaces import VectorSearchResult, VectorStore


class FaissVectorStore(VectorStore):
    """FAISS-based vector store using inner product (cosine similarity for normalized vectors)."""

    def __init__(self):
        self.index: faiss.Index | None = None
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict] = []

    def build(self, ids: list[str], vectors: list[list[float]],
              texts: list[str], metadatas: list[dict]) -> None:
        """
        Build a new FAISS index from vectors.
        
        Args:
            ids: Document IDs.
            vectors: L2-normalized embedding vectors.
            texts: Original text content.
            metadatas: Document metadata dicts.
        """
        if not vectors:
            raise ValueError("Cannot build index with empty vectors")
        
        matrix = np.array(vectors, dtype=np.float32)
        dim = matrix.shape[1]
        
        # Use IndexFlatIP for exact inner product search (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)
        
        self.ids = ids
        self.texts = texts
        self.metadatas = metadatas

    def search(self, query_vector: list[float], top_k: int,
               filters: Optional[dict] = None) -> list[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: L2-normalized query embedding.
            top_k: Number of results to return.
            filters: Optional exact-match constraints (e.g., {"equipment": "12002267"}).
                    If filters match nothing, falls back to unfiltered search.
        
        Returns:
            List of VectorSearchResult sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded")
        
        # If filters provided, try filtered search first
        if filters:
            filtered_results = self._search_with_filters(query_vector, top_k, filters)
            if filtered_results:
                return filtered_results
            # Fall back to unfiltered if no matches
        
        # Unfiltered search
        query = np.array([query_vector], dtype=np.float32)
        scores, indices = self.index.search(query, min(top_k, len(self.ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(VectorSearchResult(
                id=self.ids[idx],
                text=self.texts[idx],
                metadata=self.metadatas[idx],
                score=float(score),
            ))
        return results

    def _search_with_filters(self, query_vector: list[float], top_k: int,
                             filters: dict) -> list[VectorSearchResult]:
        """
        Search with exact metadata filters.
        
        Since FAISS doesn't support filtered search natively, we do a
        larger unfiltered search and filter results in Python.
        """
        # Search more results to have candidates after filtering
        search_k = min(top_k * 10, len(self.ids))
        query = np.array([query_vector], dtype=np.float32)
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            metadata = self.metadatas[idx]
            if all(metadata.get(k) == v for k, v in filters.items()):
                results.append(VectorSearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    metadata=metadata,
                    score=float(score),
                ))
                if len(results) >= top_k:
                    break
        return results

    def save(self, path: str) -> None:
        """
        Persist index and metadata to disk.
        
        Creates two files:
        - {path}.faiss: FAISS binary index
        - {path}.meta.json: IDs, texts, metadatas
        """
        if self.index is None:
            raise RuntimeError("No index to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        
        meta = {
            "ids": self.ids,
            "texts": self.texts,
            "metadatas": self.metadatas,
        }
        with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """
        Load index and metadata from disk.
        
        Args:
            path: Base path (without .faiss/.meta.json extension).
        """
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.ids = meta["ids"]
        self.texts = meta["texts"]
        self.metadatas = meta["metadatas"]