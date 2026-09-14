"""
Retriever module: embed query -> search -> rerank -> return top-k suggestions.

Backend-agnostic: uses EmbeddingClient and VectorStore interfaces from config.
"""

from dataclasses import dataclass
from typing import Optional

from services.interfaces import EmbeddingClient, VectorStore, VectorSearchResult


@dataclass
class Suggestion:
    """A single ranked suggestion for the technician."""
    rank: int
    issue_text: str
    resolution_text: str
    source_notification: str
    source_order: str
    equipment: Optional[str]
    func_location: str
    similarity_score: float
    confidence: str


# Confidence thresholds for cosine similarity scores (to be calibrated per embedding model)
# These are PLACEHOLDERS - must be calibrated against text-embedding-3-small score distribution
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.80,    # >= 0.80 -> HIGH confidence
    "MEDIUM": 0.65,  # >= 0.65 -> MEDIUM confidence
    # < 0.65 -> LOW confidence
}

# Rerank weights
RERANK_WEIGHTS = {
    "vector": 0.7,
    "equipment_match": 0.2,
    "func_location_match": 0.1,
}


class Retriever:
    """
    Retrieves and reranks similar historical maintenance cases.
    
    Flow:
    1. Embed query text using configured EmbeddingClient
    2. Search vector store for top-20 candidates (with optional equipment/location filters)
    3. Rerank by weighted score: vector similarity (0.7) + equipment match (0.2) + location match (0.1)
    4. Return top-k Suggestion objects with confidence labels
    """

    def __init__(self, embedder: EmbeddingClient, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def _parse_issue_resolution(self, text: str) -> tuple[str, str]:
        """
        Parse the combined text back into issue and resolution parts.
        
        Expected format: "Issue: <issue>\nResolution: <resolution>"
        """
        if "\nResolution: " in text:
            issue_part, resolution_part = text.split("\nResolution: ", 1)
            issue = issue_part.replace("Issue: ", "", 1)
            return issue.strip(), resolution_part.strip()
        return text.strip(), ""

    def _compute_rerank_score(self, result: VectorSearchResult,
                               query_equipment: Optional[str],
                               query_func_location: Optional[str]) -> float:
        """
        Compute rerank score combining vector similarity with metadata matches.
        
        Score = 0.7 * vector_score + 0.2 * equipment_match + 0.1 * location_match
        """
        vector_score = result.score
        
        equipment_match = 0.0
        if query_equipment and result.metadata.get("equipment"):
            equipment_match = 1.0 if result.metadata["equipment"] == query_equipment else 0.0
        
        location_match = 0.0
        if query_func_location and result.metadata.get("func_location"):
            location_match = 1.0 if result.metadata["func_location"] == query_func_location else 0.0
        
        return (
            RERANK_WEIGHTS["vector"] * vector_score +
            RERANK_WEIGHTS["equipment_match"] * equipment_match +
            RERANK_WEIGHTS["func_location_match"] * location_match
        )

    def _get_confidence(self, score: float) -> str:
        """Map cosine similarity score to confidence label."""
        if score >= CONFIDENCE_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif score >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def retrieve(self, issue_text: str, equipment: Optional[str] = None,
                 func_location: Optional[str] = None, top_k: int = 3) -> list[Suggestion]:
        """
        Retrieve and rerank similar past maintenance cases.
        
        Args:
            issue_text: The technician's reported issue text.
            equipment: Optional equipment number to filter/boost.
            func_location: Optional functional location to filter/boost.
            top_k: Number of suggestions to return (default 3).
        
        Returns:
            List of Suggestion objects, ranked by rerank score descending.
        """
        # Step 1: Embed query
        query_vector = self.embedder.embed_batch([issue_text])[0]
        
        # Step 2: Search vector store (fetch more for reranking)
        search_k = max(top_k * 5, 20)
        filters = {}
        if equipment:
            filters["equipment"] = equipment
        if func_location:
            filters["func_location"] = func_location
        
        results = self.vector_store.search(query_vector, search_k, filters if filters else None)
        
        if not results:
            return []
        
        # Step 3: Rerank
        scored_results = []
        for r in results:
            rerank_score = self._compute_rerank_score(r, equipment, func_location)
            scored_results.append((rerank_score, r))
        
        # Sort by rerank score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Step 4: Build Suggestion objects
        suggestions = []
        for rank, (rerank_score, result) in enumerate(scored_results[:top_k], 1):
            issue, resolution = self._parse_issue_resolution(result.text)
            confidence = self._get_confidence(result.score)
            equipment = result.metadata.get("equipment")
            
            suggestions.append(Suggestion(
                rank=rank,
                issue_text=issue,
                resolution_text=resolution,
                source_notification=result.id,
                source_order=str(result.metadata.get("order_number", "")),
                equipment=str(equipment) if equipment is not None else None,
                func_location=result.metadata.get("func_location_desc", ""),
                similarity_score=result.score,
                confidence=confidence,
            ))
        
        return suggestions