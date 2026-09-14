"""
Configuration and backend factory for MIRA.

This is the ONLY file that knows which concrete adapters are active.
All other modules import from services.interfaces and get instances
via the get_* functions below.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from services.interfaces import EmbeddingClient, VectorStore, FeedbackStore, LLMClient

# Load environment variables from .env file
load_dotenv()

# --- Backend selection via environment variables ---
EMBEDDING_BACKEND = os.getenv("MIRA_EMBEDDING_BACKEND", "openai")
LLM_BACKEND = os.getenv("MIRA_LLM_BACKEND", "openai")
VECTOR_STORE_BACKEND = os.getenv("MIRA_VECTOR_STORE_BACKEND", "faiss")
FEEDBACK_STORE_BACKEND = os.getenv("MIRA_FEEDBACK_STORE_BACKEND", "sqlite")

# --- Backend-specific configuration ---
VECTOR_STORE_PATH = os.getenv("MIRA_VECTOR_STORE_PATH", "data/vector_store/index")
FEEDBACK_DB_PATH = os.getenv("MIRA_FEEDBACK_DB_PATH", "data/feedback.db")


def get_embedding_client() -> EmbeddingClient:
    """
    Get the configured embedding client.
    
    Returns:
        An EmbeddingClient implementation.
    
    Raises:
        ValueError: If unknown backend is configured.
        NotImplementedError: If backend is not yet implemented.
    """
    if EMBEDDING_BACKEND == "openai":
        from services.adapters.embedding_openai import OpenAIEmbeddingClient
        return OpenAIEmbeddingClient()
    if EMBEDDING_BACKEND == "genai_hub":
        from services.adapters.embedding_genai_hub import GenAIHubEmbeddingClient
        return GenAIHubEmbeddingClient()
    raise ValueError(f"Unknown MIRA_EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")


def get_llm_client() -> Optional[LLMClient]:
    """
    Get the configured LLM client, or None if synthesis is disabled.
    
    Returns:
        An LLMClient implementation, or None if MIRA_LLM_BACKEND=none.
    
    Raises:
        ValueError: If unknown backend is configured.
        NotImplementedError: If backend is not yet implemented.
    """
    if LLM_BACKEND == "none":
        return None
    if LLM_BACKEND == "openai":
        from services.adapters.llm_openai import OpenAILLMClient
        return OpenAILLMClient()
    if LLM_BACKEND == "genai_hub":
        from services.adapters.llm_genai_hub import GenAIHubLLMClient
        # Default to OpenAI model; can be changed via env var if needed
        model_provider = os.getenv("MIRA_GENAI_HUB_MODEL_PROVIDER", "openai")
        return GenAIHubLLMClient(model_provider=model_provider)
    raise ValueError(f"Unknown MIRA_LLM_BACKEND: {LLM_BACKEND}")


def get_vector_store() -> VectorStore:
    """
    Get the configured vector store, loading persisted index.
    
    Returns:
        A VectorStore implementation with loaded index.
    
    Raises:
        ValueError: If unknown backend is configured.
        NotImplementedError: If backend is not yet implemented.
    """
    if VECTOR_STORE_BACKEND == "faiss":
        from services.adapters.vector_store_faiss import FaissVectorStore
        store = FaissVectorStore()
        store.load(VECTOR_STORE_PATH)
        return store
    if VECTOR_STORE_BACKEND == "hana":
        raise NotImplementedError("adapters/vector_store_hana.py - needs BTP access")
    raise ValueError(f"Unknown MIRA_VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}")


def get_feedback_store() -> FeedbackStore:
    """
    Get the configured feedback store.
    
    Returns:
        A FeedbackStore implementation.
    
    Raises:
        ValueError: If unknown backend is configured.
        NotImplementedError: If backend is not yet implemented.
    """
    if FEEDBACK_STORE_BACKEND == "sqlite":
        from services.adapters.feedback_store_sqlite import SqliteFeedbackStore
        return SqliteFeedbackStore(FEEDBACK_DB_PATH)
    if FEEDBACK_STORE_BACKEND == "hana":
        raise NotImplementedError("adapters/feedback_store_hana.py - needs BTP access")
    raise ValueError(f"Unknown MIRA_FEEDBACK_STORE_BACKEND: {FEEDBACK_STORE_BACKEND}")