"""
Build the local vector index from corpus.jsonl using the configured
embedding backend (OpenAI - see services/config.py).

Usage: python -m scripts.build_index
"""

import json
from pathlib import Path

from services import config
from services.adapters.vector_store_faiss import FaissVectorStore

CORPUS_PATH = Path("data/corpus.jsonl")
INDEX_OUT = Path("data/vector_store/index")


def load_corpus(path: Path) -> list[dict]:
    """Load corpus records from JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    """Build FAISS index from corpus using OpenAI embeddings."""
    records = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(records)} corpus records from {CORPUS_PATH}")

    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    metadatas = [r["metadata"] for r in records]

    embedder = config.get_embedding_client()
    print("Embedding documents via OpenAI...")
    print("This calls the OpenAI API in batches and incurs a small cost - "
          "check current published pricing before running at full corpus size.")
    vectors = embedder.embed_batch(texts)
    print(f"Done. Embedding dimension: {embedder.dimension}")

    store = FaissVectorStore()
    store.build(ids=ids, vectors=vectors, texts=texts, metadatas=metadatas)
    store.save(str(INDEX_OUT))
    print(f"Saved FAISS index to {INDEX_OUT}.faiss / {INDEX_OUT}.meta.json")


if __name__ == "__main__":
    main()