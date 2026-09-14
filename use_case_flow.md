**MIRA Use Case — Architecture & Invocation Flow

**Overview**
- **Purpose:** Explain how the MIRA (Maintenance Intelligence & Resolution Assistant) use case is implemented so a new developer can follow the code, run it locally, and reason about the end-to-end flow from a user issue text to ranked suggestions and optional grounded synthesis.
- **Scope:** Frontend → API → retrieval (embeddings + FAISS) → optional LLM synthesis → feedback persistence.

**Architecture (high level)**
- **Frontend:** single-page UI that posts issue text and receives suggestions + optional synthesis.
- **API (FastAPI):** exposes `/suggest`, `/feedback`, `/health` and initializes service singletons at startup.
- **Service layer:** `services` contains orchestrators and business logic (retriever + synthesizer).
- **Adapters:** concrete implementations for embedding provider, LLM provider, vector store, and feedback store (adapter pattern via `services/interfaces.py`).
- **Data & Index:** `data/corpus.jsonl` and `data/vector_store/` (FAISS index + metadata).

**Mermaid Flowchart**
```mermaid
flowchart LR
  A[User (frontend)] -->|POST /suggest| B[FastAPI `services.api`]
  B --> C[services.config -> clients]
  B --> D[Retriever `services.retriever`]
  D --> E[Embedding client (adapter)]
  E --> F[Vector store (FAISS) adapter]
  F --> D
  D -->|top-k suggestions| B
  B -->|include_synthesis| G[Synthesizer `services.synthesizer`]
  G --> H[LLM client (adapter)]
  H --> G
  G --> B
  B --> A
  A -->|user feedback| I[`/feedback`]
  I --> J[Feedback store adapter (SQLite)]
```

**Sequence & File-level Invocation Order**
1. Frontend: [frontend/index.html](frontend/index.html) — user enters issue text and clicks "Get Suggestions". It sends a POST to `/suggest`.
2. API entrypoint: [services/api.py](services/api.py)
   - Receives request body (issue_text, top_k, include_synthesis, optional filters).
   - Uses `services.config` to obtain concrete `EmbeddingClient`, `LLMClient`, `VectorStore`, and `FeedbackStore` singletons.
   - Calls the `Retriever` instance to perform retrieval.
3. Config/Factory: [services/config.py](services/config.py)
   - Reads env vars (`MIRA_EMBEDDING_BACKEND`, `MIRA_LLM_BACKEND`, etc.) and returns adapters.
4. Retriever: [services/retriever.py](services/retriever.py)
   - Normalizes input and parameters.
   - Calls embedding client: `embedding_client.embed_batch([issue_text])` (see `services/adapters/embedding_genai_hub.py`).
   - Calls vector store search: `vector_store.search(vector, top_k)` (see `services/adapters/vector_store_faiss.py`).
   - Re-ranks results, computes `confidence`, `similarity_score`, and builds `Suggestion` objects.
   - Returns top-K suggestion list to API.
5. Optionally, Synthesizer: [services/synthesizer.py](services/synthesizer.py)
   - Builds a grounded prompt using top suggestions and their metadata.
   - Calls LLM client: `llm_client.generate(prompt)` (see `services/adapters/llm_genai_hub.py` or `llm_openai.py`).
   - Returns 2–4 sentence grounded synthesis referencing suggestion notification IDs.
6. API Response: [services/api.py](services/api.py)
   - Combines suggestions and optional synthesis into structured JSON and returns to frontend.
7. Feedback: [services/api.py -> /feedback](services/api.py)
   - Frontend or user actions call `/feedback` with selected suggestion(s) and rating.
   - API persists feedback via `FeedbackStore` adapter (see `services/adapters/feedback_store_sqlite.py`).

**Key Files & What They Do**
- **`frontend/index.html`** — Minimal UI; posts to `/suggest` and renders results.
- **`services/api.py`** — FastAPI app, route handlers, startup initialization, CORS middleware for local frontend.
- **`services/config.py`** — Factory logic for wiring concrete adapters using env vars.
- **`services/interfaces.py`** — Abstract interfaces: `EmbeddingClient`, `VectorStore`, `LLMClient`, `FeedbackStore`, and `Suggestion`/response schemas.
- **`services/retriever.py`** — Convert issue text → embedding → FAISS search → rerank → Suggestion objects.
- **`services/synthesizer.py`** — Build grounded LLM prompt and call LLM adapter for concise synthesis.
- **`services/adapters/embedding_genai_hub.py`** — Embedding adapter using SAP Gen AI Hub (or chosen provider).
- **`services/adapters/llm_genai_hub.py`** — LLM adapter using SAP Gen AI Hub (or chosen provider) with optional Amazon/Claude logic.
- **`services/adapters/embedding_openai.py`**, **`services/adapters/llm_openai.py`** — Alternative OpenAI-compatible adapters (if present/configured).
- **`services/adapters/vector_store_faiss.py`** — FAISS index loading/searching and metadata handling.
- **`scripts/build_index.py`** — Script to embed all records in `data/corpus.jsonl` and write `data/vector_store/index.faiss` + metadata.
- **`data/corpus.jsonl`** — Source documents (notifications/cases) used to build the vector index.
- **`data/vector_store/`** — Directory holding `index.faiss` and `index.meta.json`.
- **`services/adapters/feedback_store_sqlite.py`** — Local SQLite feedback persistence adapter.
- **`tests/evaluate.py`** — Evaluation helper to run self-retrieval and export labeled samples.

**Type/Schema Notes & Gotchas**
- Metadata types: normalize numeric IDs (order, equipment) to strings when returning `Suggestion` objects; mismatches can break FastAPI response validation.
- Provider optional imports: adapters that depend on provider-specific SDKs (Amazon/Claude, Gen AI Hub) guard imports so unused providers do not crash startup.
- Embedding dimensionality: ensure embedding client returns vectors of expected length (e.g., 1536) consistent with index.
- Local dev ports: API often runs on port `8000` or `8001`; confirm `frontend/index.html` target matches `services.api` port.

**How to run locally (quick)**
```bash
# (1) Create and activate venv with Python 3.12.x
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (2) Build index (if not already present)
python3 scripts/build_index.py
# (3) Start API
uv run uvicorn services.api:app --reload --port 8001
# (4) Serve frontend (optional)
cd frontend && python3 -m http.server 8080
# (5) Use the UI at http://localhost:8080 or cURL:
curl -X POST http://127.0.0.1:8001/suggest -H "Content-Type: application/json" -d '{"issue_text":"Troca de Olio hidraulico","top_k":3,"include_synthesis":true}'
```

**Next recommended developer tasks**
- Run `tests/evaluate.py` to obtain similarity score distributions and calibrate `CONFIDENCE_THRESHOLDS` in `services/retriever.py`.
- Add a small README snippet linking to this file for quick onboarding.
- Add CI checks that the index is present or `scripts/build_index.py` is executed in setup.

If you'd like, I can also: generate a short README snippet linking to this document, or produce a sequence diagram PNG for presentations.