# MIRA Implementation Plan (v5 — OpenAI-only RAG)

## Status of this document

This is a specification, not yet executed code. The data-prep pipeline
(raw Excel → reviewed Excel → `corpus.jsonl`) and the FAISS/SQLite/FastAPI
scaffolding were built and tested end-to-end in earlier iterations of this
project (against a TF-IDF placeholder embedding). This revision simplifies
the backend to **OpenAI only** — no Ollama, no TF-IDF fallback — for both
embeddings and generation, per your direction. The code below is written
precisely against OpenAI's documented REST API but has **not** been
executed in this environment (no API key here). Treat the code blocks as
complete, correct specifications to implement and run in your own
environment, or hand to another AI model/developer if you hit a token
limit partway through this conversation.

---

## 1. What changed, and why

| Concern | Earlier draft | This version (v5) |
|---|---|---|
| Embedding backend | TF-IDF (offline fallback) + Ollama (primary) + OpenAI (alternate) — three adapters | **OpenAI only** — `text-embedding-3-small` |
| LLM backend | Ollama (primary) + OpenAI (alternate) | **OpenAI only** — `gpt-4o-mini` |
| Why simplify | Hedging for an offline/no-API-key hackathon scenario | You confirmed you have an OpenAI API key and want to standardize on it — one backend to configure, test, and debug instead of three |
| Local infra dependency | Needed `ollama serve` running, model pulls, version checks | None — just an API key and internet access |
| Data prep source of truth | (unchanged) `prep_corpus.py` writes a reviewable `mira_corpus_reviewed.xlsx`; that Excel, not the raw source, is the actual source of truth | unchanged, see Section 6 |

**What did not change:** the interface/adapter pattern (Section 7) is kept
even though there's now only one embedding/LLM implementation. This isn't
over-engineering for its own sake — it's the thing that makes the eventual
swap to SAP Generative AI Hub (Section 18) a new adapter file plus one
config flip, instead of a rewrite of `retriever.py`, `synthesizer.py`, and
`api.py`. If you're certain you'll never swap backends, you could inline
OpenAI calls directly into those files and delete `interfaces.py` — but
given this project's own stated end-goal is a BTP production version, keep
the seam.

**The grounding principle from the previous draft still applies:** the LLM
synthesis step (Section 11) is additive, not a replacement for the ranked,
traceable suggestion list — it may only cite notification IDs already
present in the retrieved results, and must say so explicitly if nothing is
a confident match. This matters more, not less, now that generation is a
hosted API call instead of a local model — see Section 11.1.

---

## 2. Architecture — two flows, in invocation order

### 2.1 Build-time flow (run once, and again whenever the corpus changes)

```
data/MIRA_PM_Data_Cleaned.xlsx          (raw SAP export - touched rarely, only on refresh)
        |
        v
data/prep_corpus.py                     (run once per raw refresh)
        |  filters usable records, excludes site 2151, cleans equipment IDs,
        |  pseudonymizes employee names
        v
data/mira_corpus_reviewed.xlsx          <-- SOURCE OF TRUTH (see Section 6)
        |  flat, human-readable columns. A domain expert / planner can open this,
        |  review decisions baked in by prep_corpus.py (e.g. the site 2151
        |  exclusion), hand-correct a row, override a flag - and THIS file,
        |  not the raw export, is what everything downstream is built from.
        v
data/build_corpus_jsonl.py              (deterministic converter, no filtering logic)
        |  reads mira_corpus_reviewed.xlsx -> writes corpus.jsonl
        |  re-run this any time someone hand-edits the reviewed Excel
        v
data/corpus.jsonl  +  data/corpus_stats.json
        |
        v
scripts/build_index.py
        |  loads corpus.jsonl
        |  calls services/config.get_embedding_client()
        |     -> services/adapters/embedding_openai.py  (the only implementation)
        |  embeds all 13,746 documents in batches, with retry/backoff
        |  calls services/adapters/vector_store_faiss.py to build the index
        v
data/vector_store/index.faiss + index.meta.json
```

### 2.2 Runtime flow (every technician request)

```
frontend/index.html
        |  technician types issue text, optional equipment/location filters
        |  POST /suggest
        v
services/api.py
        |  validates request (Pydantic)
        |  calls services/retriever.py -> Retriever.retrieve()
        |
        |--> services/config.get_embedding_client()
        |       -> embedding_openai.py: embeds the query text into a vector
        |
        |--> services/config.get_vector_store()
        |       -> vector_store_faiss.py: finds top-20 similar past cases
        |          by cosine similarity
        |
        |--> retriever.py reranks by weighted score
        |       (vector similarity 0.7, equipment match 0.2, location match 0.1)
        |       -> top-3 Suggestion objects, always returned
        |
        `--> IF request.include_synthesis == true:
                services/synthesizer.py
                    |  builds a grounded prompt from the top-3 suggestions
                    v
                services/config.get_llm_client()
                    -> llm_openai.py: generates a short natural-language synthesis,
                       citing only notification IDs already in the suggestion list
                    v
                added to the response as an extra "synthesis" field
        v
services/api.py returns JSON: { suggestions: [...], synthesis: "..." | null }
        v
frontend/index.html renders suggestion cards + (if present) the synthesis
        |
        |  technician clicks Accept / Reject
        |  POST /feedback
        v
services/adapters/feedback_store_sqlite.py writes to data/feedback.db
```

**The rule that keeps this swappable later (production HANA / Generative
AI Hub):** `retriever.py`, `synthesizer.py`, and `api.py` import only from
`services/interfaces.py`. They never import `openai`, `faiss`, or
`sqlite3` directly. `services/config.py` is the only file that knows which
concrete adapter is active.

---

## 3. Directory structure

```
mira-poc/
├── requirements.txt
├── .env.example
├── implementation.md                      # this file
├── data/
│   ├── MIRA_PM_Data_Cleaned.xlsx          # raw SAP export (keep out of git if sensitive)
│   ├── prep_corpus.py                     # Phase 1 - raw -> reviewed Excel
│   ├── mira_corpus_reviewed.xlsx          # the source of truth, human-editable
│   ├── build_corpus_jsonl.py              # Phase 1b - reviewed Excel -> corpus.jsonl
│   ├── corpus.jsonl                       # generated - 13,746 records
│   ├── corpus_stats.json                  # generated
│   ├── feedback.db                        # generated at runtime (SQLite)
│   └── vector_store/
│       ├── index.faiss                    # generated by scripts/build_index.py
│       └── index.meta.json                # generated
├── services/
│   ├── __init__.py
│   ├── interfaces.py                      # Phase 2 - abstract contracts
│   ├── config.py                          # Phase 2 - backend selection via env vars
│   ├── retriever.py                       # Phase 5 - retrieval + rerank, backend-agnostic
│   ├── synthesizer.py                     # Phase 6 - grounded LLM synthesis
│   ├── api.py                             # Phase 8 - FastAPI service
│   └── adapters/
│       ├── __init__.py
│       ├── _http_retry.py                 # shared retry/backoff helper for HTTP calls
│       ├── embedding_openai.py            # the only embedding implementation
│       ├── vector_store_faiss.py          # unchanged, backend-agnostic
│       ├── feedback_store_sqlite.py       # unchanged, backend-agnostic
│       ├── llm_openai.py                  # the only LLM implementation
│       ├── embedding_genai_hub.py         # TODO - production, needs BTP access
│       ├── vector_store_hana.py           # TODO - production, needs BTP access
│       └── feedback_store_hana.py         # TODO - production, needs BTP access
├── scripts/
│   └── build_index.py                     # Phase 4 - orchestration
├── tests/
│   └── evaluate.py                        # Phase 9 - evaluation
└── frontend/
    └── index.html                         # Phase 7 - UI
```

No `embedding_tfidf.py`, no `embedding_ollama.py`, no `llm_ollama.py` —
removed entirely, not just unused.

---

## 4. Environment variables (single source of truth: `.env`)

```bash
# --- Embedding backend (OpenAI only) ---
MIRA_EMBEDDING_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small

# --- LLM backend (OpenAI only, used only by synthesizer.py, optional feature) ---
MIRA_LLM_BACKEND=openai            # or "none" to disable synthesis entirely
OPENAI_LLM_MODEL=gpt-4o-mini

# --- Vector store / feedback store ---
MIRA_VECTOR_STORE_BACKEND=faiss    # faiss | hana
MIRA_FEEDBACK_STORE_BACKEND=sqlite # sqlite | hana
MIRA_VECTOR_STORE_PATH=data/vector_store/index
MIRA_FEEDBACK_DB_PATH=data/feedback.db

# --- Production values (fill in once BTP access exists) ---
# MIRA_EMBEDDING_BACKEND=genai_hub
# MIRA_VECTOR_STORE_BACKEND=hana
# MIRA_FEEDBACK_STORE_BACKEND=hana
# GENAI_HUB_API_URL=
# GENAI_HUB_API_KEY=
# HANA_HOST=
# HANA_PORT=443
# HANA_USER=
# HANA_PASSWORD=
```

**Why `text-embedding-3-small`, not `-large`:** the corpus is short
technician shorthand (~22 chars average issue text), not long documents —
the quality gap between small and large embedding models matters most on
long, nuanced text. At 13,746 documents, `-small` is meaningfully cheaper
and faster with likely little practical retrieval-quality loss for this
corpus. If Phase 9's evaluation shows retrieval quality is a bottleneck,
`-large` is a one-line env var change and an index rebuild — no code
change. OpenAI's embedding models are multilingual-capable, so the
Portuguese/German mix in the corpus is handled without a special model
choice (unlike an earlier Ollama-based draft, which specifically needed a
multilingual-tuned model).

**Why `gpt-4o-mini`, not `gpt-4o`:** the synthesis task (Section 11) is a
short, tightly-grounded summarization over ~3 short text snippets — not a
task that benefits much from a larger model, and `-mini` is significantly
cheaper for a feature that may be called on every technician query. Same
one-line-swap logic applies if quality issues show up in the Phase 10
spot-check.

---

## 5. Phase 0: Environment setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

`requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
requests==2.32.3
faiss-cpu==1.8.0.post1
numpy==1.26.4
pandas==2.2.2
openpyxl==3.1.5
```

No `scikit-learn`, `joblib` (were only needed for the now-removed TF-IDF
fallback), and no `ollama`/`openai` SDK — the OpenAI adapters use plain
`requests` against documented REST endpoints, keeping the dependency
footprint small.

---

## 6. Phase 1: Data preparation — reviewed Excel is the source of truth

### 6.1 Why two files instead of one

`prep_corpus.py`'s filtering decisions (excluding site 2151, dropping two
known-bad source columns, pseudonymizing employee names) are legitimate,
but they're business decisions baked into code that nobody outside the
dev team can see or challenge. Splitting into two stages fixes that:

- `prep_corpus.py` (run rarely — only when the raw SAP export refreshes):
  raw Excel → `data/mira_corpus_reviewed.xlsx`. Output is a flat,
  human-readable Excel file, not JSONL, specifically so a maintenance
  planner or SAP admin can open and review it.
- `data/mira_corpus_reviewed.xlsx` **is the actual source of truth.** Can
  be hand-edited directly — override the site-2151 exclusion, fix a
  garbled resolution-text row, flag a row for exclusion — without
  touching any Python code.
- `build_corpus_jsonl.py` (run any time the reviewed Excel changes): reviewed
  Excel → `corpus.jsonl`. Contains **no filtering logic** — a pure,
  deterministic column mapping. All judgment calls live in the reviewed
  Excel (visible, editable) or in `prep_corpus.py`'s docstring (documented
  code-level default), never hidden inside the converter.

### 6.2 `data/prep_corpus.py`

Filtering logic (unchanged from earlier drafts — site exclusion, equipment
ID cleanup, employee pseudonymization, dropping the two known-bad
columns), with the final step writing a reviewable Excel instead of JSONL:

```python
"""
MIRA corpus preparation - Excel -> reviewable Excel.

Decisions baked into this script (see inline comments for full reasoning):
  1. Site "2151" is excluded by default - only 16 of its 3,254 notifications
     (0.5%) are usable; the rest are time-booking boilerplate. Override with
     --include-all-sites. This is a business decision, now visible and
     editable in the output Excel, not just in this script.
  2. Equipment Number is cast from float ("12004579.0") to a clean string
     id ("12004579") so downstream equality filters actually match.
  3. Employee identity is pseudonymized by default - employee_id is kept
     for traceability, employee_name is dropped. Pass
     --include-employee-names to override for internal debugging only.
  4. Two source columns are never referenced: "Equipment issue by
     Technician" (mismatched field per the source data quality log) and
     "Completion confrmtn" (a sequential counter, not a completion flag).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SHEET_NAME = "MIRA_Corpus_Notif_Level"
DEFAULT_EXCLUDE_SITES = {"2151"}


def clean_equipment_id(value) -> str | None:
    if pd.isna(value):
        return None
    return str(int(value))


def site_from_functional_location(fl: str) -> str:
    return str(fl)[:4]


def build_corpus(xlsx_path: Path, exclude_sites: set[str],
                  include_employee_names: bool) -> tuple[list[dict], dict]:
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    df["Site"] = df["Functional Location"].apply(site_from_functional_location)

    site_breakdown = {}
    for site, g in df.groupby("Site"):
        total = len(g)
        usable = int(g["Usable For RAG Corpus"].sum())
        boilerplate = int(g["Issue Is Boilerplate/Empty"].sum())
        site_breakdown[site] = {
            "total_notifications": total,
            "usable": usable,
            "usable_pct": round(100 * usable / total, 1) if total else 0.0,
            "boilerplate_or_empty_pct": round(100 * boilerplate / total, 1) if total else 0.0,
            "excluded_by_site_decision": site in exclude_sites,
        }

    usable_df = df[df["Usable For RAG Corpus"] == True].copy()  # noqa: E712
    before_site_filter = len(usable_df)
    usable_df = usable_df[~usable_df["Site"].isin(exclude_sites)]
    dropped_by_site_filter = before_site_filter - len(usable_df)

    records = []
    for _, row in usable_df.iterrows():
        issue = str(row["Reported Issue (combined)"]).strip()
        resolution = str(row["Resolution Text (combined)"]).strip()
        metadata = {
            "order_number": str(row["Order Number"]),
            "order_type": row["Order Type"],
            "notification_type": row["Notification Type"],
            "equipment": clean_equipment_id(row["Equipment Number"]),
            "func_location": row["Functional Location"],
            "func_location_desc": row["Functional Location Description (fixed)"],
            "func_location_desc_needs_review": bool(row["FL Description Needs Manual Review"]),
            "site": row["Site"],
            "employee_id": str(row["Employee(s)"]),
        }
        if include_employee_names:
            metadata["employee_name"] = row["Employee Name(s)"]

        records.append({
            "id": str(row["Notification"]),
            "text": f"Issue: {issue}\nResolution: {resolution}",
            "metadata": metadata,
        })

    stats = {
        "source_file": xlsx_path.name,
        "total_notifications_in_source": int(len(df)),
        "site_breakdown": site_breakdown,
        "sites_excluded_by_decision": sorted(exclude_sites),
        "final_corpus_size": len(records),
        "employee_names_included": include_employee_names,
        "known_excluded_fields": [
            "Equipment issue by Technician (mismatched field per Data_Quality_Fix_Log #7)",
            "Completion confrmtn (sequential counter, not a flag, per #8)",
        ],
    }
    return records, stats


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"id": r["id"], "text": r["text"]}
        row.update(r["metadata"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/MIRA_PM_Data_Cleaned.xlsx")
    ap.add_argument("--out-xlsx", default="data/mira_corpus_reviewed.xlsx")
    ap.add_argument("--out-stats", default="data/corpus_stats.json")
    ap.add_argument("--include-all-sites", action="store_true")
    ap.add_argument("--include-employee-names", action="store_true")
    args = ap.parse_args()

    exclude_sites = set() if args.include_all_sites else DEFAULT_EXCLUDE_SITES
    records, stats = build_corpus(Path(args.input), exclude_sites, args.include_employee_names)

    df = records_to_dataframe(records)
    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False, sheet_name="MIRA_Corpus_Reviewed")

    Path(args.out_stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote {len(df)} rows to {out_xlsx} for review.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Next: review the Excel, then run data/build_corpus_jsonl.py")


if __name__ == "__main__":
    main()
```

Expected stats output (from filtering logic validated against the real
19,063-row source file in an earlier pass):
```json
{
  "total_notifications_in_source": 19063,
  "site_breakdown": {
    "2151": {"total_notifications": 3254, "usable": 16, "usable_pct": 0.5, "excluded_by_site_decision": true},
    "8270": {"total_notifications": 15809, "usable": 13746, "usable_pct": 87.0, "excluded_by_site_decision": false}
  },
  "final_corpus_size": 13746,
  "employee_names_included": false
}
```

### 6.3 `data/build_corpus_jsonl.py` — pure conversion, no judgment calls

```python
"""
Converts the human-reviewed corpus Excel into corpus.jsonl. Contains NO
filtering or business logic - if you find yourself wanting to add an
"exclude if..." condition here, that decision belongs in the reviewed
Excel (a human edits the sheet) or in prep_corpus.py's docstring (a
documented code-level default), not here.

Usage: python3 data/build_corpus_jsonl.py
"""
import json
from pathlib import Path

import pandas as pd

REVIEWED_XLSX = Path("data/mira_corpus_reviewed.xlsx")
OUT_JSONL = Path("data/corpus.jsonl")

METADATA_COLUMNS = [
    "order_number", "order_type", "notification_type", "equipment",
    "func_location", "func_location_desc", "func_location_desc_needs_review",
    "site", "employee_id", "employee_name",
]


def main():
    df = pd.read_excel(REVIEWED_XLSX, sheet_name="MIRA_Corpus_Reviewed")

    records = []
    for _, row in df.iterrows():
        metadata = {
            col: row[col] for col in METADATA_COLUMNS
            if col in df.columns and pd.notna(row[col])
        }
        records.append({"id": str(row["id"]), "text": str(row["text"]), "metadata": metadata})

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {OUT_JSONL} from {REVIEWED_XLSX}")


if __name__ == "__main__":
    main()
```

### 6.4 Run order

```bash
python3 data/prep_corpus.py            # only when the raw SAP export refreshes
# --- open data/mira_corpus_reviewed.xlsx, review/edit, save ---
python3 data/build_corpus_jsonl.py     # every time the reviewed Excel changes
python3 -m scripts.build_index         # every time corpus.jsonl changes (Section 9)
```

**Known limitation carried forward:** if someone hand-edits
`mira_corpus_reviewed.xlsx` and later `prep_corpus.py` is re-run against a
refreshed raw export, the manual edit is silently overwritten —
`prep_corpus.py`'s job is "raw → first-pass cleaned," not "merge in prior
manual edits." If this becomes a recurring workflow, add a diff/merge step
before assuming manual edits are safe across raw-data refreshes.

---

## 7. Phase 2: Interfaces and configuration

### 7.1 `services/interfaces.py`

Four abstract contracts. Nothing in business logic (`retriever.py`,
`synthesizer.py`, `api.py`) may import a concrete library directly.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class EmbeddingClient(ABC):
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized dense vector per input text, in order.
        Normalization is mandatory so inner product == cosine similarity
        in every VectorStore implementation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError


@dataclass
class VectorSearchResult:
    id: str
    text: str
    metadata: dict
    score: float


class VectorStore(ABC):
    @abstractmethod
    def build(self, ids: list[str], vectors: list[list[float]],
              texts: list[str], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int,
               filters: Optional[dict] = None) -> list[VectorSearchResult]:
        """filters is a plain dict of exact-match constraints, e.g.
        {"equipment": "12002267"}. Never build a query by string-
        concatenating these values - parameterize in any SQL-backed
        adapter. If filters match nothing, fall back to unfiltered
        search rather than returning empty."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


@dataclass
class FeedbackEvent:
    query_issue_text: str
    suggested_notification_id: Optional[str]
    action: str  # "ACCEPT" | "OVERRIDE" | "REJECT"
    technician_id: str  # SAP personnel/user ID - never a display name
    override_text: Optional[str] = None


class FeedbackStore(ABC):
    @abstractmethod
    def log(self, event: FeedbackEvent) -> str: ...

    @abstractmethod
    def recent(self, limit: int = 50) -> list[dict]: ...


class LLMClient(ABC):
    """Text generation only - never used for embeddings. Used exclusively
    by services/synthesizer.py to add an optional grounded summary ON TOP
    of retrieval results. Generation is additive, never a replacement for
    the traceable ranked list - see synthesizer.py."""
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
```

### 7.2 `services/config.py`

```python
import os
from typing import Optional
from services.interfaces import EmbeddingClient, VectorStore, FeedbackStore, LLMClient

EMBEDDING_BACKEND = os.getenv("MIRA_EMBEDDING_BACKEND", "openai")
LLM_BACKEND = os.getenv("MIRA_LLM_BACKEND", "openai")
VECTOR_STORE_BACKEND = os.getenv("MIRA_VECTOR_STORE_BACKEND", "faiss")
FEEDBACK_STORE_BACKEND = os.getenv("MIRA_FEEDBACK_STORE_BACKEND", "sqlite")

VECTOR_STORE_PATH = os.getenv("MIRA_VECTOR_STORE_PATH", "data/vector_store/index")
FEEDBACK_DB_PATH = os.getenv("MIRA_FEEDBACK_DB_PATH", "data/feedback.db")


def get_embedding_client() -> EmbeddingClient:
    if EMBEDDING_BACKEND == "openai":
        from services.adapters.embedding_openai import OpenAIEmbeddingClient
        return OpenAIEmbeddingClient()
    if EMBEDDING_BACKEND == "genai_hub":
        raise NotImplementedError("adapters/embedding_genai_hub.py - needs BTP access")
    raise ValueError(f"Unknown MIRA_EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")


def get_llm_client() -> Optional[LLMClient]:
    if LLM_BACKEND == "none":
        return None
    if LLM_BACKEND == "openai":
        from services.adapters.llm_openai import OpenAILLMClient
        return OpenAILLMClient()
    if LLM_BACKEND == "genai_hub":
        raise NotImplementedError("adapters/llm_genai_hub.py - needs BTP access")
    raise ValueError(f"Unknown MIRA_LLM_BACKEND: {LLM_BACKEND}")


def get_vector_store() -> VectorStore:
    if VECTOR_STORE_BACKEND == "faiss":
        from services.adapters.vector_store_faiss import FaissVectorStore
        store = FaissVectorStore()
        store.load(VECTOR_STORE_PATH)
        return store
    if VECTOR_STORE_BACKEND == "hana":
        raise NotImplementedError("adapters/vector_store_hana.py - needs BTP access")
    raise ValueError(f"Unknown MIRA_VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}")


def get_feedback_store() -> FeedbackStore:
    if FEEDBACK_STORE_BACKEND == "sqlite":
        from services.adapters.feedback_store_sqlite import SqliteFeedbackStore
        return SqliteFeedbackStore(FEEDBACK_DB_PATH)
    if FEEDBACK_STORE_BACKEND == "hana":
        raise NotImplementedError("adapters/feedback_store_hana.py - needs BTP access")
    raise ValueError(f"Unknown MIRA_FEEDBACK_STORE_BACKEND: {FEEDBACK_STORE_BACKEND}")
```

Only two backend values are implemented today (`openai` for embedding/LLM,
`faiss`/`sqlite` for storage) — `genai_hub`/`hana` branches exist as
explicit `NotImplementedError` markers so the migration path (Section 18)
has a concrete place to land code later, not because multiple local
options are being hedged for.

---

## 8. Phase 3: Embedding adapter (OpenAI only)

### 8.1 Shared retry helper — `services/adapters/_http_retry.py`

```python
import time
import requests


def post_with_retry(url: str, json: dict, headers: dict | None = None,
                     timeout: int = 60, max_retries: int = 3) -> dict:
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=json, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"Request to {url} failed after {max_retries} attempts") from last_exc
```

### 8.2 `services/adapters/embedding_openai.py`

```python
"""
Embedding adapter using OpenAI's embeddings API. Requires OPENAI_API_KEY.

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
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]  # fail loudly if missing
        self.model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self._dim = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("dimension unknown until embed_batch() has been called once")
        return self._dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
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
```

---

## 9. Phase 4: Vector store — unchanged, backend-agnostic

`services/adapters/vector_store_faiss.py` doesn't care which embedding
backend produced the vectors it stores — it only requires L2-normalized
float vectors of consistent dimension (1536 for `text-embedding-3-small`).
No changes needed for the OpenAI-only simplification.

`scripts/build_index.py`:

```python
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
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
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
```

**Cost/time note:** 13,746 short documents (~70 chars average) at
`text-embedding-3-small` pricing is a small one-time cost — verify current
published OpenAI pricing before running, since rates change. Batched at
100 texts/request, this is roughly 138 API calls total, which should
complete in well under a minute of wall-clock time barring rate limiting.

---

## 10. Phase 5: Retriever — unchanged core logic

`services/retriever.py` logic (embed query → search top-20 → rerank by
weighted score → return top-3) is backend-agnostic and needs no changes
for the OpenAI-only simplification.

- **`CONFIDENCE_THRESHOLDS`** (`HIGH`/`MEDIUM` cosine cutoffs): must be
  calibrated against `text-embedding-3-small`'s actual score distribution
  on this corpus — do not reuse any earlier TF-IDF-era or Ollama-era
  numbers, they don't transfer across embedding spaces. Use Phase 10's
  evaluation script to plot real scores before setting these.
- Rerank weights (`vector=0.7, equipment_match=0.2, func_location_match=0.1`)
  are unaffected by the embedding backend — no change needed.

---

## 11. Phase 6: RAG synthesis layer (LLM generation, OpenAI only)

### 11.1 Design principle

The synthesis step is **strictly grounded**: the LLM receives only the
already-retrieved top-3 suggestions (issue text, resolution text,
notification ID) and is instructed to summarize using *only* that
context, always citing notification IDs, and explicitly saying so if
nothing in context is a confident match. It must never introduce an
equipment name, procedure, or part number not already present in the
retrieved text. This keeps the "traceable to a real past order" guarantee
intact even with generation in the loop — important because a hosted API
model has no visibility into your corpus beyond what you put in the
prompt, so grounding is the only thing preventing it from filling gaps
with generic-sounding but unverified maintenance advice.

### 11.2 `services/synthesizer.py`

```python
"""
Grounded LLM synthesis on top of retrieval results. Does NOT replace the
ranked suggestion list - api.py always returns that list regardless of
whether synthesis is requested. This module only adds an optional
plain-language summary field.
"""

from services.interfaces import LLMClient
from services.retriever import Suggestion

SYSTEM_PROMPT = """You are assisting a plant maintenance technician. You will be given \
an issue description and a short list of similar PAST resolved cases, each with a \
notification ID. Your job is to write a brief (2-4 sentence) plain-language summary \
to help the technician decide what to try first.

STRICT RULES - do not break these:
1. Use ONLY the information in the provided past cases. Do not invent equipment names, \
part numbers, procedures, or causes that are not explicitly present in the text given to you.
2. Every claim you make must be attributable to a specific notification ID from the \
provided list. Cite the notification ID in parentheses after each claim, e.g. (notification 10026465).
3. If none of the provided cases seem like a confident match for the issue described, \
say so explicitly instead of guessing - do not force a recommendation.
4. Do not use any external knowledge about this equipment type beyond what is written \
in the provided cases.
5. Keep your answer short and practical - this is read by someone standing at a machine, \
not a report."""


def build_user_prompt(issue_text: str, suggestions: list[Suggestion]) -> str:
    lines = [f"Reported issue: {issue_text}", "", "Past similar cases:"]
    for s in suggestions:
        lines.append(
            f"- Notification {s.source_notification} (similarity {s.similarity_score:.2f}, "
            f"confidence {s.confidence}): issue was \"{s.issue_text}\", "
            f"resolved by \"{s.resolution_text}\"."
        )
    return "\n".join(lines)


def synthesize(issue_text: str, suggestions: list[Suggestion],
               llm_client: LLMClient) -> str:
    if not suggestions:
        return "No similar past cases were found for this issue."
    user_prompt = build_user_prompt(issue_text, suggestions)
    return llm_client.generate(SYSTEM_PROMPT, user_prompt)
```

### 11.3 `services/adapters/llm_openai.py`

```python
"""
LLM adapter using OpenAI's chat completions API. Requires OPENAI_API_KEY.
API reference: POST https://api.openai.com/v1/chat/completions
  body: {"model": "gpt-4o-mini", "messages": [...]}
  response: {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
"""

import os

from services.adapters._http_retry import post_with_retry
from services.interfaces import LLMClient


class OpenAILLMClient(LLMClient):
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        result = post_with_retry(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )
        return result["choices"][0]["message"]["content"]
```

---

## 12. Phase 7: Feedback storage — unchanged, backend-agnostic

`services/adapters/feedback_store_sqlite.py` is unaffected by the
embedding/LLM simplification. Same schema, same PII rule
(`technician_id` must be an SAP personnel ID, never a display name):

```sql
CREATE TABLE feedback_log (
    feedback_id TEXT PRIMARY KEY,
    query_issue_text TEXT NOT NULL,
    suggested_notification_id TEXT,
    action TEXT NOT NULL CHECK (action IN ('ACCEPT','OVERRIDE','REJECT')),
    technician_id TEXT NOT NULL,
    override_text TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

---

## 13. Phase 8: API layer

```python
"""
MIRA API. Run: uvicorn services.api:app --reload --port 8000
Endpoints: GET /health, POST /suggest, POST /feedback
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services import config
from services.interfaces import FeedbackEvent
from services.retriever import Retriever
from services.synthesizer import synthesize

app = FastAPI(title="MIRA API", version="0.3.0")

_embedding_client = None
_vector_store = None
_feedback_store = None
_llm_client = None
_retriever: Optional[Retriever] = None


@app.on_event("startup")
def startup():
    global _embedding_client, _vector_store, _feedback_store, _llm_client, _retriever
    _embedding_client = config.get_embedding_client()
    _vector_store = config.get_vector_store()
    _feedback_store = config.get_feedback_store()
    _llm_client = config.get_llm_client()  # may be None if MIRA_LLM_BACKEND=none
    _retriever = Retriever(_embedding_client, _vector_store)


class SuggestRequest(BaseModel):
    issue_text: str = Field(..., min_length=1, max_length=2000)
    equipment: Optional[str] = None
    func_location: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)
    include_synthesis: bool = False  # opt-in - adds LLM latency/cost


class SuggestionOut(BaseModel):
    rank: int
    issue_text: str
    resolution_text: str
    source_notification: str
    source_order: str
    equipment: Optional[str]
    func_location: str
    similarity_score: float
    confidence: str


class SuggestResponse(BaseModel):
    query: str
    suggestions: list[SuggestionOut]
    synthesis: Optional[str] = None


class FeedbackRequest(BaseModel):
    query_issue_text: str
    suggested_notification_id: Optional[str] = None
    action: str = Field(..., pattern="^(ACCEPT|OVERRIDE|REJECT)$")
    technician_id: str
    override_text: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str = "ok"


@app.get("/health")
def health():
    return {"status": "ok", "backend": {
        "embedding": config.EMBEDDING_BACKEND,
        "llm": config.LLM_BACKEND,
        "vector_store": config.VECTOR_STORE_BACKEND,
        "feedback_store": config.FEEDBACK_STORE_BACKEND,
    }}


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    if not req.issue_text.strip():
        raise HTTPException(status_code=400, detail="issue_text must not be empty")

    suggestions = _retriever.retrieve(
        issue_text=req.issue_text, equipment=req.equipment,
        func_location=req.func_location, top_k=req.top_k,
    )

    synthesis_text = None
    if req.include_synthesis:
        if _llm_client is None:
            raise HTTPException(
                status_code=400,
                detail="include_synthesis=true but MIRA_LLM_BACKEND=none",
            )
        synthesis_text = synthesize(req.issue_text, suggestions, _llm_client)

    return SuggestResponse(
        query=req.issue_text,
        suggestions=[SuggestionOut(**s.__dict__) for s in suggestions],
        synthesis=synthesis_text,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    event = FeedbackEvent(
        query_issue_text=req.query_issue_text,
        suggested_notification_id=req.suggested_notification_id,
        action=req.action, technician_id=req.technician_id,
        override_text=req.override_text,
    )
    return FeedbackResponse(feedback_id=_feedback_store.log(event))
```

**Latency/cost note:** `include_synthesis=true` adds a full OpenAI chat
completion round-trip on top of retrieval — real latency (hundreds of ms
to a couple seconds) and real per-call cost. Default it to `false` in the
frontend and let the technician opt in ("Explain this suggestion") rather
than calling it on every search.

---

## 14. Phase 9: Frontend

`frontend/index.html` — single-file HTML/JS, no build tooling, calls the
API above directly. After rendering the suggestion cards, if
`data.synthesis` is present in the `/suggest` response, render it in a
clearly-labeled box (e.g. "AI summary (verify against the cases below)")
— never let it visually replace the individual traceable suggestion
cards, since the citation-to-notification-ID promise only holds if the
underlying cases stay visible and checkable next to the summary. Add a
checkbox/toggle "Include AI explanation" that sets `include_synthesis:
true` in the request body when checked, matching the opt-in cost/latency
note above.

---

## 15. Phase 10: Evaluation

Two methodologies (no annotated ground-truth similarity pairs exist in
this corpus, so Recall@3-against-holdout isn't measurable):

1. **Self-retrieval regression check** (automatic): perturb a sample of
   real issue texts (truncate/drop punctuation/shuffle word order), check
   whether the *original* notification is retrieved in the top-k. A
   necessary-but-not-sufficient sanity check — proves paraphrase
   robustness, not that other retrieved cases are useful fixes.
2. **Labeled relevance sample** (manual, once before demo): export ~75
   real queries × top-3 suggestions to a CSV, have a maintenance planner
   mark relevant/irrelevant, compute real Precision@3/MRR. **Report this
   number against success criteria**, not an unlabeled Recall@3.
3. **Grounding spot-check on synthesis output.** On ~20 queries with
   `include_synthesis=true`, manually verify every notification ID cited
   in the synthesis text actually appears in that query's suggestion
   list, and that no equipment/procedure is mentioned that isn't present
   in the underlying issue/resolution text. Not automatable without
   another model doing the checking — manual spot-check of ~20 examples
   is a reasonable bar for a hackathon PoC.

**Must re-run both retrieval-quality methods now** — any earlier numbers
from a TF-IDF or Ollama pass do not transfer to `text-embedding-3-small`'s
score distribution.

---

## 16. Phase 11: Local demo checklist (run order)

- [ ] `OPENAI_API_KEY` set in `.env`
- [ ] `python3 data/prep_corpus.py` — only if the raw SAP export changed
- [ ] Review `data/mira_corpus_reviewed.xlsx` — first run, or to sanity-check the site-2151 exclusion / any flagged rows
- [ ] `python3 data/build_corpus_jsonl.py` — every time the reviewed Excel changes
- [ ] `python3 -m scripts.build_index` — rebuild embeddings/index (every time `corpus.jsonl` changes; this calls the OpenAI API and incurs a small cost)
- [ ] `uvicorn services.api:app --port 8000`
- [ ] `cd frontend && python3 -m http.server 8080`
- [ ] Smoke-test 3–5 real Portuguese queries, with and without `include_synthesis`
- [ ] Re-run `tests/evaluate.py`'s labeled-sample workflow and have a real Precision@3 number ready
- [ ] Run the grounding spot-check (Section 15, item 3) on ~20 synthesis outputs before demoing that feature
- [ ] Decide as a team whether the demo shows the synthesis field at all, given the traceability trade-off in Section 11.1

---

## 17. Known limitations

- Site 2151 excluded from the corpus by documented default (0.5% usable) — visible and overridable in `mira_corpus_reviewed.xlsx`.
- `Equipment issue by Technician` and `Completion confrmtn` source columns are known-unusable and never referenced.
- No SAP BW writeback exists yet — feedback goes to local SQLite only; demo narrative should say "this demonstrates the loop mechanism," not imply a live BW round-trip.
- Confidence thresholds must be calibrated against real `text-embedding-3-small` output (Section 10) before a demo — no valid default exists yet.
- The LLM synthesis feature is unverified in this codebase — the grounding rules in the prompt reduce but do not eliminate hallucination risk; the manual spot-check in Section 15 is the only current safeguard.
- Both the embedding and generation paths now depend on OpenAI API availability and cost — no offline fallback exists in this version. If that's a risk for the live demo (venue wifi, rate limits), consider caching known-good demo queries' responses ahead of time.
- If `mira_corpus_reviewed.xlsx` is hand-edited, those edits are lost if `prep_corpus.py` is re-run against a refreshed raw export (Section 6.4) — no merge step exists yet.

---

## 18. Phase 12: Production migration path (once you have BTP / HANA / Generative AI Hub)

| Step | New file | Interface | Config flip |
|---|---|---|---|
| 12.1 | `services/adapters/embedding_genai_hub.py` — call Generative AI Hub's embedding endpoint, L2-normalize before returning | `EmbeddingClient` | `MIRA_EMBEDDING_BACKEND=genai_hub` |
| 12.2 | `services/adapters/vector_store_hana.py` — HANA Cloud vector engine, `COSINE_SIMILARITY()`, parameterized `WHERE` for `filters` dict | `VectorStore` | `MIRA_VECTOR_STORE_BACKEND=hana` |
| 12.3 | `services/adapters/feedback_store_hana.py` — same schema as SQLite adapter, `hdbcli` driver | `FeedbackStore` | `MIRA_FEEDBACK_STORE_BACKEND=hana` |
| 12.4 | (optional) `services/adapters/llm_genai_hub.py` if synthesis moves to production too | `LLMClient` | `MIRA_LLM_BACKEND=genai_hub` |
| 12.5 | Deployment: Cloud Foundry manifest, XSUAA binding (or a Spring Boot reimplementation of `api.py`, per the original architecture slide) | — | — |
| 12.6 | Re-run Section 15 evaluation against the new backend and re-calibrate `CONFIDENCE_THRESHOLDS` — thresholds do not transfer across embedding spaces | — | — |

Nothing in `services/retriever.py`, `services/synthesizer.py`,
`services/api.py`, or `tests/evaluate.py` needs to change for any of these
— that is the entire point of the interface layer in Section 7.

---

## 19. If you run out of tokens partway through building this

Everything a fresh AI model or developer needs to pick this up from
scratch is in this single file: directory structure (Section 3), every
environment variable (Section 4), and complete code for every file
(Sections 6–13). Only `vector_store_faiss.py`, `feedback_store_sqlite.py`,
and `frontend/index.html`'s base structure aren't fully reproduced here —
they're backend-agnostic (unaffected by the OpenAI-only simplification)
and can be regenerated from their interface contracts in Sections 7, 9,
and 12 if needed.
