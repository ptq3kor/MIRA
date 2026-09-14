"""
MIRA API Service.

FastAPI application exposing endpoints for suggestion retrieval and feedback logging.

Endpoints:
    GET  /health                    - Health check and backend info
    POST /suggest                   - Get maintenance suggestions for an issue
    POST /feedback                  - Log technician feedback

Run: uvicorn services.api:app --reload --port 8000
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services import config
from services.interfaces import FeedbackEvent
from services.retriever import Retriever
from services.synthesizer import synthesize

app = FastAPI(title="MIRA API", version="0.3.0")

# Global singletons initialized at startup
_embedding_client = None
_vector_store = None
_feedback_store = None
_llm_client = None
_retriever: Optional[Retriever] = None

# Allow frontend app served at localhost:8080 to call this API during local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Initialize all backend clients on application startup."""
    global _embedding_client, _vector_store, _feedback_store, _llm_client, _retriever
    _embedding_client = config.get_embedding_client()
    _vector_store = config.get_vector_store()
    _feedback_store = config.get_feedback_store()
    _llm_client = config.get_llm_client()  # may be None if MIRA_LLM_BACKEND=none
    _retriever = Retriever(_embedding_client, _vector_store)


class SuggestRequest(BaseModel):
    """Request model for /suggest endpoint."""
    issue_text: str = Field(..., min_length=1, max_length=2000, description="Technician's reported issue text")
    equipment: Optional[str] = Field(None, description="Optional equipment number filter")
    func_location: Optional[str] = Field(None, description="Optional functional location filter")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of suggestions to return")
    include_synthesis: bool = Field(default=False, description="Include LLM-generated summary (adds latency/cost)")


class SuggestionOut(BaseModel):
    """Response model for a single suggestion."""
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
    """Response model for /suggest endpoint."""
    query: str
    suggestions: list[SuggestionOut]
    synthesis: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request model for /feedback endpoint."""
    query_issue_text: str = Field(..., description="Original issue text queried")
    suggested_notification_id: Optional[str] = Field(None, description="Notification ID that was suggested")
    action: str = Field(..., pattern="^(ACCEPT|OVERRIDE|REJECT)$", description="Feedback action")
    technician_id: str = Field(..., description="SAP personnel ID of technician")
    override_text: Optional[str] = Field(None, description="Custom resolution if OVERRIDE")


class FeedbackResponse(BaseModel):
    """Response model for /feedback endpoint."""
    feedback_id: str
    status: str = "ok"


@app.get("/health")
def health():
    """Health check endpoint with backend configuration info."""
    return {"status": "ok", "backend": {
        "embedding": config.EMBEDDING_BACKEND,
        "llm": config.LLM_BACKEND,
        "vector_store": config.VECTOR_STORE_BACKEND,
        "feedback_store": config.FEEDBACK_STORE_BACKEND,
    }}


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    """
    Get maintenance suggestions for a reported issue.
    
    Retrieves similar historical cases, reranks by vector similarity + metadata matches,
    optionally generates a grounded LLM synthesis summary.
    """
    if not req.issue_text.strip():
        raise HTTPException(status_code=400, detail="issue_text must not be empty")

    suggestions = _retriever.retrieve(
        issue_text=req.issue_text,
        equipment=req.equipment,
        func_location=req.func_location,
        top_k=req.top_k,
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
    """
    Log technician feedback on a suggestion.
    
    Records whether the suggestion was accepted, overridden with custom text, or rejected.
    """
    event = FeedbackEvent(
        query_issue_text=req.query_issue_text,
        suggested_notification_id=req.suggested_notification_id,
        action=req.action,
        technician_id=req.technician_id,
        override_text=req.override_text,
    )
    return FeedbackResponse(feedback_id=_feedback_store.log(event))