# MIRA POC

This repository contains the MIRA (Maintenance Intelligence & Resolution Assistant) proof-of-concept.

**Use case flow:** See the implementation overview and invocation flow in [use_case_flow.md](use_case_flow.md).

Quick start (local):

```bash
# Create and activate a Python 3.12 venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the API (adjust port if 8000 is in use)
uv run uvicorn services.api:app --reload --port 8001

# Serve frontend
cd frontend && python3 -m http.server 8080
```

If you'd like a visual PNG of the flowchart or a short onboarding snippet added to other docs, tell me which and I will add it.

