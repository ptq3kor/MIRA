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
    """Convert reviewed Excel to JSONL format for indexing."""
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