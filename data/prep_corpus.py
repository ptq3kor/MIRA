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
    """Convert Equipment Number from float (e.g., 12004579.0) to clean string ID (e.g., '12004579')."""
    if pd.isna(value):
        return None
    return str(int(value))


def site_from_functional_location(fl: str) -> str:
    """Extract site code from functional location (first 4 characters)."""
    return str(fl)[:4]


def build_corpus(xlsx_path: Path, exclude_sites: set[str],
                  include_employee_names: bool) -> tuple[list[dict], dict]:
    """
    Build corpus records and statistics from raw SAP export.
    
    Args:
        xlsx_path: Path to raw SAP export Excel file.
        exclude_sites: Set of site codes to exclude from corpus.
        include_employee_names: Whether to include employee names (for debugging only).
    
    Returns:
        Tuple of (records list, statistics dict).
    """
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
    """Convert records to a flat DataFrame for Excel output."""
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