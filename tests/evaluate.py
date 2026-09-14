"""
MIRA Evaluation Script.

Two methodologies (no annotated ground-truth similarity pairs exist):

1. Self-retrieval regression check (automatic): perturb real issue texts,
   check whether the original notification is retrieved in top-k.
   Proves paraphrase robustness, not that other retrieved cases are useful fixes.

2. Labeled relevance sample (manual): export ~75 real queries × top-3 suggestions
   to CSV, have a maintenance planner mark relevant/irrelevant,
   compute real Precision@3/MRR. Report this against success criteria.

3. Grounding spot-check on synthesis output: on ~20 queries with
   include_synthesis=true, manually verify every notification ID cited
   in the synthesis text actually appears in that query's suggestion list,
   and that no equipment/procedure is mentioned that isn't present in the
   underlying issue/resolution text.
"""

import argparse
import csv
import json
import random
import string
from pathlib import Path

from services import config
from services.retriever import Retriever


def load_corpus_sample(jsonl_path: Path, n: int = 100, seed: int = 42) -> list[dict]:
    """Load a random sample of corpus records for evaluation."""
    random.seed(seed)
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return random.sample(records, min(n, len(records)))


def perturb_text(text: str, noise_level: float = 0.15) -> str:
    """
    Apply random perturbations to simulate real-world query variations.
    
    Operations: character deletion, word shuffle, punctuation removal,
    word dropout, typo injection.
    """
    words = text.split()
    
    # Word dropout
    words = [w for w in words if random.random() > noise_level * 0.5]
    
    # Word shuffle (partial)
    if len(words) > 3 and random.random() < noise_level:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    
    # Character-level noise on remaining words
    noisy_words = []
    for w in words:
        if random.random() < noise_level * 0.3 and len(w) > 3:
            # Inject typo: swap adjacent chars
            idx = random.randint(1, len(w) - 2)
            w = w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
        noisy_words.append(w)
    
    # Remove some punctuation
    result = ' '.join(noisy_words)
    result = ''.join(c for c in result if random.random() > noise_level * 0.2 or c not in '.,;:!?')
    
    return result


def self_retrieval_test(retriever: Retriever, samples: list[dict], top_k: int = 3) -> dict:
    """
    Test if original notification is retrieved when querying with perturbed issue text.
    
    Returns dict with hit rates at k=1,3,5.
    """
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    total = 0
    
    for record in samples:
        # Parse original issue from combined text
        text = record["text"]
        if "\nResolution: " not in text:
            continue
        original_issue = text.split("\nResolution: ")[0].replace("Issue: ", "", 1).strip()
        original_id = record["id"]
        
        # Test with original text
        suggestions = retriever.retrieve(original_issue, top_k=5)
        retrieved_ids = [s.source_notification for s in suggestions]
        
        if original_id in retrieved_ids[:1]:
            hits_at_1 += 1
        if original_id in retrieved_ids[:3]:
            hits_at_3 += 1
        if original_id in retrieved_ids[:5]:
            hits_at_5 += 1
        total += 1
        
        # Test with perturbed text
        perturbed = perturb_text(original_issue)
        suggestions = retriever.retrieve(perturbed, top_k=5)
        retrieved_ids = [s.source_notification for s in suggestions]
        
        if original_id in retrieved_ids[:1]:
            hits_at_1 += 1
        if original_id in retrieved_ids[:3]:
            hits_at_3 += 1
        if original_id in retrieved_ids[:5]:
            hits_at_5 += 1
        total += 1
    
    return {
        "total_queries": total,
        "hit_at_1": hits_at_1 / total if total else 0,
        "hit_at_3": hits_at_3 / total if total else 0,
        "hit_at_5": hits_at_5 / total if total else 0,
    }


def export_labeled_sample(retriever: Retriever, samples: list[dict], 
                          output_csv: Path, queries_per_sample: int = 1) -> None:
    """
    Export queries × top-3 suggestions to CSV for manual labeling.
    
    Each row: query_id, query_text, suggestion_rank, suggested_notification,
    suggested_issue, suggested_resolution, similarity_score, confidence, label (empty)
    """
    rows = []
    for record in samples:
        text = record["text"]
        if "\nResolution: " not in text:
            continue
        original_issue = text.split("\nResolution: ")[0].replace("Issue: ", "", 1).strip()
        
        for _ in range(queries_per_sample):
            suggestions = retriever.retrieve(original_issue, top_k=3)
            for s in suggestions:
                rows.append({
                    "query_id": record["id"],
                    "query_text": original_issue,
                    "suggestion_rank": s.rank,
                    "suggested_notification": s.source_notification,
                    "suggested_issue": s.issue_text,
                    "suggested_resolution": s.resolution_text,
                    "similarity_score": round(s.similarity_score, 4),
                    "confidence": s.confidence,
                    "label_relevant": "",  # To be filled by human: 1=relevant, 0=irrelevant
                })
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Exported {len(rows)} rows to {output_csv}")
    print("Have a maintenance planner fill in 'label_relevant' column (1=relevant, 0=irrelevant)")


def compute_precision_from_labeled(csv_path: Path) -> dict:
    """
    Compute Precision@k and MRR from labeled CSV.
    
    Expects 'label_relevant' column filled with 1/0.
    """
    query_labels = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            label = row.get("label_relevant", "").strip()
            if label not in ("1", "0"):
                continue
            if qid not in query_labels:
                query_labels[qid] = []
            query_labels[qid].append(int(label))
    
    if not query_labels:
        return {"error": "No labeled data found"}
    
    # Precision@k
    precisions = {1: [], 3: [], 5: []}
    reciprocal_ranks = []
    
    for labels in query_labels.values():
        for k in [1, 3, 5]:
            relevant_at_k = sum(labels[:k])
            precisions[k].append(relevant_at_k / k if k <= len(labels) else 0)
        
        # MRR: 1/rank of first relevant
        for rank, label in enumerate(labels, 1):
            if label == 1:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return {
        "num_queries": len(query_labels),
        "precision_at_1": round(sum(precisions[1]) / len(precisions[1]), 4) if precisions[1] else 0,
        "precision_at_3": round(sum(precisions[3]) / len(precisions[3]), 4) if precisions[3] else 0,
        "precision_at_5": round(sum(precisions[5]) / len(precisions[5]), 4) if precisions[5] else 0,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0,
    }


def export_synthesis_spotcheck(retriever: Retriever, samples: list[dict],
                                output_csv: Path, n: int = 20) -> None:
    """
    Export queries for synthesis grounding spot-check.
    
    Run with include_synthesis=true via API, then manually verify:
    - Every cited notification ID appears in suggestions
    - No equipment/procedure mentioned that isn't in source text
    """
    random.seed(42)
    selected = random.sample(samples, min(n, len(samples)))
    
    rows = []
    for record in selected:
        text = record["text"]
        if "\nResolution: " not in text:
            continue
        original_issue = text.split("\nResolution: ")[0].replace("Issue: ", "", 1).strip()
        suggestions = retriever.retrieve(original_issue, top_k=3)
        
        rows.append({
            "query_id": record["id"],
            "query_text": original_issue,
            "suggestion_1_notif": suggestions[0].source_notification if len(suggestions) > 0 else "",
            "suggestion_1_issue": suggestions[0].issue_text if len(suggestions) > 0 else "",
            "suggestion_1_resolution": suggestions[0].resolution_text if len(suggestions) > 0 else "",
            "suggestion_2_notif": suggestions[1].source_notification if len(suggestions) > 1 else "",
            "suggestion_2_issue": suggestions[1].issue_text if len(suggestions) > 1 else "",
            "suggestion_2_resolution": suggestions[1].resolution_text if len(suggestions) > 1 else "",
            "suggestion_3_notif": suggestions[2].source_notification if len(suggestions) > 2 else "",
            "suggestion_3_issue": suggestions[2].issue_text if len(suggestions) > 2 else "",
            "suggestion_3_resolution": suggestions[2].resolution_text if len(suggestions) > 2 else "",
            "synthesis_text": "",  # Fill after calling API with include_synthesis=true
            "grounding_ok": "",    # Manual: 1=all citations valid, 0=hallucination found
            "notes": "",
        })
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Exported {len(rows)} spot-check rows to {output_csv}")
    print("Call /suggest with include_synthesis=true for each query, paste synthesis_text,")
    print("then verify grounding manually (grounding_ok=1/0)")


def main():
    ap = argparse.ArgumentParser(description="MIRA Evaluation")
    ap.add_argument("--mode", choices=["self-retrieval", "export-labeled", "compute-labeled", "export-synthesis"],
                    required=True, help="Evaluation mode")
    ap.add_argument("--corpus", default="data/corpus.jsonl", help="Path to corpus.jsonl")
    ap.add_argument("--sample-size", type=int, default=100, help="Number of corpus records to sample")
    ap.add_argument("--output", default="tests/eval_output.csv", help="Output CSV path")
    ap.add_argument("--labeled-csv", help="Path to labeled CSV (for compute-labeled mode)")
    args = ap.parse_args()

    # Initialize retriever
    embedder = config.get_embedding_client()
    vector_store = config.get_vector_store()
    retriever = Retriever(embedder, vector_store)

    # Load sample
    samples = load_corpus_sample(Path(args.corpus), args.sample_size)
    print(f"Loaded {len(samples)} sample records")

    if args.mode == "self-retrieval":
        print("Running self-retrieval test...")
        results = self_retrieval_test(retriever, samples)
        print(json.dumps(results, indent=2))
        
    elif args.mode == "export-labeled":
        print("Exporting labeled sample...")
        export_labeled_sample(retriever, samples, Path(args.output))
        
    elif args.mode == "compute-labeled":
        if not args.labeled_csv:
            print("Error: --labeled-csv required for compute-labeled mode")
            return
        print("Computing metrics from labeled data...")
        results = compute_precision_from_labeled(Path(args.labeled_csv))
        print(json.dumps(results, indent=2))
        
    elif args.mode == "export-synthesis":
        print("Exporting synthesis spot-check sample...")
        export_synthesis_spotcheck(retriever, samples, Path(args.output))


if __name__ == "__main__":
    main()