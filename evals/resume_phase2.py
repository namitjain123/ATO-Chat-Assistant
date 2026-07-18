"""
Resume-only driver: Phase 1 and guardrails already succeeded and were saved
(evals/enriched_dataset.json, evals/guardrails_results.json). This just runs
Phase 2 (RAGAS metrics) against the already-collected live responses, so we
don't have to redo the ~35-minute live pipeline run after fixing the
sentence-transformers/huggingface_hub version mismatch.

At full scale (75 samples, GENERAL_BATCH_SIZE=1, 40s inter-sample cooldown)
Phase 2 takes ~4-4.5 hours. SAMPLE_LIMIT below caps it to a fast subset —
10 samples -> ~30-35 min total.

Usage: python -m evals.resume_phase2
"""
import asyncio
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from evals.metrics import run_all_metrics

SAMPLE_LIMIT = 10


def status_cb(msg: str):
    print(f"[Phase 2] {msg}")


def main():
    with open("evals/enriched_dataset.json", encoding="utf-8") as f:
        enriched = json.load(f)
    with open("evals/guardrails_results.json", encoding="utf-8") as f:
        g_data = json.load(f)
    g_metrics = g_data["metrics"]

    total = len(enriched["rag_samples"])
    valid = [
        s for s in enriched["rag_samples"]
        if s.get("actual_response", "").strip() and "internal error" not in s["actual_response"]
    ]
    subset = valid[:SAMPLE_LIMIT]
    enriched = {**enriched, "rag_samples": subset}

    print(f"Loaded {len(valid)} valid responses out of {total} total; using first {len(subset)}")
    print("\n=== PHASE 2: RAGAS Metrics ===")
    metric_results = asyncio.run(run_all_metrics(enriched, status_cb=status_cb))

    summary = {}
    averages = {}
    for key, df in metric_results.items():
        summary[key] = df.to_dict(orient="records")
        numeric_col = [c for c in df.columns if c != "question"][0]
        averages[key] = round(float(df[numeric_col].mean()), 3)

    with open("evals/metric_results.json", "w", encoding="utf-8") as f:
        json.dump({"per_sample": summary, "averages": averages}, f, indent=2, ensure_ascii=False)

    print("\n=== FINAL SUMMARY ===")
    for k, v in averages.items():
        print(f"  {k}: {v}")
    print(f"  guardrails_accuracy: {g_metrics['correct']}/{g_metrics['total']} (precision={g_metrics['precision']}, recall={g_metrics['recall']})")
    print("\nSaved evals/metric_results.json")
    print("DONE.")


if __name__ == "__main__":
    main()
