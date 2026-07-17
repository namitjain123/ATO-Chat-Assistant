"""
Non-interactive driver for the eval suite (same functions evals/app.py uses,
without needing to click through Streamlit). Runs Phase 1 (live pipeline),
guardrails tests, and Phase 2 (RAGAS metrics), saving results to JSON files.

Usage: python -m evals.run_full_eval
"""
import asyncio
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from evals.pipeline import run_pipeline, load_golden_dataset, save_results
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics


def pipeline_cb(i, total, question, stage, response=""):
    if stage == "calling":
        print(f"[Phase 1] [{i + 1}/{total}] Calling: {question[:70]}")
    else:
        preview = (response[:80] + "...") if len(response) > 80 else response
        print(f"[Phase 1] [{i + 1}/{total}] Done: {preview or '(empty response)'}")


def guardrails_cb(i, total, input_text):
    print(f"[Guardrails] [{i + 1}/{total}] Testing: {input_text[:70]}")


def status_cb(msg: str):
    print(f"[Phase 2] {msg}")


def main():
    golden = load_golden_dataset()
    print(f"Loaded golden dataset: {len(golden['rag_samples'])} rag_samples, {len(golden['guardrails_samples'])} guardrails_samples")

    print("\n=== PHASE 1: Live Pipeline ===")
    enriched = run_pipeline(golden, progress_callback=pipeline_cb)
    save_results(enriched, "evals/enriched_dataset.json")
    print("Phase 1 complete. Saved evals/enriched_dataset.json")

    print("\n=== GUARDRAILS TESTS ===")
    g_results = run_guardrails_eval(enriched["guardrails_samples"], progress_callback=guardrails_cb)
    g_metrics = compute_guardrails_metrics(g_results)
    print(f"Guardrails metrics: {g_metrics}")
    with open("evals/guardrails_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": g_results, "metrics": g_metrics}, f, indent=2, ensure_ascii=False)
    print("Saved evals/guardrails_results.json")

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
