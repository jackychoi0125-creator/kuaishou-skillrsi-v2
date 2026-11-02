"""One-call Source-only procedural extraction under a frozen R1.7 budget."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from skillrsi_m3.client import KimiClient
from skillrsi_m3.core import run as run_m3, write_json
from .r17_plan import freeze
from .r17_source import audit

MAX_GROUPS = {"spreadsheetbench": 4, "livemath": 2}


def _usage(root: Path) -> dict[str, int]:
    result = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for benchmark in ("spreadsheetbench", "alfworld", "livemath"):
        for path in (root / "results/r17" / benchmark / "isolated_kimi_cache").glob("*.json"):
            value = json.loads(path.read_text()).get("usage") or {}
            for key in result:
                result[key] += int(value.get(key) or 0)
    return result


def run_one(root: Path, benchmark: str, *, dry_run: bool = False) -> dict[str, Any]:
    plan = freeze(root)
    if benchmark not in MAX_GROUPS:
        return {"benchmark": benchmark, "status": "skipped_no_new_call_planned", "new_model_calls": 0}
    isolation = audit(root, benchmark, plan)
    if isolation["status"] != "clean":
        return {"benchmark": benchmark, "status": "blocked_isolation", "errors": isolation["errors"], "new_model_calls": 0}
    base = root / "results/r17" / benchmark
    selected = base / "m3/selected_evidence.json"
    insights = base / "m3/insights.jsonl"
    if insights.is_file():
        if not (base / "m3/usage.json").is_file():
            raise ValueError("M3 resume output lacks usage record")
        existing = base / "generation_manifest.json"
        if existing.is_file():
            value = json.loads(existing.read_text())
            return {"benchmark": benchmark, "status": "resumed", "new_model_calls": 0,
                    "origin_new_kimi_usage": value["new_kimi_usage"],
                    "insights": len(insights.read_text().splitlines()), "isolation": isolation["status"]}
        audit_m3 = json.loads((base / "m3/insight_audit.json").read_text())
        value = {"benchmark": benchmark, "status": "procedural_hypotheses_generated",
                 "selected_groups": len(json.loads((base / "m3/selected_evidence.json").read_text())),
                 "insights": len(insights.read_text().splitlines()),
                 "verified_observations": audit_m3["verified"],
                 "pending_math_validation": audit_m3["pending"],
                 "new_kimi_usage": json.loads((base / "m3/usage.json").read_text()),
                 "isolation": isolation["status"], "independent_utility": "not_tested", "harbor_tasks": 0,
                 "recovered_after_post_generation_audit": True}
        write_json(existing, value)
        return {"benchmark": benchmark, "status": "resumed", "new_model_calls": 0,
                "origin_new_kimi_usage": value["new_kimi_usage"],
                "insights": value["insights"], "isolation": isolation["status"]}
    run_m3(base / "m2", benchmark, base / "m3", None, dry_run=True,
           max_groups=MAX_GROUPS[benchmark], balanced_selection=True, rich_payload=True)
    payload = json.loads(selected.read_text())
    if any(event.get("truncated") for group in payload for event in group["events"]):
        # Full original event remains available through TraceStore; no math root is upgraded from this view.
        truncation = "some events truncated in LLM view; full event SHA/source retained"
    else:
        truncation = "none"
    current = _usage(root)
    budget = plan["policy"]
    remaining = budget["max_new_kimi_total_tokens"] - current["prompt_tokens"] - current["completion_tokens"]
    admission_proxy = len(json.dumps(payload, ensure_ascii=False)) // 2 + 3000
    if dry_run:
        return {"benchmark": benchmark, "status": "dry_run", "selected_groups": len(payload),
                "payload_events": sum(len(group["events"]) for group in payload),
                "admission_proxy_not_api_usage": admission_proxy, "remaining_actual_token_budget": remaining,
                "model_calls": 0, "harbor_tasks": 0, "truncation": truncation}
    if (current["api_calls"] >= budget["max_new_kimi_calls_total"] or
        admission_proxy > remaining or
        (base / "isolated_kimi_cache").is_dir() and
        len(list((base / "isolated_kimi_cache").glob("*.json"))) > budget["max_new_kimi_calls_per_benchmark"]):
        return {"benchmark": benchmark, "status": "budget_stopped", "remaining_actual_token_budget": remaining,
                "admission_proxy_not_api_usage": admission_proxy, "new_model_calls": 0}
    diagnostics: list[dict[str, Any]] = []
    client = KimiClient(root.parent / "model.txt", base / "isolated_kimi_cache")
    start = time.monotonic()
    try:
        result = run_m3(base / "m2", benchmark, base / "m3", client,
                        max_groups=MAX_GROUPS[benchmark], balanced_selection=True, rich_payload=True,
                        math_generalize=False, diagnostics=diagnostics)
    except Exception as exc:
        write_json(base / "m3/generation_error.json", {"stage": "m3", "error_type": type(exc).__name__,
                   "message": str(exc), "actual_usage_so_far": client.usage()})
        raise
    write_json(base / "m3/extraction_diagnostics.json", diagnostics)
    post = audit(root, benchmark, plan)
    if post["status"] != "clean":
        raise ValueError("M3 Source/Holdout isolation failed after generation")
    usage = client.usage()
    manifest = {"benchmark": benchmark, "status": "procedural_hypotheses_generated",
                "selected_groups": result["selected_groups"], "insights": result["insights"],
                "verified_observations": result["verified"], "pending_math_validation": result["pending"],
                "new_kimi_usage": usage, "elapsed_sec": round(time.monotonic() - start, 3),
                "isolation": post["status"], "truncation": truncation,
                "independent_utility": "not_tested", "harbor_tasks": 0}
    write_json(base / "generation_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("spreadsheetbench", "livemath", "alfworld", "all"), default="all")
    parser.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    benchmarks = ("spreadsheetbench", "alfworld", "livemath") if args.benchmark == "all" else (args.benchmark,)
    print(json.dumps([run_one(args.root.resolve(), benchmark, dry_run=args.mode == "dry-run") for benchmark in benchmarks],
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
