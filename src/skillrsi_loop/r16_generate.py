"""Run Source-only M3 and compile at most one unaccepted single-rule candidate."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from skillrsi_m3.client import KimiClient
from skillrsi_m3.core import EvidenceVerifier, TraceStore, run as run_m3, write_json
from skillrsi_m4.core import compile_single_rule
from .crossfit import audit_bundle
from .r16_plan import BENCHMARKS, freeze
from .stability import file_sha


def _prior_usage(root: Path) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0}
    for benchmark in BENCHMARKS:
        path = root / "results/r16" / benchmark / "m3/usage.json"
        if path.is_file():
            usage = json.loads(path.read_text())
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return totals


def run_one(root: Path, benchmark: str, *, dry_run: bool = False) -> dict[str, Any]:
    plan = freeze(root)
    base = root / "results/r16" / benchmark
    preflight = audit_bundle(root, benchmark, plan)
    if preflight["status"] not in {"clean_source_m2", "clean"}:
        return {"benchmark": benchmark, "status": "blocked_isolation", "audit": preflight, "model_calls": 0}
    m3_dir = base / "m3"
    m4_dir = base / "m4"
    deterministic_only = benchmark == "livemath"
    max_groups = 8 if benchmark == "spreadsheetbench" else 3 if benchmark == "alfworld" else 2
    if dry_run:
        selected = run_m3(base / "m2", benchmark, m3_dir, None, dry_run=True, max_groups=max_groups)
        compiled = compile_single_rule(benchmark, m3_dir / "insights.jsonl", base / "skills", m4_dir, dry_run=True)
        return {"benchmark": benchmark, "status": "dry_run", "m3": selected, "m4": compiled,
                "model_calls": 0, "harbor_tasks": 0}
    current = _prior_usage(root)
    budget = plan["policy"]
    if current["prompt_tokens"] >= budget["max_new_kimi_prompt_tokens"] or current["completion_tokens"] >= budget["max_new_kimi_completion_tokens"]:
        return {"benchmark": benchmark, "status": "budget_stopped", "usage": current}
    remaining_prompt = budget["max_new_kimi_prompt_tokens"] - current["prompt_tokens"]
    if not deterministic_only and not (m3_dir / "insights.jsonl").is_file():
        selected_path = m3_dir / "selected_evidence.json"
        if not selected_path.is_file():
            run_m3(base / "m2", benchmark, m3_dir, None, dry_run=True, max_groups=max_groups)
        selected = json.loads(selected_path.read_text())
        # Admission bound only. Actual model tokens are always recorded from the Kimi API.
        while max_groups and (len(json.dumps(selected[:max_groups], ensure_ascii=False)) // 2 + 1000) > remaining_prompt:
            max_groups -= 1
        if not max_groups:
            return {"benchmark": benchmark, "status": "budget_stopped", "reason": "prompt_admission_bound",
                    "remaining_prompt_tokens": remaining_prompt, "model_calls": 0}
    insight_path = m3_dir / "insights.jsonl"
    if insight_path.is_file():
        old_usage = m3_dir / "usage.json"
        if not old_usage.is_file() or preflight["status"] != "clean":
            raise ValueError("existing M3 output lacks clean resume audit")
        original = m3_dir / "insights_pre_operationalization.jsonl"
        if not original.is_file():
            original.write_bytes(insight_path.read_bytes())
        pool = [json.loads(line) for line in insight_path.read_text().splitlines() if line.strip()]
        if pool:
            audit = EvidenceVerifier(TraceStore(base / "m2/trace_profiles.jsonl", benchmark)).verify(pool)
            insight_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in pool))
            write_json(m3_dir / "insight_audit.json", audit)
        m3_result: dict[str, Any] = {"status": "resumed", "insights_sha256": file_sha(insight_path)}
    else:
        cache = base / "isolated_kimi_cache"
        if cache.is_dir() and any(cache.iterdir()):
            raise ValueError("fresh crossfit M3 must start with empty isolated cache")
        client = None if deterministic_only else KimiClient(root.parent / "model.txt", cache)
        start = time.monotonic()
        try:
            m3_result = run_m3(base / "m2", benchmark, m3_dir, client,
                               max_groups=max_groups, deterministic_only=deterministic_only)
        except Exception as exc:
            write_json(base / "generation_error.json", {"benchmark": benchmark, "stage": "m3",
                "error_type": type(exc).__name__, "message": str(exc),
                "usage": client.usage() if client is not None else {"api_calls": 0}})
            raise
        m3_result["elapsed_sec"] = round(time.monotonic() - start, 3)
    audit = audit_bundle(root, benchmark, plan)
    if audit["status"] != "clean":
        return {"benchmark": benchmark, "status": "blocked_post_m3_isolation", "audit": audit,
                "m3": m3_result}
    compiled = compile_single_rule(benchmark, insight_path, base / "skills", m4_dir)
    audit = audit_bundle(root, benchmark, plan)
    if audit["status"] != "clean":
        raise ValueError("M4 source prompt or candidate failed holdout isolation")
    result = {"benchmark": benchmark, "status": "candidate_for_internal_validation" if compiled["status"] == "candidate_for_internal_validation" else "no_candidate",
              "m3": m3_result, "m4": compiled, "audit": audit,
              "m3_max_groups": max_groups, "m3_deterministic_only": deterministic_only,
              "cumulative_kimi_usage": _prior_usage(root)}
    write_json(base / "generation_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", *BENCHMARKS), default="all")
    parser.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    values = [run_one(root, benchmark, dry_run=args.mode == "dry-run")
              for benchmark in BENCHMARKS if args.benchmark == "all"] if args.benchmark == "all" else [
                  run_one(root, args.benchmark, dry_run=args.mode == "dry-run")]
    print(json.dumps(values, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
