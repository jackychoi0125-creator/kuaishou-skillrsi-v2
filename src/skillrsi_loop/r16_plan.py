"""Freeze a Train-only R1.6 source/holdout split and cost policy before new runs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .stability import digest, train_metrics

BENCHMARKS = ("spreadsheetbench", "alfworld", "livemath")
POLICY = {"schema_version": 1, "holdout_tasks_per_benchmark": 4,
          "harbor_arms": ["A", "B", "C", "D"], "min_repeats_for_promotion": 2,
          "max_new_kimi_prompt_tokens": 30000, "max_new_kimi_completion_tokens": 8000,
          "max_harbor_task_executions": 96, "max_harbor_input_tokens": 7_000_000,
          "max_harbor_output_tokens": 700_000, "max_wall_seconds": 14400,
          "max_input_ratio_D_over_B": 1.5, "max_duration_ratio_D_over_B": 1.5,
          "min_distinct_improved_tasks": 2, "max_distinct_regressed_tasks": 0,
          "require_clean_crossfit": True, "require_measurable_behavior_change": True,
          "val_gate": "R1 official gate unchanged; no Val before clean Train validation"}


def _stratified_holdout(rows: dict[str, dict[str, str]], benchmark: str) -> list[str]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for task_id, row in rows.items():
        outcome = "success" if float(row.get("reward") or 0) >= .999 else "failure"
        buckets[outcome].append(task_id)
    if not buckets["success"] or not buckets["failure"]:
        raise ValueError(f"Train split lacks both outcomes: {benchmark}")
    targets = {"success": 1 if benchmark == "livemath" else 2,
               "failure": 3 if benchmark == "livemath" else 2}
    chosen: list[str] = []
    for outcome, count in targets.items():
        ordered = sorted(buckets[outcome], key=lambda task: (
            rows[task].get("task_type", "unknown"),
            int(rows[task].get("input_tokens") or 0) // 50000,
            int(rows[task].get("tool_calls") or 0) // 10,
            digest([benchmark, task])))
        families: dict[str, list[str]] = defaultdict(list)
        for task in ordered:
            families[rows[task].get("task_type") or "unknown"].append(task)
        while len([task for task in chosen if task in buckets[outcome]]) < count:
            progressed = False
            for family in sorted(families, key=lambda name: (len(families[name]), name)):
                if families[family]:
                    chosen.append(families[family].pop(0))
                    progressed = True
                    if len([task for task in chosen if task in buckets[outcome]]) == count:
                        break
            if not progressed:
                raise ValueError(f"not enough {outcome} Train tasks")
    return sorted(chosen)


def build_plan(root: Path) -> dict[str, Any]:
    studies = {}
    for benchmark in BENCHMARKS:
        rows = train_metrics(root, benchmark)
        holdout = _stratified_holdout(rows, benchmark)
        source = sorted(set(rows) - set(holdout))
        studies[benchmark] = {"source_train_task_ids": source, "holdout_task_ids": holdout,
                              "source_sha256": digest(source), "holdout_sha256": digest(holdout),
                              "full_train_sha256": digest(sorted(rows)),
                              "holdout_strata": {task: {"task_type": rows[task].get("task_type"),
                                "historical_outcome": "success" if float(rows[task].get("reward") or 0) >= .999 else "failure",
                                "input_tokens": int(rows[task].get("input_tokens") or 0),
                                "tool_calls": int(rows[task].get("tool_calls") or 0)} for task in holdout}}
    value = {"schema_version": 1, "scope": "Train only", "origin": "baseline metrics before R1.6 candidate generation",
             "policy": POLICY, "studies": studies,
             "candidate_rule": "one structured Source-Train Insight per Benchmark at most",
             "experiment_order": ["isolation", "exposure_read_only", "static_tests", "dry_run", "source_m2_m3_m4", "holdout_abcd_if_candidate"],
             "previous_R1_full_train_insights_are_not_independent": True}
    value["plan_sha256"] = digest(value)
    return value


def freeze(root: Path) -> dict[str, Any]:
    path = root / "results/r16/EXPERIMENT_PLAN.json"
    proposal = build_plan(root)
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != proposal:
            raise ValueError("frozen R1.6 Train plan changed")
        return existing
    write_json(path, proposal)
    return proposal


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    plan = freeze(args.root.resolve())
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "model_calls": 0,
                      "holdout_hashes": {b: x["holdout_sha256"] for b, x in plan["studies"].items()}}))


if __name__ == "__main__":
    main()
