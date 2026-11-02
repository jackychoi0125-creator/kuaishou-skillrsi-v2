"""Freeze R1.7 offline budgets and a future Train-only crossfit boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .r16_plan import BENCHMARKS
from .stability import digest, train_metrics

POLICY = {"schema_version": 1, "new_harbor_tasks": 0,
          "max_new_kimi_total_tokens": 50000,
          "max_new_kimi_calls_per_benchmark": 2,
          "max_new_kimi_calls_total": 4,
          "max_candidates_per_benchmark": 1,
          "max_extra_tool_actions_per_rule": 1,
          "max_extra_checks_per_rule": 1,
          "require_fresh_source_m2_m3_before_independent_validation": True,
          "r16_holdout_used_for_development_decisions": True,
          "val_gate": "unchanged; no new Val in R1.7 without independent Train validation"}


def build(root: Path) -> dict[str, Any]:
    prior = json.loads((root / "results/r16/EXPERIMENT_PLAN.json").read_text())
    studies = {}
    for benchmark in BENCHMARKS:
        train = train_metrics(root, benchmark)
        old_holdout = set(prior["studies"][benchmark]["holdout_task_ids"])
        available = sorted(set(train) - old_holdout)
        successes = [task for task in available if float(train[task].get("reward") or 0) >= .999]
        failures = [task for task in available if float(train[task].get("reward") or 0) < .999]
        if not successes or not failures:
            raise ValueError("future holdout requires historical success and failure")
        # Metadata-only fixed selection. No R1.6 arm score participates.
        choose = lambda ids, count: sorted(ids, key=lambda task: (train[task].get("task_type") or "unknown",
                        digest([benchmark, task, "r17-holdout"])))[:count]
        future = sorted(choose(successes, 1 if benchmark == "livemath" else 2) +
                        choose(failures, 3 if benchmark == "livemath" else 2))
        source = sorted(set(train) - old_holdout - set(future))
        studies[benchmark] = {"future_holdout_task_ids": future, "future_holdout_sha256": digest(future),
                              "source_train_task_ids": source, "source_sha256": digest(source),
                              "consumed_r16_holdout_task_ids": sorted(old_holdout),
                              "excluded_from_source_and_future_holdout": sorted(old_holdout),
                              "future_validation_status": "planned_not_run",
                              "independence_limit": "all original Train traces appeared in historical R1; only new pipeline isolation is possible"}
    plan = {"schema_version": 1, "scope": "Train only", "policy": POLICY, "studies": studies,
            "start_commit": "be2d8dc", "new_harbor_default": 0,
            "historical_r16_cache_replay_is_diagnostic_only": True,
            "future_holdout_scores_not_used_to_choose_rules": True}
    plan["plan_sha256"] = digest(plan)
    return plan


def freeze(root: Path) -> dict[str, Any]:
    path = root / "results/r17/EXPERIMENT_PLAN.json"
    proposed = build(root)
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved != proposed:
            raise ValueError("frozen R1.7 plan or budget changed")
        return saved
    write_json(path, proposed)
    return proposed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    value = freeze(args.root.resolve())
    print(json.dumps({"plan_sha256": value["plan_sha256"], "new_harbor_tasks": 0,
                      "new_kimi_calls": 0, "new_kimi_total_tokens": 0}))


if __name__ == "__main__":
    main()
