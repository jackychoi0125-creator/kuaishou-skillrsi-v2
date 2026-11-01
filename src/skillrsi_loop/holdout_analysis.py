"""Descriptive, task-paired analysis of a frozen official Train holdout."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .crossfit import audit_bundle
from .exposure import audit_trial
from .r16_plan import freeze
from .rule_selection import paired_frequency_signal


def analyze(root: Path, benchmark: str) -> dict[str, Any]:
    base = root / "results/r16" / benchmark
    summary = json.loads((base / "train_internal_validation.json").read_text())
    plan = freeze(root)
    clean = audit_bundle(root, benchmark, plan)["status"] == "clean"
    paired_path = base / "holdout_paired.json"
    paired = json.loads(paired_path.read_text()) if paired_path.is_file() else []
    arms = ("A", "B", "C", "D")
    run_manifest_path = base / "holdout_runs_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text()) if run_manifest_path.is_file() else {"runs": {}}
    action_totals = {arm: {"pi_tool_calls": 0, "submitted_commands": 0,
                           "official_executed_steps": 0, "observed_no_progress_events": 0} for arm in arms}
    for repeat in summary["complete_repeats"]:
        for arm in arms:
            record = run_manifest["runs"][f"repeat{repeat}-{arm}"]
            for trial in record["attempts"][0]["rows"]:
                actions = trial.get("alfworld") or {}
                action_totals[arm]["pi_tool_calls"] += int(trial.get("pi_tool_calls") or 0)
                action_totals[arm]["submitted_commands"] += int(actions.get("submitted_commands") or 0)
                action_totals[arm]["official_executed_steps"] += int(actions.get("official_executed_steps") or 0)
                action_totals[arm]["observed_no_progress_events"] += len(actions.get("observed_no_progress_event_ids") or [])
    counts = {arm: {"successes": sum(row[arm]["reward"] >= .999 for row in paired),
                    "task_runs": len(paired),
                    "input_tokens": sum(row[arm]["input_tokens"] for row in paired),
                    "output_tokens": sum(row[arm]["output_tokens"] for row in paired),
                    "duration_sec": round(sum(row[arm]["duration_sec"] for row in paired), 3)}
              for arm in arms}
    def ratio(numerator: str, denominator: str, field: str) -> float | None:
        value = counts[denominator][field]
        return round(counts[numerator][field] / value, 4) if value else None
    signal_rows = [{"task_id": row["task_id"], "repeat": row["repeat"],
                    "A_reward": row["A"]["reward"], "B_reward": row["B"]["reward"],
                    "D1_reward": row["D"]["reward"]} for row in paired]
    signal = paired_frequency_signal(signal_rows, min_repeats=2)
    descriptive = paired_frequency_signal(signal_rows, min_repeats=1)
    matched_rows = [{"task_id": row["task_id"], "repeat": row["repeat"],
                     "A_reward": row["B"]["reward"], "B_reward": row["C"]["reward"],
                     "D1_reward": row["D"]["reward"]} for row in paired]
    matched_signal = paired_frequency_signal(matched_rows, min_repeats=2)
    exposure = []
    errors = []
    controls = json.loads((base / "controls/controls_manifest.json").read_text())
    for row in paired:
        for arm in arms:
            try:
                skill = controls["arms"].get(arm, {})
                skill_text = Path(skill["skill_path"]).read_text() if skill.get("skill_path") else ""
                match = re.search(r"(?m)^name:\s*(\S+)", skill_text)
                record = audit_trial(Path(row[arm]["trial_path"]), benchmark, configured=arm != "A", arm=arm,
                                     skill_name=match.group(1) if match else None,
                                     skill_sha256=skill.get("skill_sha256"))
                exposure.append({"task_id": row["task_id"], "repeat": row["repeat"], "arm": arm,
                                 "configured": record["configured"], "advertised": record["advertised"],
                                 "explicitly_read": record["explicitly_read"],
                                 "read_event_ids": record["read_event_ids"],
                                 "behavior_consistent": record["behavior_consistent"]})
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append({"task_id": row["task_id"], "repeat": row["repeat"], "arm": arm,
                               "error": type(exc).__name__ + ": " + str(exc)})
    budget = plan["policy"]
    complete = summary["status"] == "complete" and len(summary["complete_repeats"]) >= budget["min_repeats_for_promotion"]
    input_ratio = ratio("D", "B", "input_tokens")
    duration_ratio = ratio("D", "B", "duration_sec")
    reasons = []
    if not clean:
        reasons.append("crossfit_not_clean")
    if not complete:
        reasons.append("fewer_than_prespecified_complete_repeats")
    if errors:
        reasons.append("exposure_audit_error")
    if signal["effective_task_count"] < budget["min_distinct_improved_tasks"]:
        reasons.append("insufficient_independent_tasks")
    if len(signal["improved_task_ids"]) < budget["min_distinct_improved_tasks"]:
        reasons.append("insufficient_improved_tasks_vs_B")
    if len(signal["regressed_task_ids"]) > budget["max_distinct_regressed_tasks"]:
        reasons.append("regression_vs_B")
    if signal["mean_rule_delta_by_task"] is None or signal["mean_rule_delta_by_task"] <= 0:
        reasons.append("no_positive_mean_rule_signal")
    if matched_signal["mean_rule_delta_by_task"] is None or matched_signal["mean_rule_delta_by_task"] <= 0:
        reasons.append("no_rule_signal_beyond_matched_metadata")
    if input_ratio is None or input_ratio > budget["max_input_ratio_D_over_B"]:
        reasons.append("input_cost_limit")
    if duration_ratio is None or duration_ratio > budget["max_duration_ratio_D_over_B"]:
        reasons.append("duration_cost_limit")
    result = {"benchmark": benchmark, "source": "official Harbor Train holdout, paired by task_id and repeat",
              "complete_repeats": summary["complete_repeats"], "effective_distinct_tasks": signal["effective_task_count"],
              "counts": counts, "action_totals": action_totals,
              "descriptive_observed_signal": descriptive,
              "promotion_signal": signal, "metadata_matched_signal_D_minus_C": matched_signal,
              "input_ratio_D_over_B": input_ratio,
              "duration_ratio_D_over_B": duration_ratio, "crossfit_clean": clean,
              "exposure": exposure, "exposure_errors": errors,
              "utility_status": "positive_signal" if not reasons else "unverified",
              "candidate_eligible_for_val": not reasons, "reason_codes": reasons,
              "uncertainty": "Task-cluster intervals are descriptive; repeated runs of one task are not new independent tasks. "
                             "An incomplete repeat block is in-sample diagnostic only and cannot establish utility."}
    write_json(base / "holdout_analysis.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("spreadsheetbench", "alfworld", "livemath"), required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = analyze(args.root.resolve(), args.benchmark)
    print(json.dumps({key: result[key] for key in ("benchmark", "complete_repeats", "counts", "candidate_eligible_for_val", "reason_codes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
