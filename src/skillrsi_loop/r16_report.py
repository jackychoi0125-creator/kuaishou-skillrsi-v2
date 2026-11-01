"""Aggregate R1.6 evidence without feeding Val audit content to generation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .crossfit import audit_bundle
from .r16_plan import BENCHMARKS, freeze


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def run(root: Path) -> dict[str, Any]:
    plan = freeze(root)
    out = root / "results/r16"
    exposure = json.loads((out / "exposure_audit.json").read_text())
    records = exposure["records"]
    provenance = [audit_bundle(root, benchmark, plan) for benchmark in BENCHMARKS]
    write_json(out / "crossfit_provenance.json", provenance)
    quality = {}
    candidates = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cache_tokens": 0, "api_calls": 0}
    for benchmark in BENCHMARKS:
        base = out / benchmark
        insights = _jsonl(base / "m3/insights.jsonl")
        quality[benchmark] = {"insights": len(insights),
                              "evidence_validity": dict(Counter(item.get("evidence_validity", "unknown") for item in insights)),
                              "utility_status": dict(Counter(item.get("utility_status", "unverified") for item in insights)),
                              "measurable": sum(bool(item.get("measurement_specificity")) for item in insights),
                              "independent_support_counts": [len({ref["trace_id"] for ref in item.get("supporting_refs", [])}) for item in insights],
                              "causal_status": dict(Counter(item.get("causal_status", "unknown") for item in insights)),
                              "generation_scope": "Source Train only"}
        usage_path = base / "m3/usage.json"
        if usage_path.is_file():
            row = json.loads(usage_path.read_text())
            for key in usage:
                usage[key] += int(row.get(key) or 0)
        candidate_path = base / "m4/candidate_manifest.json"
        candidate = json.loads(candidate_path.read_text()) if candidate_path.is_file() else {
            "benchmark": benchmark, "status": "not_generated"}
        candidate["crossfit_status"] = next(item["status"] for item in provenance if item["benchmark"] == benchmark)
        candidate["accepted"] = False
        candidate["val_eligible"] = False
        candidate["selection_status"] = "no_candidate" if candidate["status"] == "no_candidate" else "not_validated"
        analysis_file = base / "holdout_analysis.json"
        if analysis_file.is_file():
            analysis = json.loads(analysis_file.read_text())
            candidate["holdout_utility_status"] = analysis["utility_status"]
            candidate["holdout_reason_codes"] = analysis["reason_codes"]
            candidate["holdout_complete_repeats"] = analysis["complete_repeats"]
            candidate["val_eligible"] = analysis["candidate_eligible_for_val"]
            candidate["selection_status"] = "eligible_for_val" if candidate["val_eligible"] else "needs_more_evidence"
        candidates.append(candidate)
    write_json(out / "insight_quality.json", quality)
    write_json(out / "rule_candidates.json", candidates)
    configured = sum(item["configured"] for item in records)
    advertised = sum(item["advertised"] == "observed" for item in records)
    reads = sum(item["explicitly_read"] for item in records)
    behavior = sum(item["behavior_consistent"]["state"] == "observed" for item in records)
    holdout_path = out / "alfworld/train_internal_validation.json"
    holdout = json.loads(holdout_path.read_text()) if holdout_path.is_file() else None
    analysis_path = out / "alfworld/holdout_analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else None
    lines = ["# Trace2Skill R1.6 run report", "",
             f"Frozen Train plan SHA: `{plan['plan_sha256']}`. R1 Val gate and no-Skill best version remain unchanged.",
             "Val trajectories were read only for exposure analysis and were not given to M2, M3 or M4.", "",
             "## Observed facts", "",
             f"Exposure audit: {len(records)} historical official executions; {configured} configured Skills, "
             f"{advertised} with independently observable list advertisement, {reads} explicit body reads, "
             f"{behavior} behavior-compatible signals. These states are not causal attribution.",
             "The Kit mounts /skills and registers it with Pi --skill. ATIF omits the Pi system prompt and full Skill list; "
             "missing cat SKILL.md does not prove no metadata effect.",
             "", "| R1 Val Benchmark | Configured candidate runs | Explicit SKILL.md reads |",
             "|---|---:|---:|"]
    for benchmark in BENCHMARKS:
        val_rows = [row for row in records if row["benchmark"] == benchmark and
                    row["source"] == "r1_official_val_read_only" and row["configured"]]
        lines.append(f"| {benchmark} | {len(val_rows)} | {sum(row['explicitly_read'] for row in val_rows)} |")
    lines += ["",
             "Source-only M2 profiles: SpreadsheetBench 76, ALFWorld 35, LiveMath 31; four frozen Holdout tasks per Benchmark were excluded.",
             f"New Kimi API usage: {usage['api_calls']} calls, {usage['prompt_tokens']} Prompt Tokens, "
             f"{usage['completion_tokens']} Completion Tokens, {usage['cache_tokens']} Cache Tokens.", "",
             "| Benchmark | Source M3 insights | Measurable | M4 status | Selection | Isolation |",
             "|---|---:|---:|---|---|---|"]
    for candidate in candidates:
        benchmark = candidate["benchmark"]
        lines.append(f"| {benchmark} | {quality[benchmark]['insights']} | {quality[benchmark]['measurable']} | "
                     f"{candidate['status']} | {candidate['selection_status']} | {candidate['crossfit_status']} |")
    lines += ["", "## Internal Holdout", ""]
    if holdout:
        lines.append(f"ALFWorld official Harbor: status={holdout['status']}; complete repeats={holdout['complete_repeats']}; "
                     f"input Tokens={holdout['usage']['input_tokens']}; output Tokens={holdout['usage']['output_tokens']}; "
                     f"stop reason={holdout['stopped_reason']}.")
        if holdout.get("arm_passes"):
            lines.append("Successes out of four per complete repeat: " + ", ".join(
                f"{arm}={scores}" for arm, scores in holdout["arm_passes"].items()) + ".")
        if analysis:
            signal = analysis["promotion_signal"]
            lines.append(f"Independent tasks with two repeats: {signal['effective_task_count']}; "
                         f"mean B−A={signal['mean_presence_delta_by_task']}, "
                         f"D−B={signal['mean_rule_delta_by_task']}, "
                         f"D−C={analysis['metadata_matched_signal_D_minus_C']['mean_rule_delta_by_task']}; "
                         f"descriptive task-cluster interval "
                         f"{signal['task_cluster_bootstrap_rule_95pct_descriptive']}.")
            if analysis["descriptive_observed_signal"]["effective_task_count"]:
                one = analysis["descriptive_observed_signal"]
                lines.append(f"Observed one-repeat task differences only (not promotion evidence): "
                             f"B−A={one['mean_presence_delta_by_task']}, "
                             f"D−B={one['mean_rule_delta_by_task']}, "
                             f"D−C={analysis['metadata_matched_descriptive_D_minus_C']['mean_rule_delta_by_task']} "
                             f"across {one['effective_task_count']} tasks; D vs B improved "
                             f"{len(one['improved_task_ids'])} and regressed {len(one['regressed_task_ids'])} tasks.")
            lines.append(f"D/B input cost ratio={analysis['input_ratio_D_over_B']}, summed per-task duration ratio="
                         f"{analysis['duration_ratio_D_over_B']}; explicit Skill reads in Holdout="
                         f"{sum(x['explicitly_read'] for x in analysis['exposure'])}/"
                         f"{len(analysis['exposure'])}. Eligible for Val={analysis['candidate_eligible_for_val']}; "
                         f"reasons={analysis['reason_codes']}.")
            lines.append("ALFWorld total Pi calls / submitted commands / official environment steps / observed no-progress events: " +
                         ", ".join(f"{arm}={values['pi_tool_calls']}/{values['submitted_commands']}/"
                                   f"{values['official_executed_steps']}/{values['observed_no_progress_events']}"
                                   for arm, values in analysis["action_totals"].items()) + ".")
            lines.append("Controller wall seconds (arms run tasks concurrently): " + ", ".join(
                f"{arm}={round(values['controller_seconds'], 1)}" for arm, values in analysis["run_errors"].items()) + ".")
    else:
        lines.append("No Holdout Harbor comparison has completed yet.")
    lines += ["", "## Interpretation and open questions", "",
              "Official R1 Val fell from 33/40 to 31/40 on SpreadsheetBench, 16/18 to 15/18 on ALFWorld, "
              "and 7/17 to 4/17 on LiveMath. Those read-only outcomes motivate this audit, not new rules.",
              "R1.5 showed no stable D1 gain across independent Train tasks. SpreadsheetBench's old C regression did not repeat; "
              "ALFWorld D1 regressed relative to B; LiveMath A/A and B varied, and D1 had high Token cost.",
              "Configured Skill, observable advertisement, body read and behavior-compatible events are separate. "
              "This audit cannot identify causal shares of metadata, body and sampling without matched repeated controls.",
              "M3 records evidence validity separately from utility and can propose measurable behavior changes. "
              "M4 compiles one conditional rule with direct Insight and Trace mapping. Utility remains unverified until "
              "independent repeated Holdout results pass cost and regression gates.",
              "No mathematical root cause was upgraded without independent verification. No R2 Skill was accepted and no new Val was run.", ""]
    (out / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"plan_sha256": plan["plan_sha256"], "exposure_records": len(records), "kimi_usage": usage,
            "candidate_statuses": {x["benchmark"]: x["status"] for x in candidates},
            "holdout_status": holdout["status"] if holdout else "not_run", "val_run": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
