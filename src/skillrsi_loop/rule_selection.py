"""Cost-aware, Train-only rule triage after repeated official pilot runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .stability import digest

POLICY = {
    "schema_version": 1,
    "min_complete_repeats": 2,
    "min_distinct_improved_tasks_vs_B": 2,
    "min_mean_reward_delta_vs_B": 0.05,
    "max_distinct_regressed_tasks_vs_B": 0,
    "max_input_token_ratio_vs_B": 1.5,
    "max_duration_ratio_vs_B": 1.5,
    "high_cost_input_ratio": 2.0,
    "high_cost_without_stable_gain": True,
    "require_clean_train_crossfit_for_promotion": True,
    "val_gate": "unchanged R1 official reward gate; no Val unless R2 passes clean Train internal validation",
}


def freeze_policy(root: Path) -> dict[str, Any]:
    path = root / "results/stability/RULE_SELECTION_POLICY.json"
    proposed = {**POLICY, "policy_sha256": digest(POLICY)}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != proposed:
            raise ValueError("predeclared rule-selection policy changed")
        return existing
    write_json(path, proposed)
    return proposed


def crossfit_audit(root: Path, benchmark: str, holdout_ids: list[str],
                   insight_ids: list[str]) -> dict[str, Any]:
    """Fail closed when any proposed holdout entered R1 M2/M3 or cache scope is unknown."""
    m2 = root / "results/m2"
    profiles_path = m2 / "trace_profiles.jsonl"
    if not profiles_path.is_file():
        return {"status": "unverifiable", "reason": "missing_m2_profiles"}
    profiles = [json.loads(line) for line in profiles_path.read_text().splitlines() if line.strip()]
    relevant = {p["trace_id"]: p for p in profiles if p.get("benchmark") == benchmark and p.get("split") == "train"}
    contaminated_profiles = [p["trace_id"] for p in relevant.values() if p.get("task_id") in holdout_ids and p.get("eligible_for_learning")]
    used_groups = set()
    insight_path = root / "results/r1" / benchmark / "m3/insights.jsonl"
    if insight_path.is_file():
        for line in insight_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("insight_id") in insight_ids:
                used_groups.update(item.get("source_group_ids", []))
    contaminated_groups = []
    for filename in ("positive_strategy_groups.json", "failure_pattern_groups.json"):
        path = m2 / filename
        if not path.is_file():
            continue
        for group in json.loads(path.read_text()):
            if group.get("group_id") in used_groups and any(
                    relevant.get(t, {}).get("task_id") in holdout_ids for t in group.get("trace_ids", [])):
                contaminated_groups.append(group["group_id"])
    for item in json.loads((m2 / "contrast_groups.json").read_text()) if (m2 / "contrast_groups.json").is_file() else []:
        if item.get("group_id") in used_groups and any(
                relevant.get(t, {}).get("task_id") in holdout_ids for t in item.get("trace_ids", [])):
            contaminated_groups.append(item["group_id"])
    status = "contaminated" if contaminated_profiles or contaminated_groups else "unverifiable"
    return {"status": status, "holdout_task_ids_sha256": digest(sorted(holdout_ids)),
            "candidate_insight_ids": insight_ids, "contaminated_profile_count": len(contaminated_profiles),
            "contaminated_group_ids": sorted(set(contaminated_groups)),
            "cache_isolation_verified": False,
            "reason": "R1 M2 was built from full Train and cache was not split by holdout"
                      if status != "clean" else "isolated"}


def _paired_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return rows


def select_one(root: Path, benchmark: str, policy: dict[str, Any]) -> dict[str, Any]:
    base = root / "results/stability" / benchmark
    summary_path = base / "stability_summary.json"
    pairs = _paired_records(base / "paired_repeats.csv")
    arms = json.loads((base / "arms_manifest.json").read_text())["arms"]
    insight_ids = arms["D1"]["insight_ids"]
    insight_path = root / "results/r1" / benchmark / "m3/insights.jsonl"
    pool = {x["insight_id"]: x for x in (json.loads(line) for line in insight_path.read_text().splitlines()
                                        if line.strip())} if insight_path.is_file() else {}
    evidence_ready = bool(insight_ids) and all(
        item_id in pool and pool[item_id].get("status") == "verified_observation" and
        pool[item_id].get("trigger_conditions") and pool[item_id].get("supporting_refs")
        for item_id in insight_ids)
    if not summary_path.is_file():
        return {"benchmark": benchmark, "rule_arm": "D1", "status": "needs_more_evidence",
                "reason": "pilot_not_run", "insight_ids": insight_ids}
    summary = json.loads(summary_path.read_text())
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_task.setdefault(row["task_id"], []).append(row)
    improved = sorted(task for task, rows in by_task.items() if len(rows) >= policy["min_complete_repeats"] and
                      all(float(r["D1_reward"]) > float(r["B_reward"]) for r in rows))
    regressed = sorted(task for task, rows in by_task.items() if any(
        float(r["D1_reward"]) < float(r["B_reward"]) for r in rows))
    input_d1 = sum(float(r["D1_input_tokens"]) for r in pairs)
    input_b = sum(float(r["B_input_tokens"]) for r in pairs)
    duration_d1 = sum(float(r["D1_duration_sec"]) for r in pairs)
    duration_b = sum(float(r["B_duration_sec"]) for r in pairs)
    input_ratio = input_d1 / input_b if input_b else None
    duration_ratio = duration_d1 / duration_b if duration_b else None
    effect = summary["effects"]["rule_effect"]["paired_mean_reward_delta"]
    presence = summary["effects"]["presence_effect"]["paired_mean_reward_delta"]
    holdout = list(summary.get("a_a_variation", []))
    crossfit = crossfit_audit(root, benchmark, [x["task_id"] for x in holdout], insight_ids)
    reasons = []
    if summary["status"] != "complete":
        reasons.append("incomplete_paired_repeats")
    if len(improved) < policy["min_distinct_improved_tasks_vs_B"]:
        reasons.append("insufficient_independent_improved_tasks")
    if effect is None or effect < policy["min_mean_reward_delta_vs_B"]:
        reasons.append("no_repeated_D1_over_B_gain")
    if regressed:
        reasons.append("paired_regression_vs_B")
    if input_ratio is not None and input_ratio > policy["max_input_token_ratio_vs_B"]:
        reasons.append("input_cost_above_promotion_limit")
    if duration_ratio is not None and duration_ratio > policy["max_duration_ratio_vs_B"]:
        reasons.append("duration_cost_above_promotion_limit")
    if crossfit["status"] != "clean":
        reasons.append("no_clean_crossfit_internal_validation")
    if not evidence_ready:
        reasons.append("missing_verified_train_insight_or_trigger")
    if summary["status"] != "complete":
        status = "needs_more_evidence"
    elif len(regressed) >= 2 and not improved:
        status = "reject"
    elif regressed:
        status = "regression_risk"
    elif input_ratio is not None and input_ratio >= policy["high_cost_input_ratio"] and len(improved) < policy["min_distinct_improved_tasks_vs_B"]:
        status = "high_cost"
    elif not reasons:
        status = "promote_candidate"
    else:
        status = "needs_more_evidence"
    a_variation = sum(x.get("changed") for x in summary.get("a_a_variation", []))
    extension_ready = (summary["status"] == "complete" and
                       len(improved) >= policy["min_distinct_improved_tasks_vs_B"] and not regressed and
                       effect is not None and effect >= policy["min_mean_reward_delta_vs_B"] and
                       input_ratio is not None and input_ratio <= policy["max_input_token_ratio_vs_B"] and
                       duration_ratio is not None and duration_ratio <= policy["max_duration_ratio_vs_B"] and
                       a_variation == 0)
    result = {"benchmark": benchmark, "rule_arm": "D1", "insight_ids": insight_ids,
              "status": status, "reason_codes": reasons,
              "extension_decision": "recommend_stratified_train_extension" if extension_ready else "skip_extension",
              "extension_reason": "pilot signal and cost met prespecified thresholds" if extension_ready else
                                  "pilot signal, stability, or cost below extension gate",
              "presence_effect_B_minus_A": presence, "rule_effect_D1_minus_B": effect,
              "rule_effect_cluster_bootstrap": summary["effects"]["rule_effect"]["task_cluster_bootstrap_95pct_descriptive"],
              "stable_improved_task_ids": improved, "regressed_task_ids": regressed,
              "n_distinct_tasks": len(by_task), "n_paired_task_runs": len(pairs),
              "input_ratio_D1_over_B": round(input_ratio, 4) if input_ratio is not None else None,
              "duration_ratio_D1_over_B": round(duration_ratio, 4) if duration_ratio is not None else None,
              "crossfit_audit": crossfit, "validation_scope": "in_sample_diagnostic_only",
              "train_insight_evidence_ready": evidence_ready,
              "policy_sha256": policy["policy_sha256"],
              "interpretation": "descriptive paired differences; small repeated Train sample, not independent generalization"}
    write_json(base / "rule_selection.json", result)
    return result


def run(root: Path) -> list[dict[str, Any]]:
    policy = freeze_policy(root)
    results = [select_one(root, benchmark, policy) for benchmark in
               ("spreadsheetbench", "alfworld", "livemath")]
    lines = ["# R1 Train rule selection", "",
             "Prespecified policy and official Train pilot. Repeated runs of one task are not independent tasks.", "",
             "| Benchmark | Status | B−A | D1−B | D1/B input ratio | Distinct stable improvements | Regressed tasks |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for item in results:
        lines.append(f"| {item['benchmark']} | {item['status']} | {item.get('presence_effect_B_minus_A')} | "
                     f"{item.get('rule_effect_D1_minus_B')} | {item.get('input_ratio_D1_over_B')} | "
                     f"{len(item.get('stable_improved_task_ids', []))} | {len(item.get('regressed_task_ids', []))} |")
    lines += ["", "R1 insights used all original Train. Cross-fitting requires rebuilding M2/M3 without held-out tasks and isolating caches.",
              "Pilot data are in-sample diagnosis; no R2 candidate or Val run is authorized by this report alone.", ""]
    report = root / "results/stability/RULE_SELECTION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "results/stability/RULE_SELECTION.json", results)
    for item in results:
        out = root / "results/r2" / item["benchmark"]
        write_json(out / "candidate_manifest.json", {"status": "no_candidate" if item["status"] != "promote_candidate" else "requires_clean_crossfit_validation",
                                                    "source": "Train stability rule selection", "rule_status": item["status"],
                                                    "val_status": "skipped", "reason_codes": item.get("reason_codes", [])})
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve()), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
