"""Cost-aware, Train-only rule triage after repeated official pilot runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .stability import digest, file_sha, train_metrics

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
    bundle = root / "results/stability/crossfit" / benchmark / "provenance.json"
    if bundle.is_file():
        return _verify_crossfit_bundle(root, benchmark, holdout_ids, insight_ids, bundle)
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


def _verify_crossfit_bundle(root: Path, benchmark: str, holdout_ids: list[str],
                            insight_ids: list[str], manifest_path: Path) -> dict[str, Any]:
    """Validate a separately generated Train-only M2/M3 bundle before internal use."""
    try:
        meta = json.loads(manifest_path.read_text())
        source = set(meta["source_train_task_ids"])
        holdout = set(meta["holdout_task_ids"])
        if holdout != set(holdout_ids) or source & holdout or not source:
            raise ValueError("holdout/source task sets overlap or differ")
        if not (source | holdout).issubset(train_metrics(root, benchmark)):
            raise ValueError("crossfit contains non-Train task")
        base = manifest_path.parent.resolve()
        profiles_path = Path(meta["m2_profiles_path"]).resolve()
        insights_path = Path(meta["m3_insights_path"]).resolve()
        cache_root = Path(meta["cache_root"]).resolve()
        for path in (profiles_path, insights_path, cache_root):
            if not path.is_relative_to(base):
                raise ValueError("crossfit artifact escaped isolated bundle")
        if cache_root == (root / "results/r1/cache").resolve() or not meta.get("cache_initially_empty"):
            raise ValueError("cache isolation not established")
        if file_sha(profiles_path) != meta["m2_profiles_sha256"] or file_sha(insights_path) != meta["m3_insights_sha256"]:
            raise ValueError("crossfit artifact SHA mismatch")
        profiles = [json.loads(line) for line in profiles_path.read_text().splitlines() if line.strip()]
        if any(p.get("benchmark") != benchmark or p.get("split") != "train" or
               p.get("task_id") not in source for p in profiles):
            raise ValueError("crossfit M2 profile leaked holdout or non-Train")
        profile_ids = {p["trace_id"] for p in profiles}
        insights = {x["insight_id"]: x for x in (json.loads(line) for line in insights_path.read_text().splitlines() if line.strip())}
        if not insight_ids or any(item_id not in insights for item_id in insight_ids):
            raise ValueError("selected insight absent from crossfit M3")
        for item_id in insight_ids:
            item = insights[item_id]
            if item.get("status") != "verified_observation" or not item.get("supporting_refs"):
                raise ValueError("crossfit insight lacks verified Train support")
            if any(ref["trace_id"] not in profile_ids for ref in item["supporting_refs"] + item.get("counterexample_refs", [])):
                raise ValueError("crossfit insight references holdout/unknown trace")
        return {"status": "clean", "holdout_task_ids_sha256": digest(sorted(holdout)),
                "source_task_ids_sha256": digest(sorted(source)),
                "m2_profiles_sha256": meta["m2_profiles_sha256"],
                "m3_insights_sha256": meta["m3_insights_sha256"],
                "cache_isolation_verified": True, "provenance_path": str(manifest_path)}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "unverifiable", "reason": type(exc).__name__ + ": " + str(exc),
                "provenance_path": str(manifest_path)}


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
    experiment_plan = json.loads((root / "results/stability/EXPERIMENT_PLAN.json").read_text())
    results = [select_one(root, benchmark, policy) for benchmark in
               ("spreadsheetbench", "alfworld", "livemath")]
    lines = ["# R1 Train rule selection", "",
             "Official Harbor Train pilot; predeclared plan SHA: `" + experiment_plan["plan_sha256"] + "`; "
             "rule policy SHA: `" + policy["policy_sha256"] + "`.",
             "Four distinct Train tasks per Benchmark, two runs per arm. Repeated runs of one task are not independent tasks.", "",
             "| Benchmark | Status | B−A | D1−B | D1/B input ratio | Distinct stable improvements | Regressed tasks |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for item in results:
        lines.append(f"| {item['benchmark']} | {item['status']} | {item.get('presence_effect_B_minus_A')} | "
                     f"{item.get('rule_effect_D1_minus_B')} | {item.get('input_ratio_D1_over_B')} | "
                     f"{len(item.get('stable_improved_task_ids', []))} | {len(item.get('regressed_task_ids', []))} |")
    lines += ["", "## Observed costs and variation", ""]
    for item in results:
        path = root / "results/stability" / item["benchmark"] / "stability_summary.json"
        if not path.is_file():
            lines.append(f"- {item['benchmark']}: pilot not completed.")
            continue
        summary = json.loads(path.read_text())
        arms = summary.get("per_arm", {})
        line = [f"- {item['benchmark']}: status={summary['status']}; "]
        for arm in ("A", "B", "D1"):
            data = arms.get(arm, {})
            line.append(f"{arm}={data.get('success_rate')}, input={data.get('total_input_tokens')}, "
                        f"output={data.get('total_output_tokens')}; ")
        line.append(f"A/A changed tasks={sum(bool(x.get('changed')) for x in summary.get('a_a_variation', []))}; "
                    f"error attempts={len(summary.get('error_attempts', []))}.")
        lines.append("".join(line))
        lines.append(f"  Reason codes: {', '.join(item.get('reason_codes', [])) or 'none'}. "
                     f"Extension: {item.get('extension_decision')}.")
    interaction = root / "results/stability/spreadsheetbench/interaction_summary.json"
    if interaction.is_file():
        study = json.loads(interaction.read_text())
        lines += ["", "## SpreadsheetBench rule-combination check", "",
                  "The separately frozen Train C/D2 experiment used the same task IDs and two runs per arm.",
                  "Observed successes and input tokens: " + ", ".join(
                      f"{arm}={value['successes']}/8, {value['input_tokens']} input" for arm, value in study["per_arm"].items()) + ".",
                  f"C versus D1 changed scripts in {study['C_vs_D1_script_changed_task_runs']} task-runs; "
                  f"C versus D2 in {study['C_vs_D2_script_changed_task_runs']} task-runs."]
    lines += ["", "## Interpretation and decision", "",
              "Paired effects and task-cluster bootstrap intervals are descriptive because the pilot has only four independent tasks per Benchmark.",
              "A/B differences measure Skill presence; D1/B differences estimate added rule effect under this controlled setup.",
              "R1 insights used all original Train. Cross-fitting requires rebuilding M2/M3 without held-out tasks and isolating caches.",
              "Pilot data are in-sample diagnosis. A rule needs both the prespecified performance/cost gate and clean Train internal validation before R2 generation.",
              "No candidate means no R2 Skill or new Val run. The official R1 Val gate remains unchanged.", ""]
    report = root / "results/stability/RULE_SELECTION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    summary_paths = [root / "results/stability" / benchmark / "stability_summary.json"
                     for benchmark in ("spreadsheetbench", "alfworld", "livemath")]
    summaries = {path.parent.name: json.loads(path.read_text()) for path in summary_paths if path.is_file()}
    pilot_usage = {key: sum(summary["budget_usage"][key] for summary in summaries.values())
                   for key in ("harbor_task_executions", "input_tokens", "output_tokens", "wall_seconds")}
    interaction_manifest = root / "results/stability/spreadsheetbench/interaction_runs_manifest.json"
    interaction_runs = json.loads(interaction_manifest.read_text())["runs"] if interaction_manifest.is_file() else {}
    interaction_attempts = [attempt for record in interaction_runs.values() for attempt in record["attempts"]]
    total_usage = {key: pilot_usage[key] + sum(
        (len(attempt.get("rows", [])) if key == "harbor_task_executions" else
         sum(row[key] for row in attempt.get("rows", [])) if key in ("input_tokens", "output_tokens") else
         attempt.get("elapsed_sec", 0)) for attempt in interaction_attempts)
        for key in pilot_usage}
    run_lines = ["# R1 stability execution report", "",
                 "All new official scores below come from frozen Train task sets; original R1 Val is read-only.",
                 f"Pilot plan SHA: `{experiment_plan['plan_sha256']}`. Rule policy SHA: `{policy['policy_sha256']}`.", "",
                 "| Benchmark | A repeats | B repeats | D1 repeats | A/A changed tasks | D1 status |",
                 "|---|---:|---:|---:|---:|---|"]
    for item in results:
        summary = summaries.get(item["benchmark"])
        if not summary:
            run_lines.append(f"| {item['benchmark']} | pending | pending | pending | pending | {item['status']} |")
            continue
        scores = [", ".join(f"{round(rate * 4)}/4" for rate in summary["per_arm"][arm]["repeat_success_rates"])
                  for arm in ("A", "B", "D1")]
        changed = sum(bool(x["changed"]) for x in summary["a_a_variation"])
        run_lines.append(f"| {item['benchmark']} | {scores[0]} | {scores[1]} | {scores[2]} | {changed} | {item['status']} |")
    run_lines += ["", "Pilot: " + ", ".join(f"{key}={round(value, 3)}" for key, value in pilot_usage.items()) + ".",
                  "Pilot plus Spreadsheet C/D2 interaction: " + ", ".join(
                      f"{key}={round(value, 3)}" for key, value in total_usage.items()) + ".",
                  "See local manifests for all Run paths, skill hashes, config hashes, scorer hashes, and attempts.",
                  f"Incomplete Benchmark pilots: {', '.join(b for b, s in summaries.items() if s['status'] != 'complete') or 'none'}. "
                  f"Missing summaries: {', '.join(sorted(set(('spreadsheetbench', 'alfworld', 'livemath')) - set(summaries))) or 'none'}.",
                  "Recorded infrastructure error attempts: " + str(sum(len(s.get("error_attempts", [])) for s in summaries.values()) +
                  sum(a.get("status") != "valid" for a in interaction_attempts)) + ".", "",
                  "## Decision", "",
                  "No D1 rule passed the prespecified Train performance, repeatability, cost, and clean cross-fitting gates."
                  if all(item["status"] != "promote_candidate" for item in results) else
                  "At least one rule requires separate clean Train validation before R2 generation.",
                  "No stratified Train extension, R2 Skill, internal R2 validation, or new official Val run was performed."
                  if all(item["status"] != "promote_candidate" for item in results) else
                  "See per-rule manifests for the next internal validation stage.",
                  "The R1 Val gate and no-Skill best version are unchanged. Pilot comparisons are in-sample diagnostics.",
                  "Possible explanations for score differences include model sampling, Skill presence, explicit rule use, "
                  "and run-time path variation; this pilot does not identify their causal shares.", ""]
    (root / "results/stability/RUN_REPORT.md").write_text("\n".join(run_lines), encoding="utf-8")
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
