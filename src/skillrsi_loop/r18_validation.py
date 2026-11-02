"""Frozen R1.8 SpreadsheetBench Train holdout; official Harbor only."""
from __future__ import annotations

import argparse
import json
import re
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from skillrsi_m1.core import events_from_trace
from .cli import config
from .gate import compare
from .holdout import _sampling
from .presence_control import build as build_controls
from skillrsi_m4.core import validate_skill
from .r17_plan import freeze as r17_freeze
from .r17_source import audit as source_audit
from .stability import _attempt_status, _scorer_sha, _trial_rows, digest, file_sha

ARMS = ("A", "B", "C", "D")
ORDER = ((0, "A"), (0, "B"), (0, "C"), (0, "D"),
         (1, "D"), (1, "C"), (1, "B"), (1, "A"))
POLICY = {"max_harbor_task_executions": 32, "max_input_tokens": 5_000_000,
          "max_output_tokens": 600_000, "max_wall_seconds": 14400,
          "max_input_ratio_d_over_b": 1.5, "max_duration_ratio_d_over_b": 1.5,
          "min_distinct_improved_tasks_vs_b_and_c": 2,
          "max_distinct_regressed_tasks": 0, "min_complete_repeats": 2,
          "infrastructure_retries": 0, "max_new_kimi_tokens": 0,
          "val_if_train_gate_passes": True, "val_gate": "existing_r1_gate_unchanged"}


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def freeze(root: Path) -> dict[str, Any]:
    prior = r17_freeze(root)
    check = source_audit(root, "spreadsheetbench", prior)
    if check["status"] != "clean":
        raise ValueError("R1.7 Source provenance failed: " + str(check["errors"]))
    candidate = json.loads((root / "results/r17/spreadsheetbench/m4/candidate_manifest.json").read_text())
    skill = Path(candidate["skill_path"])
    if candidate.get("status") != "candidate_for_internal_validation" or candidate.get("accepted") or not skill.is_file():
        raise ValueError("R1.7 candidate unavailable or already accepted")
    if file_sha(skill) != candidate["skill_sha256"]:
        raise ValueError("R1.7 Skill SHA mismatch")
    source_insights = [json.loads(line) for line in (root / "results/r17/spreadsheetbench/m3/insights.jsonl").read_text().splitlines() if line.strip()]
    selected = [item for item in source_insights if item["insight_id"] in candidate["rule"]["source_insight_ids"]]
    if len(selected) != len(candidate["rule"]["source_insight_ids"]):
        raise ValueError("candidate Insight source missing")
    static_errors = validate_skill(skill.read_text(), "spreadsheetbench", selected)
    if static_errors:
        raise ValueError("candidate static safety check failed: " + str(static_errors))
    base = root / "results/r18"
    write_json(base / "spreadsheetbench/preflight.json", {"candidate_exists": True,
               "candidate_skill_sha256": candidate["skill_sha256"], "candidate_accepted": False,
               "source_audit": check, "source_insight_ids": candidate["rule"]["source_insight_ids"],
               "static_safety_errors": static_errors, "future_holdout_excluded": True})
    controls = build_controls(root / "results/r17/spreadsheetbench/m4/candidate_manifest.json",
                              base / "spreadsheetbench/controls")
    controls["arms"]["D"] = {"skills_dir": candidate["skills_dir"],
                                "skill_path": candidate["skill_path"], "skill_sha256": candidate["skill_sha256"]}
    write_json(base / "spreadsheetbench/controls/controls_manifest.json", controls)
    study = prior["studies"]["spreadsheetbench"]
    holdout = study["future_holdout_task_ids"]
    kit = (root.parent / "kits/spreadsheet/spreadsheetbench-harbor-kit-v3.3-layout2").resolve()
    endpoint = config(kit, "spreadsheetbench")
    selection = _sampling(root, "spreadsheetbench", set(holdout))
    if selection["mode"] != "official_seed_sample":
        raise ValueError("Kit cannot select exact frozen Holdout")
    plan = {"schema_version": 1, "round": "r18", "start_commit": "23861d6",
            "r17_plan_sha256": prior["plan_sha256"], "benchmark": "spreadsheetbench", "split": "train",
            "source_task_ids_sha256": study["source_sha256"],
            "holdout_task_ids": holdout, "holdout_task_ids_sha256": study["future_holdout_sha256"],
            "sampling": selection, "order": [list(x) for x in ORDER], "concurrent": 5,
            "arms": {a: {"skill_sha256": controls["arms"][a].get("skill_sha256"),
                          "skills_dir": controls["arms"][a].get("skills_dir")} for a in ARMS},
            "candidate_insight_ids": candidate["rule"]["source_insight_ids"],
            "candidate_skill_sha256": candidate["skill_sha256"],
            "kit_run_sha256": file_sha(kit / "run.sh"),
            "scorer_sha256": _scorer_sha(kit, "spreadsheetbench"),
            "model_id": endpoint["model_id"], "model_api": endpoint["api"],
            "base_url": endpoint["base_url"], "reasoning": "off",
            "model_sampling": "Kit default; fixed task seed is not a deterministic model seed",
            "timeout_seconds_per_job": 1800, "policy": POLICY,
            "independence_limit": "R1 historically used full Train; only R1.7 Source pipeline is isolated"}
    plan["plan_sha256"] = digest(plan)
    path = base / "EXPERIMENT_PLAN.json"
    if path.is_file() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen R1.8 plan changed")
    write_json(path, plan)
    return plan


def freeze_val(root: Path, train_plan: dict[str, Any]) -> dict[str, Any]:
    """Freeze a separate 40-task Val budget before reading Holdout outcomes."""
    old = json.loads((root / "results/r1/spreadsheetbench/val/acceptance.json").read_text())
    prior_state = json.loads((root / "results/r1/spreadsheetbench/stage_manifest.json").read_text())
    baseline_conf = prior_state["stages"]["baseline"]["config"]
    kit = (root.parent / "kits/spreadsheet/spreadsheetbench-harbor-kit-v3.3-layout2").resolve()
    current_conf = config(kit, "spreadsheetbench")
    if baseline_conf != current_conf or old["baseline_pass"] != 33 or old["tasks"] != 40:
        raise ValueError("existing no-Skill Val baseline config does not match current Kit")
    value = {"schema_version": 1, "train_plan_sha256": train_plan["plan_sha256"],
             "baseline_job": old["baseline_job"], "baseline_config": baseline_conf,
             "candidate_skill_sha256": train_plan["candidate_skill_sha256"],
             "benchmark": "spreadsheetbench-val", "max_new_harbor_tasks": 40,
             "max_input_tokens": 3_000_000, "max_output_tokens": 300_000,
             "max_wall_seconds": 7200, "concurrent": 5, "reasoning": "off",
             "infrastructure_retries": 0, "gate": "positive_net_gain_and_zero_regressions",
             "execute_only_if_train_gate_eligible": True}
    value["plan_sha256"] = digest(value)
    path = root / "results/r18/VAL_PLAN.json"
    if path.is_file() and json.loads(path.read_text()) != value:
        raise ValueError("frozen Val plan changed")
    write_json(path, value)
    return value


def usage(state: dict[str, Any]) -> dict[str, float]:
    attempts = [a for run in state["runs"].values() for a in run.get("attempts", [])]
    rows = [row for a in attempts for row in a.get("rows", [])]
    return {"task_executions": 4 * len(attempts),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "wall_seconds": sum(a.get("elapsed_sec", 0) for a in attempts)}


def attempt(root: Path, plan: dict[str, Any], repeat: int, arm: str) -> dict[str, Any]:
    kit = (root.parent / "kits/spreadsheet/spreadsheetbench-harbor-kit-v3.3-layout2").resolve()
    jobs = kit / "workspace/runs/spreadsheetbench-train/r18-holdout" / f"repeat{repeat}-{arm}"
    if jobs.exists() and any(jobs.iterdir()):
        raise ValueError("populated unregistered Job directory")
    jobs.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(kit / "run.sh"), "--bench", "spreadsheetbench-train", "--n", "4",
           "--seed", str(plan["sampling"]["seed"]), "--concurrent", str(plan["concurrent"]),
           "--reasoning", "off", "--jobs-dir", str(jobs)]
    skills = plan["arms"][arm]["skills_dir"]
    if skills:
        cmd += ["--skills-dir", skills]
    log = jobs.with_name(jobs.name + "-controller.log")
    start = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(cmd, cwd=kit, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=plan["timeout_seconds_per_job"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    summaries = list(jobs.glob("*/summary.csv"))
    job = summaries[0].parent if len(summaries) == 1 else None
    result = {"job": str(job) if job else None, "jobs_root": str(jobs),
              "controller_log": str(log), "exit_code": code,
              "timed_out": code == 124,
              "elapsed_sec": round(time.monotonic() - start, 3)}
    if job:
        try:
            result["rows"], result["errors"] = _trial_rows(job, "spreadsheetbench")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            result["parse_error"] = str(error)
    result["status"] = _attempt_status(result, set(plan["holdout_task_ids"]))
    return result


def analyze(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    observed_attempts = {key: {"status": run["status"],
                               "official_passes": sum(row["reward"] >= .999 for row in run["attempts"][0].get("rows", []))
                               if run.get("attempts") else None,
                               "official_scored_tasks": len(run["attempts"][0].get("rows", []))
                               if run.get("attempts") else 0,
                               "configured": key.rsplit("-", 1)[-1] != "A",
                               "discovery_evidence_count": sum(bool(row.get("skill_discovery_event_ids"))
                                                               for attempt in run.get("attempts", [])
                                                               for row in attempt.get("rows", [])),
                               "explicit_read_count": sum(bool(row.get("skill_read_event_ids"))
                                                          for attempt in run.get("attempts", [])
                                                          for row in attempt.get("rows", [])),
                               "input_tokens": sum(row.get("input_tokens", 0) for attempt in run.get("attempts", [])
                                                   for row in attempt.get("rows", [])),
                               "output_tokens": sum(row.get("output_tokens", 0) for attempt in run.get("attempts", [])
                                                    for row in attempt.get("rows", [])),
                               "duration_sec": sum(row.get("duration_sec", 0) for attempt in run.get("attempts", [])
                                                   for row in attempt.get("rows", [])),
                               "pi_tool_calls": sum(row.get("pi_tool_calls", 0) for attempt in run.get("attempts", [])
                                                    for row in attempt.get("rows", []))}
                         for key, run in state["runs"].items()}
    observed_arm_cost = {arm: {metric: sum(item[metric] for key, item in observed_attempts.items()
                                                   if key.endswith("-" + arm))
                               for metric in ("input_tokens", "output_tokens", "duration_sec", "pi_tool_calls")}
                         for arm in ARMS}
    observed_d_triggers = [_formula_trigger_check(row)
                           for key, run in state["runs"].items() if key.endswith("-D")
                           for attempt in run.get("attempts", []) for row in attempt.get("rows", [])]
    completed = [r for r in (0, 1) if all(state["runs"].get(f"repeat{r}-{a}", {}).get("status") == "valid" for a in ARMS)]
    paired = []
    for repeat in completed:
        by_arm = {arm: {row["task_id"]: row for row in state["runs"][f"repeat{repeat}-{arm}"]["attempts"][0]["rows"]}
                  for arm in ARMS}
        for task in plan["holdout_task_ids"]:
            paired.append({"task_id": task, "repeat": repeat, "arms": {a: by_arm[a][task] for a in ARMS}})
    counts = {a: sum(x["arms"][a]["reward"] >= .999 for x in paired) for a in ARMS}
    frequencies = {task: {a: {"passes": sum(x["arms"][a]["reward"] >= .999 for x in paired if x["task_id"] == task),
                                 "repeats": sum(x["task_id"] == task for x in paired)} for a in ARMS}
                   for task in plan["holdout_task_ids"]}
    sums = {a: {metric: sum(x["arms"][a][metric] for x in paired)
                for metric in ("input_tokens", "output_tokens", "duration_sec", "pi_tool_calls")} for a in ARMS}
    exposure = {a: {"configured_runs": len(paired) if a != "A" else 0,
                    "discovery_evidence_runs": sum(bool(x["arms"][a].get("skill_discovery_event_ids")) for x in paired),
                    "explicit_read_runs": sum(bool(x["arms"][a].get("skill_read_event_ids")) for x in paired),
                    "advertisement_without_event": "unknown"} for a in ARMS}
    trigger_checks = [_formula_trigger_check(x["arms"]["D"]) for x in paired]
    transitions = {a: {name: sum((x["arms"][a]["reward"] >= .999) == old and
                                (x["arms"]["D"]["reward"] >= .999) == new for x in paired)
                       for name, old, new in (("fail_to_pass", False, True), ("pass_to_fail", True, False),
                                              ("pass_to_pass", True, True), ("fail_to_fail", False, False))}
                   for a in ("A", "B", "C")}
    improved, regressed = set(), set()
    for x in paired:
        d = x["arms"]["D"]["reward"] >= .999
        if d and all(x["arms"][a]["reward"] < .999 for a in ("B", "C")):
            improved.add(x["task_id"])
        if not d and any(x["arms"][a]["reward"] >= .999 for a in ("A", "B", "C")):
            regressed.add(x["task_id"])
    complete = len(completed) == plan["policy"]["min_complete_repeats"]
    cost_ok = bool(paired) and sums["D"]["input_tokens"] <= plan["policy"]["max_input_ratio_d_over_b"] * max(1, sums["B"]["input_tokens"]) and sums["D"]["duration_sec"] <= plan["policy"]["max_duration_ratio_d_over_b"] * max(1, sums["B"]["duration_sec"])
    stable = complete and all(sum(x["arms"]["D"]["reward"] >= .999 for x in paired if x["repeat"] == r) >
                              max(sum(x["arms"][a]["reward"] >= .999 for x in paired if x["repeat"] == r) for a in ("A", "B", "C")) for r in completed)
    decision = ("incomplete" if not complete else "regression_risk" if regressed else
                "high_cost" if not cost_ok else "needs_more_evidence" if not stable or len(improved) < plan["policy"]["min_distinct_improved_tasks_vs_b_and_c"] else
                "eligible_for_val")
    return {"status": decision, "complete_repeats": completed, "paired_runs": len(paired),
            "observed_attempts_not_all_paired": observed_attempts,
            "observed_arm_cost_not_fully_paired": observed_arm_cost,
            "official_score_source": "Harbor verifier_result.rewards.reward", "arm_passes": counts,
            "per_task_success_frequency": frequencies,
            "mean_reward_delta_d_minus_a": (counts["D"] - counts["A"]) / len(paired) if paired else None,
            "mean_reward_delta_d_minus_b": (counts["D"] - counts["B"]) / len(paired) if paired else None,
            "mean_reward_delta_d_minus_c": (counts["D"] - counts["C"]) / len(paired) if paired else None,
            "per_arm_cost": sums, "distinct_improved_vs_b_and_c": sorted(improved),
            "transitions_d_vs_controls": transitions,
            "cost_delta_d_minus_controls": {a: {metric: sums["D"][metric] - sums[a][metric]
                                                 for metric in sums["D"]} for a in ("A", "B", "C")},
            "skill_exposure": exposure, "candidate_trigger_checks": trigger_checks,
            "all_observed_candidate_trigger_checks": observed_d_triggers,
            "distinct_regressed_vs_any_control": sorted(regressed),
            "cost_within_gate": cost_ok, "stable_vs_all_controls": stable,
            "usage": usage(state), "stopped_reason": state.get("stopped_reason"),
            "eligible_for_val": decision == "eligible_for_val", "paired": paired}


def _formula_trigger_check(row: dict[str, Any]) -> dict[str, Any]:
    """Report visible rule-condition and readback events, never causal utility."""
    path = Path(row.get("trial_path") or "") / "agent/trajectory.json"
    if not path.is_file():
        return {"task_id": row["task_id"], "status": "unknown", "reason": "missing_trace"}
    events = events_from_trace(json.loads(path.read_text()))
    calls = [x for x in events if x["kind"] == "tool_call"]
    content = lambda x: str(x.get("arguments") or x["text"])
    trigger = [x["id"] for x in calls if re.search(r"(?s)(?:f['\"].*?=|\.format\(|formula.*\{)", content(x), re.I)]
    readbacks = [x["id"] for x in calls if re.search(r"load_workbook|read_excel|openpyxl", content(x), re.I)
                 and re.search(r"data_only\s*=\s*False|\.value|formula", content(x), re.I)]
    positions = {event["id"]: index for index, event in enumerate(calls)}
    followed = any(positions[right] > positions[left] for left in trigger for right in readbacks)
    return {"task_id": row["task_id"], "trigger_status": "observed" if trigger else "unknown",
            "trigger_event_ids": trigger[:5], "readback_event_ids": readbacks[:5],
            "behavior_consistent": followed, "utility": "not_inferred_from_behavior"}


def run(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    plan = freeze(root)
    val_plan = freeze_val(root, plan)
    if dry_run:
        return {"status": "dry_run", "plan_sha256": plan["plan_sha256"], "model_calls": 0, "harbor_tasks": 0}
    out = root / "results/r18/spreadsheetbench"
    manifest = out / "runs_manifest.json"
    state = json.loads(manifest.read_text()) if manifest.is_file() else {"plan_sha256": plan["plan_sha256"], "runs": {}, "stopped_reason": None}
    if state["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("R1.8 resume plan mismatch")
    for repeat, arm in ORDER:
        key = f"repeat{repeat}-{arm}"
        existing = state["runs"].get(key)
        if existing:
            if existing["skill_sha256"] != plan["arms"][arm]["skill_sha256"]:
                raise ValueError("R1.8 resume Skill mismatch")
            if existing["status"] != "valid":
                state["stopped_reason"] = "recorded_invalid_run:" + key + ":" + existing["status"]
                break
            rows, errors = _trial_rows(Path(existing["attempts"][0]["job"]), "spreadsheetbench")
            if errors or digest(rows) != digest(existing["attempts"][0]["rows"]):
                raise ValueError("R1.8 official Run changed")
            continue
        used = usage(state)
        policy = plan["policy"]
        if (used["task_executions"] + 4 > policy["max_harbor_task_executions"] or
            used["input_tokens"] >= policy["max_input_tokens"] or
            used["output_tokens"] >= policy["max_output_tokens"] or
            used["wall_seconds"] >= policy["max_wall_seconds"]):
            state["stopped_reason"] = "prespecified_budget"
            break
        state["runs"][key] = {"status": "running", "skill_sha256": plan["arms"][arm]["skill_sha256"], "attempts": []}
        write_json(manifest, state)
        trial = attempt(root, plan, repeat, arm)
        state["runs"][key]["attempts"].append(trial)
        state["runs"][key]["status"] = trial["status"]
        write_json(manifest, state)
        if trial["status"] != "valid":
            state["stopped_reason"] = "invalid_run:" + key
            break
    write_json(manifest, state)
    result = analyze(plan, state)
    write_json(out / "holdout_analysis.json", result)
    if result["eligible_for_val"]:
        result["val"] = run_val(root, plan, val_plan)
    accepted = bool(result.get("val", {}).get("accepted"))
    write_json(root / "results/r18/BEST_VERSION.json", {
        "spreadsheetbench": {"version": "r18" if accepted else "no_skill",
                             "accepted": accepted,
                             "candidate_skill_sha256": plan["candidate_skill_sha256"] if accepted else None},
        "alfworld": {"version": "no_skill", "accepted": True},
        "livemath": {"version": "no_skill", "accepted": True},
        "train_gate_status": result["status"],
        "val_decision": result.get("val", {}).get("decision")})
    return {k: v for k, v in result.items() if k != "paired"}


def run_val(root: Path, train_plan: dict[str, Any], val_plan: dict[str, Any]) -> dict[str, Any]:
    """One official full Val only after the frozen Train gate passes."""
    train = json.loads((root / "results/r18/spreadsheetbench/holdout_analysis.json").read_text())
    if not train.get("eligible_for_val") or train_plan["plan_sha256"] != val_plan["train_plan_sha256"]:
        return {"status": "skipped_train_gate", "accepted": False}
    candidate = json.loads((root / "results/r17/spreadsheetbench/m4/candidate_manifest.json").read_text())
    skill = Path(candidate["skill_path"])
    if file_sha(skill) != val_plan["candidate_skill_sha256"]:
        raise ValueError("frozen Val candidate Skill SHA changed")
    kit = (root.parent / "kits/spreadsheet/spreadsheetbench-harbor-kit-v3.3-layout2").resolve()
    if config(kit, "spreadsheetbench") != val_plan["baseline_config"]:
        raise ValueError("Val baseline and current endpoint config differ")
    out = root / "results/r18/spreadsheetbench/val"
    acceptance_path = out / "acceptance.json"
    if acceptance_path.is_file():
        return json.loads(acceptance_path.read_text())
    jobs = kit / "workspace/runs/spreadsheetbench-val/r18-candidate"
    if jobs.exists() and any(jobs.iterdir()):
        raise ValueError("unregistered R1.8 Val Job directory")
    jobs.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(kit / "run.sh"), "--bench", "spreadsheetbench-val", "--reasoning", "off",
           "--concurrent", str(val_plan["concurrent"]), "--skills-dir", candidate["skills_dir"],
           "--jobs-dir", str(jobs)]
    log = jobs.with_name(jobs.name + "-controller.log")
    started = time.monotonic()
    with log.open("w") as output:
        try:
            completed = subprocess.run(cmd, cwd=kit, stdout=output, stderr=subprocess.STDOUT,
                                       timeout=val_plan["max_wall_seconds"], check=False)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    if code:
        return {"status": "invalid_run", "decision": "evaluation_invalid", "exit_code": code,
                "accepted": False}
    summaries = list(jobs.glob("*/summary.csv"))
    if len(summaries) != 1:
        return {"status": "invalid_run", "decision": "evaluation_invalid",
                "reason": "missing_official_summary", "accepted": False}
    gate = compare("spreadsheetbench", Path(val_plan["baseline_job"]), summaries[0].parent,
                   out, val_plan["baseline_config"], config(kit, "spreadsheetbench"))
    if gate["candidate_totals"]["input_tokens"] > val_plan["max_input_tokens"] or gate["candidate_totals"]["output_tokens"] > val_plan["max_output_tokens"]:
        gate["decision"] = "reject_budget"
        gate["cost_budget_exceeded"] = True
        write_json(acceptance_path, gate)
    gate["accepted"] = gate["decision"] == "accept"
    gate["elapsed_sec"] = round(time.monotonic() - started, 3)
    write_json(acceptance_path, gate)
    write_json(root / "results/r18/spreadsheetbench/acceptance.json", gate)
    return gate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_dir())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve(), dry_run=args.dry_run), ensure_ascii=False))


if __name__ == "__main__":
    main()
