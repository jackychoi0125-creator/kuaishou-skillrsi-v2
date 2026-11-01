"""Focused Train-only Spreadsheet R1 combination check (C versus short rules)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .stability import (_attempt, _attempt_status, _budget_allows, _budget_use, _manifest,
                        _recover_pending, _trial_rows, digest, run_fingerprint)
from .triage import _script_diff

BENCHMARK = "spreadsheetbench"
ORDER = ((0, "C"), (0, "D2"), (1, "D2"), (1, "C"))
INTERACTION_BUDGET = {"max_harbor_task_executions": 20, "max_input_tokens": 3_000_000,
                      "max_output_tokens": 350_000, "max_wall_seconds": 5400,
                      "per_benchmark_task_executions": 20, "per_benchmark_input_tokens": 3_000_000,
                      "per_arm_task_executions": 12, "max_infra_retries_per_run": 1,
                      "timeout_seconds": {BENCHMARK: 900}}


def prepare(root: Path) -> dict[str, Any]:
    main = json.loads((root / "results/stability/EXPERIMENT_PLAN.json").read_text())
    original = json.loads((root / "results/train_ablation/spreadsheetbench/summary.json").read_text())
    study = main["studies"][BENCHMARK]
    if not set(original["task_ids"]).issubset(study["task_ids"]):
        raise ValueError("interaction tasks must include prior Train diagnostic tasks")
    plan = {"schema_version": 1, "benchmark": BENCHMARK, "split": "train",
            "pilot_plan_sha256": main["plan_sha256"], "study": study,
            "run_order": list(ORDER), "budget": INTERACTION_BUDGET,
            "hypothesis": "full R1 rule combination may alter execution more than isolated D1/D2",
            "source": "prior Train A/B/C/D; no Val task feedback"}
    plan["plan_sha256"] = digest(plan)
    path = root / "results/stability/spreadsheetbench/INTERACTION_PLAN.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen interaction plan changed")
    write_json(path, plan)
    return plan


def run(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    plan = prepare(root)
    out = root / "results/stability/spreadsheetbench"
    arms = json.loads((out / "arms_manifest.json").read_text())["arms"]
    for arm in ("C", "D2"):
        path = Path(arms[arm]["skill_path"])
        from .stability import file_sha
        if file_sha(path) != arms[arm]["skill_sha256"]:
            raise ValueError(f"Skill SHA changed: {arm}")
    if dry_run:
        return {"status": "dry_run", "model_calls": 0, "official_scores": None,
                "task_ids_sha256": plan["study"]["task_ids_sha256"]}
    state_path = out / "interaction_runs_manifest.json"
    state = _manifest(state_path, plan["plan_sha256"])
    expected = set(plan["study"]["task_ids"])
    for repeat, arm in ORDER:
        key = f"repeat{repeat}-{arm}"
        conf = run_fingerprint(root, plan["study"], arm, repeat, arms[arm]["skill_sha256"], plan)
        prior = state["runs"].get(key)
        if prior:
            if prior["config"]["config_sha256"] != conf["config_sha256"]:
                raise ValueError("interaction resume config mismatch")
            _recover_pending(root, BENCHMARK, prior, expected)
            if prior["status"] == "running" and prior["attempts"] and prior["attempts"][-1].get("status") == "valid":
                prior["status"], prior["chosen_attempt"] = "valid", len(prior["attempts"]) - 1
            if prior["status"] == "valid":
                rows, errors = _trial_rows(Path(prior["attempts"][prior["chosen_attempt"]]["job"]), BENCHMARK)
                if errors or {r["task_id"] for r in rows} != expected or digest(rows) != digest(prior["attempts"][prior["chosen_attempt"]]["rows"]):
                    raise ValueError("interaction resumed official result changed")
                continue
            if prior["status"] != "running":
                raise ValueError("incomplete interaction run requires audit")
        usage = _budget_use({BENCHMARK: state})
        allowed, reason = _budget_allows(plan, usage, BENCHMARK, len(expected),
                                          int(sum(x["input_tokens"] for x in plan["study"]["historical_strata"].values()) * 1.5))
        main = json.loads((root / "results/stability/EXPERIMENT_PLAN.json").read_text())
        pilot_manifests = {b: _manifest(root / "results/stability" / b / "runs_manifest.json", main["plan_sha256"])
                           for b in main["studies"]}
        global_usage = _budget_use(pilot_manifests)
        estimate = int(sum(x["input_tokens"] for x in plan["study"]["historical_strata"].values()) * 1.5)
        if allowed and (global_usage["harbor_task_executions"] + usage["harbor_task_executions"] + len(expected) >
                        main["budget"]["max_harbor_task_executions"]):
            allowed, reason = False, "pilot_global_task_budget"
        if allowed and (global_usage["input_tokens"] + usage["input_tokens"] + estimate >
                        main["budget"]["max_input_tokens"]):
            allowed, reason = False, "pilot_global_input_budget"
        if allowed and (global_usage["output_tokens"] + usage["output_tokens"] >=
                        main["budget"]["max_output_tokens"]):
            allowed, reason = False, "pilot_global_output_budget"
        if allowed and (global_usage["wall_seconds"] + usage["wall_seconds"] >=
                        main["budget"]["max_wall_seconds"]):
            allowed, reason = False, "pilot_global_wall_budget"
        if allowed and usage["per_arm_task_executions"].get(f"{BENCHMARK}:{arm}", 0) + len(expected) > INTERACTION_BUDGET["per_arm_task_executions"]:
            allowed, reason = False, "interaction_per_arm_task_budget"
        if not allowed:
            state["stopped_reason"] = reason
            break
        record = prior if prior else {"config": conf, "status": "running", "attempts": [], "chosen_attempt": None}
        state["runs"][key] = record
        write_json(state_path, state)
        for attempt_index in range(len(record["attempts"]), INTERACTION_BUDGET["max_infra_retries_per_run"] + 1):
            current = _budget_use({BENCHMARK: state})
            can_attempt, cap = _budget_allows(plan, current, BENCHMARK, len(expected), estimate)
            if can_attempt and current["per_arm_task_executions"].get(f"{BENCHMARK}:{arm}", 0) + len(expected) > INTERACTION_BUDGET["per_arm_task_executions"]:
                can_attempt, cap = False, "interaction_per_arm_task_budget"
            if can_attempt:
                combined = {metric: global_usage[metric] + current[metric] for metric in
                            ("harbor_task_executions", "input_tokens", "output_tokens", "wall_seconds")}
                if combined["harbor_task_executions"] + len(expected) > main["budget"]["max_harbor_task_executions"]:
                    can_attempt, cap = False, "pilot_global_task_budget"
                elif combined["input_tokens"] + estimate > main["budget"]["max_input_tokens"]:
                    can_attempt, cap = False, "pilot_global_input_budget"
                elif combined["output_tokens"] >= main["budget"]["max_output_tokens"]:
                    can_attempt, cap = False, "pilot_global_output_budget"
                elif combined["wall_seconds"] >= main["budget"]["max_wall_seconds"]:
                    can_attempt, cap = False, "pilot_global_wall_budget"
            if not can_attempt:
                record["status"] = "budget_stopped"
                state["stopped_reason"] = cap
                break
            record["pending_attempt"] = attempt_index
            write_json(state_path, state)
            result = _attempt(root, BENCHMARK, arm, repeat, conf, Path(arms[arm]["skills_dir"]), None,
                              attempt_index)
            record["pending_attempt"] = None
            result["status"] = _attempt_status(result, expected)
            record["attempts"].append(result)
            if result["status"] == "valid":
                record["status"], record["chosen_attempt"] = "valid", attempt_index
                break
            if not result["status"].startswith("infrastructure_"):
                record["status"] = result["status"]
                break
        else:
            record["status"] = "infrastructure_unrecovered"
        write_json(state_path, state)
        if record["status"] != "valid":
            state["stopped_reason"] = state.get("stopped_reason") or f"invalid_run:{key}:{record['status']}"
            break
    write_json(state_path, state)
    if all(state["runs"].get(f"repeat{r}-{a}", {}).get("status") == "valid" for r, a in ORDER):
        summarize(root, plan, state)
    return {"status": "complete" if len(state["runs"]) == 4 and not state.get("stopped_reason") else "incomplete",
            "runs": len(state["runs"]), "stopped_reason": state.get("stopped_reason")}


def summarize(root: Path, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    out = root / "results/stability/spreadsheetbench"
    pilot = json.loads((out / "runs_manifest.json").read_text())
    rows: list[dict[str, Any]] = []
    for repeat in (0, 1):
        source = {}
        for arm in ("A", "B", "D1"):
            record = pilot["runs"][f"repeat{repeat}-{arm}"]
            source[arm] = {x["task_id"]: x for x in record["attempts"][record["chosen_attempt"]]["rows"]}
        for arm in ("C", "D2"):
            record = state["runs"][f"repeat{repeat}-{arm}"]
            source[arm] = {x["task_id"]: x for x in record["attempts"][record["chosen_attempt"]]["rows"]}
        for task_id in plan["study"]["task_ids"]:
            record = {"task_id": task_id, "repeat": repeat,
                      "task_ids_sha256": plan["study"]["task_ids_sha256"]}
            for arm in ("A", "B", "C", "D1", "D2"):
                value = source[arm][task_id]
                record[f"{arm}_reward"] = value["reward"]
                record[f"{arm}_input_tokens"] = value["input_tokens"]
                record[f"{arm}_tool_calls"] = value["pi_tool_calls"]
                record[f"{arm}_trial_path"] = value["trial_path"]
            for other in ("D1", "D2"):
                diff = _script_diff(Path(source[other][task_id]["trial_path"]), Path(source["C"][task_id]["trial_path"]))
                record[f"C_vs_{other}_script_changed"] = not diff["same_script"]
                record[f"C_vs_{other}_changed_script_count"] = len(diff["changed_scripts"])
            rows.append(record)
    with (out / "interaction_paired.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    effects = {arm: {"successes": sum(x[f"{arm}_reward"] >= .999 for x in rows),
                     "input_tokens": sum(x[f"{arm}_input_tokens"] for x in rows)}
               for arm in ("A", "B", "C", "D1", "D2")}
    result = {"benchmark": BENCHMARK, "split": "train", "status": "complete",
              "task_ids_sha256": plan["study"]["task_ids_sha256"], "plan_sha256": plan["plan_sha256"],
              "per_arm": effects,
              "C_vs_D1_script_changed_task_runs": sum(x["C_vs_D1_script_changed"] for x in rows),
              "C_vs_D2_script_changed_task_runs": sum(x["C_vs_D2_script_changed"] for x in rows),
              "interpretation": "Train observational rule-combination check; single-task repeats are not independent generalization"}
    write_json(out / "interaction_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve(), dry_run=args.mode == "dry-run"), ensure_ascii=False))


if __name__ == "__main__":
    main()
