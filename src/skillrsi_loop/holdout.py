"""Official Harbor Train holdout replay for a frozen, uncontaminated candidate."""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from .crossfit import audit_bundle
from .r16_plan import freeze
from .stability import _attempt_status, _scorer_sha, _trial_rows, digest, file_sha
from .cli import CONCURRENT, KIT_REL, config

ARMS = ("A", "B", "C", "D")
ORDER = ((0, "A"), (0, "B"), (0, "C"), (0, "D"),
         (1, "D"), (1, "C"), (1, "B"), (1, "A"))


def find_sampling_seed(ordered_ids: list[str], target: set[str], *, limit: int = 1_000_000) -> int | None:
    if not target or not target.issubset(ordered_ids):
        raise ValueError("target must be a subset of Kit Train IDs")
    for seed in range(limit):
        if {ordered_ids[index] for index in random.Random(seed).sample(range(len(ordered_ids)), len(target))} == target:
            return seed
    return None


def _sampling(root: Path, benchmark: str, holdout: set[str]) -> dict[str, Any]:
    kit = root.parent / "kits" / KIT_REL[benchmark]
    if benchmark == "livemath":
        id_file = root / "results/r16" / benchmark / "holdout_ids.json"
        ids = sorted(holdout)
        if id_file.is_file() and json.loads(id_file.read_text()) != ids:
            raise ValueError("frozen holdout ID file changed")
        write_json(id_file, ids)
        return {"mode": "official_id_file", "seed": 0, "id_file": str(id_file),
                "id_file_sha256": file_sha(id_file)}
    if benchmark == "alfworld":
        data = json.loads((kit / "benchmark/splits/train/items.json").read_text())
        ordered = [item["id"].replace(":", "__") for item in data]
    else:
        data = json.loads((kit / "benchmark/data/dataset.json").read_text())
        allowed = {str(item["id"]) for item in json.loads((kit / "benchmark/splits/train/items.json").read_text())}
        ordered = [str(item["id"]) for item in data if str(item["id"]) in allowed]
    seed = find_sampling_seed(ordered, holdout)
    if seed is None:
        return {"mode": "unsupported_fixed_holdout", "reason": "no_exact_kit_seed_within_search_budget",
                "search_limit": 1_000_000}
    return {"mode": "official_seed_sample", "seed": seed, "id_file": None,
            "ordered_train_ids_sha256": digest(ordered), "search_limit": 1_000_000}


def freeze_run_plan(root: Path, benchmark: str) -> dict[str, Any]:
    plan = freeze(root)
    audit = audit_bundle(root, benchmark, plan)
    if audit["status"] != "clean":
        raise ValueError("crossfit bundle is not clean")
    base = root / "results/r16" / benchmark
    candidate = json.loads((base / "m4/candidate_manifest.json").read_text())
    if candidate["status"] != "candidate_for_internal_validation" or candidate.get("accepted"):
        raise ValueError("no unaccepted Source-Train candidate")
    controls = json.loads((base / "controls/controls_manifest.json").read_text())
    for arm in ("B", "C", "D"):
        item = controls["arms"][arm]
        if file_sha(Path(item["skill_path"])) != item["skill_sha256"]:
            raise ValueError("control Skill SHA changed")
    holdout = set(plan["studies"][benchmark]["holdout_task_ids"])
    sampling = _sampling(root, benchmark, holdout)
    kit = (root.parent / "kits" / KIT_REL[benchmark]).resolve()
    model = config(kit, benchmark)
    value = {"benchmark": benchmark, "split": "train", "source_plan_sha256": plan["plan_sha256"],
             "holdout_task_ids": sorted(holdout), "holdout_sha256": digest(sorted(holdout)),
             "sampling": sampling, "order": [list(x) for x in ORDER],
             "arms": {arm: {"skill_sha256": controls["arms"][arm].get("skill_sha256"),
                            "skills_dir": controls["arms"][arm].get("skills_dir")}
                      for arm in ARMS},
             "candidate_rule_id": candidate["rule"]["rule_id"],
             "candidate_insight_sha256": audit["m3_insights_sha256"],
             "kit_run_sha256": file_sha(kit / "run.sh"), "scorer_sha256": _scorer_sha(kit, benchmark),
             "model_id": model["model_id"], "model_api": model["api"], "base_url": model["base_url"],
             "reasoning": "off", "concurrent": CONCURRENT[benchmark],
             "timeout_seconds": 1800 if benchmark == "alfworld" else 900,
             "infrastructure_retries": 0, "budget": plan["policy"],
             "validation_claim": "holdout repeated comparison required; never accepted from one run"}
    value["run_plan_sha256"] = digest(value)
    path = base / "holdout_run_plan.json"
    if path.is_file() and json.loads(path.read_text()) != value:
        raise ValueError("frozen holdout run config changed")
    write_json(path, value)
    return value


def _usage(state: dict[str, Any]) -> dict[str, float]:
    rows = [row for record in state["runs"].values() for attempt in record.get("attempts", [])
            for row in attempt.get("rows", [])]
    attempts = [attempt for record in state["runs"].values() for attempt in record.get("attempts", [])]
    return {"task_executions": sum(record["sampling_n"] for record in state["runs"].values() for _ in record.get("attempts", [])),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "wall_seconds": sum(attempt.get("elapsed_sec", 0) for attempt in attempts)}


def _attempt(root: Path, plan: dict[str, Any], repeat: int, arm: str) -> dict[str, Any]:
    benchmark = plan["benchmark"]
    kit = (root.parent / "kits" / KIT_REL[benchmark]).resolve()
    jobs_root = kit / "workspace/runs" / f"{benchmark}-train/r16-holdout/repeat{repeat}-{arm}"
    if jobs_root.exists() and any(jobs_root.iterdir()):
        raise ValueError("populated unregistered holdout Run directory")
    jobs_root.parent.mkdir(parents=True, exist_ok=True)
    args = [str(kit / "run.sh"), "--bench", f"{benchmark}-train", "--n", str(len(plan["holdout_task_ids"])),
            "--seed", str(plan["sampling"]["seed"]), "--reasoning", "off", "--concurrent",
            str(plan["concurrent"]), "--jobs-dir", str(jobs_root)]
    skills_dir = plan["arms"][arm]["skills_dir"]
    if skills_dir:
        args += ["--skills-dir", skills_dir]
    if plan["sampling"].get("id_file"):
        args += ["--id-file", plan["sampling"]["id_file"]]
    log = jobs_root.with_name(jobs_root.name + "-controller.log")
    started = time.monotonic()
    timed_out = False
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(args, cwd=kit, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=plan["timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = 124
    summaries = list(jobs_root.glob("*/summary.csv"))
    job = summaries[0].parent if len(summaries) == 1 else None
    attempt = {"attempt": 0, "job": str(job) if job else None, "jobs_root": str(jobs_root),
               "controller_log": str(log), "exit_code": exit_code, "timed_out": timed_out,
               "elapsed_sec": round(time.monotonic() - started, 3)}
    if job:
        try:
            attempt["rows"], attempt["errors"] = _trial_rows(job, benchmark)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            attempt["parse_error"] = type(exc).__name__ + ": " + str(exc)
    attempt["status"] = _attempt_status(attempt, set(plan["holdout_task_ids"]))
    return attempt


def run(root: Path, benchmark: str, *, dry_run: bool = False) -> dict[str, Any]:
    candidate_path = root / "results/r16" / benchmark / "m4/candidate_manifest.json"
    if candidate_path.is_file():
        candidate = json.loads(candidate_path.read_text())
        if candidate.get("status") == "no_candidate":
            return {"benchmark": benchmark, "status": "skipped_no_candidate", "model_calls": 0,
                    "harbor_tasks": 0, "eligible_for_val": False}
    plan = freeze_run_plan(root, benchmark)
    if plan["sampling"]["mode"] == "unsupported_fixed_holdout":
        return {"status": "blocked", "reason": plan["sampling"]["reason"], "model_calls": 0}
    if dry_run:
        return {"status": "dry_run", "benchmark": benchmark, "run_plan_sha256": plan["run_plan_sha256"],
                "holdout_sha256": plan["holdout_sha256"], "sampling": plan["sampling"],
                "model_calls": 0, "harbor_tasks": 0}
    base = root / "results/r16" / benchmark
    state_path = base / "holdout_runs_manifest.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"run_plan_sha256": plan["run_plan_sha256"],
                   "runs": {}, "stopped_reason": None}
    if state["run_plan_sha256"] != plan["run_plan_sha256"]:
        raise ValueError("holdout resume config mismatch")
    for repeat in (0, 1):
        block = [(r, arm) for r, arm in ORDER if r == repeat]
        if all(state["runs"].get(f"repeat{r}-{arm}", {}).get("status") == "valid" for r, arm in block):
            continue
        usage = _usage(state)
        remaining = len([1 for r, arm in block if f"repeat{r}-{arm}" not in state["runs"]])
        estimate = sum(int(x["input_tokens"]) for x in freeze(root)["studies"][benchmark]["holdout_strata"].values())
        budget = plan["budget"]
        if (usage["task_executions"] + remaining * len(plan["holdout_task_ids"]) > budget["max_harbor_task_executions"] or
            usage["input_tokens"] + remaining * estimate > budget["max_harbor_input_tokens"] or
            usage["output_tokens"] >= budget["max_harbor_output_tokens"] or
            usage["wall_seconds"] >= budget["max_wall_seconds"]):
            state["stopped_reason"] = "prespecified_budget_before_complete_repeat_block"
            break
        for _, arm in block:
            key = f"repeat{repeat}-{arm}"
            prior = state["runs"].get(key)
            if prior:
                if prior["status"] != "valid":
                    raise ValueError("incomplete holdout Run requires explicit audit")
                rows, errors = _trial_rows(Path(prior["attempts"][0]["job"]), benchmark)
                if errors or digest(rows) != digest(prior["attempts"][0]["rows"]):
                    raise ValueError("official resumed holdout result changed")
                continue
            usage = _usage(state)
            if (usage["task_executions"] + len(plan["holdout_task_ids"]) > budget["max_harbor_task_executions"] or
                usage["input_tokens"] + estimate > budget["max_harbor_input_tokens"] or
                usage["output_tokens"] >= budget["max_harbor_output_tokens"] or
                usage["wall_seconds"] >= budget["max_wall_seconds"]):
                state["stopped_reason"] = "prespecified_budget_before_attempt"
                break
            state["runs"][key] = {"status": "running", "sampling_n": len(plan["holdout_task_ids"]),
                                   "skill_sha256": plan["arms"][arm]["skill_sha256"], "attempts": []}
            write_json(state_path, state)
            attempt = _attempt(root, plan, repeat, arm)
            state["runs"][key]["attempts"].append(attempt)
            state["runs"][key]["status"] = attempt["status"]
            write_json(state_path, state)
            if attempt["status"] != "valid":
                state["stopped_reason"] = f"invalid_run:{key}:{attempt['status']}"
                break
        if state["stopped_reason"]:
            break
    write_json(state_path, state)
    complete = [repeat for repeat in (0, 1) if all(state["runs"].get(f"repeat{repeat}-{arm}", {}).get("status") == "valid" for arm in ARMS)]
    summary = {"benchmark": benchmark, "status": "complete" if len(complete) == 2 else "incomplete",
               "complete_repeats": complete, "usage": _usage(state), "stopped_reason": state["stopped_reason"],
               "candidate_accepted": False, "eligible_for_val": False,
               "score_source": "official Harbor verifier_result.rewards.reward"}
    if complete:
        paired = []
        for repeat in complete:
            by_arm = {arm: {row["task_id"]: row for row in state["runs"][f"repeat{repeat}-{arm}"]["attempts"][0]["rows"]}
                      for arm in ARMS}
            for task in plan["holdout_task_ids"]:
                paired.append({"task_id": task, "repeat": repeat,
                               **{arm: {"reward": by_arm[arm][task]["reward"],
                                        "input_tokens": by_arm[arm][task]["input_tokens"],
                                        "output_tokens": by_arm[arm][task]["output_tokens"],
                                        "duration_sec": by_arm[arm][task]["duration_sec"],
                                        "trial_path": by_arm[arm][task]["trial_path"]} for arm in ARMS}})
        write_json(base / "holdout_paired.json", paired)
        summary["arm_passes"] = {arm: [sum(row[arm]["reward"] >= .999 for row in paired if row["repeat"] == repeat)
                                       for repeat in complete] for arm in ARMS}
    write_json(base / "train_internal_validation.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("alfworld", "spreadsheetbench", "livemath"), required=True)
    parser.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve(), args.benchmark, dry_run=args.mode == "dry-run"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
