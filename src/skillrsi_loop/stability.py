"""Train-only repeated official-Harbor replay with immutable run fingerprints.

The pilot measures A/A variability, Skill-presence (B/A), and D1/B differences.
It never reads Val trajectories or feeds outcomes to M3/M4.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import signal
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from skillrsi_m1.core import BENCHMARKS, events_from_trace
from skillrsi_m3.core import write_json
from .ablation import prepare_arms
from .cli import CONCURRENT, KIT_REL, config
from .triage import _alf_summary, _command, _skill_discoveries, _skill_reads

PILOT_ARMS = ("A", "B", "D1")
REPETITIONS = 2
DEFAULT_BUDGET = {
    "max_harbor_task_executions": 96,
    "max_input_tokens": 20_000_000,
    "max_output_tokens": 2_000_000,
    "max_wall_seconds": 28_800,
    "per_benchmark_task_executions": 32,
    "per_benchmark_input_tokens": 12_000_000,
    "per_arm_task_executions": 32,
    "cost_warning_input_ratio": 2.0,
    "max_infra_retries_per_run": 1,
    "timeout_seconds": {"spreadsheetbench": 900, "alfworld": 1800, "livemath": 900},
}
INFRA_NOTE = re.compile(r"(?i)429|rate.?limit|timeout|timed out|connection error|terminated|provider|network|container unavailable|docker daemon")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_metrics(root: Path, benchmark: str) -> dict[str, dict[str, str]]:
    path = root.parent / "metrics" / f"{benchmark}-baseline.csv"
    with path.open(newline="", encoding="utf-8") as source:
        return {row["task_id"]: row for row in csv.DictReader(source) if row["split"] == "train"}


def _fixed_ids(root: Path, benchmark: str) -> tuple[list[str], str, Path | None]:
    """Reproduce the Kit's pinned Train sample order without opening Val/Test data."""
    kit = root.parent / "kits" / KIT_REL[benchmark]
    if benchmark == "spreadsheetbench":
        base = kit / "benchmark"
        data = json.loads((base / "data/dataset.json").read_text())
        allowed = {str(x["id"]) for x in json.loads((base / "splits/train/items.json").read_text())}
        ordered = [str(x["id"]) for x in data if str(x["id"]) in allowed]
        ids = random.Random(0).sample(ordered, 4)
        mechanism = "kit_train_split_sample_seed_0_n_4"
        id_file = None
    elif benchmark == "alfworld":
        data = json.loads((kit / "benchmark/splits/train/items.json").read_text())
        ids = [x["id"].replace(":", "__") for x in random.Random(0).sample(data, 4)]
        mechanism = "kit_train_split_sample_seed_0_n_4"
        id_file = None
    else:
        metrics = train_metrics(root, benchmark)
        prior = root / "results/train_ablation/livemath/success_probe/selection.json"
        if not prior.is_file():
            raise ValueError("LiveMath Train success-probe selection is missing")
        successes = json.loads(prior.read_text())["task_ids"][:3]
        failures = [task for task, row in metrics.items() if float(row["reward"]) < .999]
        if len(successes) != 3 or not failures:
            raise ValueError("LiveMath Train strata unavailable")
        ids = successes + [failures[0]]
        mechanism = "train_metadata_three_historical_success_one_failure"
        id_file = root / "results/stability/livemath/train_ids.json"
    metrics = train_metrics(root, benchmark)
    if len(ids) != 4 or len(set(ids)) != 4 or not set(ids).issubset(metrics):
        raise ValueError(f"pilot sample is not four distinct Train tasks: {benchmark}")
    return ids, mechanism, id_file


def build_plan(root: Path, budget: dict[str, Any] | None = None) -> dict[str, Any]:
    budget = {**DEFAULT_BUDGET, **(budget or {})}
    studies = {}
    for benchmark in BENCHMARKS:
        ids, mechanism, id_file = _fixed_ids(root, benchmark)
        metrics = train_metrics(root, benchmark)
        studies[benchmark] = {"benchmark": benchmark, "split": "train", "task_ids": sorted(ids),
                              "task_ids_sha256": digest(sorted(ids)), "selection": mechanism,
                              "sample_seed": 0, "independent_repeats": REPETITIONS,
                              "arms": list(PILOT_ARMS), "run_order": ["A", "B", "D1", "D1", "B", "A"],
                              "historical_strata": {task: {"historical_reward": float(metrics[task]["reward"]),
                                                           "task_type": metrics[task]["task_type"],
                                                           "input_tokens": int(metrics[task].get("input_tokens") or 0),
                                                           "tool_calls": int(metrics[task].get("tool_calls") or 0)}
                                                    for task in ids},
                              "id_file": str(id_file) if id_file else None}
    plan = {"schema_version": 1, "study": "r1-stability-pilot", "data_policy": "Train only",
            "studies": studies, "budget": budget,
            "retry_policy": "one retry only for explicit infrastructure fault; never retry ordinary reward=0",
            "promotion_policy": "no R2 without repeated D1>B, acceptable cost, and clean cross-fitting validation",
            "inference_limit": "same Train trajectories contributed to R1 Insights; pilot is in-sample diagnosis"}
    plan["plan_sha256"] = digest(plan)
    return plan


def prepare_plan(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    out = root / "results/stability"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "EXPERIMENT_PLAN.json"
    proposed = build_plan(root)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != proposed:
            raise ValueError("frozen experiment plan differs from current data or budget")
    else:
        write_json(path, proposed)
    ids = proposed["studies"]["livemath"]["task_ids"]
    id_file = out / "livemath/train_ids.json"
    if id_file.exists() and json.loads(id_file.read_text()) != ids:
        raise ValueError("LiveMath fixed ID file changed")
    write_json(id_file, ids)
    return proposed


def _scorer_sha(kit: Path, benchmark: str) -> str:
    base = kit / "benchmark"
    paths = sorted(base.rglob("*score*.py")) + sorted(base.rglob("*official*.py"))
    if not paths:
        raise ValueError(f"no scoring code found in {kit}")
    return digest([(str(p.relative_to(kit)), file_sha(p)) for p in dict.fromkeys(paths)])


def run_fingerprint(root: Path, study: dict[str, Any], arm: str, repeat: int,
                    skill_sha: str | None, plan: dict[str, Any]) -> dict[str, Any]:
    benchmark = study["benchmark"]
    kit = (root.parent / "kits" / KIT_REL[benchmark]).resolve()
    endpoint = config(kit, benchmark)
    conf = {"benchmark": benchmark, "split": "train", "task_ids": study["task_ids"],
            "task_ids_sha256": study["task_ids_sha256"], "arm": arm, "repeat": repeat,
            "skill_sha256": skill_sha, "kit_script_sha256": file_sha(kit / "run.sh"),
            "scorer_sha256": _scorer_sha(kit, benchmark), "model_id": endpoint["model_id"],
            "model_api": endpoint["api"], "base_url": endpoint["base_url"], "reasoning": "off",
            "sample_seed": study["sample_seed"] + repeat if benchmark == "livemath" else study["sample_seed"],
            "sampling_n": len(study["task_ids"]),
            "concurrent": CONCURRENT[benchmark], "infra_retries": plan["budget"]["max_infra_retries_per_run"],
            "timeout_seconds": plan["budget"]["timeout_seconds"][benchmark],
            "plan_sha256": plan["plan_sha256"]}
    if study.get("id_file"):
        ids_path = Path(study["id_file"])
        if json.loads(ids_path.read_text()) != study["task_ids"]:
            raise ValueError("fixed Train ID file differs from frozen task set")
        conf["task_id_file_sha256"] = file_sha(ids_path)
    conf["config_sha256"] = digest(conf)
    return conf


def _trial_rows(job: Path, benchmark: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read every trial; preserve model/infra errors instead of filtering them."""
    with (job / "summary.csv").open(newline="", encoding="utf-8") as source:
        source_rows = list(csv.DictReader(source))
    if len({x["qid"] for x in source_rows}) != len(source_rows):
        raise ValueError("duplicate task in official summary")
    rows, errors = [], []
    for value in source_rows:
        task_id = value["qid"]
        matches = list(job.glob(f"{task_id}__*/result.json"))
        if len(matches) != 1:
            raise ValueError(f"missing/duplicate official result: {task_id}")
        result = json.loads(matches[0].read_text())
        reward = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
        if reward is None or abs(float(reward) - float(value["reward"])) > 1e-6:
            raise ValueError(f"official reward mismatch: {task_id}")
        trial = matches[0].parent
        trace_path = trial / "agent/trajectory.json"
        if not trace_path.is_file():
            raise ValueError(f"missing ATIF trace: {task_id}")
        events = events_from_trace(json.loads(trace_path.read_text()))
        verifier_path = trial / "verifier/metrics.json"
        verifier_metrics = json.loads(verifier_path.read_text()) if verifier_path.is_file() else {}
        model_error = value.get("model_error", "").lower() == "true"
        exception = value.get("exception", "").lower() == "true"
        note = value.get("note") or ""
        infra = (model_error or exception) and bool(INFRA_NOTE.search(note))
        if model_error or exception:
            errors.append({"task_id": task_id, "model_error": model_error,
                           "exception": exception, "note": note, "classification":
                           "infrastructure" if infra else "unclassified_run_error"})
        row = {"task_id": task_id, "reward": float(reward),
               "input_tokens": int(value.get("input_tokens") or 0),
               "output_tokens": int(value.get("output_tokens") or 0),
               "cache_tokens": int(value.get("cache_tokens") or 0),
               "duration_sec": float(value.get("duration_sec") or 0),
               "model_error": model_error, "execution_error": exception, "error_note": note,
               "pi_tool_calls": sum(x["kind"] == "tool_call" for x in events),
               "skill_discovery_event_ids": _skill_discoveries(events),
               "skill_read_event_ids": _skill_reads(events), "trial_path": str(trial),
               "trace_sha256": file_sha(trace_path), "official_result_path": str(matches[0])}
        row["grader_revision"] = verifier_metrics.get("grading_revision")
        if benchmark == "alfworld":
            row["alfworld"] = alfworld_counts(trial, events)
        rows.append(row)
    return rows, errors


def alfworld_counts(trial: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep Pi calls, bridge calls, submissions and official replay steps distinct."""
    calls = [x for x in events if x["kind"] == "tool_call"]
    bridge_send, bridge_status, bridge_start, other = [], [], [], []
    for call in calls:
        command = _command(call)
        if re.search(r"\bbridge\.sh\s+send\b", command):
            bridge_send.append(call)
        elif re.search(r"\bbridge\.sh\s+status\b", command):
            bridge_status.append(call)
        elif re.search(r"\bbridge\.sh\s+start\b", command):
            bridge_start.append(call)
        else:
            other.append(call)
    transcript = list((trial / "artifacts/bench/answers").glob("*/trajectory.jsonl"))
    if len(transcript) != 1:
        raise ValueError("ALFWorld official action transcript missing or ambiguous")
    commands = [json.loads(line)["command"] for line in transcript[0].read_text().splitlines() if line.strip()]
    verifier = json.loads((trial / "verifier/metrics.json").read_text())
    if verifier.get("n_commands") != len(commands):
        raise ValueError("ALFWorld transcript differs from official verifier command count")
    observations = [x for x in events if x["kind"] == "observation"]
    invalid_text = [x["id"] for x in observations if re.search(
        r"(?i)nothing happens|not a valid action|can't do that|cannot do that", x["text"])]
    invalid_actions = [x["id"] for x in observations if re.search(
        r"(?i)not a valid action|can't do that|cannot do that", x["text"])]
    repeated = sum(commands[i] == commands[i-1] for i in range(1, len(commands)))
    return {"pi_tool_calls": len(calls), "bridge_send_tool_calls": len(bridge_send),
            "bridge_status_tool_calls": len(bridge_status), "bridge_start_tool_calls": len(bridge_start),
            "other_pi_tool_calls": len(other),
            "submitted_commands": len(commands), "official_executed_steps": int(verifier["executed_steps"]),
            "official_max_steps": int(verifier["max_steps"]), "official_won": bool(verifier["won"]),
            "repeated_submitted_commands": repeated, "observed_no_progress_event_ids": invalid_text,
            "observed_invalid_action_event_ids": invalid_actions,
            "accepted_admissible_actions": None,
            "accepted_action_note": "not directly reported by official verifier; executed_steps counts env.step calls, including possibly invalid commands",
            "grader_revision": verifier.get("grading_revision"), "transcript_sha256": file_sha(transcript[0])}


def _attempt(root: Path, benchmark: str, arm: str, repeat: int, conf: dict[str, Any],
             skills_dir: Path | None, id_file: Path | None, attempt: int) -> dict[str, Any]:
    kit = (root.parent / "kits" / KIT_REL[benchmark]).resolve()
    jobs_root = (kit / "workspace/runs" / f"{benchmark}-train" / "r1-stability-pilot" /
                 f"repeat{repeat}-{arm}-attempt{attempt}")
    if jobs_root.exists() and any(jobs_root.iterdir()):
        raise ValueError(f"unregistered populated jobs root: {jobs_root}")
    jobs_root.parent.mkdir(parents=True, exist_ok=True)
    args = [str(kit / "run.sh"), "--bench", f"{benchmark}-train", "--n", str(conf["sampling_n"]),
            "--seed", str(conf["sample_seed"]), "--reasoning", "off",
            "--concurrent", str(conf["concurrent"]), "--jobs-dir", str(jobs_root)]
    if skills_dir:
        args += ["--skills-dir", str(skills_dir)]
    if id_file:
        args += ["--id-file", str(id_file)]
    log = jobs_root.with_name(jobs_root.name + "-controller.log")
    started = time.monotonic()
    timed_out = False
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(args, cwd=kit, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=conf["timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    job_paths = sorted(jobs_root.glob("*/summary.csv"))
    job = job_paths[-1].parent if len(job_paths) == 1 else None
    result: dict[str, Any] = {"attempt": attempt, "jobs_root": str(jobs_root), "job": str(job) if job else None,
                              "controller_log": str(log), "exit_code": code, "timed_out": timed_out,
                              "elapsed_sec": round(time.monotonic() - started, 3)}
    if job:
        try:
            rows, errors = _trial_rows(job, benchmark)
            result["rows"] = rows
            result["errors"] = errors
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            result["parse_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def _attempt_status(attempt: dict[str, Any], expected_ids: set[str]) -> str:
    if attempt["timed_out"]:
        return "infrastructure_timeout"
    if "parse_error" in attempt or not attempt.get("rows"):
        return "infrastructure_missing_or_invalid_score" if attempt["exit_code"] else "invalid_score_or_trace"
    if {r["task_id"] for r in attempt["rows"]} != expected_ids:
        return "task_set_mismatch"
    errors = attempt.get("errors", [])
    if errors:
        return "infrastructure_error" if all(e["classification"] == "infrastructure" for e in errors) else "unclassified_run_error"
    if attempt["exit_code"]:
        return "infrastructure_runner_failure"
    return "valid"


def _manifest(path: Path, plan_sha: str) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("plan_sha256") != plan_sha:
            raise ValueError("resume plan hash mismatch")
        return value
    return {"plan_sha256": plan_sha, "runs": {}, "stopped_reason": None}


def _recover_pending(root: Path, benchmark: str, record: dict[str, Any], expected: set[str]) -> None:
    pending = record.get("pending_attempt")
    if pending is None:
        return
    conf = record["config"]
    jobs_root = (root.parent / "kits" / KIT_REL[benchmark] / "workspace/runs" /
                 f"{benchmark}-train/r1-stability-pilot/repeat{conf['repeat']}-{conf['arm']}-attempt{pending}")
    summaries = list(jobs_root.glob("*/summary.csv")) if jobs_root.exists() else []
    if len(summaries) == 1:
        job = summaries[0].parent
        rows, errors = _trial_rows(job, benchmark)
        recovered = {"attempt": pending, "job": str(job), "jobs_root": str(jobs_root),
                     "controller_log": str(jobs_root.with_name(jobs_root.name + "-controller.log")),
                     "exit_code": 0, "timed_out": False, "elapsed_sec": 0,
                     "rows": rows, "errors": errors, "recovered_after_controller_interruption": True}
        recovered["status"] = _attempt_status(recovered, expected)
        record["attempts"].append(recovered)
        if recovered["status"] == "valid":
            record["status"] = "valid"
            record["chosen_attempt"] = pending
    elif jobs_root.exists() and any(jobs_root.iterdir()):
        raise ValueError(f"interrupted run has partial Harbor output requiring audit: {jobs_root}")
    record["pending_attempt"] = None


def _budget_use(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = {"harbor_task_executions": 0, "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0.0}
    per_bench: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_arm: dict[str, float] = defaultdict(float)
    for benchmark, manifest in manifests.items():
        for record in manifest["runs"].values():
            for attempt in record.get("attempts", []):
                total["harbor_task_executions"] += record["config"]["sampling_n"]
                total["wall_seconds"] += attempt.get("elapsed_sec", 0)
                per_bench[benchmark]["harbor_task_executions"] += record["config"]["sampling_n"]
                per_arm[f"{benchmark}:{record['config']['arm']}"] += record["config"]["sampling_n"]
                for row in attempt.get("rows", []):
                    total["input_tokens"] += row["input_tokens"]
                    total["output_tokens"] += row["output_tokens"]
                    per_bench[benchmark]["input_tokens"] += row["input_tokens"]
    return {**total, "per_benchmark": {k: dict(v) for k, v in per_bench.items()},
            "per_arm_task_executions": dict(per_arm)}


def _budget_allows(plan: dict[str, Any], usage: dict[str, Any], benchmark: str,
                   block_task_executions: int, historical_input_estimate: int) -> tuple[bool, str | None]:
    budget = plan["budget"]
    existing = usage["per_benchmark"].get(benchmark, {})
    if usage["harbor_task_executions"] + block_task_executions > budget["max_harbor_task_executions"]:
        return False, "max_harbor_task_executions"
    if existing.get("harbor_task_executions", 0) + block_task_executions > budget["per_benchmark_task_executions"]:
        return False, "per_benchmark_task_executions"
    if usage["input_tokens"] + historical_input_estimate > budget["max_input_tokens"]:
        return False, "max_input_tokens_estimate"
    if existing.get("input_tokens", 0) + historical_input_estimate > budget["per_benchmark_input_tokens"]:
        return False, "per_benchmark_input_tokens_estimate"
    if usage["output_tokens"] >= budget["max_output_tokens"]:
        return False, "max_output_tokens"
    if usage["wall_seconds"] >= budget["max_wall_seconds"]:
        return False, "max_wall_seconds"
    return True, None


def run_benchmark(root: Path, benchmark: str, plan: dict[str, Any], *, dry_run: bool,
                  stop_after: str | None = None) -> dict[str, Any]:
    study = plan["studies"][benchmark]
    out = root / "results/stability" / benchmark
    out.mkdir(parents=True, exist_ok=True)
    arms_path = out / "arms_manifest.json"
    if arms_path.exists():
        arms = json.loads(arms_path.read_text())["arms"]
        for arm in PILOT_ARMS:
            path = arms[arm].get("skill_path")
            if path and file_sha(Path(path)) != arms[arm]["skill_sha256"]:
                raise ValueError(f"Skill SHA changed for {benchmark}/{arm}; refusing resume")
    else:
        arms = prepare_arms(root, benchmark, out)
    if dry_run:
        return {"benchmark": benchmark, "mode": "dry_run", "model_calls": 0,
                "official_scores": None, "task_ids_sha256": study["task_ids_sha256"]}
    manifests = {name: _manifest(root / "results/stability" / name / "runs_manifest.json", plan["plan_sha256"])
                 for name in BENCHMARKS}
    manifest = manifests[benchmark]
    id_file = Path(study["id_file"]) if study["id_file"] else None
    expected = set(study["task_ids"])
    order = [(0, "A"), (0, "B"), (0, "D1"), (1, "D1"), (1, "B"), (1, "A")]
    started = time.monotonic()
    for repeat in range(REPETITIONS):
        block = [(r, a) for r, a in order if r == repeat]
        if all(manifest["runs"].get(f"repeat{r}-{a}", {}).get("status") == "valid" for r, a in block):
            continue
        usage = _budget_use(manifests)
        remaining = sum(f"repeat{r}-{a}" not in manifest["runs"] for r, a in block)
        estimated = int(sum(x["input_tokens"] for x in study["historical_strata"].values()) * remaining * 1.5)
        allowed, reason = _budget_allows(plan, usage, benchmark, remaining * len(expected), estimated)
        if allowed and any(usage["per_arm_task_executions"].get(f"{benchmark}:{arm}", 0) + len(expected) >
                           plan["budget"]["per_arm_task_executions"] for _, arm in block):
            allowed, reason = False, "per_arm_task_executions"
        if not allowed:
            manifest["stopped_reason"] = reason
            write_json(out / "runs_manifest.json", manifest)
            break
        for rep, arm in block:
            key = f"repeat{rep}-{arm}"
            skill_sha = arms[arm]["skill_sha256"]
            conf = run_fingerprint(root, study, arm, rep, skill_sha, plan)
            prior = manifest["runs"].get(key)
            if prior:
                if prior["config"]["config_sha256"] != conf["config_sha256"]:
                    raise ValueError(f"resume config hash mismatch: {benchmark}/{key}")
                _recover_pending(root, benchmark, prior, expected)
                if prior["status"] == "running" and prior["attempts"] and prior["attempts"][-1].get("status") == "valid":
                    prior["status"] = "valid"
                    prior["chosen_attempt"] = len(prior["attempts"]) - 1
                if prior["status"] == "valid":
                    # Revalidate original evidence and reward before reusing it.
                    chosen = prior["attempts"][prior["chosen_attempt"]]
                    rows, errors = _trial_rows(Path(chosen["job"]), benchmark)
                    if errors or {r["task_id"] for r in rows} != expected or digest(rows) != digest(chosen["rows"]):
                        raise ValueError(f"resumed official run differs: {benchmark}/{key}")
                    continue
                if prior["status"] != "running" or any(a["status"] not in
                        {"infrastructure_timeout", "infrastructure_missing_or_invalid_score",
                         "infrastructure_error", "infrastructure_runner_failure", "valid"}
                        for a in prior["attempts"]):
                    raise ValueError(f"incomplete run requires explicit audit: {benchmark}/{key}")
                record = prior
            else:
                record = {"config": conf, "status": "running", "attempts": [], "chosen_attempt": None}
                manifest["runs"][key] = record
            write_json(out / "runs_manifest.json", manifest)
            for attempt_index in range(len(record["attempts"]), plan["budget"]["max_infra_retries_per_run"] + 1):
                current_usage = _budget_use(manifests)
                estimate_one = int(sum(x["input_tokens"] for x in study["historical_strata"].values()) * 1.5)
                can_attempt, cap = _budget_allows(plan, current_usage, benchmark, len(expected), estimate_one)
                if can_attempt and current_usage["per_arm_task_executions"].get(f"{benchmark}:{arm}", 0) + len(expected) > plan["budget"]["per_arm_task_executions"]:
                    can_attempt, cap = False, "per_arm_task_executions"
                if not can_attempt:
                    record["status"] = "budget_stopped"
                    manifest["stopped_reason"] = cap
                    break
                record["pending_attempt"] = attempt_index
                write_json(out / "runs_manifest.json", manifest)
                attempt = _attempt(root, benchmark, arm, rep, conf,
                                   Path(arms[arm]["skills_dir"]) if arms[arm]["skills_dir"] else None,
                                   id_file, attempt_index)
                record["pending_attempt"] = None
                attempt["status"] = _attempt_status(attempt, expected)
                record["attempts"].append(attempt)
                write_json(out / "runs_manifest.json", manifest)
                if attempt["status"] == "valid":
                    record["status"], record["chosen_attempt"] = "valid", attempt_index
                    break
                if not attempt["status"].startswith("infrastructure_"):
                    record["status"] = attempt["status"]
                    break
            else:
                record["status"] = "infrastructure_unrecovered"
            write_json(out / "runs_manifest.json", manifest)
            if record["status"] != "valid":
                manifest["stopped_reason"] = manifest.get("stopped_reason") or f"invalid_run:{key}:{record['status']}"
                write_json(out / "runs_manifest.json", manifest)
                break
            if stop_after == key:
                return manifest
        if manifest.get("stopped_reason"):
            break
    manifest["controller_elapsed_sec"] = round(time.monotonic() - started, 3)
    write_json(out / "runs_manifest.json", manifest)
    summarize_benchmark(root, benchmark, plan, manifest)
    return manifest


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 3)


def _cluster_interval(pairs: list[dict[str, Any]], numerator: str, denominator: str) -> list[float] | None:
    ids = sorted({x["task_id"] for x in pairs})
    if len(ids) < 2:
        return None
    by_id = {task: [x for x in pairs if x["task_id"] == task] for task in ids}
    rng = random.Random(781)
    effects = []
    for _ in range(1000):
        sampled = [row for _task in (rng.choice(ids) for _ in ids) for row in by_id[_task]]
        effects.append(statistics.mean(float(x[f"{numerator}_reward"]) - float(x[f"{denominator}_reward"])
                                       for x in sampled))
    return [_percentile(effects, .025), _percentile(effects, .975)]


def summarize_benchmark(root: Path, benchmark: str, plan: dict[str, Any],
                        manifest: dict[str, Any]) -> dict[str, Any]:
    out = root / "results/stability" / benchmark
    study = plan["studies"][benchmark]
    pairs: list[dict[str, Any]] = []
    for repeat in range(REPETITIONS):
        keys = {arm: f"repeat{repeat}-{arm}" for arm in PILOT_ARMS}
        if not all(manifest["runs"].get(key, {}).get("status") == "valid" for key in keys.values()):
            continue
        lookup = {arm: {r["task_id"]: r for r in manifest["runs"][key]["attempts"][
                  manifest["runs"][key]["chosen_attempt"]]["rows"]} for arm, key in keys.items()}
        for task_id in study["task_ids"]:
            pair: dict[str, Any] = {"benchmark": benchmark, "task_id": task_id, "repeat": repeat,
                                    "sample_seed": (study["sample_seed"] + repeat if benchmark == "livemath"
                                                    else study["sample_seed"]),
                                    "historical_reward": study["historical_strata"][task_id]["historical_reward"]}
            for arm in PILOT_ARMS:
                row = lookup[arm][task_id]
                for metric in ("reward", "input_tokens", "output_tokens", "duration_sec", "pi_tool_calls"):
                    pair[f"{arm}_{metric}"] = row[metric]
                pair[f"{arm}_skill_read"] = bool(row["skill_read_event_ids"])
                pair[f"{arm}_skill_discovery"] = bool(row["skill_discovery_event_ids"])
                pair[f"{arm}_trial_path"] = row["trial_path"]
                pair[f"{arm}_config_sha256"] = manifest["runs"][keys[arm]]["config"]["config_sha256"]
                if benchmark == "alfworld":
                    for metric in ("submitted_commands", "official_executed_steps", "official_max_steps",
                                   "bridge_send_tool_calls", "bridge_status_tool_calls", "repeated_submitted_commands"):
                        pair[f"{arm}_{metric}"] = row["alfworld"][metric]
                    pair[f"{arm}_observed_no_progress"] = len(row["alfworld"]["observed_no_progress_event_ids"])
            pairs.append(pair)
    paired_path = out / "paired_repeats.csv"
    if pairs:
        with paired_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    else:
        paired_path.write_text("benchmark,task_id,repeat\n")
    attempts = [attempt for record in manifest["runs"].values() for attempt in record.get("attempts", [])]
    complete_repeats = len({p["repeat"] for p in pairs})
    per_arm = {}
    for arm in PILOT_ARMS:
        rows = [record["attempts"][record["chosen_attempt"]]["rows"] for key, record in manifest["runs"].items()
                if key.endswith("-" + arm) and record["status"] == "valid"]
        observations = [r for group in rows for r in group]
        by_task = {task: [r for r in observations if r["task_id"] == task] for task in study["task_ids"]}
        if not observations:
            per_arm[arm] = {"n_task_runs": 0}
            continue
        repeat_rates = [round(statistics.mean(r["reward"] >= .999 for r in group), 4) for group in rows]
        historical_success = [r for r in observations if study["historical_strata"][r["task_id"]]["historical_reward"] >= .999]
        historical_failure = [r for r in observations if study["historical_strata"][r["task_id"]]["historical_reward"] < .999]
        per_arm[arm] = {"n_task_runs": len(observations), "n_distinct_tasks": len(study["task_ids"]),
                        "success_rate": round(statistics.mean(r["reward"] >= .999 for r in observations), 4),
                        "repeat_success_rates": repeat_rates,
                        "repeat_success_rate_range": round(max(repeat_rates) - min(repeat_rates), 4),
                        "per_task_success_frequency": {task: round(statistics.mean(r["reward"] >= .999 for r in values), 4)
                                                       for task, values in by_task.items() if values},
                        "historical_success_regression_rate": round(statistics.mean(r["reward"] < .999 for r in historical_success), 4) if historical_success else None,
                        "historical_failure_fix_rate": round(statistics.mean(r["reward"] >= .999 for r in historical_failure), 4) if historical_failure else None,
                        "input_tokens_p50_p90": [_percentile([r["input_tokens"] for r in observations], p) for p in (.5, .9)],
                        "output_tokens_p50_p90": [_percentile([r["output_tokens"] for r in observations], p) for p in (.5, .9)],
                        "duration_sec_p50_p90": [_percentile([r["duration_sec"] for r in observations], p) for p in (.5, .9)],
                        "tool_calls_p50_p90": [_percentile([r["pi_tool_calls"] for r in observations], p) for p in (.5, .9)],
                        "total_input_tokens": sum(r["input_tokens"] for r in observations),
                        "total_output_tokens": sum(r["output_tokens"] for r in observations),
                        "model_error_rate": round(statistics.mean(r.get("model_error", False) for r in observations), 4),
                        "execution_error_rate": round(statistics.mean(r.get("execution_error", False) for r in observations), 4),
                        "explicit_discovery_rate": round(statistics.mean(bool(r["skill_discovery_event_ids"]) for r in observations), 4),
                        "explicit_read_rate": round(statistics.mean(bool(r["skill_read_event_ids"]) for r in observations), 4)}
        if benchmark == "alfworld":
            per_arm[arm]["environment_actions_p50_p90"] = [
                _percentile([r["alfworld"]["official_executed_steps"] for r in observations], p) for p in (.5, .9)]
            per_arm[arm]["submitted_commands_p50_p90"] = [
                _percentile([r["alfworld"]["submitted_commands"] for r in observations], p) for p in (.5, .9)]
            per_arm[arm]["bridge_send_calls_total"] = sum(r["alfworld"]["bridge_send_tool_calls"] for r in observations)
            per_arm[arm]["bridge_status_calls_total"] = sum(r["alfworld"]["bridge_status_tool_calls"] for r in observations)
            per_arm[arm]["no_progress_observations_total"] = sum(
                len(r["alfworld"]["observed_no_progress_event_ids"]) for r in observations)
            per_arm[arm]["successful_run_costs"] = [{"task_id": r["task_id"],
                "official_executed_steps": r["alfworld"]["official_executed_steps"],
                "official_max_steps": r["alfworld"]["official_max_steps"],
                "submitted_commands": r["alfworld"]["submitted_commands"],
                "no_progress_observations": len(r["alfworld"]["observed_no_progress_event_ids"]),
                "input_tokens": r["input_tokens"], "duration_sec": r["duration_sec"],
                "trial_path": r["trial_path"]}
                for r in observations if r["reward"] >= .999]
    aa = []
    for task in study["task_ids"]:
        values = [x["A_reward"] for x in pairs if x["task_id"] == task]
        if len(values) == REPETITIONS:
            aa.append({"task_id": task, "rewards": values, "changed": len(set(values)) > 1})
    effects = {}
    for label, numerator, denominator in (("presence_effect", "B", "A"),
                                           ("rule_effect", "D1", "B"), ("candidate_vs_no_skill", "D1", "A")):
        changes = [x[f"{numerator}_reward"] - x[f"{denominator}_reward"] for x in pairs]
        effects[label] = {"paired_mean_reward_delta": round(statistics.mean(changes), 4) if changes else None,
                          "task_cluster_bootstrap_95pct_descriptive": _cluster_interval(pairs, numerator, denominator),
                          "improved_task_runs": sum(x > 0 for x in changes),
                          "regressed_task_runs": sum(x < 0 for x in changes),
                          "n_distinct_tasks": len(study["task_ids"]), "n_paired_task_runs": len(changes),
                          "interpretation": "controlled descriptive contrast; repeated runs of one task are not independent tasks"}
    summary = {"benchmark": benchmark, "split": "train", "status": "complete" if complete_repeats == REPETITIONS else "incomplete",
               "task_ids_sha256": study["task_ids_sha256"], "plan_sha256": plan["plan_sha256"],
               "complete_repeats": complete_repeats, "expected_repeats": REPETITIONS,
               "a_a_variation": aa, "per_arm": per_arm, "effects": effects,
               "error_attempts": [{"attempt": a["attempt"], "status": a.get("status"), "job": a.get("job"),
                                   "errors": a.get("errors", []), "parse_error": a.get("parse_error")}
                                  for a in attempts if a.get("status") != "valid"],
               "error_rates": {"run_attempt_infrastructure_error_rate": round(
                   sum(str(a.get("status", "")).startswith("infrastructure_") for a in attempts) / len(attempts), 4)
                   if attempts else None,
                   "task_row_model_error_rate": round(
                       sum(r.get("model_error", False) for a in attempts for r in a.get("rows", [])) /
                       sum(len(a.get("rows", [])) for a in attempts), 4)
                   if any(a.get("rows") for a in attempts) else None},
               "budget_usage": _budget_use({benchmark: manifest}), "stopped_reason": manifest.get("stopped_reason"),
               "validation_scope": "in_sample_diagnostic_only", "score_source": "official Harbor verifier_result.rewards.reward"}
    write_json(out / "stability_summary.json", summary)
    cost = {"benchmark": benchmark, "per_arm": {arm: {k: v for k, v in values.items() if k in
            {"input_tokens_p50_p90", "output_tokens_p50_p90", "duration_sec_p50_p90", "tool_calls_p50_p90",
             "environment_actions_p50_p90", "total_input_tokens", "total_output_tokens", "no_progress_observations_total"}}
            for arm, values in per_arm.items()}, "budget_usage": summary["budget_usage"]}
    if per_arm.get("D1", {}).get("total_input_tokens") is not None and per_arm.get("B", {}).get("total_input_tokens"):
        cost["d1_over_b_input_ratio"] = round(per_arm["D1"]["total_input_tokens"] / per_arm["B"]["total_input_tokens"], 4)
    write_json(out / "cost_analysis.json", cost)
    write_json(out / "rule_effects.json", effects)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run", "summarize"))
    parser.add_argument("--benchmark", choices=("all", *BENCHMARKS), default="all")
    parser.add_argument("--mode", choices=("real", "dry-run"), default="dry-run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = prepare_plan(root, dry_run=args.mode == "dry-run")
    if args.command == "plan":
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "studies":
                          {b: x["task_ids_sha256"] for b, x in plan["studies"].items()}, "model_calls": 0}))
        return
    for benchmark in BENCHMARKS if args.benchmark == "all" else (args.benchmark,):
        if args.command == "summarize":
            manifest = _manifest(root / "results/stability" / benchmark / "runs_manifest.json", plan["plan_sha256"])
            result = summarize_benchmark(root, benchmark, plan, manifest)
        else:
            manifest = run_benchmark(root, benchmark, plan, dry_run=args.mode == "dry-run",
                                     stop_after=args.stop_after)
            result = {"benchmark": benchmark, "status": "dry_run" if args.mode == "dry-run" else
                      ("complete" if all(x.get("status") == "valid" for x in manifest["runs"].values()) and
                       len(manifest["runs"]) == len(PILOT_ARMS) * REPETITIONS else "incomplete"),
                      "runs": len(manifest.get("runs", {}))}
        print(json.dumps(result if args.command == "summarize" else result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
