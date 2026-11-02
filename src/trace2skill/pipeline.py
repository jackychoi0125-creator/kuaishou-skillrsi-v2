"""Unified Train-only M1-M4 pipeline with conservative no-candidate rollback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from skillrsi_m1.m2_pipeline import run as run_m2
from skillrsi_m1.reliability import validate_one
from skillrsi_m3.client import KimiClient
from skillrsi_m3.core import run as run_m3, write_json
from skillrsi_m4.core import run as run_m4
from .adapters import BenchmarkAdapter


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _prepare(adapter: BenchmarkAdapter, traces_path: Path, metrics_path: Path,
             output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = adapter.load_trace(traces_path)
    metrics = adapter.load_metrics(metrics_path)
    if len({entry.get("case_id") for entry in entries}) != len(entries):
        raise ValueError("duplicate or missing ATIF case_id")
    if len({row.get("task_id") for row in metrics}) != len(metrics):
        raise ValueError("duplicate or missing metric task_id")
    if any(row.get("split", "train") != "train" for row in metrics):
        raise ValueError("M1-M4 accepts Train input only")
    by_id = {entry["case_id"]: entry for entry in entries}
    if set(by_id) != {row["task_id"] for row in metrics}:
        raise ValueError("ATIF and Metrics Train task sets differ")
    records = []
    for row in metrics:
        task = row["task_id"]
        entry = by_id[task]
        trace = entry.get("trace", entry)
        raw = json.dumps(trace, ensure_ascii=False).encode()
        path = output / "private_raw" / f"{hashlib.sha256(task.encode()).hexdigest()[:16]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        row = {**row, "task_id": task, "split": "train", "trace_path": str(path),
               "trial_path": row.get("trial_path") or "", "attempt": ""}
        wrapped = {"case_id": task, "source_trace_path": str(path),
                   "trace_sha256": hashlib.sha256(raw).hexdigest(), "trace": trace}
        records.append(validate_one(adapter.name, row, wrapped))
    return records, {"tasks": len(records), "valid": sum(r["trace_validity"] == "valid" for r in records),
                     "success": sum(r["task_outcome"] == "success" for r in records),
                     "failure": sum(r["task_outcome"] == "failure" for r in records)}


def _safe_config(config: Path | None) -> dict[str, Any]:
    if config is None:
        return {}
    value = json.loads(config.read_text())
    allowed = {"val_bench", "kit_run_args", "generation_model_config", "allow_val",
               "train_gate_manifest", "baseline_job", "baseline_config", "train_holdout",
               "expected_val_tasks"}
    if set(value) - allowed:
        raise ValueError("unsupported configuration keys")
    return value


def evolve(*, benchmark: str, train_traces: Path, train_metrics: Path, output_dir: Path,
           kit_dir: Path | None = None, config: Path | None = None, mode: str = "dry-run",
           resume: bool = False, max_rounds: int = 1, max_generation_tokens: int = 50000,
           max_evaluation_tasks: int = 100, skip_val: bool = False,
           model_config: Path | None = None) -> dict[str, Any]:
    if not benchmark or not benchmark.replace("_", "").replace("-", "").isalnum():
        raise ValueError("invalid benchmark identifier")
    if max_rounds < 1 or max_generation_tokens < 0 or max_evaluation_tasks < 0:
        raise ValueError("invalid budget")
    if max_rounds != 1:
        raise ValueError("one supplied Train snapshot supports exactly one evolution round")
    settings = _safe_config(config)
    adapter = BenchmarkAdapter(benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"benchmark": benchmark, "train_traces_sha256": _sha(train_traces),
                "train_metrics_sha256": _sha(train_metrics), "mode": mode,
                "config_sha256": _sha(config) if config else None,
                "model_config_sha256": _sha(model_config) if model_config else None,
                "endpoint_env_sha256": hashlib.sha256(json.dumps([
                    os.environ.get("TRACE2SKILL_MODEL_ID"), os.environ.get("TRACE2SKILL_GENERATION_URL"),
                    os.environ.get("TRACE2SKILL_GENERATION_API_KEY")], sort_keys=True).encode()).hexdigest(),
                "max_rounds": max_rounds, "max_generation_tokens": max_generation_tokens,
                "max_evaluation_tasks": max_evaluation_tasks, "skip_val": skip_val,
                "kit_run_sha256": _sha(kit_dir / "run.sh") if kit_dir and (kit_dir / "run.sh").is_file() else None}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state_path = output_dir / "run_manifest.json"
    if mode == "dry-run":
        result = {"fingerprint": fingerprint, "status": "dry_run", "model_calls": 0,
                  "harbor_tasks": 0, "best": {"version": "no_skill", "skill_path": None}}
        write_json(output_dir / "dry_run_plan.json", result)
        return result
    if state_path.is_file():
        old = json.loads(state_path.read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("resume configuration or input SHA changed")
        if not resume:
            raise ValueError("output exists; use --resume")
        candidate = old.get("candidate") or {}
        if candidate.get("status") == "candidate":
            skill = Path(candidate["skill_path"])
            if not skill.is_file() or _sha(skill) != candidate["sha256"]:
                raise ValueError("resume candidate Skill SHA changed")
        return old
    if mode != "real":
        raise ValueError("mode must be dry-run or real")
    records, m1 = _prepare(adapter, train_traces, train_metrics, output_dir)
    frozen_holdout = (settings.get("train_holdout") or {}).get("task_ids") or []
    if set(frozen_holdout) & {record["task_id"] for record in records}:
        raise ValueError("frozen Train Holdout overlaps Source Train input")
    m1_dir = output_dir / "m1"
    _jsonl(m1_dir / "validated.jsonl", records)
    write_json(m1_dir / "summary.json", m1)
    m2_dir = output_dir / "m2"
    m2 = run_m2(m1_dir / "validated.jsonl", None, m2_dir)
    if not m2.get("eligible_success") and not m2.get("eligible_failure"):
        candidate = {"benchmark": benchmark, "status": "no_candidate", "reason": "no_valid_train_trace"}
        write_json(output_dir / "m4/candidate_manifest.json", candidate)
        best = {"version": "no_skill", "skill_path": None, "accepted": True}
        result = {"fingerprint": fingerprint, "status": "complete", "benchmark": benchmark,
                  "m1": m1, "m2": m2, "candidate": candidate, "best": best,
                  "generation_usage": {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                  "harbor_tasks": 0, "stop_reason": "no_valid_train_trace"}
        write_json(output_dir / "best_manifest.json", best)
        write_json(state_path, result)
        return result
    m3_dir = output_dir / "m3"
    model_path = model_config or (Path(settings["generation_model_config"]) if settings.get("generation_model_config") else None)
    env_ready = all(os.environ.get(key) for key in ("TRACE2SKILL_GENERATION_API_KEY",
                                                 "TRACE2SKILL_MODEL_ID", "TRACE2SKILL_GENERATION_URL"))
    client = KimiClient(model_path, output_dir / "private_model_cache") if (model_path or env_ready) and max_generation_tokens > 0 else None
    m3 = run_m3(m2_dir, benchmark, m3_dir, client, deterministic_only=client is None,
                math_generalize=False, balanced_selection=True, rich_payload=True)
    used = json.loads((m3_dir / "usage.json").read_text())
    if used.get("prompt_tokens", 0) + used.get("completion_tokens", 0) > max_generation_tokens:
        raise ValueError("generation Token budget exceeded; no Skill compiled")
    if client is None:
        candidate = {"benchmark": benchmark, "status": "no_candidate", "reason": "generation_endpoint_not_configured"}
        write_json(output_dir / "m4/candidate_manifest.json", candidate)
    else:
        candidate = run_m4(benchmark, m3_dir, output_dir / "candidate_skills", output_dir / "m4", client)
        used = client.usage()
        if used["prompt_tokens"] + used["completion_tokens"] > max_generation_tokens:
            candidate = {"benchmark": benchmark, "status": "no_candidate",
                         "reason": "generation_token_budget_exceeded_after_m4",
                         "model_calls": used["api_calls"]}
            write_json(output_dir / "m4/candidate_manifest.json", candidate)
    best = {"version": "no_skill", "skill_path": None, "accepted": True}
    val = {"status": "skipped_no_candidate" if candidate["status"] != "candidate" else "skipped_by_policy"}
    train_gate = None
    if candidate["status"] == "candidate" and settings.get("train_holdout") and kit_dir is not None:
        from .evaluation import validate_train
        train_gate = validate_train(candidate, kit_dir, output_dir / "train_gate",
                                    settings["train_holdout"], max_evaluation_tasks)
    if candidate["status"] == "candidate" and not skip_val and settings.get("allow_val") is True:
        if kit_dir is None or not (kit_dir / "run.sh").is_file():
            val = {"status": "unverifiable_external_requirement", "reason": "Kit run.sh missing"}
        elif max_evaluation_tasks == 0:
            val = {"status": "skipped_budget"}
        else:
            gate_file = Path(settings["train_gate_manifest"]) if settings.get("train_gate_manifest") else None
            train_gate = train_gate or (json.loads(gate_file.read_text()) if gate_file and gate_file.is_file() else None)
            if not train_gate or not train_gate.get("eligible_for_val") or train_gate.get("candidate_skill_sha256") != candidate["sha256"]:
                val = {"status": "requires_independent_train_validation",
                       "reason": "candidate must pass frozen Train gate with matching Skill SHA before Val"}
            elif settings.get("baseline_job") and not settings.get("baseline_config"):
                val = {"status": "baseline_config_required", "reason": "external baseline needs recorded matching config"}
            else:
                baseline_config = settings.get("baseline_config")
                if baseline_config and not _baseline_matches_kit(baseline_config, kit_dir,
                                                                 settings.get("val_bench") or f"{benchmark}-val"):
                    raise ValueError("baseline Kit/model/reasoning configuration differs")
                val = _evaluate(adapter, kit_dir, output_dir,
                                Path(settings["baseline_job"]) if settings.get("baseline_job") else None,
                                Path(candidate["skill_path"]).parent.parent,
                                max_evaluation_tasks - int((train_gate or {}).get("harbor_tasks") or 0),
                                settings.get("val_bench") or f"{benchmark}-val", settings.get("kit_run_args") or [],
                                int(settings.get("expected_val_tasks") or 0))
                if val["decision"] == "accept":
                    best = {"version": "r1", "skill_path": candidate["skill_path"],
                            "skill_sha256": candidate["sha256"], "accepted": True}
    elif candidate["status"] == "candidate" and not skip_val:
        val = {"status": "skipped_val_not_enabled"}
    result = {"fingerprint": fingerprint, "status": "complete", "benchmark": benchmark,
              "m1": m1, "m2": m2, "m3": m3, "candidate": candidate,
              "validation": val, "best": best, "rounds_completed": 1,
              "train_gate": train_gate,
              "stop_reason": "no_candidate" if candidate["status"] != "candidate" else "candidate_requires_train_validation",
              "generation_usage": used, "harbor_tasks": 0,
              "input_sha256": {"traces": identity["train_traces_sha256"], "metrics": identity["train_metrics_sha256"]}}
    write_json(output_dir / "best_manifest.json", best)
    write_json(state_path, result)
    return result


def _evaluate(adapter: BenchmarkAdapter, kit: Path, output: Path, baseline_job: Path | None,
              skills_dir: Path, max_tasks: int, val_bench: str, extra_args: list[str],
              expected_val_tasks: int = 0) -> dict[str, Any]:
    """Run only a prevalidated candidate; scores come from Harbor result.json."""
    if baseline_job is None:
        if expected_val_tasks <= 0 or 2 * expected_val_tasks > max_tasks:
            return {"status": "skipped_budget_or_unknown_val_size", "decision": "reject_budget",
                    "tasks_required": 2 * expected_val_tasks if expected_val_tasks else None,
                    "remaining_task_budget": max_tasks}
        baseline_root = output / "val_baseline_jobs"
        if baseline_root.exists() and any(baseline_root.iterdir()):
            raise ValueError("unregistered no-Skill Val Job directory")
        command = adapter.get_evaluation_command(kit, "val", skills_dir=None, jobs_dir=baseline_root,
                                                  bench_name=val_bench, extra_args=extra_args)
        with (output / "val_baseline_controller.log").open("w") as target:
            completed = subprocess.run(command, cwd=kit, stdout=target, stderr=subprocess.STDOUT, check=False)
        summaries = list(baseline_root.glob("*/summary.csv"))
        if completed.returncode or len(summaries) != 1:
            return {"status": "invalid_baseline", "decision": "evaluation_invalid"}
        baseline_job = summaries[0].parent
    baseline = _read_official(baseline_job)
    remaining = max_tasks - (len(baseline) if (output / "val_baseline_jobs").is_dir() else 0)
    if len(baseline) > remaining or (expected_val_tasks and len(baseline) != expected_val_tasks):
        return {"status": "skipped_budget_or_val_size_mismatch", "decision": "reject_budget",
                "tasks_required": len(baseline), "remaining_task_budget": remaining}
    jobs = output / "val_jobs"
    if jobs.exists() and any(jobs.iterdir()):
        raise ValueError("unregistered Val Job directory")
    cmd = adapter.get_evaluation_command(kit, "val", skills_dir=skills_dir, jobs_dir=jobs,
                                         bench_name=val_bench, extra_args=extra_args)
    log = output / "val_controller.log"
    with log.open("w") as target:
        completed = subprocess.run(cmd, cwd=kit, stdout=target, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        return {"status": "invalid_run", "decision": "evaluation_invalid", "exit_code": completed.returncode}
    summaries = list(jobs.glob("*/summary.csv"))
    if len(summaries) != 1:
        return {"status": "invalid_run", "decision": "evaluation_invalid", "reason": "missing_official_summary"}
    candidate = _read_official(summaries[0].parent)
    if set(candidate) != set(baseline):
        return {"status": "invalid_run", "decision": "evaluation_invalid", "reason": "task_id_mismatch"}
    fixes = sum(a["official_reward"] < .999 and candidate[t]["official_reward"] >= .999 for t, a in baseline.items())
    regressions = sum(a["official_reward"] >= .999 and candidate[t]["official_reward"] < .999 for t, a in baseline.items())
    decision = "accept" if fixes > regressions and regressions == 0 else "reject"
    return {"status": "complete", "decision": decision, "score_source": "official_harbor_result_json",
            "tasks": len(baseline), "baseline_pass": sum(x["official_reward"] >= .999 for x in baseline.values()),
            "candidate_pass": sum(x["official_reward"] >= .999 for x in candidate.values()),
            "fixes": fixes, "regressions": regressions, "baseline_job": str(baseline_job),
            "candidate_job": str(summaries[0].parent)}


def _read_official(job: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with (job / "summary.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            task = row["qid"]
            matches = list(job.glob(f"{task}__*/result.json"))
            if not task or task in rows or len(matches) != 1:
                raise ValueError("duplicate or missing official task result")
            result = json.loads(matches[0].read_text())
            score = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
            if score is None or abs(float(score) - float(row["reward"])) > 1e-6:
                raise ValueError("official Reward and summary mismatch")
            rows[task] = {"official_reward": float(score), "input_tokens": int(row.get("input_tokens") or 0),
                          "output_tokens": int(row.get("output_tokens") or 0),
                          "duration_sec": float(row.get("duration_sec") or 0)}
    return rows


def _baseline_matches_kit(baseline: dict[str, Any], kit: Path, val_bench: str) -> bool:
    if baseline.get("kit_run_sha256") != _sha(kit / "run.sh") or baseline.get("reasoning") != "off" or baseline.get("bench") != val_bench or baseline.get("seed") != 0:
        return False
    env = (kit / "kit.env").read_text() if (kit / "kit.env").is_file() else ""
    match = re.search(r"(?m)^(?:export\s+)?KIT_MODEL_ID=['\"]?([^'\"\s]+)", env)
    return bool(match and baseline.get("model_id") == match.group(1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trace2skill")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evolve")
    run.add_argument("--benchmark", required=True)
    run.add_argument("--train-traces", type=Path, required=True)
    run.add_argument("--train-metrics", type=Path, required=True)
    run.add_argument("--kit-dir", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--model-config", type=Path)
    run.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-rounds", type=int, default=1)
    run.add_argument("--max-generation-tokens", type=int, default=50000)
    run.add_argument("--max-evaluation-tasks", type=int, default=100)
    run.add_argument("--skip-val", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(evolve(**{k: v for k, v in vars(args).items() if k != "command"}), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
