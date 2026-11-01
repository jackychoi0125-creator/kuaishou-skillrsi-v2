"""Read-only Skill exposure audit; never sends Val content to M3/M4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from skillrsi_m1.core import events_from_trace
from skillrsi_m3.core import write_json
from .stability import file_sha
from .triage import _alf_summary, _command, _skill_discoveries, _skill_reads

BENCHMARKS = ("spreadsheetbench", "alfworld", "livemath")


def _jobs(root: Path) -> list[dict[str, Any]]:
    jobs = []
    for benchmark in BENCHMARKS:
        val = root / "results/r1" / benchmark / "val/run_manifest.json"
        if val.is_file():
            manifest = json.loads(val.read_text())
            for arm, key in (("A", "baseline"), ("R1", "candidate")):
                jobs.append({"benchmark": benchmark, "split": "val", "source": "r1_official_val_read_only",
                             "arm": arm, "job": manifest[key]["job"],
                             "configured": arm != "A", "skill_sha256": manifest[key].get("skill_sha256"),
                             "skill_path": json.loads((root / "results/r1" / benchmark / "m4/candidate_manifest.json").read_text()).get("skill_path")
                             if arm != "A" else None})
        ablation = root / "results/train_ablation" / benchmark / "summary.json"
        if ablation.is_file():
            for arm, value in json.loads(ablation.read_text())["arms"].items():
                jobs.append({"benchmark": benchmark, "split": "train", "source": "r1_train_ablation",
                             "arm": arm, "job": value["job"], "configured": arm != "A",
                             "skill_path": str(next((root / "results/train_ablation" / benchmark / "skills" / arm).glob("*/SKILL.md"), ""))
                             if arm != "A" else None})
        stability = root / "results/stability" / benchmark / "runs_manifest.json"
        if stability.is_file():
            data = json.loads(stability.read_text())
            for key, record in data["runs"].items():
                if record["status"] != "valid":
                    continue
                attempt = record["attempts"][record["chosen_attempt"]]
                arm = record["config"]["arm"]
                jobs.append({"benchmark": benchmark, "split": "train", "source": "r15_stability",
                             "arm": arm, "repeat": record["config"]["repeat"], "job": attempt["job"],
                             "configured": arm != "A", "skill_sha256": record["config"].get("skill_sha256"),
                             "skill_path": str(next((root / "results/stability" / benchmark / "skills" / arm).glob("*/SKILL.md"), ""))
                             if arm != "A" else None})
        interaction = root / "results/stability" / benchmark / "interaction_runs_manifest.json"
        if interaction.is_file():
            data = json.loads(interaction.read_text())
            for record in data["runs"].values():
                if record["status"] != "valid":
                    continue
                attempt = record["attempts"][record["chosen_attempt"]]
                jobs.append({"benchmark": benchmark, "split": "train", "source": "r15_interaction",
                             "arm": record["config"]["arm"], "repeat": record["config"]["repeat"],
                             "job": attempt["job"], "configured": True,
                             "skill_sha256": record["config"].get("skill_sha256"),
                             "skill_path": str(next((root / "results/stability" / benchmark / "skills" /
                                                     record["config"]["arm"]).glob("*/SKILL.md"), ""))})
    return jobs


def _advertisement(events: list[dict[str, Any]], skill_name: str | None) -> list[str]:
    if not skill_name:
        return []
    read_ids = set(_skill_reads(events))
    first_read = next((index for index, event in enumerate(events) if event["id"] in read_ids), len(events))
    return [event["id"] for index, event in enumerate(events[:first_read]) if
            skill_name.lower() in event["text"].lower() and
            ((event["kind"] == "message" and event.get("source") == "agent") or
             (event["kind"] == "observation" and re.search(r"(?i)available skills|skills list|description:", event["text"])))][:8]


def _behavior(benchmark: str, arm: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Conservative observable compatibility, never causal rule application."""
    if arm not in {"D1", "D2", "D"}:
        return {"state": "unknown", "event_ids": [], "signal": "multi_or_no_rule_cannot_be_attributed"}
    if benchmark == "alfworld":
        summary = _alf_summary(events)
        sequence = summary["action_observation_sequence"]
        refs = [item["action_event_id"] for left, item in zip(sequence, sequence[1:])
                if left["explicit_no_progress"] and item["action"] != left["action"]] if arm in {"D2", "D"} else [
                    item["action_event_id"] for item in sequence if item["explicit_no_progress"]]
        return {"state": "observed" if refs else "unknown", "event_ids": refs[:8],
                "signal": "changed_action_after_no_progress" if arm in {"D2", "D"} else "admissible_action_check_trigger_observed_only",
                "specificity": "trigger_only" if arm == "D1" else "action_sequence_only_admissibility_unverified",
                "no_progress_events": len(summary["no_progress_event_ids"]),
                "repeated_actions": len(summary["repeated_action_event_ids"])}
    if benchmark == "spreadsheetbench":
        refs = [event["id"] for event in events if event["kind"] == "tool_call" and
                re.search(r"(?i)load_workbook|sheetnames|iter_rows|cell\(", _command(event))] if arm == "D1" else [
                    event["id"] for event in events if event["kind"] == "tool_call" and
                    re.search(r"(?i)assert|verify|check|test", _command(event))]
        return {"state": "observed" if refs else "unknown", "event_ids": refs[:8],
                "signal": "relevant_workbook_inspection_candidate" if arm == "D1" else "result_check_candidate"}
    refs = [event["id"] for event in events if event["kind"] == "message" and
            re.search(r"(?i)definition|notation|defined|check|verify|recheck|定义|核对", event["text"])]
    return {"state": "observed" if refs else "unknown", "event_ids": refs[:8],
            "signal": "visible_definition_or_answer_check"}


def audit_trial(trial: Path, benchmark: str, *, configured: bool, arm: str,
                skill_name: str | None = None, skill_sha256: str | None = None) -> dict[str, Any]:
    trace_path = trial / "agent/trajectory.json"
    events = events_from_trace(json.loads(trace_path.read_text()))
    reads = _skill_reads(events)
    discoveries = _skill_discoveries(events)
    advertised = _advertisement(events, skill_name)
    result = json.loads((trial / "result.json").read_text())
    reward = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
    if reward is None:
        raise ValueError("missing official reward")
    metrics = json.loads((trial / "agent/trajectory.json").read_text()).get("final_metrics") or {}
    behavior = _behavior(benchmark, arm, events)
    return {"benchmark": benchmark, "arm": arm, "task_id": trial.name.split("__", 1)[0] if benchmark != "alfworld" else
            "__".join(trial.name.split("__")[:2]), "trial_path": str(trial), "trace_sha256": file_sha(trace_path),
            "skill_sha256": skill_sha256, "configured": configured,
            "advertised": "observed" if advertised else "unknown",
            "advertised_event_ids": advertised[:8], "explicitly_read": bool(reads),
            "read_event_ids": reads, "discovery_event_ids": discoveries,
            "behavior_consistent": behavior, "causal_application": "unknown",
            "kit_registration_expected": configured,
            "kit_registration_note": "Kit passes --skill and mounts /skills; Pi system prompt/list is not persisted in ATIF",
            "reward": float(reward), "input_tokens": int(metrics.get("input_tokens") or 0),
            "output_tokens": int(metrics.get("output_tokens") or 0)}


def run(root: Path, out_path: Path) -> dict[str, Any]:
    rows = []
    problems = []
    for job in _jobs(root):
        skill_file = Path(job["skill_path"]) if job.get("skill_path") else None
        skill_name = None
        if skill_file and skill_file.is_file():
            name = re.search(r"(?m)^name:\s*(\S+)", skill_file.read_text())
            skill_name = name.group(1) if name else None
            if job.get("skill_sha256") and file_sha(skill_file) != job["skill_sha256"]:
                problems.append({"job": job["job"], "reason": "skill_sha_mismatch"})
                continue
        path = Path(job["job"])
        if not path.is_dir():
            problems.append({"job": str(path), "reason": "missing_job"})
            continue
        with (path / "summary.csv").open(newline="", encoding="utf-8") as source:
            summary = list(csv.DictReader(source))
        for item in summary:
            matches = list(path.glob(f"{item['qid']}__*/result.json"))
            if len(matches) != 1:
                problems.append({"job": str(path), "task_id": item["qid"], "reason": "missing_trial"})
                continue
            try:
                record = audit_trial(matches[0].parent, job["benchmark"], configured=job["configured"],
                                     arm=job["arm"], skill_name=skill_name, skill_sha256=job.get("skill_sha256"))
                if record["task_id"] != item["qid"]:
                    raise ValueError("trial_id_mismatch")
                record.update({"split": job["split"], "source": job["source"], "repeat": job.get("repeat"),
                               "task_id": item["qid"], "input_tokens": int(item.get("input_tokens") or 0),
                               "output_tokens": int(item.get("output_tokens") or 0),
                               "duration_sec": float(item.get("duration_sec") or 0),
                               "pi_tool_calls": sum(event["kind"] == "tool_call" for event in
                                                    events_from_trace(json.loads((matches[0].parent / "agent/trajectory.json").read_text())))})
                if record["reward"] != float(item["reward"]):
                    raise ValueError("official_reward_mismatch")
                rows.append(record)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                problems.append({"job": str(path), "task_id": item["qid"], "reason": type(exc).__name__ + ": " + str(exc)})
    coverage = {}
    for key in sorted({(x["benchmark"], x["split"], x["source"], x["arm"]) for x in rows}):
        group = [x for x in rows if (x["benchmark"], x["split"], x["source"], x["arm"]) == key]
        coverage["/".join(key)] = {"runs": len(group), "configured": sum(x["configured"] for x in group),
                                  "advertised_observed": sum(x["advertised"] == "observed" for x in group),
                                  "advertised_unknown": sum(x["advertised"] == "unknown" for x in group),
                                  "explicitly_read": sum(x["explicitly_read"] for x in group),
                                  "behavior_consistent_observed": sum(x["behavior_consistent"]["state"] == "observed" for x in group),
                                  "input_tokens": sum(x["input_tokens"] for x in group),
                                  "duration_sec": round(sum(x["duration_sec"] for x in group), 3)}
    result = {"schema_version": 1, "scope": "read_only_prior_runs", "model_calls": 0,
              "kit_mechanism": "--skills-dir mounted at /skills and explicitly passed to Pi via --skill; advertised list absent from ATIF",
              "records": rows, "coverage": coverage, "problems": problems,
              "uncertainty": "No explicit read does not imply no metadata exposure or no Skill effect; behavior consistency is not causality"}
    write_json(out_path, result)
    return {"records": len(rows), "problems": len(problems), "coverage_groups": len(coverage)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    print(json.dumps(run(root, root / "results/r16/exposure_audit.json")))


if __name__ == "__main__":
    main()
