"""Align ALFWorld official commands to ATIF feedback; never count polls as actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from skillrsi_m1.core import _alf_action, events_from_trace
from skillrsi_m3.core import write_json
from .r17_plan import freeze
from .stability import file_sha
from .triage import _command

NO_PROGRESS = re.compile(r"(?i)nothing happens|not a valid action|can't do that|cannot do that")
INVALID = re.compile(r"(?i)not a valid action|can't do that|cannot do that|invalid action")


def normalize_command(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _feedback(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("steps"), int) else None


def align_trial(row: dict[str, Any]) -> dict[str, Any]:
    trial = Path(row["trial_path"])
    trace_path = trial / "agent/trajectory.json"
    if file_sha(trace_path) != row["trace_sha256"]:
        raise ValueError("official ATIF SHA changed")
    transcript_files = list((trial / "artifacts/bench/answers").glob("*/trajectory.jsonl"))
    if len(transcript_files) != 1:
        raise ValueError("missing or ambiguous official command transcript")
    transcript = [json.loads(line)["command"] for line in transcript_files[0].read_text().splitlines() if line.strip()]
    if file_sha(transcript_files[0]) != row["alfworld"]["transcript_sha256"]:
        raise ValueError("official command transcript SHA changed")
    if len(transcript) != row["alfworld"]["official_executed_steps"]:
        raise ValueError("official command/step count mismatch")
    events = events_from_trace(json.loads(trace_path.read_text()))
    observations = {event["id"]: event for event in events if event["kind"] == "observation"}
    linked: dict[int, dict[str, Any]] = {}
    send_calls = []
    duplicate_step_feedback = 0
    for event in events:
        if event["kind"] != "tool_call":
            continue
        command = _alf_action(_command(event))
        if command is None:
            continue
        observation = observations.get(event["id"].replace(".call", ".obs"))
        parsed = _feedback(observation["text"]) if observation else None
        send_calls.append({"command_event_id": event["id"], "feedback_event_id": observation["id"] if observation else None,
                           "parsed_step": parsed.get("steps") if parsed else None})
        if not parsed:
            continue
        step = parsed["steps"]
        if not 1 <= step <= len(transcript) or normalize_command(command) != normalize_command(transcript[step - 1]):
            continue
        if step in linked:
            duplicate_step_feedback += 1
            continue
        feedback_text = str(parsed.get("observation") or "")
        admissible = [normalize_command(str(x)) for x in parsed.get("admissible_commands") or []]
        state = {key: parsed.get(key) for key in ("observation", "inventory", "won", "done")}
        linked[step] = {"official_step": step, "command_event_id": event["id"],
                        "feedback_event_id": observation["id"], "feedback_text_sha256": hashlib.sha256(feedback_text.encode()).hexdigest(),
                        "feedback_excerpt": feedback_text[:160], "command": transcript[step - 1],
                        "admissible_commands": admissible,
                        "no_progress": bool(NO_PROGRESS.search(feedback_text)),
                        "invalid_action": bool(INVALID.search(feedback_text)),
                        "state_signature": hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]}
    commands = []
    for step, value in enumerate(transcript, 1):
        item = linked.get(step, {"official_step": step, "command": value,
                                 "command_event_id": None, "feedback_event_id": None,
                                 "no_progress": None, "invalid_action": None,
                                 "state_signature": None, "admissible_commands": []})
        previous = commands[-1] if commands else None
        item["repeated_previous_command"] = bool(previous and normalize_command(item["command"]) == normalize_command(previous["command"]))
        item["state_changed_from_previous"] = (item["state_signature"] != previous["state_signature"]
                                               if previous and item["state_signature"] and previous["state_signature"] else None)
        commands.append(item)
    triggers = []
    for index, item in enumerate(commands):
        if item["no_progress"] is not True or not item["admissible_commands"]:
            continue
        next_item = commands[index + 1] if index + 1 < len(commands) else None
        changed = (normalize_command(next_item["command"]) != normalize_command(item["command"])) if next_item else None
        admissible = normalize_command(next_item["command"]) in item["admissible_commands"] if next_item else None
        triggers.append({"official_step": item["official_step"], "trigger_event_ids": [x for x in
                         (item["command_event_id"], item["feedback_event_id"]) if x],
                         "next_command_event_id": next_item["command_event_id"] if next_item else None,
                         "next_command_changed": changed, "next_command_admissible": admissible,
                         "rule_behavior_observed": bool(changed and admissible) if next_item else None})
    observed_no_progress = sum(item["no_progress"] is True for item in commands)
    known_feedback = sum(item["no_progress"] is not None for item in commands)
    no_progress_followups = [item for item in commands[:-1] if item["no_progress"] is True]
    repeated_after_no_progress = sum(commands[item["official_step"]]["repeated_previous_command"] for item in no_progress_followups)
    return {"task_id": row["task_id"], "trial_path": str(trial), "reward": row["reward"],
            "trace_sha256": row["trace_sha256"], "transcript_sha256": file_sha(transcript_files[0]),
            "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
            "duration_sec": row["duration_sec"], "pi_tool_calls": row["pi_tool_calls"],
            "bridge_send_tool_calls": len(send_calls),
            "bridge_status_tool_calls": row["alfworld"]["bridge_status_tool_calls"],
            "official_executed_steps": len(commands), "linked_feedback_steps": known_feedback,
            "unlinked_official_steps": len(commands) - known_feedback,
            "send_calls_without_unique_official_step": len(send_calls) - len(linked),
            "duplicate_step_feedback": duplicate_step_feedback,
            "old_no_progress_observation_events": len(row["alfworld"]["observed_no_progress_event_ids"]),
            "unique_no_progress_commands": observed_no_progress,
            "no_progress_command_rate_known": round(observed_no_progress / known_feedback, 4) if known_feedback else None,
            "invalid_action_commands_known": sum(item["invalid_action"] is True for item in commands),
            "repeated_commands": sum(item["repeated_previous_command"] for item in commands),
            "no_progress_with_next_command": len(no_progress_followups),
            "repeated_after_no_progress": repeated_after_no_progress,
            "repeated_after_no_progress_rate": round(repeated_after_no_progress / len(no_progress_followups), 4)
                                             if no_progress_followups else None,
            "visible_state_changes": sum(item["state_changed_from_previous"] is True for item in commands),
            "skill_read_event_ids": row["skill_read_event_ids"], "triggers": triggers, "commands": commands,
            "uncertainty": "Missing parsed feedback is unknown, not counted as progress; official transcript fixes command count."}


def run(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    plan = freeze(root)
    manifest = json.loads((root / "results/r16/alfworld/holdout_runs_manifest.json").read_text())
    if manifest.get("stopped_reason") != "prespecified_budget_before_complete_repeat_block":
        raise ValueError("unexpected historical Holdout manifest")
    trials = [(arm, row) for arm in "ABCD" for row in manifest["runs"][f"repeat0-{arm}"]["attempts"][0]["rows"]]
    if dry_run:
        return {"status": "dry_run", "historical_trials": len(trials), "new_harbor_tasks": 0,
                "new_model_calls": 0, "plan_sha256": plan["plan_sha256"]}
    records = []
    for arm, row in trials:
        record = align_trial(row)
        record["arm"] = arm
        records.append(record)
    summary = {}
    for arm in "ABCD":
        subset = [item for item in records if item["arm"] == arm]
        summary[arm] = {"trials": len(subset), "successes": sum(item["reward"] >= .999 for item in subset),
                        "input_tokens": sum(item["input_tokens"] for item in subset),
                        "output_tokens": sum(item["output_tokens"] for item in subset),
                        "summed_task_duration_sec": round(sum(item["duration_sec"] for item in subset), 3),
                        "pi_tool_calls": sum(item["pi_tool_calls"] for item in subset),
                        "bridge_send_tool_calls": sum(item["bridge_send_tool_calls"] for item in subset),
                        "official_executed_steps": sum(item["official_executed_steps"] for item in subset),
                        "old_no_progress_observation_events": sum(item["old_no_progress_observation_events"] for item in subset),
                        "unique_no_progress_commands": sum(item["unique_no_progress_commands"] for item in subset),
                        "known_feedback_steps": sum(item["linked_feedback_steps"] for item in subset),
                        "repeated_commands": sum(item["repeated_commands"] for item in subset),
                        "no_progress_with_next_command": sum(item["no_progress_with_next_command"] for item in subset),
                        "repeated_after_no_progress": sum(item["repeated_after_no_progress"] for item in subset),
                        "visible_state_changes": sum(item["visible_state_changes"] for item in subset),
                        "invalid_action_commands_known": sum(item["invalid_action_commands_known"] for item in subset),
                        "unlinked_official_steps": sum(item["unlinked_official_steps"] for item in subset),
                        "triggers": sum(len(item["triggers"]) for item in subset),
                        "trigger_behavior_observed": sum(trigger["rule_behavior_observed"] is True for item in subset
                                                         for trigger in item["triggers"]),
                        "explicit_skill_reads": sum(bool(item["skill_read_event_ids"]) for item in subset)}
    by_arm_task = {(item["arm"], item["task_id"]): item for item in records}
    paired = []
    for task_id in sorted({item["task_id"] for item in records}):
        row = {"task_id": task_id}
        for arm in "ABCD":
            item = by_arm_task[(arm, task_id)]
            row[arm] = {key: item[key] for key in ("reward", "input_tokens", "output_tokens", "duration_sec",
                        "official_executed_steps", "unique_no_progress_commands", "bridge_send_tool_calls")}
        paired.append(row)
    delta_vs_a = {arm: {key: summary[arm][key] - summary["A"][key]
                        for key in ("input_tokens", "output_tokens", "summed_task_duration_sec",
                                    "pi_tool_calls", "bridge_send_tool_calls", "official_executed_steps",
                                    "unique_no_progress_commands")}
                  for arm in "BCD"}
    result = {"scope": "read_only_official_r16_train_holdout", "plan_sha256": plan["plan_sha256"],
              "new_harbor_tasks": 0, "new_model_calls": 0,
              "summary": summary, "delta_vs_A": delta_vs_a, "paired_task_costs": paired, "trials": records,
              "finding_boundary": "Command/feedback alignment is observable; D versus B/C is a four-task single-repeat association, not causality."}
    write_json(root / "results/r17/alfworld/behavior_cost_audit.json", result)
    trigger = {"benchmark": "alfworld", "source": "official R1.6 Train Holdout",
               "arms": {arm: {"configured": arm != "A", "explicit_reads": summary[arm]["explicit_skill_reads"],
                              "trigger_count": summary[arm]["triggers"],
                              "behavior_observed_count": summary[arm]["trigger_behavior_observed"],
                              "trigger_evidence_examples": [
                                  {"task_id": item["task_id"], "trial_path": item["trial_path"],
                                   "trigger_event_ids": event["trigger_event_ids"],
                                   "next_command_event_id": event["next_command_event_id"],
                                   "rule_behavior_observed": event["rule_behavior_observed"],
                                   "explicitly_read": bool(item["skill_read_event_ids"])}
                                  for item in records if item["arm"] == arm for event in item["triggers"]][:12]}
                       for arm in "ABCD"},
               "causal_utility": "unverified", "reason": "D tied B at 3/4 with one repeat and higher cost"}
    write_json(root / "results/r17/skill_trigger_analysis.json", trigger)
    return {"summary": summary, "new_harbor_tasks": 0, "new_model_calls": 0}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve(), dry_run=args.dry_run), ensure_ascii=False))


if __name__ == "__main__":
    main()
