"""M2 unified task, behavior, outcome and diagnosis profiles."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS, public_task
from .core import events_from_trace, sha256
from .m2_diagnostics import candidate_hypotheses

DIMENSIONS = ("planning", "action", "verification", "efficiency", "recovery")
VERIFICATION_RE = re.compile(r"\b(verified|checked|confirmed|asserted|tested|validated)\b|已验证|已检查|核对过", re.I)
CHECK_CALL_RE = re.compile(r"\b(status|look|verify|check|test|assert|pytest)\b", re.I)
PLAN_RE = re.compile(r"\b(first|then|next|plan|steps|begin by|after that|首先|然后|计划)\b", re.I)
ERROR_RE = re.compile(r"Traceback|SyntaxError|Permission denied|not a valid action|Nothing happens|can't do that|failed", re.I)


def _label(dimension: str, name: str, status: str, ids: list[str]) -> dict[str, Any]:
    return {"dimension": dimension, "label": name, "status": status, "event_ids": list(dict.fromkeys(ids))}


def behavior_signature(events: list[dict[str, Any]], extra: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    labels: dict[str, list[dict[str, Any]]] = {dimension: [] for dimension in DIMENSIONS}
    messages = [event for event in events if event["kind"] == "message" and event.get("source") == "agent"]
    calls = [event for event in events if event["kind"] == "tool_call"]
    observations = [event for event in events if event["kind"] == "observation"]
    for event in messages:
        if PLAN_RE.search(event["text"]):
            labels["planning"].append(_label("planning", "explicit_step_plan", "observed", [event["id"]]))
        if VERIFICATION_RE.search(event["text"]):
            labels["verification"].append(_label("verification", "agent_reported_result_check", "observed", [event["id"]]))
    if calls:
        labels["action"].append(_label("action", "tool_calls_executed", "observed", [call["id"] for call in calls[:6]]))
        for call in calls:
            if CHECK_CALL_RE.search(call["text"]):
                labels["verification"].append(_label("verification", "state_or_result_check_call", "observed", [call["id"]]))
    elif messages:
        labels["action"].append(_label("action", "textual_solution_produced", "observed", [messages[-1]["id"]]))
    for previous, current in zip(calls, calls[1:]):
        if previous.get("function") == current.get("function") and previous.get("arguments") == current.get("arguments"):
            labels["efficiency"].append(_label("efficiency", "repeated_identical_tool_call", "observed",
                                                [previous["id"], current["id"]]))
    for index, event in enumerate(events):
        if event["kind"] != "observation" or not (event.get("extra", {}).get("is_error") or ERROR_RE.search(event["text"])):
            continue
        following = next((item for item in events[index + 1:] if item["kind"] == "tool_call"), None)
        if following:
            labels["recovery"].append(_label("recovery", "continued_after_error_feedback", "observed",
                                             [event["id"], following["id"]]))
    for item in extra:
        if item["status"] == "observed" and item["event_ids"]:
            labels[item["dimension"]].append(item)
    for dimension in DIMENSIONS:
        unique = {(item["label"], tuple(item["event_ids"])): item for item in labels[dimension]}
        labels[dimension] = list(unique.values())[:12]
        if labels[dimension]:
            continue
        if dimension == "recovery" and not any(ERROR_RE.search(item["text"]) for item in observations):
            status = "not_observable"
        elif dimension == "efficiency" and len(calls) < 2:
            status = "not_observable"
        elif dimension == "verification" and not messages:
            status = "not_observable"
        else:
            status = "not_observed"
        labels[dimension] = [_label(dimension, "none_detected", status, [])]
    return labels


def profile_one(validated: dict[str, Any], legacy: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = validated["benchmark"]
    adapter = ADAPTERS[benchmark]
    public = public_task(validated)
    task = adapter.task_signature(validated, public)
    path = Path(validated["source_trace_path"])
    eligible = validated["trace_validity"] == "valid" and validated["task_outcome"] in {"success", "failure"}
    if path.is_file() and validated.get("trace_sha256") and sha256(path) == validated["trace_sha256"]:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
            events = events_from_trace(trace) if isinstance(trace.get("steps"), list) else []
        except (OSError, TypeError, ValueError):
            events = []
    else:
        events = []
    if not events:
        eligible = False
    symptoms, extra_behavior = adapter.signals(validated, events) if eligible else ([], [])
    event_ids = {event["id"] for event in events}
    hypotheses = []
    if eligible and validated["task_outcome"] == "failure":
        hypotheses = candidate_hypotheses(benchmark, events, validated["metric_trial_path"])
    if legacy and legacy.get("trace_sha256") == validated.get("trace_sha256"):
        for mode in legacy.get("observed_failure_modes", []):
            if mode.get("event_id") in event_ids and not any(item["code"] == mode["code"] for item in symptoms):
                symptoms.append({"code": mode["code"], "event_ids": [mode["event_id"]]})
    confirmed: dict[str, Any] | str = "undetermined"
    if eligible and legacy and legacy.get("diagnosis_status") == "judge_supported" and legacy.get("trace_sha256") == validated.get("trace_sha256"):
        cited = legacy.get("supporting_events") or []
        if cited and set(cited).issubset(event_ids):
            confirmed = {"module": legacy["error_module"], "type": legacy["error_type"],
                         "root_cause": legacy["root_cause"], "event_ids": cited,
                         "confidence": legacy["confidence"], "source": "legacy_m1_judge_import"}
    behaviors = behavior_signature(events, extra_behavior) if eligible else {
        dimension: [_label(dimension, "not_analyzed", "unknown", [])] for dimension in DIMENSIONS}
    critical = list(dict.fromkeys(event_id for item in symptoms + hypotheses for event_id in item.get("event_ids", [])
                                  if event_id in event_ids))
    if isinstance(confirmed, dict):
        critical = list(dict.fromkeys(confirmed["event_ids"] + critical))
    goal_text = public["goal"]
    if benchmark == "alfworld":
        first_observation = next((event for event in events if event["kind"] == "observation"
                                  and "Your task is to:" in event["text"]), None)
        if first_observation:
            match = re.search(r"Your task is to:\s*([^\\\"\n]+)", first_observation["text"])
            if match:
                goal_text = match.group(1).strip()
    profile = {
        "schema_version": "2.0", "trace_id": validated["trace_id"], "benchmark": benchmark,
        "task_id": validated["task_id"], "split": validated["split"], "attempt": validated["attempt"],
        "source_trace_path": validated["source_trace_path"], "trace_sha256": validated["trace_sha256"],
        "trace_validity": validated["trace_validity"], "reliability_flags": validated["reliability_flags"],
        "eligible_for_learning": eligible, "task_signature": task, "behavior_signature": behaviors,
        "outcome": {"status": validated["task_outcome"], "reward": validated["reward"],
                    "metrics": validated["metrics"]},
        "diagnosis": {"observed_symptoms": symptoms, "root_cause_hypotheses": hypotheses,
                      "confirmed_root_cause": confirmed, "critical_events": critical,
                      "diagnosis_confidence": confirmed["confidence"] if isinstance(confirmed, dict) else 0.0},
        "task_goal_excerpt": goal_text[:700],
        "event_count": len(events),
    }
    return profile, events


def observed_labels(profile: dict[str, Any]) -> set[str]:
    return {f"{dimension}:{item['label']}" for dimension, items in profile["behavior_signature"].items()
            for item in items if item["status"] == "observed"}


def validate_profile(profile: dict[str, Any], events: list[dict[str, Any]]) -> None:
    required = {"task_family", "task_subtype", "task_complexity", "required_operations",
                "environment_characteristics", "task_constraints"}
    if not required.issubset(profile["task_signature"]) or set(profile["behavior_signature"]) != set(DIMENSIONS):
        raise ValueError(f"invalid unified profile schema: {profile['trace_id']}")
    event_ids = {event["id"] for event in events}
    for labels in profile["behavior_signature"].values():
        for label in labels:
            if label["status"] not in {"observed", "not_observed", "not_observable", "unknown"}:
                raise ValueError("invalid behavior status")
            if label["status"] == "observed" and (not label["event_ids"] or not set(label["event_ids"]).issubset(event_ids)):
                raise ValueError(f"ungrounded behavior label: {profile['trace_id']}")
    for symptom in profile["diagnosis"]["observed_symptoms"]:
        if not symptom["event_ids"] or not set(symptom["event_ids"]).issubset(event_ids):
            raise ValueError(f"ungrounded symptom: {profile['trace_id']}")
    confirmed = profile["diagnosis"]["confirmed_root_cause"]
    if isinstance(confirmed, dict) and not set(confirmed["event_ids"]).issubset(event_ids):
        raise ValueError(f"ungrounded confirmed root: {profile['trace_id']}")
