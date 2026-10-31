"""Benchmark adapters for public task signatures and observable trace signals."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .core import ALF_FAILURE_RE, BOX_RE, _alf_action, _alf_observation

TOPIC_PATTERNS = [
    ("probability", r"probability|random variable|expectation|variance|distribution"),
    ("geometry", r"triangle|circle|polygon|euclidean|geodesic"),
    ("number_theory", r"prime|divisib|congruence|diophantine|modulo"),
    ("combinatorics", r"graph|permutation|combination|partition|coloring|counting"),
    ("analysis", r"integral|derivative|fourier|operator|measure|continuous"),
    ("topology", r"topolog|homolog|cohomolog|manifold|homotopy"),
    ("algebra", r"group|ring|field|polynomial|matrix|linear algebra|representation"),
    ("logic", r"logic|theory|definab|o-minimal|model-theoretic"),
]


def public_task(record: dict[str, Any]) -> dict[str, Any]:
    """Read agent-visible material only; never inspect solution, tests or verifier details."""
    trial = Path(record.get("metric_trial_path") or "")
    try:
        result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        task_dir = Path(result["task_id"]["path"])
    except (OSError, KeyError, TypeError, ValueError):
        return {"instruction": "", "question": "", "goal": "", "task_dir": ""}
    instruction = task_dir / "instruction.md"
    question = task_dir / "environment/public/question.json"
    try:
        instruction_text = instruction.read_text(encoding="utf-8") if instruction.is_file() else ""
        question_text = str(json.loads(question.read_text(encoding="utf-8")).get("question", "")) if question.is_file() else ""
    except (OSError, ValueError, TypeError):
        instruction_text, question_text = "", ""
    match = re.search(r"(?ms)^### instruction\s*\n(.*?)(?=^### |\Z)", instruction_text)
    goal = match.group(1).strip() if match else question_text or instruction_text
    return {"instruction": instruction_text, "question": question_text, "goal": goal,
            "task_dir": str(task_dir)}


def _bucket(value: int, low: int, high: int) -> str:
    return "small" if value < low else "medium" if value < high else "large"


class BaseAdapter:
    name = "base"

    def task_signature(self, record: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
        return {"task_family": "unknown", "task_subtype": "unknown", "task_complexity": "unknown",
                "required_operations": [], "environment_characteristics": [], "task_constraints": []}

    def signals(self, record: dict[str, Any], events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return [], []

    def contrast_features(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        calls = [event for event in events if event["kind"] == "tool_call"]
        observations = [event for event in events if event["kind"] == "observation"]
        return {"action_sequence": [{"event_id": event["id"], "action": event["text"][:180]}
                                    for event in calls[:8]],
                "feedback": [{"event_id": event["id"], "observation": event["text"][:180]}
                             for event in observations[-4:]],
                "final_option": None}


class SpreadsheetAdapter(BaseAdapter):
    name = "spreadsheetbench"

    def task_signature(self, record: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
        metric = record["metrics"]
        task_type = (metric.get("task_type") or "").lower()
        family = "cell_level" if "cell" in task_type else "sheet_level" if "sheet" in task_type else "unknown"
        instruction = public["goal"].lower()
        operations = [name for name, pattern in (
            ("formula", r"formula|function|公式"), ("sort", r"sort|排序"),
            ("filter", r"filter|筛选"), ("transform", r"transform|convert|split|join|转换"),
            ("format", r"format|style|格式"), ("lookup", r"lookup|match|查找"))
            if re.search(pattern, instruction)]
        if not operations:
            operations = ["spreadsheet_edit"]
        cells = int(metric.get("spreadsheet_cell_count") or 0)
        return {"task_family": family, "task_subtype": operations[0],
                "task_complexity": _bucket(cells, 500, 5000), "required_operations": operations,
                "environment_characteristics": ["python_spreadsheet", "file_artifact"],
                "task_constraints": ["save_solution_py"]}

    def signals(self, record: dict[str, Any], events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        symptoms = []
        for event in events:
            if event["kind"] == "observation" and (event.get("extra", {}).get("is_error") or
                    re.search(r"Traceback|SyntaxError|Permission denied|FileNotFoundError", event["text"], re.I)):
                symptoms.append({"code": "tool_execution_error_observed", "event_ids": [event["id"]]})
        if record["task_outcome"] == "failure":
            last = next((event for event in reversed(events) if event["kind"] in {"message", "tool_call"}), None)
            if last:
                answer_dir = Path(record["metric_trial_path"]) / "artifacts/bench/answers"
                script_present = answer_dir.is_dir() and any(answer_dir.rglob("solution.py"))
                symptoms.append({"code": "submitted_script_reward_zero" if script_present else "missing_submitted_script",
                                 "event_ids": [last["id"]]})
        return symptoms[:8], []

    def contrast_features(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        result = super().contrast_features(events)
        result["tool_error_events"] = [event["id"] for event in events if event["kind"] == "observation"
                                       and (event.get("extra", {}).get("is_error") or
                                            re.search(r"Traceback|SyntaxError|Permission denied", event["text"], re.I))][:6]
        return result


class ALFAdapter(BaseAdapter):
    name = "alfworld"

    def task_signature(self, record: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
        subtype = record["metrics"].get("task_type") or "unknown"
        operations = [name for name in ("pick", "clean", "heat", "cool", "place", "look", "two") if name in subtype]
        family = "multi_stage_processing" if any(x in operations for x in ("clean", "heat", "cool")) else (
            "multi_step_interaction" if "look" in operations or "two" in operations else "simple_pick_place")
        return {"task_family": family, "task_subtype": subtype, "task_complexity":
                "large" if len(operations) >= 3 else "medium" if len(operations) == 2 else "small",
                "required_operations": operations, "environment_characteristics": ["text_world", "stateful_actions"],
                "task_constraints": ["max_50_actions", "admissible_commands"]}

    def signals(self, record: dict[str, Any], events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        symptoms: list[dict[str, Any]] = []
        behaviors: list[dict[str, Any]] = []
        by_id = {event["id"]: event for event in events}
        previous: tuple[str, str, str] | None = None
        last_state: tuple[dict[str, Any], str] | None = None
        for event in events:
            if event["kind"] != "tool_call":
                continue
            action = _alf_action(str(event.get("arguments", {}).get("command", "")))
            if not action:
                continue
            observation_id = event["id"].replace(".call", ".obs")
            observation_event = by_id.get(observation_id)
            if not observation_event:
                continue
            observation, _ = _alf_observation(observation_event["text"])
            if ALF_FAILURE_RE.search(observation):
                symptoms.append({"code": "environment_rejected_action", "event_ids": [event["id"], observation_id]})
            if previous and previous[0] == action.lower() and previous[1] == observation:
                behaviors.append({"dimension": "efficiency", "label": "repeated_action_no_observed_progress",
                                  "status": "observed", "event_ids": [previous[2], event["id"], observation_id]})
            previous = (action.lower(), observation, event["id"])
            try:
                state = json.loads(observation_event["text"])
            except (TypeError, ValueError):
                state = {}
            if isinstance(state, dict) and state.get("won") is False:
                last_state = (state, observation_id)
                if state.get("done") is True and state.get("steps") == state.get("max_steps"):
                    symptoms.append({"code": "action_budget_exhausted", "event_ids": [observation_id]})
        if record["task_outcome"] == "failure" and last_state and last_state[0].get("done") is False:
            symptoms.append({"code": "unfinished_episode_reward_zero", "event_ids": [last_state[1]]})
        return symptoms[:12], behaviors[:12]

    def contrast_features(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        result = super().contrast_features(events)
        by_id = {event["id"]: event for event in events}
        actions = []
        previous_observation = None
        for event in events:
            if event["kind"] != "tool_call":
                continue
            action = _alf_action(str(event.get("arguments", {}).get("command", "")))
            if not action:
                continue
            observation_id = event["id"].replace(".call", ".obs")
            obs_event = by_id.get(observation_id)
            observation, _ = _alf_observation(obs_event["text"]) if obs_event else ("", set())
            actions.append({"event_id": event["id"], "action": action,
                            "observation_event_id": observation_id if obs_event else None,
                            "observation": observation[:180],
                            "observation_changed": None if previous_observation is None else observation != previous_observation})
            previous_observation = observation
        result["action_sequence"] = actions[:8]
        result["total_actions"] = len(actions)
        result["rejected_action_events"] = [item["observation_event_id"] for item in actions
                                             if item["observation_event_id"] and ALF_FAILURE_RE.search(item["observation"])]
        return result


class MathAdapter(BaseAdapter):
    name = "livemath"

    def task_signature(self, record: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
        question = public["question"].lower()
        topics = [name for name, pattern in TOPIC_PATTERNS if re.search(pattern, question)]
        family = topics[0] if topics else "unknown"
        length = len(public["question"])
        return {"task_family": family, "task_subtype": "multiple_choice_math",
                "task_complexity": _bucket(length, 900, 1800),
                "required_operations": ["derive_answer", "select_option"],
                "environment_characteristics": ["text_only", "no_tool_environment"],
                "task_constraints": ["boxed_A_to_E"], "topic_labels": topics or ["unknown"]}

    def signals(self, record: dict[str, Any], events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        messages = [event for event in events if event["kind"] == "message" and event.get("source") == "agent"]
        if not messages:
            return [], []
        last = messages[-1]
        symptoms = []
        if not BOX_RE.search(last["text"]):
            symptoms.append({"code": "boxed_answer_absent", "event_ids": [last["id"]]})
        elif record["task_outcome"] == "failure":
            symptoms.append({"code": "boxed_option_present_reward_zero", "event_ids": [last["id"]]})
        return symptoms, []

    def contrast_features(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        result = super().contrast_features(events)
        messages = [event for event in events if event["kind"] == "message" and event.get("source") == "agent"]
        final = messages[-1] if messages else None
        matches = BOX_RE.findall(final["text"]) if final else []
        result["final_option"] = {"option": matches[-1], "event_id": final["id"]} if matches else None
        result["reasoning_events"] = [event["id"] for event in messages if re.search(r"therefore|because|thus|hence|所以|因此", event["text"], re.I)][:6]
        return result


ADAPTERS = {adapter.name: adapter for adapter in (SpreadsheetAdapter(), ALFAdapter(), MathAdapter())}
