"""Read original ATIF, filter unreliable trials, and check observable constraints."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any


BENCHMARKS = ("spreadsheetbench", "alfworld", "livemath")
EXPECTED_TRAIN = {"spreadsheetbench": 80, "alfworld": 39, "livemath": 35}
TAXONOMY = {
    "memory": {"hallucination", "memory_retrieval_failure", "over_simplification"},
    "reflection": {"progress_misjudge", "outcome_misinterpretation", "causal_misattribution", "hallucination"},
    "planning": {"constraint_ignorance", "impossible_action", "inefficient_plan"},
    "action": {"misalignment", "invalid_action", "format_error", "parameter_error"},
    "system": {"step_limit", "tool_execution_error", "llm_limit", "environment_error"},
}
INFRA_TYPES = {"model", "execution", "scoring"}
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\):|(?:^|\n)(?:[A-Za-z_]+Error|SyntaxError|Exception):|"
    r"Permission denied|command not found|No such file or directory|\bexit (?:code |status )?[1-9]",
    re.I,
)
ALF_FAILURE_RE = re.compile(
    r"not a valid action|can't do that|cannot do that|nothing happens|failed|is not (?:open|clean|hot|cold)|"
    r"you can't|you cannot|not found|there is no",
    re.I,
)
PROCESS_RE = re.compile(r"\b(clean|heat|cool)\b", re.I)
PUT_RE = re.compile(r"^put\s", re.I)
BOX_RE = re.compile(r"\\boxed\s*\{\s*([A-E])\s*\}")
FORMULA_ERROR_RE = re.compile(r"(?m)^\s*#(?:REF!|DIV/0!|VALUE!|NAME\?|NUM!|N/A)\s*$", re.I)
CELL_ASSIGN_RE = re.compile(r"\[\s*['\"]([A-Z]{1,3}\d+)['\"]\s*\]\s*=", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def metric_rows(metrics_dir: Path, benchmark: str) -> list[dict[str, str]]:
    path = metrics_dir / f"{benchmark}-baseline-original.csv"
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row.get("split") == "train"]
    if len(rows) != EXPECTED_TRAIN[benchmark]:
        raise ValueError(f"{path}: expected {EXPECTED_TRAIN[benchmark]} train rows, got {len(rows)}")
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate train task_id")
    return rows


def prepare_train(metrics_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Preserve every original ATIF object in a Kit-style JSONL wrapper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        rows = metric_rows(metrics_dir, benchmark)
        output = output_dir / f"{benchmark}-train.atif.jsonl"
        count = 0
        with output.open("w", encoding="utf-8") as target:
            for row in rows:
                path = Path(row["trace_path"])
                if not path.is_file():
                    # Preserve a missing-trace marker; reliability filtering decides its fate.
                    item = {"case_id": row["task_id"], "trace": None,
                            "source_trace_path": str(path), "trace_sha256": None}
                else:
                    try:
                        trace = json.loads(path.read_text(encoding="utf-8"))
                        error = None
                    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                        trace, error = None, type(exc).__name__
                    item = {"case_id": row["task_id"], "trace": trace,
                            "source_trace_path": str(path), "trace_sha256": sha256(path),
                            "trace_error": error}
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        summary[benchmark] = {"tasks": count, "jsonl": str(output), "bytes": output.stat().st_size}
    return summary


def load_atif_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSON: {exc}") from exc
            task_id = item.get("case_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"{path}:{line_no}: missing case_id")
            if task_id in items:
                raise ValueError(f"{path}:{line_no}: duplicate case_id {task_id}")
            items[task_id] = item
    return items


def task_context(row: dict[str, str]) -> dict[str, Any]:
    """Read only agent-visible or public task data, never solution/tests/reference."""
    trial = Path(row["trial_path"])
    try:
        result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        task = Path(result["task_id"]["path"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {"task_path": "", "instruction": "", "task_toml": "", "public": {}, "read_error": True}
    instruction = task / "instruction.md"
    task_toml = task / "task.toml"
    public_dir = task / "environment/public"
    public: dict[str, Any] = {}
    if public_dir.is_dir():
        for path in sorted(public_dir.glob("*.json")):
            try:
                public[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                public[path.name] = {"_read_error": True}
    return {
        "task_path": str(task),
        "instruction": instruction.read_text(encoding="utf-8") if instruction.is_file() else "",
        "task_toml": task_toml.read_text(encoding="utf-8") if task_toml.is_file() else "",
        "public": public,
    }


def trace_validity(item: dict[str, Any] | None, row: dict[str, str]) -> tuple[str, list[str]]:
    flags: list[str] = []
    if item and item.get("trace_error"):
        return "malformed", ["malformed_source_trace"]
    if item is None or item.get("trace") is None:
        return "missing", ["missing_trace"]
    trace = item["trace"]
    if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list) or not trace["steps"]:
        return "malformed", ["malformed_atif"]
    ids = [s.get("step_id") for s in trace["steps"] if isinstance(s, dict)]
    if len(ids) != len(trace["steps"]) or any(not isinstance(x, int) for x in ids) or len(ids) != len(set(ids)):
        return "malformed", ["invalid_step_ids"]
    if item.get("case_id") != row["task_id"]:
        return "mismatch", ["case_id_mismatch"]
    source = Path(item.get("source_trace_path") or "")
    if source.is_file() and item.get("trace_sha256") and sha256(source) != item["trace_sha256"]:
        return "mismatch", ["trace_hash_mismatch"]
    if source.is_file():
        try:
            if json.loads(source.read_text(encoding="utf-8")) != trace:
                return "mismatch", ["embedded_trace_mismatch"]
        except (OSError, UnicodeError, json.JSONDecodeError):
            return "malformed", ["malformed_source_trace"]
    if truth(row.get("model_error")):
        flags.append("terminal_model_error")
    if int(row.get("agent_429_errors") or 0) > 0:
        flags.append("observed_429")
    if truth(row.get("execution_error")):
        flags.append("execution_error")
    if truth(row.get("scoring_error")):
        flags.append("scoring_error")
    return "valid", flags


def outcome(row: dict[str, str], validity: str) -> str:
    if validity != "valid":
        return "invalid_trace"
    if truth(row.get("scoring_error")) or row.get("reward") in {None, ""}:
        return "scoring_failure"
    if float(row["reward"]) >= 0.999:
        return "success"
    if truth(row.get("execution_error")) or truth(row.get("model_error")):
        return "infrastructure_failure"
    return "task_failure"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def events_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step in trace["steps"]:
        step_id = step["step_id"]
        if step.get("message"):
            events.append({"id": f"s{step_id}.message", "step": step_id, "kind": "message",
                           "text": _text(step["message"]), "source": step.get("source", "")})
        calls = step.get("tool_calls") or []
        observations = (step.get("observation") or {}).get("results") or []
        by_call = {o.get("source_call_id"): o for o in observations if isinstance(o, dict)}
        for index, call in enumerate(calls, 1):
            call_id = call.get("tool_call_id")
            arguments = call.get("arguments") or {}
            events.append({"id": f"s{step_id}.call{index}", "step": step_id, "kind": "tool_call",
                           "text": _text(arguments), "function": call.get("function_name", ""),
                           "arguments": arguments, "source_call_id": call_id})
            obs = by_call.get(call_id) or (observations[index - 1] if index <= len(observations) else {})
            if isinstance(obs, dict):
                events.append({"id": f"s{step_id}.obs{index}", "step": step_id, "kind": "observation",
                               "text": _text(obs.get("content", "")), "extra": obs.get("extra") or {},
                               "source_call_id": call_id})
    return events


def _candidate(rule: str, module: str, error_type: str, root: str, steps: list[int],
               evidence_ids: list[str], reason: str, confidence: float) -> dict[str, Any]:
    return {"rule_id": rule, "candidate_id": f"{rule}@{evidence_ids[0]}",
            "error_module": module, "error_type": error_type,
            "root_cause": root, "critical_steps": sorted(set(steps)),
            "supporting_events": list(dict.fromkeys(evidence_ids)), "reason": reason,
            "confidence": confidence}


def _alf_action(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for index, part in enumerate(parts):
        if part.endswith("bridge.sh") and index + 3 < len(parts) and parts[index + 1] == "send":
            return parts[index + 3].strip()
    return None


def _alf_observation(text: str) -> tuple[str, set[str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, set()
    if not isinstance(payload, dict):
        return text, set()
    commands = payload.get("admissible_commands") or []
    return str(payload.get("observation") or ""), {str(x).strip().lower() for x in commands}


def check_constraints(benchmark: str, events: list[dict[str, Any]], context: dict[str, Any],
                      row: dict[str, str]) -> list[dict[str, Any]]:
    """AgentRx-style guarded checks; candidates carry exact ATIF event IDs."""
    candidates: list[dict[str, Any]] = []
    by_id = {event["id"]: event for event in events}
    for position, event in enumerate(events):
        if event["kind"] != "observation":
            continue
        text = event["text"]
        if benchmark != "livemath" and (truth(event.get("extra", {}).get("is_error")) or ERROR_RE.search(text)):
            call_id = event["id"].replace(".obs", ".call")
            module, error_type = ("action", "parameter_error") if benchmark == "spreadsheetbench" else ("system", "tool_execution_error")
            later_calls = any(e["kind"] == "tool_call" for e in events[position + 1:])
            confidence = 0.45 if later_calls else 0.78
            candidates.append(_candidate("tool_error", module, error_type, "tool_or_script_error",
                                         [event["step"]], [call_id, event["id"]],
                                         "Tool observation reports an explicit execution error; later calls may have repaired it." if later_calls else
                                         "Final tool observation reports an explicit execution error.", confidence))
    if benchmark == "alfworld":
        admissible: set[str] = set()
        previous: tuple[str, str, str, int] | None = None
        failed_process: tuple[int, str] | None = None
        for event in events:
            if event["kind"] != "tool_call":
                continue
            command = event.get("arguments", {}).get("command", "")
            action = _alf_action(str(command))
            if not action:
                continue
            obs_id = event["id"].replace(".call", ".obs")
            obs = by_id.get(obs_id, {})
            observation, new_admissible = _alf_observation(obs.get("text", ""))
            normalized = action.lower()
            if admissible and normalized not in admissible and normalized not in {"inventory", "look"} and ALF_FAILURE_RE.search(observation):
                candidates.append(_candidate("inadmissible_action", "action", "invalid_action", "inadmissible_environment_action",
                                             [event["step"]], [event["id"], obs_id],
                                             "Submitted action is absent from the preceding admissible command list.", 0.88))
            if normalized not in {"inventory", "look"} and ALF_FAILURE_RE.search(observation):
                candidates.append(_candidate("environment_rejection", "action", "invalid_action", "rejected_environment_action",
                                             [event["step"]], [event["id"], obs_id],
                                             "Environment observation explicitly rejects or fails the action.", 0.80))
            if previous and previous[0] == normalized and previous[1] == observation and observation:
                candidates.append(_candidate("repeated_no_progress", "planning", "inefficient_plan", "repeated_action_without_state_change",
                                             [previous[3], event["step"]], [previous[2], event["id"], obs_id],
                                             "Same action was repeated with unchanged environment observation.", 0.76))
            if PROCESS_RE.search(normalized) and ALF_FAILURE_RE.search(observation):
                failed_process = (event["step"], obs_id)
            elif PUT_RE.search(normalized) and failed_process:
                candidates.append(_candidate("failed_subgoal_then_place", "reflection", "outcome_misinterpretation",
                                             "failed_processing_subgoal_not_rechecked", [failed_process[0], event["step"]],
                                             [failed_process[1], event["id"]],
                                             "Agent attempted placement after an explicit processing failure.", 0.84))
                failed_process = None
            previous = (normalized, observation, event["id"], event["step"])
            if new_admissible:
                admissible = new_admissible
    elif benchmark == "spreadsheetbench":
        instruction = context.get("instruction", "")
        wants_output = bool(re.search(r"save|output|保存|结果文件|/bench/answers", instruction, re.I))
        answer_dir = Path(row["trial_path"]) / "artifacts/bench/answers"
        scripts = list(answer_dir.rglob("solution.py")) if answer_dir.is_dir() else []
        if wants_output and not scripts:
            last_call = next((e for e in reversed(events) if e["kind"] == "tool_call"), None)
            if last_call:
                candidates.append(_candidate("missing_output_artifact", "action", "misalignment", "result_not_saved",
                                             [last_call["step"]], [last_call["id"]],
                                             "Task requires solution.py, but no submitted script was saved.", 0.62))
        for event in events:
            if event["kind"] == "observation" and FORMULA_ERROR_RE.search(event["text"]):
                call_id = event["id"].replace(".obs", ".call")
                call = by_id.get(call_id, {})
                if "/bench/answers" not in call.get("text", ""):
                    continue  # Input workbooks may already contain error values.
                candidates.append(_candidate("formula_error_observed", "action", "parameter_error", "spreadsheet_formula_error",
                                             [event["step"]], [call_id, event["id"]],
                                             "Tool output contains an explicit spreadsheet formula error value.", 0.55))
        target_match = re.search(r"### answer_position\s*(?:'[^']+'!|[A-Za-z0-9_]+!)?([A-Z]{1,3}\d+)\s*(?:\n|$)", instruction, re.I)
        if target_match and scripts:
            target = target_match.group(1).upper()
            script = scripts[0].read_text(encoding="utf-8", errors="replace")
            writes = {cell.upper() for cell in CELL_ASSIGN_RE.findall(script)}
            has_dynamic_writes = bool(re.search(r"\.cell\s*\(|iter_rows\s*\(|\bfor\s+\w+\s+in\b", script))
            if writes and target not in writes and not has_dynamic_writes:
                call = next((e for e in reversed(events) if e["kind"] == "tool_call" and
                             "solution.py" in e["text"] and any(cell in e["text"] for cell in writes)), None)
                if call:
                    candidates.append(_candidate("explicit_target_mismatch", "action", "parameter_error", "wrong_target_cell",
                                                 [call["step"]], [call["id"]],
                                                 f"answer_position requires {target}, but explicit script writes target {sorted(writes)}.", 0.60))
    elif benchmark == "livemath":
        # A wrong option is never by itself evidence for a faulty derivation.
        agent_messages = [e for e in events if e["kind"] == "message" and e.get("source") == "agent"]
        if agent_messages:
            final = agent_messages[-1]
            if final["text"].strip() and not BOX_RE.search(final["text"]):
                candidates.append(_candidate("invalid_final_format", "action", "format_error", "invalid_answer_format",
                                             [final["step"]], [final["id"]],
                                             "Final agent message does not contain a boxed A–E option.", 0.80))
    return candidates


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x["confidence"], -min(x["critical_steps"])))


def make_record(benchmark: str, row: dict[str, str], item: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validity, flags = trace_validity(item, row)
    status = outcome(row, validity)
    context = task_context(row)
    if context.get("read_error"):
        flags.append("missing_task_context")
    trace = item.get("trace") if item else None
    events = events_from_trace(trace) if validity == "valid" else []
    candidates = check_constraints(benchmark, events, context, row) if status == "task_failure" else []
    selected = choose_candidate(candidates)
    event_index = {event["id"]: event for event in events}
    evidence = []
    if selected:
        for event_id in selected["supporting_events"]:
            event = event_index.get(event_id)
            if event:
                evidence.append({"event_id": event_id, "step_id": event["step"], "kind": event["kind"],
                                 "excerpt": event["text"][:500]})
    record = {
        "benchmark": benchmark, "task_id": row["task_id"], "task_type": row.get("task_type", ""),
        "outcome": status, "reward": float(row["reward"]) if row.get("reward") else None,
        "trace_validity": validity, "reliability_flags": flags,
        "error_module": selected["error_module"] if selected else None,
        "error_type": selected["error_type"] if selected else None,
        "critical_steps": selected["critical_steps"] if selected else [],
        "supporting_events": selected["supporting_events"] if selected else [],
        "root_cause": selected["root_cause"] if selected and selected["confidence"] >= 0.65 else "undetermined" if status == "task_failure" else None,
        "confidence": selected["confidence"] if selected else 0.0,
        "diagnosis_status": "rule_supported" if selected and selected["confidence"] >= 0.65 else "needs_review" if status == "task_failure" else "filtered",
        "rule_id": selected["rule_id"] if selected else None,
        "evidence": evidence,
        "trace_sha256": item.get("trace_sha256") if item else None,
        "source_trace_path": item.get("source_trace_path") if item else None,
        "metric_trial_path": row["trial_path"],
    }
    return record, candidates, events, context


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "by_benchmark": {b: dict(Counter(r["outcome"] for r in records if r["benchmark"] == b)) for b in BENCHMARKS},
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "diagnoses": dict(Counter(r["diagnosis_status"] for r in records)),
        "rule_hits": dict(Counter(r["rule_id"] for r in records if r["rule_id"])),
    }
