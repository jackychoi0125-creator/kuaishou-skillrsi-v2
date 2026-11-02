"""M1: deterministic Trace reliability and outcome validation only."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import BENCHMARKS, EXPECTED_TRAIN, events_from_trace


def _flag(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _load_wrapped_atif(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    entries: dict[str, dict[str, Any]] = {}
    issues = []
    if not path.is_file():
        return entries, [f"{path.name}:missing_atif_jsonl"]
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                entry = json.loads(line)
                case_id = entry["case_id"]
                if not isinstance(case_id, str) or case_id in entries:
                    raise ValueError("missing or duplicate case_id")
                entries[case_id] = entry
            except (ValueError, KeyError, TypeError) as exc:
                issues.append(f"{path.name}:{number}:{type(exc).__name__}")
    return entries, issues


def validate_one(benchmark: str, row: dict[str, str], entry: dict[str, Any] | None) -> dict[str, Any]:
    flags: list[str] = []
    if row.get("benchmark") and row["benchmark"] != benchmark:
        flags.append("benchmark_mismatch")
    if row.get("split") != "train":
        flags.append("split_mismatch")
    trial_path = Path(row.get("trial_path") or "")
    if row.get("attempt") and trial_path.is_dir() and row["attempt"] not in trial_path.parts:
        flags.append("attempt_source_mismatch")
    reward = _number(row.get("reward"))
    if reward is None:
        flags.append("missing_or_invalid_reward")
    elif not 0 <= reward <= 1:
        flags.append("reward_out_of_range")
    elif row.get("success") and _flag(row["success"]) != (reward >= .999):
        flags.append("success_reward_disagreement")
    if _flag(row.get("model_error")):
        flags.append("terminal_model_error")
    if "timeout" in (row.get("error_type") or "").lower():
        flags.append("provider_timeout")
    if int(row.get("agent_429_errors") or 0):
        flags.append("observed_429")
    for key in ("execution_error", "scoring_error"):
        if _flag(row.get(key)):
            flags.append(key)
    if row.get("original_error_type"):
        flags.append("infrastructure_rerun")
    if row.get("attempt") and row["attempt"] != "baseline-v3":
        flags.append("nonbaseline_attempt")
    trace_path = Path(row.get("trace_path") or "")
    trace = entry.get("trace") if entry else None
    trace_hash = entry.get("trace_sha256") if entry else None
    validity = "valid"
    if any(flag in flags for flag in ("benchmark_mismatch", "split_mismatch", "attempt_source_mismatch")):
        validity = "invalid"
    if entry and entry.get("trace_error"):
        flags.append("malformed_trace")
        validity = "invalid"
    elif entry is None or trace is None or not trace_path.is_file():
        flags.append("missing_trace" if entry is None or trace is None else "missing_source_trace")
        validity = "invalid"
    elif entry.get("case_id") != row["task_id"] or entry.get("source_trace_path") != str(trace_path):
        flags.append("trace_identity_mismatch")
        validity = "invalid"
    elif not isinstance(trace, dict) or not isinstance(trace.get("steps"), list) or not trace["steps"]:
        flags.append("missing_required_steps")
        validity = "invalid"
    else:
        steps = trace["steps"]
        step_ids = [step.get("step_id") for step in steps if isinstance(step, dict)]
        if len(step_ids) != len(steps) or any(type(value) is not int for value in step_ids) or len(set(step_ids)) != len(step_ids):
            flags.append("invalid_step_ids")
            validity = "invalid"
        try:
            source_bytes = trace_path.read_bytes()
            if trace_hash and hashlib.sha256(source_bytes).hexdigest() != trace_hash:
                flags.append("trace_hash_mismatch")
                validity = "invalid"
            elif json.loads(source_bytes) != trace:
                flags.append("embedded_atif_source_mismatch")
                validity = "invalid"
        except (OSError, ValueError, TypeError):
            flags.append("malformed_source_trace")
            validity = "invalid"
        if validity == "valid" and (not any(step.get("source") == "agent" for step in steps)
                                    or _flag(row.get("model_error")) and reward != 1):
            flags.append("incomplete_agent_run")
            validity = "partial"
    if _flag(row.get("scoring_error")) or reward is None or "reward_out_of_range" in flags:
        task_outcome = "uncertain"
    elif reward >= .999:
        task_outcome = "success"
    elif validity != "valid" or any(flag in flags for flag in ("terminal_model_error", "execution_error")):
        task_outcome = "uncertain"
    else:
        task_outcome = "failure"
    event_refs = []
    if validity in {"valid", "partial"} and isinstance(trace, dict):
        try:
            event_refs = [event["id"] for event in events_from_trace(trace)]
        except (KeyError, TypeError, ValueError):
            flags.append("event_index_unavailable")
    return {
        "schema_version": "2.0", "trace_id": f"{benchmark}:{row['task_id']}",
        "benchmark": benchmark, "task_id": row["task_id"], "split": row.get("split"),
        "attempt": row.get("attempt"), "source_run": row.get("attempt"),
        "source_trace_path": str(trace_path), "trace_sha256": trace_hash,
        "trace_validity": validity, "task_outcome": task_outcome,
        "event_refs": event_refs, "event_count": len(event_refs),
        "reward": reward, "reliability_flags": sorted(set(flags)),
        "metrics": {key: row.get(key) for key in ("task_type", "input_tokens", "output_tokens", "total_tokens",
                    "trace_chars", "message_turns", "tool_calls", "end_to_end_sec", "model_sec", "tool_sec",
                    "spreadsheet_file_bytes", "spreadsheet_sheet_count", "spreadsheet_cell_count",
                    "alfworld_actions", "livemath_question_chars")},
        "metric_trial_path": row.get("trial_path"), "original_error_type": row.get("original_error_type") or None,
    }


def _direct_entry(row: dict[str, str]) -> dict[str, Any]:
    path = Path(row.get("trace_path") or "")
    try:
        source_bytes = path.read_bytes()
        trace = json.loads(source_bytes)
        return {"case_id": row["task_id"], "source_trace_path": str(path),
                "trace_sha256": hashlib.sha256(source_bytes).hexdigest(), "trace": trace}
    except (OSError, ValueError, TypeError) as exc:
        return {"case_id": row["task_id"], "source_trace_path": str(path),
                "trace_sha256": None, "trace": None, "trace_error": type(exc).__name__}


def run(metrics_dir: Path, input_dir: Path, output_dir: Path, dry_run: bool = False,
        metrics_variant: str = "baseline") -> dict[str, Any]:
    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    input_issues: list[str] = []
    counts: dict[str, Any] = {}
    replacements: list[dict[str, Any]] = []
    original_outcomes: Counter[str] = Counter()
    for benchmark in BENCHMARKS:
        with (metrics_dir / f"{benchmark}-{metrics_variant}.csv").open(newline="", encoding="utf-8") as source:
            all_rows = list(csv.DictReader(source))
        with (metrics_dir / f"{benchmark}-baseline-original.csv").open(newline="", encoding="utf-8") as source:
            original_rows = {row["task_id"]: row for row in csv.DictReader(source) if row.get("split") == "train"}
        original_outcomes.update("success" if _flag(row.get("success")) else "failure" for row in original_rows.values())
        rows = [row for row in all_rows if row.get("split") == "train"]
        if len(rows) != EXPECTED_TRAIN[benchmark]:
            input_issues.append(f"{benchmark}: expected {EXPECTED_TRAIN[benchmark]} train rows, got {len(rows)}")
        entries, issues = _load_wrapped_atif(input_dir / f"{benchmark}-train.atif.jsonl")
        input_issues.extend(issues)
        metrics_ids = [row["task_id"] for row in rows]
        if len(metrics_ids) != len(set(metrics_ids)):
            input_issues.append(f"{benchmark}: duplicate metric task_id")
        for task_id in sorted(set(entries) - set(metrics_ids)):
            input_issues.append(f"{benchmark}: ATIF-only {task_id}")
        for row in rows:
            original = original_rows.get(row["task_id"])
            source_replaced = bool(original and original.get("trace_path") != row.get("trace_path"))
            if source_replaced:
                replacements.append({"trace_id": f"{benchmark}:{row['task_id']}",
                                     "original_success": _flag(original.get("success")),
                                     "final_success": _flag(row.get("success")),
                                     "original_trace_path": original.get("trace_path"),
                                     "final_trace_path": row.get("trace_path"),
                                     "original_attempt": original.get("attempt"), "final_attempt": row.get("attempt")})
            entry = _direct_entry(row) if source_replaced else entries.get(row["task_id"])
            record = validate_one(benchmark, row, entry)
            record["original_source_trace_path"] = original.get("trace_path") if original else None
            record["source_replaced_by_retry"] = source_replaced
            records.append(record)
        counts[benchmark] = {"raw_train_metrics": len(rows), "raw_train_atif": len(entries),
                             "success": sum(record["task_outcome"] == "success" for record in records if record["benchmark"] == benchmark),
                             "failure": sum(record["task_outcome"] == "failure" for record in records if record["benchmark"] == benchmark),
                             "uncertain": sum(record["task_outcome"] == "uncertain" for record in records if record["benchmark"] == benchmark)}
    audit = {
        "schema_version": "2.0", "raw_input_count": sum(item["raw_train_metrics"] for item in counts.values()),
        "processed_count": len(records), "by_benchmark": counts,
        "trace_validity": dict(Counter(record["trace_validity"] for record in records)),
        "task_outcome": dict(Counter(record["task_outcome"] for record in records)),
        "flag_counts": dict(Counter(flag for record in records for flag in record["reliability_flags"])),
        "excluded_or_uncertain": [{"trace_id": record["trace_id"], "reason": record["reliability_flags"]}
                                  for record in records if record["trace_validity"] != "valid" or record["task_outcome"] == "uncertain"],
        "sources": dict(Counter(record["source_run"] or "unknown" for record in records)),
        "source_replacements": replacements,
        "input_issues": input_issues, "model_calls": 0,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "baseline_reconciliation": {"original_reward_outcomes": dict(original_outcomes),
            "final_reward_outcomes": dict(Counter("success" if record["reward"] is not None and record["reward"] >= .999 else "failure" for record in records)),
            "replaced_trials": len(replacements), "source_csv_variant": metrics_variant},
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "validated.jsonl").open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
