"""M2: progressive, reversible trace views and mechanism-level evidence groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import (ALF_FAILURE_RE, BENCHMARKS, _alf_action, _alf_observation,
                   events_from_trace, sha256)


REPO = Path(__file__).resolve().parents[2]
BENCH_SHORT = {"spreadsheetbench": "sheet", "alfworld": "alf", "livemath": "math"}
TOPIC_PATTERNS = [
    ("probability", r"probability|random variable|expectation|variance|distribution|martingale"),
    ("geometry", r"triangle|circle|polygon|euclidean|projective space|riemannian|geodesic"),
    ("number_theory", r"prime|divisib|congruence|integer solutions|diophantine|modulo"),
    ("combinatorics", r"graph|permutation|combination|partition|coloring|counting"),
    ("analysis", r"integral|derivative|fourier|operator|measure|continuous|l\^p|variation"),
    ("topology", r"topolog|homolog|cohomolog|manifold|homotopy"),
    ("algebra", r"group|ring|field|polynomial|matrix|linear algebra|representation"),
    ("logic", r"logic|theory|definab|o-minimal|model-theoretic"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def token_estimate(chars: int) -> int:
    """Deterministic analysis-context proxy, never reported as billed model tokens."""
    return math.ceil(max(0, chars) / 4)


def percentile75(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(0.75 * len(ordered)) - 1]


def _topic_hint(metric: dict[str, str]) -> str:
    """A cheap public-question metadata tag; no ATIF is opened at L0."""
    trial = Path(metric.get("trial_path") or "")
    try:
        result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        question_path = Path(result["task_id"]["path"]) / "environment/public/question.json"
        question = str(json.loads(question_path.read_text(encoding="utf-8")).get("question", ""))
    except (OSError, KeyError, ValueError, TypeError):
        return "unknown"
    text = question.lower()
    for topic, pattern in TOPIC_PATTERNS:
        if re.search(pattern, text):
            return topic
    return "other"


def _family(benchmark: str, metric: dict[str, str], topic_hint: str) -> str:
    task_type = (metric.get("task_type") or "").lower()
    if benchmark == "spreadsheetbench":
        return "cell_level" if "cell" in task_type else "sheet_level" if "sheet" in task_type else "unknown_spreadsheet"
    if benchmark == "alfworld":
        if any(word in task_type for word in ("clean", "heat", "cool")):
            return "multi_stage_processing"
        if any(word in task_type for word in ("two", "look_at")):
            return "multi_step_interaction"
        return "simple_pick_place"
    return topic_hint


def _scale(benchmark: str, metric: dict[str, str]) -> str:
    if benchmark == "spreadsheetbench":
        cells = int(metric.get("spreadsheet_cell_count") or 0)
        return "small" if cells < 500 else "medium" if cells < 5000 else "large"
    if benchmark == "alfworld":
        actions = int(metric.get("alfworld_actions") or 0)
        return "short" if actions < 10 else "medium" if actions < 30 else "long"
    length = int(metric.get("livemath_question_chars") or 0)
    return "short" if length < 900 else "medium" if length < 1800 else "long"


def build_l0(metrics_dir: Path, m1_records_path: Path) -> list[dict[str, Any]]:
    """L0 touches only Metrics, M1 summaries and public metadata; never ATIF."""
    m1_records = read_jsonl(m1_records_path)
    by_key = {(record["benchmark"], record["task_id"]): record for record in m1_records}
    if len(by_key) != len(m1_records):
        raise ValueError("duplicate M1 task keys")
    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        with (metrics_dir / f"{benchmark}-baseline-original.csv").open(newline="", encoding="utf-8") as source:
            metrics = [row for row in csv.DictReader(source) if row.get("split") == "train"]
        for metric in metrics:
            key = (benchmark, metric["task_id"])
            m1 = by_key.get(key)
            if m1 is None:
                raise ValueError(f"M1 record missing for {key}")
            topic = _topic_hint(metric) if benchmark == "livemath" else ""
            row = {
                "trace_id": f"{benchmark}:{metric['task_id']}", "benchmark": benchmark,
                "task_id": metric["task_id"], "task_type": metric.get("task_type", ""),
                "task_family": _family(benchmark, metric, topic), "topic_hint": topic,
                "topic_hint_source": "public_question_keyword_heuristic" if topic else None,
                "complexity_bucket": _scale(benchmark, metric),
                "outcome": m1["outcome"], "reward": m1["reward"],
                "trace_validity": m1["trace_validity"], "reliability_flags": m1["reliability_flags"],
                "error_module": m1["error_module"], "error_type": m1["error_type"],
                "root_cause": m1["root_cause"], "rule_id": m1["rule_id"],
                "diagnosis_status": m1["diagnosis_status"], "confidence": m1["confidence"],
                "diagnostic_gap": m1.get("diagnostic_gap"),
                "candidate_rule_ids": m1.get("candidate_rule_ids", []),
                "observed_failure_modes": m1.get("observed_failure_modes", []),
                "critical_steps": m1["critical_steps"], "supporting_events": m1["supporting_events"],
                "source_trace_path": m1["source_trace_path"], "trace_sha256": m1["trace_sha256"],
                "trace_chars": int(metric.get("trace_chars") or 0),
                "message_turns": int(metric.get("message_turns") or 0),
                "tool_calls": int(metric.get("tool_calls") or 0),
                "task_tokens": int(metric.get("total_tokens") or 0),
                "alfworld_actions": int(metric.get("alfworld_actions") or 0) if benchmark == "alfworld" else None,
                "spreadsheet_cell_count": int(metric.get("spreadsheet_cell_count") or 0) if benchmark == "spreadsheetbench" else None,
                "livemath_question_chars": int(metric.get("livemath_question_chars") or 0) if benchmark == "livemath" else None,
            }
            rows.append(row)
    if len(rows) != len(m1_records) or len({row["trace_id"] for row in rows}) != len(rows):
        raise ValueError("Metrics and M1 record sets do not match")
    for benchmark in BENCHMARKS:
        subset = [row for row in rows if row["benchmark"] == benchmark]
        limits = {name: percentile75([row[name] for row in subset])
                  for name in ("task_tokens", "tool_calls", "trace_chars")}
        for row in subset:
            row["anomaly_flags"] = [name for name in limits if row[name] > limits[name]]
            row["estimated_full_trace_tokens"] = token_estimate(row["trace_chars"])
    return rows


def read_verified_trace(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["source_trace_path"] or "")
    if not path.is_file():
        raise FileNotFoundError(f"missing original ATIF: {row['trace_id']}")
    if row.get("trace_sha256") and sha256(path) != row["trace_sha256"]:
        raise ValueError(f"ATIF hash changed: {row['trace_id']}")
    trace = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(trace.get("steps"), list):
        raise ValueError(f"malformed ATIF: {row['trace_id']}")
    return trace


def alfworld_sequence(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact Action → Observation → State Change → Next Action IR."""
    events = events_from_trace(trace)
    by_id = {event["id"]: event for event in events}
    sequence: list[dict[str, Any]] = []
    prior_observation: str | None = None
    prior_commands: set[str] = set()
    prior_action: str | None = None
    for event in events:
        if event["kind"] != "tool_call":
            continue
        action = _alf_action(str(event.get("arguments", {}).get("command", "")))
        if not action:
            continue
        observation_id = event["id"].replace(".call", ".obs")
        observation, commands = _alf_observation(by_id.get(observation_id, {}).get("text", ""))
        unchanged = prior_observation == observation and prior_commands == commands if prior_observation is not None else None
        rejected = bool(ALF_FAILURE_RE.search(observation)) and action.lower() not in {"look", "inventory"}
        item = {
            "step_id": event["step"], "action_event_id": event["id"],
            "observation_event_id": observation_id if observation_id in by_id else None,
            "action": action, "observation_excerpt": observation[:180],
            "state_change": "unchanged" if unchanged else "changed" if unchanged is False else "unknown",
            "invalid_return": rejected,
            "repeated_action": prior_action == action.lower(),
            "next_action": None,
        }
        if sequence:
            sequence[-1]["next_action"] = action
        sequence.append(item)
        prior_observation, prior_commands, prior_action = observation, commands, action.lower()
    return sequence


def enrich_l1(rows: list[dict[str, Any]]) -> None:
    """Read ALFWorld traces mechanically; no model reads full trajectories."""
    for row in rows:
        row["trace_signals"] = {}
        row["signal_events"] = []
        if row["benchmark"] != "alfworld" or row["trace_validity"] != "valid":
            continue
        sequence = alfworld_sequence(read_verified_trace(row))
        row["trace_signals"] = {
            "invalid_returns": sum(item["invalid_return"] for item in sequence),
            "repeated_actions": sum(item["repeated_action"] for item in sequence),
            "unchanged_observations": sum(item["state_change"] == "unchanged" for item in sequence),
            "action_count": len(sequence),
        }
        row["signal_events"] = [
            item["observation_event_id"] or item["action_event_id"]
            for item in sequence if item["invalid_return"] or item["repeated_action"]
            or item["state_change"] == "unchanged"
        ][:8]


def mechanism(row: dict[str, Any]) -> tuple[str, str]:
    """Return a mechanism key and epistemic status, never promote a signal to a root cause."""
    if row["outcome"] == "success":
        return f"success:{row['task_family']}", "observed_outcome"
    if row["outcome"] != "task_failure":
        return f"filtered:{row['outcome']}", "reliability_filter"
    if row["diagnosis_status"] == "judge_supported" and row.get("root_cause") not in (None, "undetermined"):
        return f"root:{row['error_module']}:{row['root_cause']}", "judge_supported"
    if row.get("rule_id"):
        return f"candidate:{row['error_module']}:{row['rule_id']}", "candidate_only"
    for mode in row.get("observed_failure_modes") or []:
        if mode["code"] in {"action_budget_exhausted", "agent_stopped_with_unfinished_episode"}:
            return f"signal:{mode['code']}", "observed_signal_only"
    if row.get("candidate_rule_ids"):
        return f"rejected_candidate:{row['candidate_rule_ids'][0]}", "candidate_not_root"
    signals = row.get("trace_signals") or {}
    if signals.get("invalid_returns"):
        return "signal:invalid_return", "observed_signal_only"
    if signals.get("repeated_actions") or signals.get("unchanged_observations"):
        return "signal:no_progress", "observed_signal_only"
    return f"undetermined:{row['task_family']}", "root_undetermined"


def _confidence_label(rows: list[dict[str, Any]]) -> str:
    if all(row["outcome"] == "success" for row in rows):
        return "high"
    if all(row["diagnosis_status"] == "judge_supported" for row in rows):
        return "high"
    if any(row["diagnosis_status"] == "judge_supported" for row in rows):
        return "medium"
    return "low"


def make_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pattern, status = mechanism(row)
        buckets[(row["benchmark"], pattern, status)].append(row)
    valid_failures = Counter(row["benchmark"] for row in rows if row["outcome"] == "task_failure")
    successes = [row for row in rows if row["outcome"] == "success"]
    groups = []
    for (benchmark, pattern, status), cases in sorted(buckets.items()):
        families = Counter(row["task_family"] for row in cases)
        family = families.most_common(1)[0][0] if len(families) == 1 else "mixed"
        frequency = len(cases)
        coverage = frequency / max(1, valid_failures[benchmark]) if cases[0]["outcome"] == "task_failure" else 0.0
        uncertainty = 1 - statistics.mean(float(row.get("confidence") or 0) for row in cases)
        cost = max(1, int(statistics.median(row["estimated_full_trace_tokens"] for row in cases)))
        priority = frequency * coverage * uncertainty / (cost + 1)
        matching_successes = [row["trace_id"] for row in successes if row["benchmark"] == benchmark
                              and row["task_family"] in families]
        key = f"{benchmark}:{pattern}:{status}"
        cluster_id = f"{BENCH_SHORT[benchmark]}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"
        critical = [f"{row['trace_id']}:{event_id}" for row in cases for event_id in row["supporting_events"]]
        signals = Counter()
        for row in cases:
            signals.update(row.get("trace_signals") or {})
        groups.append({
            "cluster_id": cluster_id, "benchmark": benchmark, "task_family": family,
            "failure_pattern": pattern, "mechanism_status": status, "case_count": frequency,
            "case_ids": [row["trace_id"] for row in cases], "representative_traces": [],
            "positive_examples": matching_successes[:3] if cases[0]["outcome"] == "task_failure" else [],
            "critical_events": critical, "summary": f"{frequency} cases; {pattern}; evidence status: {status}",
            "evidence_confidence": _confidence_label(cases), "observed_signals": dict(signals),
            "priority": {"frequency": frequency, "coverage_proxy": round(coverage, 4),
                         "uncertainty_proxy": round(uncertainty, 4), "cost_proxy_tokens": cost,
                         "score": round(priority, 8)},
        })
    return groups


def select_representatives(rows: list[dict[str, Any]], groups: list[dict[str, Any]],
                           max_l2: int, per_group: int) -> list[dict[str, Any]]:
    if max_l2 < 1 or per_group < 1:
        raise ValueError("max_l2 and per_group must be positive")
    group_for = {case_id: group for group in groups for case_id in group["case_ids"]}
    failures = [row for row in rows if row["outcome"] == "task_failure"]
    supported = sorted((row for row in failures if row["diagnosis_status"] == "judge_supported"),
                       key=lambda row: row["trace_id"])
    if len(supported) > max_l2:
        raise ValueError(f"L2 budget {max_l2} cannot preserve {len(supported)} supported cases")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()

    def add(row: dict[str, Any]) -> None:
        if row["trace_id"] in seen:
            return
        group = group_for[row["trace_id"]]
        selected.append(row)
        seen.add(row["trace_id"])
        counts[group["cluster_id"]] += 1
        group["representative_traces"].append(row["trace_id"])

    for row in supported:
        add(row)
    # Give each benchmark at least one failure example when the budget allows.
    for benchmark in BENCHMARKS:
        if len(selected) >= max_l2:
            break
        if any(row["benchmark"] == benchmark for row in selected):
            continue
        candidates = [row for row in failures if row["benchmark"] == benchmark and row["trace_id"] not in seen]
        if candidates:
            candidates.sort(key=lambda row: (-group_for[row["trace_id"]]["priority"]["score"],
                                             row["estimated_full_trace_tokens"], row["trace_id"]))
            add(candidates[0])
    while len(selected) < max_l2:
        candidates = [row for row in failures if row["trace_id"] not in seen
                      and counts[group_for[row["trace_id"]]["cluster_id"]] < per_group]
        if not candidates:
            break
        observed = {(row["benchmark"], row["task_family"], row["complexity_bucket"])
                    for row in selected}
        observed_family = {(row["benchmark"], row["task_family"]) for row in selected}
        def score(row: dict[str, Any]) -> tuple[float, str]:
            group = group_for[row["trace_id"]]
            novelty = (2 if (row["benchmark"], row["task_family"]) not in observed_family else 0)
            novelty += (1 if (row["benchmark"], row["task_family"], row["complexity_bucket"]) not in observed else 0)
            first_group = 2 if counts[group["cluster_id"]] == 0 else 0
            uncertainty = 1 - float(row.get("confidence") or 0)
            return (first_group + novelty + uncertainty + 1000 * group["priority"]["score"]
                    - 0.01 * math.log1p(row["estimated_full_trace_tokens"]), row["trace_id"])
        best = max(candidates, key=score)
        add(best)
    return selected


def make_view(row: dict[str, Any], window: int, excerpt_chars: int) -> dict[str, Any]:
    trace = read_verified_trace(row)
    events = events_from_trace(trace)
    steps = sorted({event["step"] for event in events})
    critical_ids = set(row["supporting_events"])
    anchors = set(row["critical_steps"])
    if not anchors and row["signal_events"]:
        anchors = {event["step"] for event in events if event["id"] in row["signal_events"][:3]}
    if not anchors:
        anchors = set(steps[-3:])
    selected_steps = {step for step in steps if any(abs(step - anchor) <= window for anchor in anchors)}
    selected_steps.update(event["step"] for event in events if event["id"] in critical_ids)
    view_events = []
    for event in events:
        if event["step"] not in selected_steps:
            continue
        full_text = event.get("text", "")
        view_events.append({
            "event_id": event["id"], "step_id": event["step"], "kind": event["kind"],
            "excerpt": full_text[:excerpt_chars], "full_chars": len(full_text),
            "text_sha256": hashlib.sha256(full_text.encode()).hexdigest(),
            "is_critical": event["id"] in critical_ids,
        })
    user_messages = [event for event in events if event["kind"] == "message" and event.get("source") == "user"]
    sequence = alfworld_sequence(trace) if row["benchmark"] == "alfworld" else None
    return {
        "trace_id": row["trace_id"], "benchmark": row["benchmark"], "task_id": row["task_id"],
        "source_trace_path": row["source_trace_path"], "trace_sha256": row["trace_sha256"],
        "failure_pattern": mechanism(row)[0], "task_family": row["task_family"],
        "diagnosis_status": row["diagnosis_status"], "task_excerpt":
            user_messages[0]["text"][:min(excerpt_chars, 300)] if user_messages else "",
        "selected_steps": sorted(selected_steps), "events": view_events,
        "alfworld_sequence": sequence,
    }


def evaluate(rows: list[dict[str, Any]], groups: list[dict[str, Any]],
             selected: list[dict[str, Any]], views: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    supported = [row for row in rows if row["diagnosis_status"] == "judge_supported"]
    expected = {f"{row['trace_id']}:{event_id}" for row in supported for event_id in row["supporting_events"]}
    l1 = {event_id for group in groups for event_id in group["critical_events"]}
    l2 = {f"{view['trace_id']}:{event['event_id']}" for view in views for event in view["events"]}
    baseline = sum(row["estimated_full_trace_tokens"] for row in rows)
    files = [output_dir / "l0.jsonl", output_dir / "groups.jsonl"]
    files.extend(output_dir / "views" / f"{BENCH_SHORT[view['benchmark']]}_{view['task_id']}.json" for view in views)
    loaded = sum(token_estimate(path.stat().st_size) for path in files)
    return {
        "tasks": len(rows), "groups": len(groups), "selected_l2": len(selected),
        "selected_by_benchmark": dict(Counter(row["benchmark"] for row in selected)),
        "unique_selected_families": len({(row["benchmark"], row["task_family"]) for row in selected}),
        "full_trace_context_token_proxy": baseline, "progressive_context_token_proxy": loaded,
        "estimated_context_reduction": round(1 - loaded / baseline, 4) if baseline else 0,
        "token_proxy_method": "ceil(UTF-8 file bytes / 4) for views; ceil(metric trace_chars / 4) for full traces; not billed tokens",
        "supported_root_cases": len(supported), "supported_critical_events": len(expected),
        "l1_critical_event_retention": len(expected & l1) / len(expected) if expected else None,
        "l2_critical_event_retention": len(expected & l2) / len(expected) if expected else None,
        "missing_l1_events": sorted(expected - l1), "missing_l2_events": sorted(expected - l2),
        "interpretation": "Retention measures M1 judge-supported event pointers, not correctness of unknown root causes.",
    }


def build(metrics_dir: Path, m1_records_path: Path, output_dir: Path,
          max_l2: int = 12, per_group: int = 2, window: int = 1,
          excerpt_chars: int = 450) -> dict[str, Any]:
    rows = build_l0(metrics_dir, m1_records_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist L0 before any trajectory is opened.
    write_jsonl(output_dir / "l0.jsonl", rows)
    enrich_l1(rows)
    groups = make_groups(rows)
    selected = select_representatives(rows, groups, max_l2, per_group)
    views = []
    views_dir = output_dir / "views"
    views_dir.mkdir(exist_ok=True)
    for old_view in views_dir.glob("*.json"):
        old_view.unlink()
    for row in selected:
        view = make_view(row, window, excerpt_chars)
        views.append(view)
        path = views_dir / f"{BENCH_SHORT[row['benchmark']]}_{row['task_id']}.json"
        path.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "groups.jsonl", groups)
    write_jsonl(output_dir / "selection.jsonl", [{"trace_id": row["trace_id"],
                 "cluster_id": next(group["cluster_id"] for group in groups if row["trace_id"] in group["case_ids"]),
                 "task_family": row["task_family"], "complexity_bucket": row["complexity_bucket"]}
                 for row in selected])
    evaluation = evaluate(rows, groups, selected, views, output_dir)
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evaluation


def fetch(m1_records_path: Path, benchmark: str, task_id: str, event_id: str) -> dict[str, Any]:
    records = read_jsonl(m1_records_path)
    record = next((item for item in records if item["benchmark"] == benchmark and item["task_id"] == task_id), None)
    if record is None:
        raise KeyError(f"unknown trace: {benchmark}:{task_id}")
    trace = read_verified_trace({**record, "trace_id": f"{benchmark}:{task_id}"})
    event = next((event for event in events_from_trace(trace) if event["id"] == event_id), None)
    if event is None:
        raise KeyError(f"unknown event: {event_id}")
    step = next(step for step in trace["steps"] if step["step_id"] == event["step"])
    return {"trace_id": f"{benchmark}:{task_id}", "event_id": event_id,
            "event": event, "raw_atif_step": step, "trace_sha256": record["trace_sha256"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M2 progressive trace classification and views")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--metrics-dir", type=Path, default=REPO.parent / "metrics")
    build_parser.add_argument("--m1-records", type=Path, default=REPO / "results/m1/records.jsonl")
    build_parser.add_argument("--output-dir", type=Path, default=REPO / "results/m2")
    build_parser.add_argument("--max-l2", type=int, default=12)
    build_parser.add_argument("--per-group", type=int, default=2)
    build_parser.add_argument("--window", type=int, default=1)
    build_parser.add_argument("--excerpt-chars", type=int, default=450)
    fetch_parser = sub.add_parser("fetch")
    fetch_parser.add_argument("--m1-records", type=Path, default=REPO / "results/m1/records.jsonl")
    fetch_parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    fetch_parser.add_argument("--task-id", required=True)
    fetch_parser.add_argument("--event-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.metrics_dir, args.m1_records, args.output_dir, args.max_l2,
                       args.per_group, args.window, args.excerpt_chars)
    else:
        result = fetch(args.m1_records, args.benchmark, args.task_id, args.event_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

