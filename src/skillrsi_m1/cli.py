"""Command line entry point for M1 reliability and evidence reports."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .core import (BENCHMARKS, TAXONOMY, diagnostic_gap, events_from_trace, load_atif_jsonl,
                   make_record, metric_rows, prepare_train, summarize)
from .judge import ChatClient, classify_and_localize


REPO = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_records(records: list[dict[str, Any]]) -> None:
    ids = [(r["benchmark"], r["task_id"]) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate benchmark/task_id records")
    for record in records:
        module, error_type = record["error_module"], record["error_type"]
        if module is not None and (module not in TAXONOMY or error_type not in TAXONOMY[module]):
            raise ValueError(f"invalid AgentDebug label: {module}/{error_type}")
        if record["root_cause"] not in {None, "undetermined"}:
            if not record["supporting_events"] or not record["critical_steps"] or not record["evidence"]:
                raise ValueError(f"ungrounded root cause: {record['benchmark']}/{record['task_id']}")
            if set(record["supporting_events"]) != {e["event_id"] for e in record["evidence"]}:
                raise ValueError(f"broken evidence references: {record['benchmark']}/{record['task_id']}")
            source = Path(record["source_trace_path"])
            if not source.is_file():
                raise ValueError(f"missing evidence source: {record['benchmark']}/{record['task_id']}")
            trace = json.loads(source.read_text(encoding="utf-8"))
            events = {event["id"]: event for event in events_from_trace(trace)}
            for cited in record["evidence"]:
                event = events.get(cited["event_id"])
                if not event or event["step"] != cited["step_id"] or not event["text"].startswith(cited["excerpt"]):
                    raise ValueError(f"evidence does not match ATIF: {record['benchmark']}/{record['task_id']}")
        if record["benchmark"] == "livemath" and record["outcome"] == "task_failure" and record["root_cause"] != "undetermined":
            if record.get("rule_id") != "invalid_final_format":
                raise ValueError(f"unverified math root cause: {record['task_id']}")


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = []
    for benchmark in BENCHMARKS:
        rows = metric_rows(args.metrics_dir, benchmark)
        atif = load_atif_jsonl(args.input_dir / f"{benchmark}-train.atif.jsonl")
        if set(atif) != {row["task_id"] for row in rows}:
            raise ValueError(f"{benchmark}: ATIF JSONL and Metrics train IDs differ")
        for row in rows:
            record, candidates, events, context = make_record(benchmark, row, atif[row["task_id"]])
            records.append(record)
            if record["outcome"] == "task_failure" and candidates:
                pending.append((record, candidates, events, context))
    reused = 0
    if args.llm:
        if args.resume and (args.output_dir / "records.jsonl").is_file():
            previous = {(r["benchmark"], r["task_id"]): r for r in
                        map(json.loads, (args.output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines())}
            remaining = []
            judge_fields = ("error_module", "error_type", "critical_steps", "supporting_events", "root_cause",
                            "confidence", "diagnosis_status", "rule_id", "evidence", "judge_input_tokens", "judge_output_tokens")
            for entry in pending:
                record = entry[0]
                old = previous.get((record["benchmark"], record["task_id"]))
                if old and not old.get("judge_error") and "judge_input_tokens" in old and old.get("trace_sha256") == record.get("trace_sha256"):
                    record.update({field: old[field] for field in judge_fields if field in old})
                    reused += 1
                else:
                    remaining.append(entry)
            pending = remaining
        client = ChatClient(args.model_config, timeout=args.llm_timeout, thinking=args.kimi_thinking)
        selected = pending[:args.max_llm] if args.max_llm is not None else pending
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(classify_and_localize, client, *entry): entry[0] for entry in selected}
            for done, future in enumerate(as_completed(future_map), 1):
                record = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    record["judge_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    record["diagnosis_status"] = "needs_review"
                print(f"judge {done}/{len(selected)} {record['benchmark']}/{record['task_id']} "
                      f"{record['diagnosis_status']}", file=sys.stderr, flush=True)
    validate_records(records)
    for record in records:
        record["diagnostic_gap"] = diagnostic_gap(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "records.jsonl", records)
    _write_jsonl(args.output_dir / "failures.jsonl", [r for r in records if r["outcome"] == "task_failure"])
    _write_jsonl(args.output_dir / "filtered.jsonl", [r for r in records if r["outcome"] != "task_failure"])
    summary = summarize(records)
    summary["llm_attempted"] = len(selected) if args.llm else 0
    summary["llm_reused"] = reused
    summary["llm_errors"] = sum(bool(r.get("judge_error")) for r in records)
    summary["judge_input_tokens"] = sum(r.get("judge_input_tokens", 0) for r in records)
    summary["judge_output_tokens"] = sum(r.get("judge_output_tokens", 0) for r in records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="build original Train ATIF JSONL from trial traces")
    prepare.add_argument("--metrics-dir", type=Path, default=REPO.parent / "metrics")
    prepare.add_argument("--output-dir", type=Path, default=REPO / "data/train")
    run = sub.add_parser("diagnose", help="create evidence-grounded FailureRecords")
    run.add_argument("--metrics-dir", type=Path, default=REPO.parent / "metrics")
    run.add_argument("--input-dir", type=Path, default=REPO / "data/train")
    run.add_argument("--output-dir", type=Path, default=REPO / "results/m1")
    run.add_argument("--llm", action="store_true", help="enable Kimi classifier and localizer")
    run.add_argument("--model-config", type=Path, default=REPO.parent / "model.txt")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-llm", type=int, help="limit LLM calls for a smoke run")
    run.add_argument("--llm-timeout", type=int, default=300, help="per-request read timeout in seconds")
    run.add_argument("--kimi-thinking", action="store_true", help="use gateway default Kimi thinking mode")
    run.add_argument("--resume", action="store_true", help="reuse completed Judge results from output directory")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_train(args.metrics_dir, args.output_dir)
    else:
        if args.workers < 1:
            parser.error("--workers must be positive")
        result = diagnose(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
