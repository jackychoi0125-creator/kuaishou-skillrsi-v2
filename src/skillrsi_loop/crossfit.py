"""Source-Train-only M2/M3 bundles with a fail-closed holdout leakage audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from skillrsi_m1.m2_pipeline import run as run_m2
from skillrsi_m1.core import events_from_trace
from skillrsi_m3.core import write_json
from .r16_plan import BENCHMARKS, freeze
from .stability import digest, file_sha, train_metrics

GROUP_FILES = ("positive_strategy_groups.json", "failure_pattern_groups.json", "contrast_groups.json")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_source(root: Path, benchmark: str, plan: dict[str, Any]) -> dict[str, Any]:
    study = plan["studies"][benchmark]
    source, holdout = set(study["source_train_task_ids"]), set(study["holdout_task_ids"])
    if not source or not holdout or source & holdout or source | holdout != set(train_metrics(root, benchmark)):
        raise ValueError("frozen Source/Holdout partition invalid")
    validated_path = root / "results/m1/validated.jsonl"
    original = _jsonl(validated_path)
    selected = [row for row in original if row["benchmark"] == benchmark and row["task_id"] in source]
    if {row["task_id"] for row in selected} != source or any(row.get("split") != "train" for row in selected):
        raise ValueError("M1 source records do not match frozen Source Train")
    base = root / "results/r16" / benchmark
    base.mkdir(parents=True, exist_ok=True)
    provenance = base / "provenance.json"
    cache_dir = base / "isolated_kimi_cache"
    if not provenance.is_file() and cache_dir.is_dir() and any(cache_dir.iterdir()):
        raise ValueError("new crossfit bundle must start with empty cache")
    source_path = base / "source_validated.jsonl"
    if source_path.is_file():
        existing = _jsonl(source_path)
        if existing != selected:
            raise ValueError("frozen Source validated input changed")
    else:
        source_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    meta = {"benchmark": benchmark, "plan_sha256": plan["plan_sha256"],
            "source_train_task_ids": sorted(source), "holdout_task_ids": sorted(holdout),
            "source_sha256": digest(sorted(source)), "holdout_sha256": digest(sorted(holdout)),
            "source_validated_path": str(source_path), "source_validated_sha256": file_sha(source_path),
            "m2_dir": str(base / "m2"), "m3_dir": str(base / "m3"), "m4_dir": str(base / "m4"),
            "cache_dir": str(base / "isolated_kimi_cache"),
            "cache_initially_empty": True,
            "old_r1_full_train_candidate_forbidden": True, "split_origin": "frozen_baseline_train_metadata"}
    if provenance.is_file() and json.loads(provenance.read_text()) != meta:
        raise ValueError("crossfit provenance changed")
    write_json(provenance, meta)
    return meta


def build_source_m2(root: Path, benchmark: str, plan: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    meta = prepare_source(root, benchmark, plan)
    output = Path(meta["m2_dir"])
    if dry_run:
        return {"benchmark": benchmark, "dry_run": True, "model_calls": 0,
                "source_count": len(meta["source_train_task_ids"]), "holdout_count": len(meta["holdout_task_ids"])}
    result = run_m2(Path(meta["source_validated_path"]), None, output, max_deep_read_traces=12,
                    max_positive_traces=6, max_negative_traces=8, max_contrast_pairs=4)
    audit = audit_bundle(root, benchmark, plan)
    if audit["status"] != "clean_source_m2":
        raise ValueError("Source M2 leakage audit failed: " + ",".join(audit["errors"]))
    return {"benchmark": benchmark, "m2": result, "isolation": audit}


def audit_bundle(root: Path, benchmark: str, plan: dict[str, Any]) -> dict[str, Any]:
    study = plan["studies"][benchmark]
    source, holdout = set(study["source_train_task_ids"]), set(study["holdout_task_ids"])
    base = root / "results/r16" / benchmark
    meta_path = base / "provenance.json"
    errors = []
    if not meta_path.is_file():
        return {"benchmark": benchmark, "status": "unverifiable", "errors": ["missing_provenance"]}
    meta = json.loads(meta_path.read_text())
    if meta.get("plan_sha256") != plan["plan_sha256"] or set(meta.get("source_train_task_ids", [])) != source or set(meta.get("holdout_task_ids", [])) != holdout:
        errors.append("frozen_partition_mismatch")
    if source & holdout or source | holdout != set(train_metrics(root, benchmark)):
        errors.append("non_train_or_overlapping_partition")
    source_path = Path(meta["source_validated_path"])
    if not source_path.is_file() or file_sha(source_path) != meta.get("source_validated_sha256"):
        errors.append("source_input_sha_mismatch")
    m2_dir = Path(meta["m2_dir"])
    profile_path = m2_dir / "trace_profiles.jsonl"
    if not profile_path.is_file():
        return {"benchmark": benchmark, "status": "unverifiable", "errors": errors + ["missing_source_m2"]}
    profiles = _jsonl(profile_path)
    trace_to_task = {p["trace_id"]: p["task_id"] for p in profiles}
    if any(p.get("benchmark") != benchmark or p.get("split") != "train" or p["task_id"] not in source for p in profiles):
        errors.append("m2_profile_holdout_or_non_train")
    if len(trace_to_task) != len(profiles) or set(trace_to_task.values()) != source:
        errors.append("m2_profile_source_coverage")
    for profile in profiles:
        trace_path = Path(profile["source_trace_path"])
        if not trace_path.is_file() or file_sha(trace_path) != profile["trace_sha256"]:
            errors.append("m2_raw_trace_sha_mismatch")
            break
    group_count = 0
    for filename in GROUP_FILES:
        path = m2_dir / filename
        if not path.is_file():
            errors.append("missing_" + filename)
            continue
        for group in json.loads(path.read_text()):
            group_count += 1
            refs = list(group.get("trace_ids", []))
            refs += [group[key] for key in ("positive_trace_id", "negative_trace_id") if group.get(key)]
            if any(trace_to_task.get(ref) not in source for ref in refs):
                errors.append("m2_group_holdout_or_unknown_trace")
    for filename in ("l2_contrasts.json",):
        path = m2_dir / filename
        if path.is_file() and any(trace_to_task.get(item.get(key)) not in source for item in json.loads(path.read_text())
                                  for key in ("positive_trace_id", "negative_trace_id")):
            errors.append("m2_contrast_holdout")
    for path in (m2_dir / "deep_views").glob("*.json"):
        if trace_to_task.get(json.loads(path.read_text())["trace_id"]) not in source:
            errors.append("m2_deep_view_holdout")
    m3_path = Path(meta["m3_dir"]) / "insights.jsonl"
    insight_count = 0
    event_ids_by_trace: dict[str, set[str]] = {}
    if m3_path.is_file():
        for item in _jsonl(m3_path):
            insight_count += 1
            if item.get("benchmark") != benchmark:
                errors.append("m3_cross_benchmark")
            for ref in item.get("supporting_refs", []) + item.get("counterexample_refs", []):
                if trace_to_task.get(ref.get("trace_id")) not in source:
                    errors.append("m3_ref_holdout_or_unknown_trace")
                    continue
                profile = next(p for p in profiles if p["trace_id"] == ref["trace_id"])
                if ref.get("trace_sha256") != profile["trace_sha256"]:
                    errors.append("m3_ref_sha_mismatch")
                if ref["trace_id"] not in event_ids_by_trace:
                    trace = json.loads(Path(profile["source_trace_path"]).read_text())
                    event_ids_by_trace[ref["trace_id"]] = {event["id"] for event in events_from_trace(trace)}
                if not ref.get("event_ids") or not set(ref["event_ids"]).issubset(event_ids_by_trace[ref["trace_id"]]):
                    errors.append("m3_ref_event_missing")
    cache_dir = Path(meta["cache_dir"])
    if not cache_dir.resolve().is_relative_to(base.resolve()) or cache_dir.resolve() == (root / "results/r1/cache").resolve():
        errors.append("cache_not_isolated")
    if meta.get("cache_initially_empty") is not True:
        errors.append("cache_initial_state_unverified")
    if cache_dir.is_dir():
        for cache in cache_dir.glob("*.json"):
            content = cache.read_text()
            if any(task in content for task in holdout):
                errors.append("cache_contains_holdout_id")
                break
    m4_dir = Path(meta["m4_dir"])
    for path in (m4_dir / "source_prompt.json", m4_dir / "candidate_manifest.json"):
        if path.is_file() and any(task in path.read_text() for task in holdout):
            errors.append("m4_prompt_or_manifest_contains_holdout_id")
    status = "contaminated" if errors else "clean" if m3_path.is_file() else "clean_source_m2"
    return {"benchmark": benchmark, "status": status, "errors": sorted(set(errors)),
            "source_sha256": study["source_sha256"], "holdout_sha256": study["holdout_sha256"],
            "m2_profiles": len(profiles), "m2_groups": group_count, "m3_insights": insight_count,
            "m2_profiles_sha256": file_sha(profile_path),
            "m3_insights_sha256": file_sha(m3_path) if m3_path.is_file() else None,
            "cache_isolation_verified": "cache_not_isolated" not in errors,
            "provenance_path": str(meta_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dry-run", "build-m2", "audit"))
    parser.add_argument("--benchmark", choices=("all", *BENCHMARKS), default="all")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = freeze(root)
    values = []
    for benchmark in BENCHMARKS if args.benchmark == "all" else (args.benchmark,):
        if args.command == "dry-run":
            values.append(build_source_m2(root, benchmark, plan, dry_run=True))
        elif args.command == "build-m2":
            values.append(build_source_m2(root, benchmark, plan))
        else:
            values.append(audit_bundle(root, benchmark, plan))
    if args.command != "dry-run":
        write_json(root / "results/r16/crossfit_provenance.json", values)
    print(json.dumps(values, ensure_ascii=False))


if __name__ == "__main__":
    main()
