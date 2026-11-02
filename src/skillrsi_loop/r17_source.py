"""Rebuild R1.7 Source-only M2 and audit its Train provenance."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from skillrsi_m1.m2_pipeline import run as run_m2
from skillrsi_m1.core import events_from_trace
from skillrsi_m3.core import write_json
from .r17_plan import freeze
from .stability import digest, file_sha, train_metrics


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def contains_task_id(content: str, task_id: str) -> bool:
    """Match a whole task identifier, not a prefix such as -2 in -27."""
    return bool(re.search(r"(?<![A-Za-z0-9_-])" + re.escape(task_id) + r"(?![A-Za-z0-9_-])", content))


def audit(root: Path, benchmark: str, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = plan or freeze(root)
    study = plan["studies"][benchmark]
    source = set(study["source_train_task_ids"])
    excluded = set(study["future_holdout_task_ids"] + study["consumed_r16_holdout_task_ids"])
    base = root / "results/r17" / benchmark
    errors = []
    if source & excluded or source | excluded != set(train_metrics(root, benchmark)):
        errors.append("source_holdout_partition_invalid")
    source_file = base / "source_validated.jsonl"
    meta_file = base / "source_provenance.json"
    if not source_file.is_file() or not meta_file.is_file():
        return {"benchmark": benchmark, "status": "unverifiable", "errors": errors + ["source_artifacts_missing"]}
    meta = json.loads(meta_file.read_text())
    if meta.get("plan_sha256") != plan["plan_sha256"] or file_sha(source_file) != meta.get("source_validated_sha256"):
        errors.append("source_provenance_mismatch")
    if meta.get("cache_initially_empty") is not True:
        errors.append("cache_initial_state_unverified")
    validated = _jsonl(source_file)
    if {row["task_id"] for row in validated} != source or any(row.get("split") != "train" or row.get("benchmark") != benchmark for row in validated):
        errors.append("source_validated_task_leak_or_gap")
    profile_file = base / "m2/trace_profiles.jsonl"
    if not profile_file.is_file():
        return {"benchmark": benchmark, "status": "unverifiable", "errors": errors + ["m2_profiles_missing"]}
    profiles = _jsonl(profile_file)
    trace_ids = {p["trace_id"]: p for p in profiles}
    if {p["task_id"] for p in profiles} != source or any(p["task_id"] in excluded or p.get("split") != "train" for p in profiles):
        errors.append("m2_profile_holdout_leak_or_gap")
    for p in profiles:
        if file_sha(Path(p["source_trace_path"])) != p["trace_sha256"]:
            errors.append("raw_trace_sha_mismatch")
            break
    group_count = 0
    for name in ("positive_strategy_groups.json", "failure_pattern_groups.json", "contrast_groups.json"):
        path = base / "m2" / name
        if not path.is_file():
            errors.append("missing_" + name)
            continue
        for group in json.loads(path.read_text()):
            group_count += 1
            refs = list(group.get("trace_ids", []))
            refs += [group[key] for key in ("positive_trace_id", "negative_trace_id") if group.get(key)]
            if any(ref not in trace_ids for ref in refs):
                errors.append("group_holdout_or_unknown_trace")
    contrast_file = base / "m2/l2_contrasts.json"
    if contrast_file.is_file() and any(item.get(key) not in trace_ids for item in json.loads(contrast_file.read_text())
                                  for key in ("positive_trace_id", "negative_trace_id")):
        errors.append("l2_contrast_holdout_or_unknown_trace")
    for path in (base / "m2/deep_views").glob("*.json"):
        if json.loads(path.read_text()).get("trace_id") not in trace_ids:
            errors.append("deep_view_holdout_or_unknown_trace")
    insight_file = base / "m3/insights.jsonl"
    if insight_file.is_file():
        event_ids_by_trace = {}
        for item in _jsonl(insight_file):
            for ref in item.get("supporting_refs", []) + item.get("counterexample_refs", []):
                trace = trace_ids.get(ref.get("trace_id"))
                if not trace or ref.get("trace_sha256") != trace["trace_sha256"]:
                    errors.append("m3_reference_holdout_or_sha_mismatch")
                    continue
                if ref["trace_id"] not in event_ids_by_trace:
                    raw = json.loads(Path(trace["source_trace_path"]).read_text())
                    event_ids_by_trace[ref["trace_id"]] = {event["id"] for event in events_from_trace(raw)}
                if not ref.get("event_ids") or not set(ref["event_ids"]).issubset(event_ids_by_trace[ref["trace_id"]]):
                    errors.append("m3_reference_event_missing")
    cache = base / "isolated_kimi_cache"
    if not cache.resolve().is_relative_to(base.resolve()):
        errors.append("cache_path_outside_bundle")
    if cache.is_dir() and any(contains_task_id(path.read_text(), task)
                              for path in cache.glob("*.json") for task in excluded):
        errors.append("cache_contains_holdout_id")
    prompt = base / "m4/source_prompt.json"
    if prompt.is_file() and any(contains_task_id(prompt.read_text(), task) for task in excluded):
        errors.append("m4_prompt_contains_holdout_id")
    for skill in (base / "skills").glob("**/SKILL.md"):
        if any(contains_task_id(skill.read_text(), task) for task in excluded):
            errors.append("m4_skill_contains_holdout_id")
    return {"benchmark": benchmark, "status": "clean" if not errors else "contaminated",
            "errors": sorted(set(errors)), "source_tasks": len(source), "m2_profiles": len(profiles),
            "m2_groups": group_count, "source_sha256": study["source_sha256"],
            "future_holdout_sha256": study["future_holdout_sha256"],
            "cache_dir": str(cache), "insight_count": len(_jsonl(insight_file)) if insight_file.is_file() else 0}


def build(root: Path, benchmark: str, *, dry_run: bool = False) -> dict[str, Any]:
    plan = freeze(root)
    study = plan["studies"][benchmark]
    source = set(study["source_train_task_ids"])
    validated = [row for row in _jsonl(root / "results/m1/validated.jsonl")
                 if row.get("benchmark") == benchmark and row.get("task_id") in source]
    if {row["task_id"] for row in validated} != source or any(row.get("split") != "train" for row in validated):
        raise ValueError("frozen Source Train does not match M1 validated records")
    if dry_run:
        return {"benchmark": benchmark, "status": "dry_run", "source_count": len(source),
                "future_holdout_count": len(study["future_holdout_task_ids"]), "model_calls": 0,
                "harbor_tasks": 0}
    base = root / "results/r17" / benchmark
    cache = base / "isolated_kimi_cache"
    provenance_file = base / "source_provenance.json"
    if not provenance_file.is_file() and cache.is_dir() and any(cache.iterdir()):
        raise ValueError("fresh R1.7 Source bundle must start with empty isolated cache")
    path = base / "source_validated.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in validated)
    if path.is_file() and path.read_text() != content:
        raise ValueError("frozen Source validated content changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    meta = {"benchmark": benchmark, "plan_sha256": plan["plan_sha256"],
            "source_validated_sha256": file_sha(path), "source_sha256": digest(sorted(source)),
            "future_holdout_sha256": study["future_holdout_sha256"],
            "old_r16_holdout_excluded": True, "cache_initially_empty": True,
            "cache_dir": str(cache), "source_path": str(path)}
    prior_meta = provenance_file
    if prior_meta.is_file() and json.loads(prior_meta.read_text()) != meta:
        raise ValueError("Source provenance changed")
    write_json(prior_meta, meta)
    run_m2(path, None, base / "m2", max_deep_read_traces=12, max_positive_traces=6,
           max_negative_traces=8, max_contrast_pairs=4)
    result = audit(root, benchmark, plan)
    if result["status"] != "clean":
        raise ValueError("R1.7 M2 source isolation failed: " + str(result["errors"]))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", "spreadsheetbench", "alfworld", "livemath"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    benchmarks = ("spreadsheetbench", "alfworld", "livemath") if args.benchmark == "all" else (args.benchmark,)
    print(json.dumps([build(args.root.resolve(), b, dry_run=args.dry_run) for b in benchmarks], ensure_ascii=False))


if __name__ == "__main__":
    main()
