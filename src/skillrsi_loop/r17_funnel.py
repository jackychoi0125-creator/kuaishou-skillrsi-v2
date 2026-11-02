"""Replay historical Source-Train M3 caches and account for every discarded Insight."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from skillrsi_m3.core import run as run_m3, write_json
from .crossfit import audit_bundle
from .r16_plan import freeze as freeze_r16
from .r17_plan import freeze as freeze_r17

GROUP_FILES = ("positive_strategy_groups.json", "failure_pattern_groups.json", "contrast_groups.json")


class OfflineReplay:
    def __init__(self, response: dict[str, Any]):
        self.response = response

    def ask(self, *_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], str]:
        return self.response, "r16_cached_response_replay"

    def usage(self) -> dict[str, Any]:
        return {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "cache_tokens": 0, "mode": "offline_replay"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fresh_generation(root: Path, benchmark: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    base = root / "results/r17" / benchmark
    m3 = base / "m3"
    if not (m3 / "insights.jsonl").is_file():
        return None, []
    m2 = base / "m2"
    cache_files = [json.loads(path.read_text()) for path in (base / "isolated_kimi_cache").glob("*.json")]
    extraction = [item for item in cache_files if item.get("usage", {}).get("stage") == "m3_extract"]
    if len(extraction) != 1:
        raise ValueError("fresh Source generation requires exactly one extraction cache")
    cached = extraction[0]
    raw = cached["response"]["insights"]
    diagnostics = json.loads((m3 / "extraction_diagnostics.json").read_text())
    pool = _jsonl(m3 / "insights.jsonl")
    audit = json.loads((m3 / "insight_audit.json").read_text())
    details = {item["insight_id"]: item for item in audit["details"]}
    selected = json.loads((m3 / "selected_evidence.json").read_text())
    ops = _jsonl(m3 / "insight_operations.jsonl")
    refinement_path = m3 / "refinement_diagnostics.json"
    refinements = json.loads(refinement_path.read_text()) if refinement_path.is_file() else []
    refinement_by_id = {item["insight_id"]: item for item in refinements if item.get("insight_id")}
    rejections = []
    by_id = {item["insight_id"]: item for item in pool}
    for record in diagnostics:
        reasons = list(record.get("reasons") or [])
        item = by_id.get(record.get("insight_id"))
        stage = record["stage"]
        if item:
            reasons.extend(details[item["insight_id"]].get("errors") or [])
            if item["status"] == "pending_math_validation":
                reasons.append("independent_math_validation_missing")
            if item.get("risk_level") == "high":
                reasons.append("high_risk")
            if not item.get("measurement_specificity"):
                reasons.append("expected_observation_not_per_insight")
            if item.get("estimated_cost_risk") == "unknown":
                reasons.append("cost_risk_not_per_insight")
            if not item.get("cost_bound"):
                reasons.append("numeric_cost_bound_missing")
            if item.get("avoid_conditions"):
                from skillrsi_m4.core import inverted_avoid_condition
                if inverted_avoid_condition(item["avoid_conditions"]):
                    reasons.append("inverted_avoid_condition")
            refinement = refinement_by_id.get(item["insight_id"])
            if refinement and refinement.get("status") == "rejected":
                reasons.extend(refinement.get("reasons") or [])
        if reasons:
            rejections.append({"raw_index": record["raw_index"], "insight_id": record.get("insight_id"),
                               "source_group_id": record.get("source_group_id"), "stage": stage,
                               "reasons": sorted(set(reasons))})
    from skillrsi_m4.core import inverted_avoid_condition
    scoped = [item for item in pool if item["status"] == "verified_observation" and
              item.get("trigger_conditions") and item.get("risk_level") != "high" and
              not inverted_avoid_condition(item.get("avoid_conditions") or [])]
    measurable = [item for item in scoped if item.get("measurement_specificity")]
    stage_counts = {"m2_input_groups": sum(len(json.loads((m2 / file).read_text())) for file in GROUP_FILES),
                    "selector_groups": len(selected), "llm_input_groups": len(selected),
                    "llm_input_events": sum(len(group["events"]) for group in selected),
                    "llm_raw_insights": len(raw), "cached_json_object_parse_success": 1,
                    "schema_compatible_insights": sum(isinstance(item, dict) and all(key in item for key in
                                                  ("source_group_id", "principle", "recommended_behavior", "supporting_refs")) for item in raw),
                    "extractor_event_refs_valid": sum(item["stage"] == "extractor_accepted" for item in diagnostics),
                    "consolidated_insights": len(pool),
                    "dedup_merge_operations": sum(item.get("operation") in {"AGREE", "EDIT"} for item in ops),
                    "evidence_verified_or_math_pending": audit["verified"] + audit["pending"],
                    "independently_semantic_validated": sum(item.get("semantic_validation") == "independently_checked"
                                                         for item in audit["details"]),
                    "risk_and_applicability_pass": len(scoped),
                    "procedural_math_hypotheses": sum(item["status"] == "pending_math_validation" for item in pool),
                    "operationalized_hypotheses": sum(bool(item.get("cost_bound")) and item.get("measurement_specificity") for item in pool),
                    "measurable_behavior_hypotheses": len(measurable),
                    "m4_cost_bounded_usable": sum(isinstance(item.get("cost_bound"), dict) for item in measurable)}
    result = {"scope": "fresh R1.7 Source Train only", "stages": stage_counts,
              "selected_kinds": dict(Counter(item["kind"] for item in selected)),
              "payload_truncated_events": sum(e.get("truncated", False) for group in selected for e in group["events"]),
              "response_global_fields_not_per_insight": [key for key in ("observed_problem", "expected_behavior_change",
                  "expected_observation", "estimated_cost_risk") if key in cached["response"] and
                  any(key not in item for item in raw if isinstance(item, dict))],
              "new_kimi_usage": {"extraction": json.loads((m3 / "usage.json").read_text()),
                                 "refinement": json.loads((m3 / "refinement_usage.json").read_text())
                                 if (m3 / "refinement_usage.json").is_file() else None},
              "math_evidence_level": "procedural_hypothesis" if benchmark == "livemath" else None,
              "utility_status": "unverified"}
    return result, rejections


def replay_one(root: Path, benchmark: str, *, dry_run: bool = False) -> dict[str, Any]:
    freeze_r17(root)
    isolation = audit_bundle(root, benchmark, freeze_r16(root))
    if isolation["status"] != "clean":
        raise ValueError("R1.6 Source bundle failed isolation recheck")
    prior = root / "results/r16" / benchmark
    out = root / "results/r17" / benchmark
    m2 = prior / "m2"
    selected = json.loads((prior / "m3/selected_evidence.json").read_text())
    group_counts = {file.replace("_groups.json", ""): len(json.loads((m2 / file).read_text()))
                    for file in GROUP_FILES}
    events = [event for group in selected for event in group["events"]]
    cache_files = list((prior / "isolated_kimi_cache").glob("*.json"))
    if len(cache_files) > 1:
        raise ValueError("ambiguous historical M3 cache")
    cached = json.loads(cache_files[0].read_text()) if cache_files else None
    if cached and (not isinstance(cached.get("response"), dict) or
                   not isinstance(cached["response"].get("insights"), list)):
        raise ValueError("historical cached response has no parsed insights array")
    raw = cached["response"]["insights"] if cached else []
    if dry_run:
        return {"benchmark": benchmark, "status": "dry_run", "m2_groups": sum(group_counts.values()),
                "selected_groups": len(selected), "payload_events": len(events),
                "cached_raw_insights": len(raw), "model_calls": 0, "harbor_tasks": 0}
    diagnostics: list[dict[str, Any]] = []
    max_groups = {"spreadsheetbench": 8, "alfworld": 3, "livemath": 2}[benchmark]
    replay = run_m3(m2, benchmark, out / "replay_m3",
                    OfflineReplay(cached["response"]) if cached else None,
                    max_groups=max_groups, deterministic_only=cached is None,
                    diagnostics=diagnostics, math_generalize=False)
    pool = _jsonl(out / "replay_m3/insights.jsonl")
    audit = json.loads((out / "replay_m3/insight_audit.json").read_text())
    by_id = {item["insight_id"]: item for item in pool}
    detail_by_id = {item["insight_id"]: item for item in audit["details"]}
    rejected = []
    for item in diagnostics:
        reasons = list(item.get("reasons") or [])
        insight = by_id.get(item.get("insight_id"))
        if insight is None and item["stage"] == "extractor_accepted":
            reasons.append("consolidated_into_other_insight")
        elif insight is not None:
            reasons.extend(detail_by_id[item["insight_id"]].get("errors") or [])
            if insight.get("status") == "pending_math_validation":
                reasons.append("independent_math_validation_missing")
            if insight.get("risk_level") == "high":
                reasons.append("high_risk")
            if not insight.get("measurement_specificity"):
                reasons.append("unmeasurable_expected_observation")
            if not insight.get("cost_bound"):
                reasons.append("numeric_cost_bound_missing")
        if reasons:
            rejected.append({"raw_index": item["raw_index"], "source_group_id": item.get("source_group_id"),
                             "insight_id": item.get("insight_id"), "stage": item["stage"],
                             "reasons": sorted(set(reasons))})
    if cached is None:
        rejected.append({"raw_index": None, "stage": "model_admission", "reasons": [
            "r16_deterministic_only_no_llm" if benchmark == "livemath" else "missing_historical_cache"]})
    verified = [item for item in pool if item.get("status") == "verified_observation"]
    scoped = [item for item in verified if item.get("trigger_conditions") and
              isinstance(item.get("avoid_conditions"), list) and item.get("risk_level") != "high"]
    measurable = [item for item in scoped if item.get("measurement_specificity")]
    usable = [item for item in measurable if isinstance(item.get("cost_bound"), dict)]
    stages = {"m2_input_groups": sum(group_counts.values()),
              "selector_groups": len(selected), "llm_input_groups": len(selected) if cached else 0,
              "llm_input_events": len(events) if cached else 0,
              "llm_raw_insights": len(raw), "cached_json_object_parse_success": int(bool(cached)),
              "schema_compatible_insights": sum(isinstance(item, dict) and all(key in item for key in
                                            ("source_group_id", "principle", "recommended_behavior", "supporting_refs")) for item in raw),
              "extractor_event_refs_valid": sum(item["stage"] == "extractor_accepted" for item in diagnostics),
              "consolidated_insights": len(pool), "verified_observations": len(verified),
              "risk_and_applicability_pass": len(scoped), "measurable_behavior_hypotheses": len(measurable),
              "m4_cost_bounded_usable": len(usable)}
    result = {"benchmark": benchmark, "scope": "R1.6 Source-only cached replay; development diagnostic, not independent validation",
              "group_counts": group_counts, "selected_kinds": dict(Counter(g["kind"] for g in selected)),
              "payload_truncated_events": sum(e.get("truncated", False) for e in events),
              "stages": stages, "replay_m3": replay,
              "cached_kimi_usage_original": cached.get("usage") if cached else None,
              "new_kimi_usage": {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
              "normalization_counts": {"singular_event_id_refs": sum(i.get("normalized_event_id_refs", 0) for i in diagnostics)},
              "result": "no_independently_validated_rule"}
    breakdown = {"benchmark": benchmark, "raw_candidate_rejections": rejected,
                 "reason_counts": dict(Counter(reason for row in rejected for reason in row["reasons"])),
                 "note": "One raw Insight may have multiple precise rejection reasons; raw cache response remains unchanged."}
    fresh, fresh_rejections = _fresh_generation(root, benchmark)
    if fresh:
        result["fresh_source_generation"] = fresh
        breakdown["fresh_source_rejections"] = fresh_rejections
        breakdown["fresh_reason_counts"] = dict(Counter(reason for row in fresh_rejections for reason in row["reasons"]))
    write_json(out / "insight_funnel.json", result)
    write_json(out / "rejection_breakdown.json", breakdown)
    return {"benchmark": benchmark, "stages": stages, "rejections": len(rejected), "new_model_calls": 0}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", "spreadsheetbench", "alfworld", "livemath"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    values = [replay_one(args.root.resolve(), benchmark, dry_run=args.dry_run)
              for benchmark in (("spreadsheetbench", "alfworld", "livemath") if args.benchmark == "all" else (args.benchmark,))]
    print(json.dumps(values, ensure_ascii=False))


if __name__ == "__main__":
    main()
