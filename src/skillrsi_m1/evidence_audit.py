"""Compare L2 analysis views with original ATIF events without model calls."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .core import BENCHMARKS


def audit_views(profiles: list[dict[str, Any]], events_cache: dict[str, list[dict[str, Any]]],
                views: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {profile["trace_id"]: profile for profile in profiles}
    view_by_id = {view["trace_id"]: view for view in views}
    details = []
    all_root, retained_root = 0, 0
    all_difference, retained_difference = 0, 0
    for view in views:
        trace_id = view["trace_id"]
        profile = by_id[trace_id]
        source = {event["id"]: event for event in events_cache[trace_id]}
        selected = {event["event_id"]: event for event in view["events"]}
        broken = []
        for event_id, item in selected.items():
            raw = source.get(event_id)
            if raw is None or len(raw["text"]) != item["full_chars"]:
                broken.append(event_id)
            elif item["compressed"]:
                head, _, tail = item["content"].partition("\n[...analysis-view middle omitted...]\n")
                if not raw["text"].startswith(head) or not raw["text"].endswith(tail):
                    broken.append(event_id)
            elif raw["text"] != item["content"]:
                broken.append(event_id)
        root = profile["diagnosis"]["confirmed_root_cause"]
        root_ids = set(root["event_ids"]) if isinstance(root, dict) else set()
        all_root += len(root_ids)
        retained_root += len(root_ids & set(selected))
        symptom_ids = [set(item["event_ids"]) for item in profile["diagnosis"]["observed_symptoms"]]
        symptom_covered = any(ids & set(selected) for ids in symptom_ids) if symptom_ids else None
        behavior_labels = [label for dimension in profile["behavior_signature"].values() for label in dimension
                           if label["status"] == "observed"]
        behavior_covered = sum(bool(set(label["event_ids"]) & set(selected)) for label in behavior_labels)
        content = "\n".join(item["content"] for item in view["events"])
        benchmark = profile["benchmark"]
        if benchmark == "alfworld" and profile["outcome"]["status"] == "success":
            terminal_present = '"won": true' in content or '"won":true' in content
        elif benchmark == "livemath":
            terminal_present = "\\boxed" in content
        else:
            terminal_present = None
        details.append({"trace_id": trace_id, "benchmark": benchmark, "outcome": profile["outcome"]["status"],
                        "event_ref_integrity": not broken, "broken_event_refs": broken,
                        "root_events_retained": len(root_ids & set(selected)), "root_events_total": len(root_ids),
                        "any_symptom_event_retained": symptom_covered,
                        "behavior_labels_with_evidence": behavior_covered,
                        "observed_behavior_labels": len(behavior_labels),
                        "terminal_evidence_present": terminal_present,
                        "compressed_events": sum(item["compressed"] for item in view["events"])})
    for comparison in comparisons:
        positive_ids = {item["event_id"] for item in view_by_id[comparison["positive_trace_id"]]["events"]}
        negative_ids = {item["event_id"] for item in view_by_id[comparison["negative_trace_id"]]["events"]}
        for difference in comparison["differences"]:
            all_difference += 1
            if set(difference["positive_event_ids"]).issubset(positive_ids) and \
                    set(difference["negative_event_ids"]).issubset(negative_ids):
                retained_difference += 1
    samples = {}
    for benchmark in BENCHMARKS:
        subset = [item for item in details if item["benchmark"] == benchmark]
        pair = next((item for item in comparisons if by_id[item["positive_trace_id"]]["benchmark"] == benchmark), None)
        samples[benchmark] = {"success": next((item["trace_id"] for item in subset if item["outcome"] == "success"), None),
                              "failure": next((item["trace_id"] for item in subset if item["outcome"] == "failure"), None),
                              "contrast_id": pair["contrast_id"] if pair else None,
                              "contrast_status": "paired" if pair else "no_matched_positive"}
    return {"summary": {"sampled_traces": len(details), "sampled_by_benchmark":
                dict(Counter(item["benchmark"] for item in details)),
                "event_ref_integrity": sum(item["event_ref_integrity"] for item in details) / len(details) if details else None,
                "supported_root_event_retention": retained_root / all_root if all_root else None,
                "contrast_difference_event_retention": retained_difference / all_difference if all_difference else None,
                "alfworld_success_terminal_feedback_retained": all(item["terminal_evidence_present"] for item in details
                    if item["benchmark"] == "alfworld" and item["outcome"] == "success"),
                "compressed_math_views_need_full_read": sum(item["benchmark"] == "livemath" and item["compressed_events"] > 0
                    for item in details)},
            "samples": samples, "details": details,
            "interpretation": "Structural retention only; semantic sufficiency requires manual review and independent task validation."}
