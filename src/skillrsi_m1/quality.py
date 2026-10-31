"""Deterministic group quality and reuse-readiness audit for M3 gating."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

from .profiles import observed_labels


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _mean_pairwise(values: list[set[str]]) -> float | None:
    pairs = list(combinations(values, 2))
    return round(sum(_jaccard(left, right) for left, right in pairs) / len(pairs), 4) if pairs else None


def _member_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    tasks = _jaccard(set(left["task_signature"]["required_operations"]),
                     set(right["task_signature"]["required_operations"]))
    behavior = _jaccard(observed_labels(left), observed_labels(right))
    subtype = float(left["task_signature"]["task_subtype"] == right["task_signature"]["task_subtype"])
    return .4 * tasks + .4 * behavior + .2 * subtype


def _group_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["benchmark"] != right["benchmark"] or left["task_family"] != right["task_family"]:
        return 0.0
    left_pattern = set(left.get("behavior_pattern", [])) | set(left.get("observed_symptoms", []))
    right_pattern = set(right.get("behavior_pattern", [])) | set(right.get("observed_symptoms", []))
    return round(.5 + .5 * _jaccard(left_pattern, right_pattern), 4)


def audit_groups(profiles: list[dict[str, Any]], positive: list[dict[str, Any]],
                 failure: list[dict[str, Any]], selected: set[str]) -> dict[str, Any]:
    by_id = {profile["trace_id"]: profile for profile in profiles}
    all_groups = [("positive", group) for group in positive] + [("failure", group) for group in failure]
    details = []
    for kind, group in all_groups:
        members = [by_id[trace_id] for trace_id in group["trace_ids"]]
        signatures = [set(member["task_signature"]["required_operations"])
                      | {"subtype:" + member["task_signature"]["task_subtype"],
                         "complexity:" + member["task_signature"]["task_complexity"]} for member in members]
        behaviors = [observed_labels(member) for member in members]
        task_cohesion = _mean_pairwise(signatures)
        behavior_cohesion = _mean_pairwise(behaviors)
        medoid = max(members, key=lambda candidate: (sum(_member_similarity(candidate, other) for other in members),
                                                      candidate["trace_id"]))["trace_id"]
        nearest = max(((other["group_id"], _group_similarity(group, other)) for other_kind, other in all_groups
                       if other_kind == kind and other["group_id"] != group["group_id"]),
                      key=lambda item: item[1], default=(None, 0.0))
        evidence = group.get("evidence_events", group.get("critical_events", []))
        adverse = kind == "positive" and any("repeated_" in label or "no_observed_progress" in label
                                              for label in group.get("behavior_pattern", []))
        if len(members) == 1:
            readiness = "single_case_hypothesis"
        elif adverse or (task_cohesion is not None and task_cohesion < .5) or \
                (behavior_cohesion is not None and behavior_cohesion < .5) or not evidence:
            readiness = "needs_review"
        else:
            readiness = "candidate_experience"
        details.append({"group_id": group["group_id"], "kind": kind, "benchmark": group["benchmark"],
                        "case_count": len(members), "representative_trace": medoid,
                        "representative_in_l2": medoid in selected,
                        "task_signature_cohesion": task_cohesion,
                        "behavior_signature_cohesion": behavior_cohesion,
                        "nearest_group_id": nearest[0], "nearest_group_similarity": nearest[1],
                        "evidence_event_count": len(evidence), "adverse_behavior_present": adverse,
                        "reuse_readiness": readiness, "causal_status": "hypothesis_only"})
    return {"groups": details, "summary": {"total_groups": len(details),
            "singletons": sum(item["case_count"] == 1 for item in details),
            "singletons_by_kind": dict(Counter(item["kind"] for item in details if item["case_count"] == 1)),
            "reuse_readiness": dict(Counter(item["reuse_readiness"] for item in details)),
            "near_duplicate_groups": sum(item["nearest_group_similarity"] >= .85 for item in details)}}
