"""M2: unified profiles, evidence groups, matching and reversible deep views."""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS
from .core import BENCHMARKS
from .evidence_audit import audit_views
from .profiles import observed_labels, profile_one, validate_profile
from .quality import audit_groups


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z]{4,}|\d+", text.lower())
            if token not in {"with", "from", "that", "this", "which", "there", "then", "where", "answer"}}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def match_pair(positive: dict[str, Any], negative: dict[str, Any]) -> dict[str, Any]:
    if positive["benchmark"] != negative["benchmark"]:
        return {"grade": None, "score": 0.0, "reason": "different_benchmark"}
    if positive["outcome"]["status"] != "success" or negative["outcome"]["status"] != "failure":
        return {"grade": None, "score": 0.0, "reason": "wrong_outcome_pair"}
    if positive["task_id"] == negative["task_id"] and positive["trace_id"] != negative["trace_id"]:
        return {"grade": "A", "score": 1.0, "reason": "same_train_task_different_attempt"}
    left, right = positive["task_signature"], negative["task_signature"]
    if left["task_family"] == "unknown" or left["task_family"] != right["task_family"]:
        return {"grade": None, "score": 0.0, "reason": "different_or_unknown_task_family"}
    operations = _jaccard(set(left["required_operations"]), set(right["required_operations"]))
    constraints = _jaccard(set(left["task_constraints"]), set(right["task_constraints"]))
    complexity = 1.0 if left["task_complexity"] == right["task_complexity"] else 0.0
    subtype = 1.0 if left["task_subtype"] == right["task_subtype"] else 0.0
    text = _jaccard(_tokens(positive["task_goal_excerpt"]), _tokens(negative["task_goal_excerpt"]))
    score = round(.3 * operations + .15 * constraints + .15 * complexity + .2 * subtype + .2 * text, 4)
    if positive["benchmark"] == "livemath" and text < .4:
        return {"grade": None, "score": score, "reason": "math_problem_content_not_comparable"}
    if operations >= .5 and score >= .65 and text >= .25:
        return {"grade": "B", "score": score, "reason": "similar_operations_constraints_and_complexity"}
    if operations > 0 and score >= .4 and text >= .12:
        return {"grade": "C", "score": score, "reason": "same_family_limited_similarity"}
    return {"grade": None, "score": score, "reason": "insufficient_operation_or_goal_similarity"}


def _evidence_labels(profile: dict[str, Any]) -> dict[str, list[str]]:
    return {item["label"]: item["event_ids"] for dimension in profile["behavior_signature"].values()
            for item in dimension if item["status"] == "observed"}


def contrast(positive: dict[str, Any], negative: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    p_labels, n_labels = _evidence_labels(positive), _evidence_labels(negative)
    common = sorted(set(p_labels) & set(n_labels))
    differences = []
    for label in sorted(set(p_labels) ^ set(n_labels)):
        differences.append({"behavior": label, "positive_event_ids": p_labels.get(label, []),
                            "negative_event_ids": n_labels.get(label, []),
                            "interpretation": "hypothesis_only"})
    return {"contrast_id": f"{positive['trace_id']}__{negative['trace_id']}",
            "positive_trace_id": positive["trace_id"], "negative_trace_id": negative["trace_id"],
            "matching_grade": match["grade"], "match_score": match["score"], "matching_reason": match["reason"],
            "shared_task_family": positive["task_signature"]["task_family"],
            "shared_constraints": sorted(set(positive["task_signature"]["task_constraints"]) &
                                         set(negative["task_signature"]["task_constraints"])),
            "common_behaviors": [{"behavior": label, "positive_event_ids": p_labels[label],
                                  "negative_event_ids": n_labels[label]} for label in common],
            "behavior_differences": differences, "causal_status": "hypothesis_only"}


def make_contrasts(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [profile for profile in profiles if profile["eligible_for_learning"] and profile["outcome"]["status"] == "success"]
    negatives = [profile for profile in profiles if profile["eligible_for_learning"] and profile["outcome"]["status"] == "failure"]
    contrasts = []
    rejected = []
    for negative in negatives:
        candidates = [(match_pair(positive, negative), positive) for positive in positives]
        eligible = [(match, positive) for match, positive in candidates if match["grade"]]
        if not eligible:
            reasons = Counter(match["reason"] for match, _ in candidates)
            rejected.append({"negative_trace_id": negative["trace_id"], "status": "no_matched_positive",
                             "rejection_reasons": dict(reasons)})
            continue
        grade_order = {"A": 3, "B": 2, "C": 1}
        match, positive = max(eligible, key=lambda pair: (grade_order[pair[0]["grade"]], pair[0]["score"],
                                                          pair[1]["trace_id"]))
        contrasts.append(contrast(positive, negative, match))
    return contrasts, rejected


def _group_id(prefix: str, key: tuple[Any, ...]) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(repr(key).encode()).hexdigest()[:10]}"


def make_groups(profiles: list[dict[str, Any]], contrasts: list[dict[str, Any]],
                rejected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pos_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    neg_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        if not profile["eligible_for_learning"]:
            continue
        signature = profile["task_signature"]
        behavior = observed_labels(profile)
        if profile["outcome"]["status"] == "success":
            salient = tuple(sorted(label for label in behavior if label.startswith(("action:", "verification:", "recovery:", "efficiency:"))))
            pos_buckets[(profile["benchmark"], signature["task_family"], salient)].append(profile)
        elif profile["outcome"]["status"] == "failure":
            symptoms = tuple(sorted({item["code"] for item in profile["diagnosis"]["observed_symptoms"]})) or ("no_observed_symptom",)
            salient = tuple(sorted(label for label in behavior if label.startswith(("verification:", "recovery:", "efficiency:"))))
            confirmed = profile["diagnosis"]["confirmed_root_cause"]
            root = confirmed["root_cause"] if isinstance(confirmed, dict) else "undetermined"
            neg_buckets[(profile["benchmark"], signature["task_family"], symptoms, salient, root)].append(profile)
    positive = []
    for key, members in sorted(pos_buckets.items()):
        evidence = list(dict.fromkeys(f"{member['trace_id']}:{event_id}" for member in members
                                  for dimension in member["behavior_signature"].values() for label in dimension
                                  if label["status"] == "observed" for event_id in label["event_ids"]))
        positive.append({"group_id": _group_id("positive", key), "benchmark": key[0], "task_family": key[1],
                         "behavior_pattern": list(key[2]), "case_count": len(members),
                         "trace_ids": [member["trace_id"] for member in members], "evidence_events": evidence,
                         "claim_status": "candidate_strategy_not_causal_proof"})
    failure = []
    for key, members in sorted(neg_buckets.items()):
        evidence = list(dict.fromkeys(f"{member['trace_id']}:{event_id}" for member in members
                                  for event_id in member["diagnosis"]["critical_events"]))
        failure.append({"group_id": _group_id("failure", key), "benchmark": key[0], "task_family": key[1],
                        "observed_symptoms": list(key[2]), "behavior_pattern": list(key[3]),
                        "confirmed_root_cause": key[4], "case_count": len(members),
                        "trace_ids": [member["trace_id"] for member in members], "critical_events": evidence})
    contrast_groups = [{"group_id": _group_id("contrast", (item["positive_trace_id"], item["negative_trace_id"])),
                        **item} for item in contrasts]
    contrast_groups.extend({"group_id": _group_id("unmatched", (item["negative_trace_id"],)), **item}
                           for item in rejected)
    return positive, failure, contrast_groups


def select_l2(profiles: list[dict[str, Any]], contrasts: list[dict[str, Any]],
              max_deep_read_traces: int, max_positive_traces: int, max_negative_traces: int,
              max_contrast_pairs: int, max_llm_input_tokens: int) -> tuple[list[str], list[str]]:
    by_id = {profile["trace_id"]: profile for profile in profiles}
    selected: list[str] = []
    pair_ids: list[str] = []
    counts = Counter()
    estimated = 0

    def cost(trace_id: str) -> int:
        metric = by_id[trace_id]["outcome"]["metrics"]
        return max(100, min(3000, math.ceil(int(metric.get("trace_chars") or 0) / 20)))

    def add(trace_id: str) -> bool:
        nonlocal estimated
        if trace_id in selected:
            return True
        status = by_id[trace_id]["outcome"]["status"]
        limit = max_positive_traces if status == "success" else max_negative_traces
        if len(selected) >= max_deep_read_traces or counts[status] >= limit or estimated + cost(trace_id) > max_llm_input_tokens:
            return False
        selected.append(trace_id)
        counts[status] += 1
        estimated += cost(trace_id)
        return True

    grade_order = {"A": 3, "B": 2, "C": 1}
    supported_roots = [profile for profile in profiles if profile["eligible_for_learning"] and
                       isinstance(profile["diagnosis"]["confirmed_root_cause"], dict)]
    for profile in sorted(supported_roots, key=lambda item: item["trace_id"]):
        add(profile["trace_id"])
    ranked_pairs = sorted(contrasts, key=lambda item: (-grade_order[item["matching_grade"]], -item["match_score"]))
    benchmark_first = [next((pair for pair in ranked_pairs if by_id[pair["positive_trace_id"]]["benchmark"] == benchmark), None)
                       for benchmark in BENCHMARKS]
    ordered_pairs = [pair for pair in benchmark_first if pair is not None]
    ordered_pairs.extend(pair for pair in ranked_pairs if pair not in ordered_pairs)
    seeded_benchmarks: set[str] = set()
    for pair in ordered_pairs:
        if len(pair_ids) >= max_contrast_pairs:
            break
        p, n = pair["positive_trace_id"], pair["negative_trace_id"]
        benchmark = by_id[p]["benchmark"]
        if benchmark in seeded_benchmarks:
            continue
        # Avoid half a pair when either side is over budget.
        if p not in selected and n not in selected and (len(selected) + 2 > max_deep_read_traces or
            counts["success"] >= max_positive_traces or counts["failure"] >= max_negative_traces or
            estimated + cost(p) + cost(n) > max_llm_input_tokens):
            continue
        if add(p) and add(n):
            pair_ids.append(pair["contrast_id"])
            seeded_benchmarks.add(benchmark)
    for benchmark in BENCHMARKS:
        if benchmark in seeded_benchmarks:
            continue
        available_positive = [item for item in profiles if item["benchmark"] == benchmark and
                              item["eligible_for_learning"] and item["outcome"]["status"] == "success"]
        available_negative = [item for item in profiles if item["benchmark"] == benchmark and
                              item["eligible_for_learning"] and item["outcome"]["status"] == "failure"]
        if available_positive and available_negative:
            candidates = [(match_pair(p, n)["score"], p, n) for p in available_positive for n in available_negative]
            _, p, n = max(candidates, key=lambda item: (item[0], -cost(item[1]["trace_id"])-cost(item[2]["trace_id"])))
            if (len(selected) + int(p["trace_id"] not in selected) + int(n["trace_id"] not in selected) <= max_deep_read_traces
                    and counts["success"] < max_positive_traces and counts["failure"] < max_negative_traces
                    and estimated + cost(p["trace_id"]) + cost(n["trace_id"]) <= max_llm_input_tokens):
                add(p["trace_id"])
                add(n["trace_id"])
    for pair in ordered_pairs:
        if len(pair_ids) >= max_contrast_pairs:
            break
        p, n = pair["positive_trace_id"], pair["negative_trace_id"]
        if pair["contrast_id"] in pair_ids:
            continue
        if p not in selected and n not in selected and (len(selected) + 2 > max_deep_read_traces or
            counts["success"] >= max_positive_traces or counts["failure"] >= max_negative_traces or
            estimated + cost(p) + cost(n) > max_llm_input_tokens):
            continue
        if add(p) and add(n):
            pair_ids.append(pair["contrast_id"])
    negatives = [profile for profile in profiles if profile["eligible_for_learning"] and profile["outcome"]["status"] == "failure"]
    negatives.sort(key=lambda item: (not isinstance(item["diagnosis"]["confirmed_root_cause"], dict),
                                     -len(item["diagnosis"]["observed_symptoms"]), item["trace_id"]))
    for profile in negatives:
        add(profile["trace_id"])
    positives = [profile for profile in profiles if profile["eligible_for_learning"] and profile["outcome"]["status"] == "success"]
    positives.sort(key=lambda item: (-len(observed_labels(item)), item["trace_id"]))
    for profile in positives:
        add(profile["trace_id"])
    return selected, pair_ids


def deep_view(profile: dict[str, Any], events: list[dict[str, Any]], contrast_event_ids: set[str]) -> dict[str, Any]:
    diagnosis = profile["diagnosis"]
    root = diagnosis["confirmed_root_cause"]
    root_ids = set(root["event_ids"]) if isinstance(root, dict) else set()
    critical = set(root_ids) | contrast_event_ids
    for symptom in diagnosis["observed_symptoms"][:3]:
        critical.update(symptom["event_ids"])
    for hypothesis in diagnosis["root_cause_hypotheses"][:2]:
        critical.update(hypothesis["event_ids"])
    for dimension in profile["behavior_signature"].values():
        for label in dimension[:2]:
            if label["status"] == "observed":
                critical.update(label["event_ids"][:2])
    first_agent = next((event["id"] for event in events if event["kind"] == "message" and event.get("source") == "agent"), None)
    last_agent = next((event["id"] for event in reversed(events) if event["kind"] == "message" and event.get("source") == "agent"), None)
    critical.update(event_id for event_id in (first_agent, last_agent) if event_id)
    last_observation = next((event for event in reversed(events) if event["kind"] == "observation"), None)
    if last_observation:
        critical.add(last_observation["id"])
        critical.add(last_observation["id"].replace(".obs", ".call"))
    selected_events = []
    for event in events:
        if event["id"] not in critical:
            continue
        content = event["text"]
        compressed = False
        if event["kind"] == "message" and len(content) > 3000 and event["id"] not in root_ids:
            # Agent monologues can be very long. Keep the evidence-bearing beginning/end;
            # the full event remains retrievable by trace_id and event_id.
            content = content[:1400] + "\n[...analysis-view middle omitted...]\n" + content[-700:]
            compressed = True
        selected_events.append({"event_id": event["id"], "step_id": event["step"],
                                "kind": event["kind"], "content": content,
                                "compressed": compressed, "full_chars": len(event["text"])})
    return {"trace_id": profile["trace_id"], "source_trace_path": profile["source_trace_path"],
            "trace_sha256": profile["trace_sha256"], "task_signature": profile["task_signature"],
            "task_goal_excerpt": profile["task_goal_excerpt"],
            "outcome": profile["outcome"]["status"], "event_count": len(events),
            "selected_event_count": len(selected_events), "events": selected_events}


def deep_contrast(pair: dict[str, Any], positive: dict[str, Any], negative: dict[str, Any],
                  positive_events: list[dict[str, Any]], negative_events: list[dict[str, Any]]) -> dict[str, Any]:
    adapter = ADAPTERS[positive["benchmark"]]
    p = adapter.contrast_features(positive_events)
    n = adapter.contrast_features(negative_events)
    differences = list(pair["behavior_differences"])
    p_actions, n_actions = p["action_sequence"], n["action_sequence"]
    first_difference = next((index for index, (left, right) in enumerate(zip(p_actions, n_actions))
                             if left["action"] != right["action"]), None)
    if first_difference is not None:
        left, right = p_actions[first_difference], n_actions[first_difference]
        differences.append({"behavior": "action_sequence_diverged", "positive_event_ids": [left["event_id"]],
                            "negative_event_ids": [right["event_id"]], "interpretation": "hypothesis_only"})
    elif len(p_actions) != len(n_actions):
        differences.append({"behavior": "observed_action_sequence_length_differs",
                            "positive_event_ids": [p_actions[-1]["event_id"]] if p_actions else [],
                            "negative_event_ids": [n_actions[-1]["event_id"]] if n_actions else [],
                            "interpretation": "hypothesis_only"})
    if p.get("final_option") and n.get("final_option") and p["final_option"]["option"] != n["final_option"]["option"]:
        differences.append({"behavior": "different_final_options",
                            "positive_event_ids": [p["final_option"]["event_id"]],
                            "negative_event_ids": [n["final_option"]["event_id"]],
                            "interpretation": "hypothesis_only; does not establish mathematical root cause"})
    return {"contrast_id": pair["contrast_id"], "positive_trace_id": positive["trace_id"],
            "negative_trace_id": negative["trace_id"], "matching_grade": pair["matching_grade"],
            "shared_task_goal_terms": sorted(_tokens(positive["task_goal_excerpt"]) &
                                             _tokens(negative["task_goal_excerpt"]))[:20],
            "shared_constraints": pair["shared_constraints"],
            "positive_execution": p, "negative_execution": n,
            "common_behaviors": pair["common_behaviors"], "differences": differences,
            "causal_status": "hypothesis_only"}


def run(validated_path: Path, legacy_path: Path | None, output_dir: Path, *,
        max_deep_read_traces: int = 12, max_positive_traces: int = 6,
        max_negative_traces: int = 8, max_contrast_pairs: int = 4,
        max_llm_input_tokens: int = 30000, dry_run: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    validated = _read_jsonl(validated_path)
    if any(row.get("split") != "train" for row in validated):
        raise ValueError("M2 learning input must contain Train only")
    if len({row["trace_id"] for row in validated}) != len(validated):
        raise ValueError("duplicate trace_id in validated input")
    legacy = {}
    if legacy_path and legacy_path.is_file():
        legacy = {(row["benchmark"], row["task_id"]): row for row in _read_jsonl(legacy_path)}
    profiles = []
    events_cache: dict[str, list[dict[str, Any]]] = {}
    for row in validated:
        profile, events = profile_one(row, legacy.get((row["benchmark"], row["task_id"])))
        validate_profile(profile, events)
        profiles.append(profile)
        events_cache[profile["trace_id"]] = events
    l0_sec = time.perf_counter() - started
    contrasts, rejected = make_contrasts(profiles)
    positive, failure, contrast_groups = make_groups(profiles, contrasts, rejected)
    l1_sec = time.perf_counter() - started - l0_sec
    selected, selected_pairs = select_l2(profiles, contrasts, max_deep_read_traces, max_positive_traces,
                                         max_negative_traces, max_contrast_pairs, max_llm_input_tokens)
    by_id = {profile["trace_id"]: profile for profile in profiles}
    evidence_by_trace: dict[str, set[str]] = defaultdict(set)
    for pair in contrasts:
        if pair["contrast_id"] not in selected_pairs:
            continue
        comparison = deep_contrast(pair, by_id[pair["positive_trace_id"]], by_id[pair["negative_trace_id"]],
                                   events_cache[pair["positive_trace_id"]], events_cache[pair["negative_trace_id"]])
        for item in pair["common_behaviors"][:2]:
            evidence_by_trace[pair["positive_trace_id"]].update(item["positive_event_ids"][:2])
            evidence_by_trace[pair["negative_trace_id"]].update(item["negative_event_ids"][:2])
        for item in comparison["differences"]:
            evidence_by_trace[pair["positive_trace_id"]].update(item["positive_event_ids"])
            evidence_by_trace[pair["negative_trace_id"]].update(item["negative_event_ids"])
    views = [deep_view(by_id[trace_id], events_cache[trace_id], evidence_by_trace[trace_id]) for trace_id in selected]
    pair_by_id = {pair["contrast_id"]: pair for pair in contrasts}
    while selected and sum(len(json.dumps(view, ensure_ascii=False)) for view in views) > 4 * max_llm_input_tokens:
        selected.pop()
        selected_pairs = [pair_id for pair_id in selected_pairs
                          if pair_by_id[pair_id]["positive_trace_id"] in selected and
                          pair_by_id[pair_id]["negative_trace_id"] in selected]
        views = [deep_view(by_id[trace_id], events_cache[trace_id], evidence_by_trace[trace_id]) for trace_id in selected]
    deep_comparisons = [deep_contrast(pair_by_id[pair_id], by_id[pair_by_id[pair_id]["positive_trace_id"]],
                        by_id[pair_by_id[pair_id]["negative_trace_id"]],
                        events_cache[pair_by_id[pair_id]["positive_trace_id"]],
                        events_cache[pair_by_id[pair_id]["negative_trace_id"]]) for pair_id in selected_pairs]
    while selected and (sum(len(json.dumps(view, ensure_ascii=False)) for view in views) +
                        sum(len(json.dumps(item, ensure_ascii=False)) for item in deep_comparisons)) > 4 * max_llm_input_tokens:
        selected.pop()
        selected_pairs = [pair_id for pair_id in selected_pairs
                          if pair_by_id[pair_id]["positive_trace_id"] in selected and
                          pair_by_id[pair_id]["negative_trace_id"] in selected]
        views = [deep_view(by_id[trace_id], events_cache[trace_id], evidence_by_trace[trace_id]) for trace_id in selected]
        deep_comparisons = [deep_contrast(pair_by_id[pair_id], by_id[pair_by_id[pair_id]["positive_trace_id"]],
                            by_id[pair_by_id[pair_id]["negative_trace_id"]],
                            events_cache[pair_by_id[pair_id]["positive_trace_id"]],
                            events_cache[pair_by_id[pair_id]["negative_trace_id"]]) for pair_id in selected_pairs]
    l2_sec = time.perf_counter() - started - l0_sec - l1_sec
    full_chars = sum(int(row["metrics"].get("trace_chars") or 0) for row in validated)
    view_chars = sum(len(json.dumps(view, ensure_ascii=False)) for view in views)
    comparison_chars = sum(len(json.dumps(item, ensure_ascii=False)) for item in deep_comparisons)
    supported = {f"{profile['trace_id']}:{event_id}" for profile in profiles
                 if isinstance(profile["diagnosis"]["confirmed_root_cause"], dict)
                 for event_id in profile["diagnosis"]["confirmed_root_cause"]["event_ids"]}
    retained = {f"{view['trace_id']}:{event['event_id']}" for view in views for event in view["events"]}
    group_quality = audit_groups(profiles, positive, failure, set(selected))
    evidence_quality = audit_views(profiles, events_cache, views, deep_comparisons)
    quality_by_id = {item["group_id"]: item for item in group_quality["groups"]}
    for group in positive + failure:
        group["quality"] = quality_by_id[group["group_id"]]
    legacy_judged = [row for row in legacy.values() if row.get("judge_input_tokens") is not None]
    summary = {"schema_version": "2.0", "input_traces": len(validated),
               "eligible_success": sum(profile["eligible_for_learning"] and profile["outcome"]["status"] == "success" for profile in profiles),
               "eligible_failure": sum(profile["eligible_for_learning"] and profile["outcome"]["status"] == "failure" for profile in profiles),
               "success_behavior_profiles": sum(profile["eligible_for_learning"] and profile["outcome"]["status"] == "success"
                   and bool(observed_labels(profile)) for profile in profiles),
               "failure_symptom_profiles": sum(profile["eligible_for_learning"] and profile["outcome"]["status"] == "failure"
                   and bool(profile["diagnosis"]["observed_symptoms"]) for profile in profiles),
               "ineligible": [profile["trace_id"] for profile in profiles if not profile["eligible_for_learning"]],
               "positive_strategy_groups": len(positive), "failure_pattern_groups": len(failure),
               "group_quality": group_quality["summary"],
               "evidence_audit": evidence_quality["summary"],
               "contrast_pairs": len(contrasts), "contrast_grades": dict(Counter(item["matching_grade"] for item in contrasts)),
               "no_matched_positive": len(rejected), "deep_read_traces": len(views),
               "deep_read_success": sum(by_id[trace_id]["outcome"]["status"] == "success" for trace_id in selected),
               "deep_read_failure": sum(by_id[trace_id]["outcome"]["status"] == "failure" for trace_id in selected),
               "deep_read_contrast_pairs": len(selected_pairs),
               "supported_critical_events": len(supported), "l2_supported_event_retention":
                   len(supported & retained) / len(supported) if supported else None,
               "full_trace_chars": full_chars, "deep_view_chars": view_chars,
               "l2_contrast_chars": comparison_chars,
               "deep_view_token_proxy": math.ceil((view_chars + comparison_chars) / 4),
               "view_to_full_length_ratio": round((view_chars + comparison_chars) / full_chars, 4) if full_chars else None,
               "model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_analysis_tokens": 0,
               "legacy_judge_import": {"cases": len(legacy_judged),
                    "prompt_tokens": sum(row.get("judge_input_tokens", 0) for row in legacy_judged),
                    "completion_tokens": sum(row.get("judge_output_tokens", 0) for row in legacy_judged)},
               "estimated_max_llm_input_tokens": max_llm_input_tokens,
               "stage_seconds": {"l0": round(l0_sec, 3), "l1": round(l1_sec, 3), "l2": round(l2_sec, 3)},
               "dry_run": dry_run}
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "trace_profiles.jsonl", profiles)
        _write_json(output_dir / "positive_strategy_groups.json", positive)
        _write_json(output_dir / "failure_pattern_groups.json", failure)
        _write_json(output_dir / "group_quality.json", group_quality)
        _write_json(output_dir / "evidence_audit.json", evidence_quality)
        _write_json(output_dir / "contrast_groups.json", contrast_groups)
        _write_json(output_dir / "l2_contrasts.json", deep_comparisons)
        _write_json(output_dir / "m2_summary.json", summary)
        views_dir = output_dir / "deep_views"
        views_dir.mkdir(exist_ok=True)
        for old in views_dir.glob("*.json"):
            old.unlink()
        for index, view in enumerate(views):
            _write_json(views_dir / f"{index:03d}.json", view)
        legacy_views_dir = output_dir / "views"
        legacy_views_dir.mkdir(exist_ok=True)
        for old in legacy_views_dir.glob("*.json"):
            old.unlink()
        for view in views:
            legacy_view = {"trace_id": view["trace_id"], "benchmark": view["trace_id"].split(":", 1)[0],
                           "task_id": view["trace_id"].split(":", 1)[1],
                           "source_trace_path": view["source_trace_path"], "trace_sha256": view["trace_sha256"],
                           "events": [{"event_id": event["event_id"], "step_id": event["step_id"],
                                       "kind": event["kind"], "excerpt": event["content"],
                                       "full_chars": event["full_chars"], "is_critical": True}
                                      for event in view["events"]]}
            _write_json(legacy_views_dir / f"{view['trace_id'].replace(':', '_')}.json", legacy_view)
        # Keep the previous flat field contract for callers still reading M2 v1 files.
        compat_l0 = [{"trace_id": profile["trace_id"], "benchmark": profile["benchmark"],
                      "task_id": profile["task_id"], "task_family": profile["task_signature"]["task_family"],
                      "task_type": profile["outcome"]["metrics"].get("task_type"),
                      "complexity_bucket": profile["task_signature"]["task_complexity"],
                      "outcome": "task_failure" if profile["outcome"]["status"] == "failure" else profile["outcome"]["status"],
                      "reward": profile["outcome"]["reward"], "trace_validity": profile["trace_validity"],
                      "reliability_flags": profile["reliability_flags"],
                      "root_cause": profile["diagnosis"]["confirmed_root_cause"]["root_cause"]
                          if isinstance(profile["diagnosis"]["confirmed_root_cause"], dict) else "undetermined",
                      "supporting_events": profile["diagnosis"]["critical_events"],
                      "source_trace_path": profile["source_trace_path"], "trace_sha256": profile["trace_sha256"],
                      "trace_chars": int(profile["outcome"]["metrics"].get("trace_chars") or 0),
                      "estimated_full_trace_tokens": math.ceil(int(profile["outcome"]["metrics"].get("trace_chars") or 0) / 4)}
                     for profile in profiles]
        compat_groups = []
        for group in positive + failure:
            positive_group = group["group_id"].startswith("positive")
            compat_groups.append({"cluster_id": group["group_id"], "benchmark": group["benchmark"],
                                  "task_family": group["task_family"],
                                  "failure_pattern": "success_strategy:" + ",".join(group.get("behavior_pattern", []))
                                      if positive_group else ",".join(group["observed_symptoms"]),
                                  "case_count": group["case_count"], "case_ids": group["trace_ids"],
                                  "representative_traces": [trace_id for trace_id in group["trace_ids"] if trace_id in selected],
                                  "positive_examples": group["trace_ids"][:3] if positive_group else [],
                                  "critical_events": group.get("evidence_events", group.get("critical_events", [])),
                                  "summary": "Unified M2 profile group", "evidence_confidence":
                                      "medium" if not positive_group and group.get("confirmed_root_cause") != "undetermined" else "low"})
        group_for = {trace_id: group["cluster_id"] for group in compat_groups for trace_id in group["case_ids"]}
        _write_jsonl(output_dir / "l0.jsonl", compat_l0)
        _write_jsonl(output_dir / "groups.jsonl", compat_groups)
        _write_jsonl(output_dir / "selection.jsonl", [{"trace_id": trace_id, "cluster_id": group_for.get(trace_id)}
                                                   for trace_id in selected])
        legacy_loaded_chars = (output_dir / "l0.jsonl").stat().st_size + (output_dir / "groups.jsonl").stat().st_size
        legacy_loaded_chars += sum(path.stat().st_size for path in legacy_views_dir.glob("*.json"))
        legacy_loaded_chars += (output_dir / "l2_contrasts.json").stat().st_size
        _write_json(output_dir / "evaluation.json", {
            **summary, "tasks": len(profiles), "groups": len(compat_groups), "selected_l2": len(selected),
            "full_trace_context_token_proxy": math.ceil(full_chars / 4),
            "progressive_context_token_proxy": math.ceil(legacy_loaded_chars / 4),
            "estimated_context_reduction": round(1 - legacy_loaded_chars / full_chars, 4) if full_chars else None,
            "supported_root_cases": sum(isinstance(profile["diagnosis"]["confirmed_root_cause"], dict) for profile in profiles),
            "supported_critical_events": len(supported), "l2_critical_event_retention": summary["l2_supported_event_retention"]})
    return summary
