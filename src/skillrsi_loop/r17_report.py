"""Summarize R1.7 facts, hypotheses and unrun validation without score invention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .r17_generate import _usage
from .r17_plan import freeze
from .r17_source import audit


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def run(root: Path) -> dict[str, Any]:
    plan = freeze(root)
    out = root / "results/r17"
    funnels = {b: _read(out / b / "insight_funnel.json") for b in ("spreadsheetbench", "alfworld", "livemath")}
    breakdowns = {b: _read(out / b / "rejection_breakdown.json") for b in funnels}
    isolation = {b: audit(root, b, plan) for b in funnels}
    behavior = _read(out / "alfworld/behavior_cost_audit.json")
    trigger = _read(out / "skill_trigger_analysis.json")
    math = _read(out / "livemath/math_evidence_audit.json")
    candidates = _read(out / "candidate_hypotheses.json")
    usage = _usage(root)
    if usage["prompt_tokens"] + usage["completion_tokens"] > plan["policy"]["max_new_kimi_total_tokens"]:
        raise ValueError("Kimi budget exceeded")
    if any(value["status"] != "clean" for value in isolation.values()):
        raise ValueError("R1.7 Source isolation not clean")
    if candidates["accepted"] or candidates["new_harbor_tasks"]:
        raise ValueError("unexpected Skill acceptance or Harbor job")
    funnel_lines = ["# R1.7 Insight generation funnel", "",
                    f"Frozen plan SHA: `{plan['plan_sha256']}`. Old R1.6 cache replay is diagnostic only; "
                    "fresh R1.7 M3 uses rebuilt Source Train groups.", "",
                    "| Benchmark / phase | M2 groups | Selected | Input events | Raw Insights | Valid refs | "
                    "Evidence observations | Measurable | Static M4 eligible |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b, value in funnels.items():
        stages = value["stages"]
        funnel_lines.append(f"| {b} / cached replay | {stages['m2_input_groups']} | {stages['selector_groups']} | "
                            f"{stages['llm_input_events']} | {stages['llm_raw_insights']} | "
                            f"{stages['extractor_event_refs_valid']} | {stages['verified_observations']} | "
                            f"{stages['measurable_behavior_hypotheses']} | {stages['m4_cost_bounded_usable']} |")
        fresh = value.get("fresh_source_generation")
        if fresh:
            stages = fresh["stages"]
            funnel_lines.append(f"| {b} / fresh Source | {stages['m2_input_groups']} | {stages['selector_groups']} | "
                                f"{stages['llm_input_events']} | {stages['llm_raw_insights']} | "
                                f"{stages['extractor_event_refs_valid']} | {stages['evidence_verified_or_math_pending']} | "
                                f"{stages['measurable_behavior_hypotheses']} | {stages['m4_cost_bounded_usable']} |")
    funnel_lines += ["", "The cached response is already parsed JSON; original unparsed response text is unavailable. "
                    "The table's valid-ref count is an extractor check, not independent semantic or utility validation.",
                    "All fresh Source Insights have `semantic_validation=not_independently_checked`; utility remains unverified.", ""]
    for b, value in breakdowns.items():
        funnel_lines += [f"## {b}", "",
                         "Cached replay rejection reasons: " + json.dumps(value["reason_counts"], ensure_ascii=False) + "."]
        if "fresh_reason_counts" in value:
            funnel_lines.append("Fresh Source rejection reasons: " + json.dumps(value["fresh_reason_counts"], ensure_ascii=False) + ".")
        fresh = funnels[b].get("fresh_source_generation")
        if fresh:
            funnel_lines.append("Fresh selected group kinds: " + json.dumps(fresh["selected_kinds"], ensure_ascii=False) +
                                f"; truncated payload events: {fresh['payload_truncated_events']}.")
        funnel_lines.append("")
    funnel_lines += ["SpreadsheetBench R1.6 Kimi returned seven Insights. Its singular `event_id` refs were valid, "
                     "but the old parser required `event_ids`; six still have source-group mismatches after normalization. "
                     "The fresh call selected failure and contrast groups instead of six of eight positive groups. "
                     "Three new contrast candidates lacked both sides and were rejected.",
                     "LiveMath's R1.6 deterministic-only path made no LLM call. The fresh full-event view produced "
                     "four procedural hypotheses; none has independently verified mathematical content or utility.", ""]
    (out / "INSIGHT_FUNNEL_REPORT.md").write_text("\n".join(funnel_lines), encoding="utf-8")
    s = behavior["summary"]
    m4 = candidates["m4_manifests"]
    skill = m4["spreadsheetbench"]
    report = ["# Trace2Skill R1.7 run report", "",
              f"Frozen plan SHA: `{plan['plan_sha256']}`. New Harbor jobs: **0**. "
              f"New Kimi calls: **{usage['api_calls']}**; actual Prompt/Completion Tokens: "
              f"**{usage['prompt_tokens']}/{usage['completion_tokens']}** "
              f"(total {usage['prompt_tokens'] + usage['completion_tokens']}).", "",
              "## Observed facts", "",
              "- All three rebuilt Source M2 bundles passed Train-only isolation. Future Holdout task IDs and "
              "the already-used R1.6 Holdout are excluded from Source. Historical R1 used all Train; "
              "the future split is algorithmically isolated, not an untouched historical dataset.",
              "- SpreadsheetBench's old zero was primarily a schema mismatch (`event_id` versus `event_ids`) "
              "and then group-reference mismatch. Fresh Source M3 yielded five evidence-valid, measurable "
              "behavior hypotheses after per-Insight refinement. Inverted exclusion wording was moved into "
              "a separate prohibited-behavior field with provenance.",
              f"- LiveMath generated {math['procedural_hypotheses']} procedural hypotheses. "
              f"Full original cited events reread: {math['full_events_reread']}; independently verified local "
              f"mathematical claims: {math['mathematically_verified_local_claims']}. No reasoning root was confirmed.",
              f"- R1.6 ALFWorld old no-progress observation events C/D={s['C']['old_no_progress_observation_events']}/"
              f"{s['D']['old_no_progress_observation_events']}; distinct linked no-progress commands="
              f"{s['C']['unique_no_progress_commands']}/{s['D']['unique_no_progress_commands']}. "
              "Polls and duplicate observations were excluded from command counts.", "",
              "| ALFWorld historical arm | Official reward | Pi calls | Send calls | Official steps | "
              "Known no-progress commands | Trigger behavior | Input Tokens | Summed task seconds |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in "ABCD":
        row = s[arm]
        report.append(f"| {arm} | {row['successes']}/4 | {row['pi_tool_calls']} | "
                      f"{row['bridge_send_tool_calls']} | {row['official_executed_steps']} | "
                      f"{row['unique_no_progress_commands']} | {row['trigger_behavior_observed']}/{row['triggers']} | "
                      f"{row['input_tokens']} | {row['summed_task_duration_sec']} |")
    report += ["", "## Interpretation and remaining uncertainty", "",
               f"D matched B at 3/4, used {s['D']['official_executed_steps'] - s['B']['official_executed_steps']} "
               "more official environment steps and more input Tokens. The rule-compatible next action occurred "
               f"{s['D']['trigger_behavior_observed']}/{s['D']['triggers']} times in D, but also "
               f"{s['B']['trigger_behavior_observed']}/{s['B']['triggers']} in B without that rule. "
               "D explicitly read the Skill in two of four tasks. These one-repeat facts do not separate body, "
               "metadata and sampling effects causally. Each trigger and feedback has ATIF event IDs in the local audit.",
               f"M4 compiled one unaccepted {skill['skill_bytes']}-byte conditional SpreadsheetBench Skill "
               "from a direct Source Insight mapping, with at most one extra tool action and check. "
               "Its bound is predicted, not measured; semantic support and utility remain unverified. "
               "LiveMath has no M4 candidate, and the old ALFWorld D is barred from reuse as an independent candidate.",
               "No R2 Skill was accepted. The newly frozen Holdout was not run, and no Val or Test was run. "
               "The existing official Val gate and no-Skill best version remain unchanged.", ""]
    (out / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return {"plan_sha256": plan["plan_sha256"], "new_kimi_usage": usage, "new_harbor_tasks": 0,
            "hypotheses": len(candidates["records"]), "static_skill_status": skill["status"],
            "accepted": False, "val_run": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
