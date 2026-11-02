"""Conservative R1.8 run and submission readiness reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(root: Path) -> dict:
    out = root / "results/r18"
    plan = json.loads((out / "EXPERIMENT_PLAN.json").read_text())
    val_plan = json.loads((out / "VAL_PLAN.json").read_text()) if (out / "VAL_PLAN.json").is_file() else None
    analysis_path = out / "spreadsheetbench/holdout_analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else None
    preflight_path = out / "spreadsheetbench/preflight.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else None
    validation_path = root / "submission_validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.is_file() else None
    generation_path = root / "submission/results/generation_tokens.json"
    generation = json.loads(generation_path.read_text()) if generation_path.is_file() else None
    harbor_path = root / "submission/results/harbor_tokens.json"
    harbor = json.loads(harbor_path.read_text()) if harbor_path.is_file() else None
    best_path = out / "BEST_VERSION.json"
    best = json.loads(best_path.read_text()) if best_path.is_file() else None
    status = analysis["status"] if analysis else "not_run"
    lines = ["# R1.8 run report", "", f"Frozen Train plan SHA: `{plan['plan_sha256']}`; "
             f"candidate Skill SHA: `{plan['candidate_skill_sha256']}`; "
             f"Source/Holdout hashes: `{plan['source_task_ids_sha256']}` / `{plan['holdout_task_ids_sha256']}`.",
             f"Frozen conditional Val plan SHA: `{val_plan['plan_sha256'] if val_plan else 'missing'}`.",
             "R1.7's original zero-Harbor plan was not changed. R1.8 uses its preselected Train Holdout only.", "",
             f"Candidate preflight: `{preflight['source_audit']['status'] if preflight else 'missing'}`; "
             f"static safety errors: {len(preflight['static_safety_errors']) if preflight else 'unknown'}.", "",
             "## SpreadsheetBench Train Holdout", ""]
    if analysis:
        lines += [f"- Official paired status: `{status}`; complete repeats: {len(analysis['complete_repeats'])}/2.",
                  f"- Harbor executions: {analysis['usage']['task_executions']}; input/output Tokens: "
                  f"{analysis['usage']['input_tokens']}/{analysis['usage']['output_tokens']}.",
                  f"- Arm passes across complete repeats: {analysis['arm_passes']}.",
                  f"- D−A, D−B, D−C mean paired Reward differences: "
                  f"{analysis['mean_reward_delta_d_minus_a']}, {analysis['mean_reward_delta_d_minus_b']}, "
                  f"{analysis['mean_reward_delta_d_minus_c']}.",
                  f"- Distinct D improvements versus both B/C: {len(analysis['distinct_improved_vs_b_and_c'])}; "
                  f"distinct regressions versus any control: {len(analysis['distinct_regressed_vs_any_control'])}.",
                  f"- Val eligibility: {analysis['eligible_for_val']}. Stop reason: {analysis.get('stopped_reason') or 'none'}."]
        observed = analysis.get("observed_attempts_not_all_paired") or {}
        lines += [f"- Across all scored attempts, explicit Skill reads: "
                  f"{sum(item['explicit_read_count'] for item in observed.values())}; "
                  f"discovery evidence: {sum(item['discovery_evidence_count'] for item in observed.values())}. "
                  "A configured Skill can still affect the Agent without an explicit body read."]
        triggers = analysis.get("all_observed_candidate_trigger_checks") or []
        lines += [f"- Candidate D rule trigger observed in {sum(x.get('trigger_status') == 'observed' for x in triggers)}"
                  f"/{len(triggers)} scored executions; behavior consistent in "
                  f"{sum(bool(x.get('behavior_consistent')) for x in triggers)}. "
                  "These event observations do not establish Reward causality."]
        if analysis["usage"]["input_tokens"] > plan["policy"]["max_input_tokens"]:
            lines += ["- Budget audit: actual input Tokens exceeded the frozen maximum within one indivisible "
                      "Harbor Job; no later Job was authorized. This is a controller limitation, not a candidate benefit."]
        val_path = out / "spreadsheetbench/val/acceptance.json"
        if val_path.is_file():
            val = json.loads(val_path.read_text())
            lines += [f"- Official Val baseline/candidate: {val['baseline_pass']}/{val['tasks']} versus "
                      f"{val['candidate_pass']}/{val['tasks']}; fixes {val['fixes']}, regressions "
                      f"{val['regressions']}; decision `{val['decision']}`."]
        else:
            lines += ["- New official Val: not run; no R1.8 Val score is claimed."]
        if status == "incomplete" and analysis.get("observed_attempts_not_all_paired"):
            lines += ["", "Individual official attempts (descriptive only; incomplete repeats cannot qualify):", "",
                      "| Attempt | Official passes | Scored tasks | Input Tokens | Output Tokens | Pi calls |",
                      "|---|---:|---:|---:|---:|---:|"]
            for name, item in analysis["observed_attempts_not_all_paired"].items():
                lines.append(f"| {name} | {item['official_passes']} | {item['official_scored_tasks']} | "
                             f"{item['input_tokens']} | {item['output_tokens']} | {item['pi_tool_calls']} |")
    else:
        lines += ["Official Holdout pairing has not completed; no Train benefit or Val eligibility is claimed."]
    lines += ["", "The four frozen Arms are A=no Skill, B=no-operation Skill, C=metadata-matched no-operation Skill, "
              "and D=the exact R1.7 single-rule candidate. All rewards come from Harbor verifier_result.rewards.reward. "
              "Historical R1 used all Train tasks, so R1.8 establishes Source-pipeline isolation rather than a never-seen population.",
              "", "## Other Benchmarks and Best state", "",
              "ALFWorld had no qualified new R1.7 candidate: historical D tied the no-operation B at 3/4 with higher "
              "Token and action cost. LiveMath had four Source-cited procedural hypotheses but no independently "
              "verified mathematical conclusion or low-risk candidate. Their accepted best remains no Skill. "
              "The public R1 candidates were rejected by paired official Val. No Val task details enter M3/M4.",
              f"Final Best states: { {b: best[b]['version'] for b in ('spreadsheetbench','alfworld','livemath')} if best else 'pending Holdout outcome'}.",
              "", "## Real execution accounting", "",
              f"R1.8 generation Kimi calls/Tokens: 0/0. R1.8 new Harbor tasks: "
              f"{analysis['usage']['task_executions'] if analysis else 0}. "
              "Generation usage from earlier rounds is reported separately in submission/results/generation_tokens.json.",
              ""]
    if generation:
        parts = [generation["historical_exploration"]["r1"],
                 *generation["historical_exploration"]["r16"].values(),
                 *generation["r17_incremental"].values()]
        lines += [f"Known R1/R1.6/R1.7 Kimi service receipts: {sum(x['api_calls'] for x in parts)} calls; "
                  f"Prompt {sum(x['prompt_tokens'] for x in parts)}, Completion "
                  f"{sum(x['completion_tokens'] for x in parts)} Tokens. "
                  "This excludes any unrecorded earlier exploration.", ""]
        legacy = generation.get("legacy_m1_judge_reported_not_r18_incremental")
        if legacy:
            lines += [f"Older M1 Judge records separately report {legacy['prompt_tokens']} Prompt and "
                      f"{legacy['completion_tokens']} Completion Tokens across {legacy['reused_cases']} reused cases; "
                      "these lack the same receipt provenance and are not added to M3/M4 service totals.", ""]
    if harbor:
        total = harbor["all_known_manifest_jobs"]
        lines += [f"Completed official Jobs referenced by local manifests: {total['jobs']} Jobs, "
                  f"{total['task_executions']} task executions, {total['input_tokens']} input and "
                  f"{total['output_tokens']} output Tokens. This scope is not all possible historical runs.", ""]
    (out / "RUN_REPORT.md").write_text("\n".join(lines))
    readiness = ["# Submission readiness", "", f"Package validator: `{validation['status'] if validation else 'not_run'}`.",
                 "No rejected R1/R1.6/R1.7 candidate is included as a Best Skill.",
                 "The package can parse and profile a synthetic unknown Benchmark and safely return no_candidate.",
                 "The current official rules page was fetched on 2026-11-02; it lists src, skills, README, DESIGN and results.",
                 "Whether the organizer permits an empty accepted Skill directory is unverified.",
                 "The generic entry point requires a Kit adapter or compatible run.sh protocol for an unknown Kit, "
                 "and an independently verified Train gate before any official Val.", ""]
    (out / "SUBMISSION_READINESS.md").write_text("\n".join(readiness))
    return {"holdout_status": status, "submission_status": validation["status"] if validation else "not_run"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve())))


if __name__ == "__main__":
    main()
