# R1.8 validation and portable submission

The historical R1 CLI binds three local Kit paths, a fixed test endpoint and existing repository results. R1.8 leaves that reproducibility path intact. `skillrsi_loop.r18_validation` is the frozen official SpreadsheetBench study; `trace2skill.pipeline` is the portable entry point shipped to reviewers.

## Source and validation boundary

The R1.7 SpreadsheetBench candidate SHA is frozen in `results/r18/EXPERIMENT_PLAN.json`. The R1.7 Source provenance audit verifies the M2 group, M3 Insight event and SHA references, generation cache, M4 prompt and Skill against the previously frozen future Holdout IDs. Static M4 checks run again before the official study. R1 historically saw all Train, so this establishes isolation of the **R1.7 generation pipeline**, not a never-observed historical population.

The Train study runs A=no Skill, B=no-operation Skill, C=metadata-matched no-operation Skill and D=the exact R1.7 Skill. Four fixed Train tasks and two full repeats are required. Every official `result.json` Reward must agree with Harbor `summary.csv`, and every arm must contain the same task IDs. The plan freezes Skill/Kit/scorer hashes, endpoint identity, task sampling seed, order, concurrency, timeout, zero infrastructure retries and Token/task/wall budgets. It requires D to beat all controls in both repeats, improve two distinct tasks versus B/C, show no distinct regressions, and keep input Token and task time within 1.5 times B. Partial results never qualify for Val. A separate full-Val budget and same-configuration baseline were frozen before Train results.

## Portable call chain

`trace2skill evolve` loads Train ATIF and Metrics, runs deterministic M1 reliability checks, then M2 profiles and evidence groups. M3 uses verified Train event references and optional Kimi; M4 generates a candidate only from qualified Insight. A configured official Train gate may then run; only an eligible candidate can reach Harbor Val. Val performs paired official Reward comparison and accepts only positive net gain with zero regressions. A missing candidate or failed gate leaves `best_manifest.json` at no Skill. For an unknown Benchmark, the generic adapter preserves task metadata and ATIF behavior events; if it cannot establish safe utility, it returns `no_candidate`.

`trace2skill.submission` copies portable code and aggregate results, but no raw Trace, private answer, endpoint key or model cache. `trace2skill.validate_submission` checks structure, accepted Skill SHA, results schemas and sensitive-content patterns. The current [official rules](https://bs3-hb1.corp.kuaishou.com/infra-wanqing-llm-gateway/evolve-inputs/skill-creator-match/RULES.html) were fetched on 2026-11-02. They list the five required package directories but do not explicitly address an empty accepted-Skill directory; the validator retains that external uncertainty.
