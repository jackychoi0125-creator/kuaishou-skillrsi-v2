# R1 regression audit and Train ablation

## Val audit

Run `skillrsi-audit-r1` from the repository root after the official R1 paired Val jobs exist. It writes `results/r1/REGRESSION_AUDIT.{jsonl,csv,md}` and `results/r1/REGRESSION_AUDIT_SUMMARY.json`. These local files are ignored by Git because they contain Val task details.

The audit uses official rewards to identify changed tasks. It aligns trajectory milestones such as skill reads, workbook inspection or script execution, environment actions and feedback, and mathematical final-option events. Each milestone retains the run-local event ID. First divergence is an observed execution difference; it is not automatically a Skill effect. Explicit Skill discovery and `SKILL.md` reads are recorded separately from Skill metadata availability. Spreadsheet script diffs, ALFWorld action/feedback sequences, and LiveMath visible answer/check summaries support domain diagnosis. A single paired execution cannot separate sampling variation from Skill influence, so rule matches remain hypotheses.

The Val audit is read-only and has no import path into M3, M4, or Train ablation construction. Detailed Val traces, answers, and verifier notes must not be used as experience-generation inputs.

## Train A/B/C/D diagnosis

Run `skillrsi-ablate-train --benchmark all --n 3 --seed 0 --resume` to execute the same three Train task IDs for each benchmark under five arms. `--dry-run` validates preparation without a model call or score. Arms run sequentially through each benchmark's official Harbor Kit and current DeepSeek configuration:

- A: no Skill.
- B: valid neutral Skill with no added task procedure.
- C: frozen full R1 Skill, verified against its M4 SHA.
- D1 and D2: separate short rule fragments, each mapped to an existing Train Insight ID.

The controller writes stage manifests, official trial paths, `paired_train.csv`, and `summary.json` under ignored `results/train_ablation/<benchmark>/`. It checks that all arms contain the same task IDs, every ID belongs to Train, each result has a matching official score, and raw trajectories exist. It does not ingest Val audit data. This is a small diagnostic sample with one replay per arm; a score difference is not a causal estimate.

## Next-round rule governance

M3 records independent supporting traces, counterexamples, semantic support status, and risk level. Insights with contradictory trigger and avoid conditions remain separate and are marked high risk. M4 excludes high-risk insights, maps generated directives to Train Insight IDs in the candidate manifest, and rejects unmapped directives when several insights could be responsible. Static safety checks target answer leakage and secrets without rejecting ordinary words such as “search” or “secret” in benign contexts. The Gate adds trace, explicit Skill-read and cost diagnostics while preserving R1's official reward and zero-regression acceptance rule. Its stability field remains `unmeasured` for a single run.
