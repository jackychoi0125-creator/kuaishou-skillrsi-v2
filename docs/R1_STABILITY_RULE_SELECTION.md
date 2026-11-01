# R1 Train stability pilot and rule selection

## What is measured

`skillrsi-stability plan` freezes a Train-only task list, run order, budget, retry policy, and task-set hashes under ignored `results/stability/EXPERIMENT_PLAN.json`. `skillrsi-stability run --benchmark <name> --mode real --resume` executes official Harbor jobs in two independent repeats. Each repeat contains A (no Skill), B (neutral Skill), and D1 (one Train Insight-linked short rule). The order is A/B/D1 then D1/B/A. The second A is the A/A variability check. Dry Run creates no model calls or official scores.

The pilot uses four Train tasks per Benchmark. SpreadsheetBench and ALFWorld reproduce the Kit's pinned Train sample with `--n 4 --seed 0`; their Kits do not accept an ID list, so both independent repeats keep sample seed 0 and use distinct Run directories. LiveMath uses a frozen Train ID file with three historical Train successes and one historical failure; its repeats use sample seeds 0 and 1 while preserving the same ID set. A seed is a sampling parameter, not proof that model output is deterministic.

Each Run records the task-set hash, arm, repeat, Skill SHA, `run.sh` SHA, scoring-code SHA, model ID/API/base URL, reasoning mode, sample seed, concurrency, timeout, retry limit, official Run path, and configuration hash. Resume checks these fields and rereads every official reward and Trace. A changed Skill or config stops reuse. An interrupted run with a complete official summary can be recovered once; partial output requires audit.

Only an explicit infrastructure error can receive one independent retry. Ordinary reward 0 is a task outcome and is not retried. The initial attempt and every retry remain in the manifest. The first valid complete attempt is selected; no best-of selection is performed.

## Budgets and analysis

The frozen pilot budget caps task executions, model input and output tokens, wall time, per-Benchmark executions and input tokens, and per-arm executions. A block of A/B/D1 is admitted only if its pre-run estimate fits; if any block is incomplete, its scores are not used as a complete paired comparison. The pilot can stop before expansion when D1 input cost exceeds twice B without stable accuracy benefit.

`paired_repeats.csv` aligns A, B, and D1 by task ID and repeat. `stability_summary.json` reports A/A variation, per-task success frequencies, historical-success regressions, historical-failure fixes, P50/P90 cost, tool calls, explicit Skill access, errors, and paired B−A and D1−B differences. Task-cluster bootstrap intervals are descriptive; repeated runs of one task do not increase the number of independent tasks.

For ALFWorld, `submitted_commands` comes from the official action transcript, and `official_executed_steps/max_steps` comes from verifier metrics. Pi tool calls, bridge send/start/status calls, repeated commands, and observed no-progress/invalid-action feedback are separate. The verifier's executed-step count is not interpreted as the count of admissible actions; that count is unknown unless separately validated.

## Promotion and data boundary

`skillrsi-select-rules` reads only Train stability outputs and frozen R1 Train Insights. Its policy is frozen before pilot results in `RULE_SELECTION_POLICY.json`. A rule needs repeated improvement over B on at least two distinct tasks, acceptable Token/time cost, no observed paired regression, a clear Train Insight trigger, and clean Train internal validation. A/B presence effects and A/A natural variation remain separate from the D1/B comparison.

R1 M2 and M3 used all original Train traces. Replaying those Train tasks is therefore `in_sample_diagnostic_only`. `crossfit_audit` flags any proposed holdout already present in the original M2 profile/group path and fails closed when cache isolation is unverified. If no rule clears the policy and clean cross-fitting gate, the selector emits `no_candidate`, skips R2 generation and Val, and leaves the existing zero-regression Val Gate unchanged. Val trajectories and answers never enter this controller or selector.
