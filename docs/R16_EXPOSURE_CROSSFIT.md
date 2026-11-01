# R1.6: Skill exposure and clean Train holdout

R1.6 separates four questions: whether a Skill was configured, whether its metadata was visibly advertised, whether the agent explicitly read its body, and whether subsequent behavior was compatible with its rule. Only the first and third are normally observable in the retained Harbor ATIF. The Kit mounts `--skills-dir` at `/skills` and registers the Skill with Pi, but the full Pi system prompt and Skill list are absent from ATIF. An absent `cat SKILL.md` event therefore has no causal interpretation.

## Frozen data and policy

`python -m skillrsi_loop.r16_plan` freezes a Train-only Source/Holdout partition and the cost and promotion policy in `results/r16/EXPERIMENT_PLAN.json`. The file is checked on every subsequent run; any changed partition or policy fails closed. SpreadsheetBench, ALFWorld and LiveMath each reserve four Holdout tasks; LiveMath reserves one historical success and three historical failures. Old R1 Skills and Insights used all Train tasks and are excluded from the clean validation bundle.

`python -m skillrsi_loop.crossfit build-m2 --benchmark all` rebuilds M2 from Source Train M1 records. `python -m skillrsi_loop.r16_generate --benchmark all --mode real` calls Kimi only for Source-only M3 groups when a budget admission check passes, and compiles at most one conditional rule per Benchmark. The M3 cache is separate for each Benchmark and is required to be initially empty. The crossfit audit checks task IDs in profiles, groups, contrasts, views, M3 references, cache and M4 prompts; it checks original Trace SHA-256 and event IDs. A missing or contaminated artifact prevents Holdout execution.

The resulting Insight separates `evidence_validity` from `utility_status`. A verified observation is still `unverified` for utility until repeated controlled feedback supports it. The M4 compiler maps one rule directly to one structured Insight and its source Trace IDs. This mapping is a provenance link, not proof of benefit.

## Experimental controls

The official Harbor Train Holdout comparison uses four arms on identical frozen task IDs: A has no Skill; B is a neutral valid Skill; C has D's front matter and description with a neutral body; D has one conditional rule. Arms are executed in a frozen order, then reverse order on a second repeat. Each run retains its official reward, full Run path, Skill SHA, Kit and scorer SHA, model configuration, cost and exposure events. Task errors are retained; task failures are never retried until they pass.

`python -m skillrsi_loop.holdout --benchmark alfworld --mode dry-run` checks the plan without model calls. Real execution requires a Source-only candidate and a clean audit. The controller stops before a repeat block that cannot fit the frozen budget. `python -m skillrsi_loop.holdout_analysis --benchmark alfworld` reports A/B presence and D/B rule signals by independent task, including a descriptive task-cluster bootstrap interval. A single repeat never qualifies a rule for Val. The existing official R1 Val gate and no-Skill best version are unchanged.

`python -m skillrsi_loop.exposure` audits retained R1 Val and Train runs read only. Val details never enter Source M2, M3 or M4. `python -m skillrsi_loop.r16_report` writes the local report and candidate decisions. Generated plans, insights, raw runs, caches, prompts and audit results remain under ignored `results/r16/`; commit code, tests and this document only.

## Tests

Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q`. Tests require no API key and cover split isolation, group and Insight leaks, strict event and SHA references, no-op and metadata-matched controls, ceiling diagnostics, task-level repeated effects, one-repeat non-promotion, and no-candidate Harbor skipping.

## First R1.6 execution

The frozen Source-only M2 generated 76 SpreadsheetBench, 35 ALFWorld and 31 LiveMath profiles. M3 used two real Kimi calls (26,125 prompt and 3,068 completion Tokens). SpreadsheetBench and LiveMath produced no responsible single-rule candidate. ALFWorld produced one unaccepted source-backed rule.

The official ALFWorld Train Holdout ran A/B/C/D once on the same four tasks: 4/4, 3/4, 1/4 and 3/4. Input Tokens were 736,071 / 1,408,852 / 2,198,547 / 1,534,982. The 16 executions used 5,878,452 input and 437,526 output Tokens. The controller stopped before the second complete repeat block under the frozen 7,000,000 input Token budget. D tied B in total success and cost about 1.98 times B in summed per-task duration; one task improved while another regressed. This single repeat does not establish stable rule utility. No R2 candidate was accepted and no new Val was run. Raw runs, task IDs and detailed audit evidence remain in ignored `results/r16/`.
