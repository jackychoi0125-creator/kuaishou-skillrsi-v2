# R1.7: evidence funnel and bounded behavior hypotheses

R1.7 diagnoses the M3/M4 pipeline before further Harbor evaluation. The frozen plan in `results/r17/EXPERIMENT_PLAN.json` allows zero new Harbor tasks and at most 50,000 new Kimi Tokens across four calls. The old R1.6 Holdout is used only for read-only cost and trigger analysis. A new four-task Train Holdout per Benchmark is frozen before new Source-only generation; the already-used R1.6 Holdout and new Holdout are excluded from Source. Since historical R1 used all Train, this gives pipeline isolation, not a previously untouched population.

## Run order

From the repository root with `PYTHONPATH=src .venv/bin/python`:

1. `-m skillrsi_loop.r17_plan` freezes task sets and budgets.
2. `-m skillrsi_loop.r17_funnel --dry-run` and `-m skillrsi_loop.r17_funnel` replay old Kimi caches without model calls, recording every raw Insight's rejection stage and reason.
3. `-m skillrsi_loop.r17_behavior --dry-run` and `-m skillrsi_loop.r17_behavior` read official R1.6 ALFWorld Train runs, align commands with ATIF feedback, and remove repeated polls from environment-action counts.
4. `-m skillrsi_loop.r17_source --dry-run` and `-m skillrsi_loop.r17_source` build new Source-only M2 groups. `-m skillrsi_loop.r17_generate --mode dry-run` checks the new LLM payload and budget. `--mode real` makes one Source-only extraction call for SpreadsheetBench and LiveMath. The isolated cache enables idempotent recovery.
5. `-m skillrsi_loop.r17_refine --benchmark spreadsheetbench|livemath --mode dry-run` checks a compact follow-up request. `--mode real` asks for per-Insight behavior observations and numerical cost bounds. It does not change evidence references or claim utility.
6. `-m skillrsi_loop.r17_hypotheses --dry-run` and `-m skillrsi_loop.r17_hypotheses` normalize inverted exclusion wording, recheck original Trace SHA and event IDs, and compile at most one unaccepted conditional Skill per Benchmark. `-m skillrsi_loop.r17_math_audit` rereads full cited mathematical events. `-m skillrsi_loop.r17_funnel` refreshes the full funnel and `-m skillrsi_loop.r17_report` writes reports.

The generated artifacts, cache, prompts, Skill files and raw task evidence stay in ignored `results/r17/`. Neither Val nor Test details enter M3/M4. The no-Skill best version and official Val gate are unchanged.

## Findings

R1.6 SpreadsheetBench Kimi returned seven raw Insights, but the parser required `event_ids` while the response used `event_id`. The compatibility fix showed one group-supported candidate; six still had source-group mismatches. The fresh balanced Source run chose a failure group and two contrasts among four groups. It yielded five evidence-valid observations, and a second compact call made their expected behavior and one-action/one-check cost bounds measurable. The model had originally placed those fields at the response root, so they were not automatically assigned to every Insight. It also confused behavior to avoid with conditions where a rule should not run. Deterministic normalization records the original text as `prohibited_behavior` and uses conservative nonapplicability guards. The compiled 994-byte rule remains unaccepted and its cost is a prediction, not a measured effect.

LiveMath's old zero came from the deterministic-only setting. One new Source call produced four cited procedural hypotheses; three obtained measurable bounds in a compact follow-up. The verifier reread seven complete original events. No local mathematical claim passed an independent mathematical check. The arithmetic verifier only checks literal cited equalities with exact rational arithmetic; it cannot establish a full solution or reasoning root. All four remain hypotheses, and no math Skill was compiled.

The ALFWorld audit aligned official command transcripts with ATIF tool responses by command and environment step. C's 434 and D's 299 old no-progress *observation events* correspond to 62 and 35 distinct linked no-progress *commands*. D showed the rule-compatible next command after 22 of 35 triggers, but B without that rule showed it after 11 of 13. D tied B at 3/4, with 152 versus 84 official environment steps and 1,534,982 versus 1,408,852 input Tokens. This single-repeat association does not isolate Skill body, metadata or sampling effects. Detailed command, feedback and trigger event IDs are in the local audit.

New Kimi usage was four calls, 28,017 Prompt and 4,812 Completion Tokens, total 32,829. No new Harbor, Val or Test run was made; no R2 Skill was accepted. Run all 84 tests with `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q`.
