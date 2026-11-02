# Trace2Skill design

## Evidence flow

M1 checks ATIF structure, Trace identity, Reward and execution flags without interpreting strategy. M2 builds the same task and behavior schema for success and failure, then groups positive strategies, observable failure patterns and comparable contrasts. Groups retain Trace SHA and event IDs; compressed L2 views are reversible analysis aids. M3 uses ExpeL-style add/edit/agree/remove and verifies citations against the original Train event. An observation can be valid while the proposed behavior has no proven utility. M4 compiles a narrow conditional rule only when an Insight has a trigger, guard, observable effect and cost bound. Static checks reject broad, repeated or unbounded work.

## Selection and rollback

Train and Val have distinct roles. Train traces generate and internally test candidates. Official Harbor Val compares identical task IDs under matched configurations. A candidate becomes best only after the frozen gate accepts it; otherwise the best snapshot remains no Skill. R1 public Val was 31/40 versus 33/40 for SpreadsheetBench, 15/18 versus 16/18 for ALFWorld, and 4/17 versus 7/17 for LiveMath. All three were rejected. R1.6 and R1.7 strengthened Source/Holdout isolation, evidence quality and cost checks after observing negative transfer. Historical full-Train exposure means a later source-isolated Holdout is not an entirely untouched population.

## Portability and uncertainty

Public Benchmarks use domain adapters. A new Benchmark falls back to generic ATIF event and Reward handling. Its task family can come from Train metadata; behavior labels come from actual event IDs. Unsupported semantic claims remain hypotheses, and an unknown domain may safely end with `no_candidate`. The runner does not assume a hidden Kit accepts the same flags as a public Kit. Full generalization is unmeasured without the organizer's hidden evaluation.

## Token use

L0/L1 grouping is deterministic. L2 selects a small number of representative event windows. Kimi calls are cached and counted from service usage, not character estimates. A rule that adds large Token or action cost can be rejected even when a small sample shows a favorable score. Generation Token and Harbor evaluation Token are reported separately. No Val answers, private traces, endpoint keys or model caches are shipped in this package.
