# Trace2Skill submission

This package runs a Train-only evidence pipeline and keeps **no Skill** as the best version until an official paired validation gate accepts a candidate. The public R1 candidates were rejected. It uses only Python standard-library dependencies at runtime.

## Install

Use Python 3.9 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m unittest discover -s tests -q
```

## Run on Train ATIF and Metrics

The ATIF JSONL must have one `case_id` and `trace.steps` per line. The CSV must have `task_id`, `split=train`, and `reward`; `task_type` is optional. Metrics and traces must have identical task IDs. Do not supply Val or Test traces to the generator.

```bash
.venv/bin/trace2skill evolve \
  --benchmark your-benchmark \
  --train-traces /path/to/train.atif.jsonl \
  --train-metrics /path/to/train-metrics.csv \
  --output-dir ./run-01 \
  --mode real --skip-val
```

For a no-call plan, use `--mode dry-run`. To resume an identical completed run without another model call, add `--resume`. A changed input or configuration hash is rejected. The CLI also accepts `--max-rounds`, `--max-generation-tokens`, and `--max-evaluation-tasks`. Only one round can run from a single supplied Train snapshot; subsequent rounds need a new accepted Skill's Train trace.

For a generation model, set `TRACE2SKILL_GENERATION_API_KEY`, `TRACE2SKILL_MODEL_ID`, and `TRACE2SKILL_GENERATION_URL` in the environment; the URL must be the full chat-completions endpoint. Alternatively, pass `--model-config /path/to/model-config.txt` with the key and model ID, while setting the URL in the environment. Keep credentials outside the package. Without a generation endpoint, M3 runs deterministic evidence checks and may return `no_candidate`; no synthetic model usage is reported. A Kit path can be supplied with `--kit-dir /path/to/kit`. Kit-specific invocation details can be set in a JSON configuration file with `val_bench`, `kit_run_args`, `generation_model_config`, `train_holdout`, `baseline_job`, `baseline_config`, and `allow_val`. Unknown Kit protocols require an adapter. The unified entry point will not run Val until a candidate passes an independent Train gate.

For an official Train gate, supply a Source-only Train JSONL and a pre-frozen Holdout in the configuration file. A compatible Kit example is:

```json
{
  "train_holdout": {
    "task_ids": ["frozen-train-id-1", "frozen-train-id-2"],
    "bench": "your-benchmark-train",
    "sampling_args": ["--n", "2", "--seed", "123"],
    "repeats": 2,
    "concurrent": 2
  },
  "allow_val": false
}
```

The controller checks that every official Run contains exactly the frozen IDs. It rejects a changed Skill SHA or Run result during resume. Set `allow_val` only after the Train gate is eligible. For Val, either provide a same-configuration official `baseline_job` and `baseline_config`, or provide `expected_val_tasks` so the CLI first runs a new no-Skill baseline. Budget for both baseline and candidate when creating a new one. The Kit must accept the specified flags; otherwise provide a compatible adapter or run configuration. This example has placeholder task IDs and does not claim an official result.

Inspect `run-01/m1/summary.json`, `run-01/m2/m2_summary.json`, `run-01/m3/usage.json`, `run-01/m4/candidate_manifest.json`, `run-01/best_manifest.json`, and `run-01/run_manifest.json`. A candidate is never the best version merely because it was generated. Raw private traces and model cache stay under the local output directory.

To check this package:

```bash
.venv/bin/python -m trace2skill.validate_submission --path .
```

The public historical official Harbor Val results and generation-token accounting are summarized in `results/`. The package contains no accepted Skill because every tested public candidate was rejected. The [organizer's current rules page](https://bs3-hb1.corp.kuaishou.com/infra-wanqing-llm-gateway/evolve-inputs/skill-creator-match/RULES.html) was fetched on 2026-11-02 and its listed package directories match this package. It does not explicitly state whether an empty accepted-Skill directory is permitted; the validator records that as an external requirement to confirm.
