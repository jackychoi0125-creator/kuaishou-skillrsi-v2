"""Paired official-Harbor validation gate; never learns from Val content."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json

EXPECTED = {"spreadsheetbench": 40, "alfworld": 18, "livemath": 17}


def _read_run(job: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with (job / "summary.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            task_id = row["qid"]
            if not task_id or task_id in rows:
                raise ValueError(f"duplicate/missing qid in {job}")
            trial = next(job.glob(f"{task_id}__*/result.json"), None)
            if trial is None:
                raise ValueError(f"missing official result for {task_id}")
            result = json.loads(trial.read_text())
            score = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
            if score is None or abs(float(score) - float(row["reward"])) > 1e-6:
                raise ValueError(f"summary disagrees with official score: {task_id}")
            rows[task_id] = {**row, "official_reward": float(score), "result_path": str(trial)}
    return rows


def _truth(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def compare(benchmark: str, baseline_job: Path, candidate_job: Path, out_dir: Path,
            baseline_config: dict[str, Any], candidate_config: dict[str, Any]) -> dict[str, Any]:
    required = ("model_id", "api", "base_url", "reasoning", "seed", "kit_root", "bench", "kit_script_sha256")
    mismatch = [key for key in required if baseline_config.get(key) != candidate_config.get(key)]
    if mismatch:
        raise ValueError(f"Val configuration mismatch: {mismatch}")
    before, after = _read_run(baseline_job), _read_run(candidate_job)
    if len(before) != EXPECTED[benchmark] or len(after) != EXPECTED[benchmark] or set(before) != set(after):
        raise ValueError(f"Val alignment failed: baseline={len(before)}, candidate={len(after)}, missing={sorted(set(before)-set(after))}, extra={sorted(set(after)-set(before))}")
    diff = []
    transitions = {key: 0 for key in ("fail_to_pass", "pass_to_fail", "pass_to_pass", "fail_to_fail")}
    invalid = []
    for task_id in sorted(before):
        old, new = before[task_id], after[task_id]
        left = old["official_reward"] >= .999
        right = new["official_reward"] >= .999
        transition = f"{'pass' if left else 'fail'}_to_{'pass' if right else 'fail'}"
        transitions[transition] += 1
        if any(_truth(row[field]) for row in (old, new) for field in ("exception", "model_error")):
            invalid.append(task_id)
        diff.append({"task_id": task_id, "baseline_reward": old["official_reward"],
                     "candidate_reward": new["official_reward"], "transition": transition,
                     "baseline_input_tokens": old["input_tokens"], "candidate_input_tokens": new["input_tokens"],
                     "baseline_output_tokens": old["output_tokens"], "candidate_output_tokens": new["output_tokens"],
                     "baseline_duration_sec": old["duration_sec"], "candidate_duration_sec": new["duration_sec"],
                     "baseline_error": _truth(old["exception"]) or _truth(old["model_error"]),
                     "candidate_error": _truth(new["exception"]) or _truth(new["model_error"])})
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diff.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(diff[0]))
        writer.writeheader()
        writer.writerows(diff)
    gain = transitions["fail_to_pass"] - transitions["pass_to_fail"]
    decision = "evaluation_invalid" if invalid else "accept" if gain > 0 and transitions["pass_to_fail"] == 0 else "reject"
    result = {"benchmark": benchmark, "tasks": EXPECTED[benchmark], "baseline_pass": sum(x["official_reward"] >= .999 for x in before.values()),
              "candidate_pass": sum(x["official_reward"] >= .999 for x in after.values()),
              "baseline_success_rate": sum(x["official_reward"] >= .999 for x in before.values()) / EXPECTED[benchmark],
              "candidate_success_rate": sum(x["official_reward"] >= .999 for x in after.values()) / EXPECTED[benchmark],
              "success_rate_delta": gain / EXPECTED[benchmark],
              "fixes": transitions["fail_to_pass"], "regressions": transitions["pass_to_fail"],
              "transitions": transitions, "net_gain": gain, "invalid_task_ids": invalid,
              "baseline_model_errors": sum(_truth(row["model_error"]) for row in before.values()),
              "candidate_model_errors": sum(_truth(row["model_error"]) for row in after.values()),
              "baseline_execution_errors": sum(_truth(row["exception"]) for row in before.values()),
              "candidate_execution_errors": sum(_truth(row["exception"]) for row in after.values()),
              "decision": decision, "confidence": "low" if gain == 1 else "unquantified",
              "baseline_job": str(baseline_job), "candidate_job": str(candidate_job),
              "baseline_totals": _totals(before), "candidate_totals": _totals(after)}
    write_json(out_dir / "acceptance.json", result)
    return result


def _totals(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {name: sum(float(row.get(field) or 0) for row in rows.values())
            for name, field in (("input_tokens", "input_tokens"), ("cache_tokens", "cache_tokens"),
                                ("output_tokens", "output_tokens"), ("duration_sec", "duration_sec"))}
