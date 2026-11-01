"""Train-only A/B/C/D skill-presence and rule-fragment diagnostic replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from skillrsi_m1.core import BENCHMARKS, events_from_trace
from skillrsi_m3.core import write_json
from .cli import CONCURRENT, KIT_REL, config
from .triage import _skill_reads

ARMS = ("A", "B", "C", "D1", "D2")
FRAGMENTS = {
    "spreadsheetbench": [
        {"keyword": "inspect", "trigger": "The task requires changing cells in a workbook with nontrivial structure.",
         "procedure": "Inspect only the relevant sheets, ranges, existing formulas and target cells before editing.",
         "avoid": "Do not dump unrelated sheets or add broad exploration for a simple cell change.",
         "validation": "The selected source and target ranges match the task's visible requirements."},
        {"keyword": "verif", "trigger": "The task requires formula or data transformation validation.",
         "procedure": "After editing, check the saved output against visible examples or invariants; use a calculation engine only if available.",
         "avoid": "Do not assume hidden expected outputs or require an unavailable engine.",
         "validation": "The output file exists and the relevant visible values or formulas were checked."},
    ],
    "alfworld": [
        {"keyword": "admiss", "trigger": "An environment action returns no progress or is rejected.",
         "procedure": "Read current admissible commands and choose an action listed for the current state.",
         "avoid": "Do not infer command syntax from memory when the environment lists valid actions.",
         "validation": "The next action is permitted by the current observation."},
        {"keyword": "repeat", "trigger": "The previous action returned no state change.",
         "procedure": "Before retrying, check whether any relevant state changed; otherwise choose a different admissible action.",
         "avoid": "Do not repeat the identical command after the same no-progress feedback.",
         "validation": "A repeated command is justified by an intervening observed state change."},
    ],
    "livemath": [
        {"keyword": "definition", "trigger": "The problem explicitly defines a local symbol or relation.",
         "procedure": "Locate and restate the local definition before using a familiar result.",
         "avoid": "Do not assume the symbol has its conventional meaning.",
         "validation": "The working meaning matches the statement's local definition."},
        {"keyword": "reasoning", "trigger": "A later reasoning step uses a locally defined symbol.",
         "procedure": "Check that this use follows the problem's stated meaning before giving the final answer.",
         "avoid": "Do not add unrelated checks when the problem has no local definition.",
         "validation": "Each relevant use is consistent with the local definition."},
    ],
}


def _skill(name: str, description: str, trigger: str, procedure: str, avoid: str, validation: str) -> str:
    return (f"---\nname: {name}\ndescription: {description}\n---\n\n"
            f"# When to use\n\n- {trigger}\n\n# Procedure\n\n- {procedure}\n\n"
            f"# Avoid\n\n- {avoid}\n\n# Validation\n\n- {validation}\n")


def prepare_arms(root: Path, benchmark: str, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Only existing Train-derived Insight/Skill artifacts are used here."""
    r1 = root / "results/r1" / benchmark
    manifest = json.loads((r1 / "m4/candidate_manifest.json").read_text())
    full = Path(manifest["skill_path"])
    raw = full.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("R1 Skill SHA does not match M4 manifest")
    pool = [json.loads(line) for line in (r1 / "m3/insights.jsonl").read_text().splitlines() if line.strip()]
    used = {item["insight_id"]: item for item in pool if item["insight_id"] in manifest["insight_ids"]}
    if not used or any(item["status"] != "verified_observation" for item in used.values()):
        raise ValueError("Ablation requires verified Train insights")
    result = {"A": {"skills_dir": None, "skill_sha256": None, "insight_ids": []}}
    neutral = _skill(f"{benchmark}-presence-only", "Neutral placeholder with no additional task procedure.",
                     "This placeholder may be present in the skill list.",
                     "Follow the task's normal instructions.",
                     "No extra action is required by this placeholder.",
                     "Use the normal task completion check.")
    content = {"B": neutral, "C": raw.decode("utf-8")}
    mapping = {}
    for index, spec in enumerate(FRAGMENTS[benchmark], 1):
        ranked = sorted(used.values(), key=lambda item: (spec["keyword"] in
                        (item["principle"] + " " + item["recommended_behavior"]).lower(),
                        item["insight_id"]), reverse=True)
        if spec["keyword"] not in (ranked[0]["principle"] + " " + ranked[0]["recommended_behavior"]).lower():
            raise ValueError(f"No Train Insight supports fragment keyword {spec['keyword']}")
        mapped = ranked[0]
        arm = f"D{index}"
        content[arm] = _skill(f"{benchmark}-fragment-{index}",
                              f"Train-supported R1 rule fragment {index} for {benchmark}.",
                              spec["trigger"], spec["procedure"], spec["avoid"], spec["validation"])
        mapping[arm] = {"insight_ids": [mapped["insight_id"]],
                        "source_rule_keyword": spec["keyword"], "causal_status": mapped["causal_status"]}
    for arm, text in content.items():
        skill_dir = out_dir / "skills" / arm
        target = skill_dir / f"{benchmark}-{arm.lower()}" / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        result[arm] = {"skills_dir": str(skill_dir), "skill_path": str(target),
                       "skill_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                       "insight_ids": mapping.get(arm, {}).get("insight_ids", list(used) if arm == "C" else []),
                       "source_rule_keyword": mapping.get(arm, {}).get("source_rule_keyword")}
    write_json(out_dir / "arms_manifest.json", {"benchmark": benchmark, "arms": result,
                                             "data_policy": "Train only; Val audit excluded"})
    return result


def _job(root: Path) -> Path | None:
    summaries = sorted(root.glob("*/summary.csv")) if root.is_dir() else []
    return summaries[-1].parent if summaries else None


def _healthy(job: Path) -> bool:
    try:
        with (job / "summary.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        return bool(rows) and all(row.get("model_error", "").lower() != "true" and
                                  row.get("exception", "").lower() != "true" for row in rows)
    except (OSError, ValueError):
        return False


def _run_harbor(kit: Path, benchmark: str, jobs_root: Path, skills_dir: Path | None,
                n: int, seed: int) -> Path:
    for attempt in range(3):
        candidate = jobs_root if attempt == 0 else jobs_root.with_name(jobs_root.name + f"-retry{attempt}")
        existing = _job(candidate)
        if existing and _healthy(existing):
            return existing
        if existing:
            continue
        try:
            return _run_once(kit, benchmark, candidate, skills_dir, n, seed)
        except (RuntimeError, TimeoutError):
            if attempt == 2:
                raise
    raise RuntimeError(f"No valid official Train run for {benchmark} at {jobs_root}")


def _run_once(kit: Path, benchmark: str, jobs_root: Path, skills_dir: Path | None,
              n: int, seed: int) -> Path:
    timeout_seconds = 1800 if benchmark == "alfworld" else 900
    jobs_root.parent.mkdir(parents=True, exist_ok=True)
    if jobs_root.is_dir() and any(jobs_root.iterdir()):
        raise ValueError(f"incomplete populated jobs directory requires inspection: {jobs_root}")
    args = [str(kit / "run.sh"), "--bench", benchmark + "-train", "--n", str(n),
            "--seed", str(seed), "--concurrent", str(CONCURRENT[benchmark]),
            "--reasoning", "off", "--jobs-dir", str(jobs_root)]
    if skills_dir:
        args += ["--skills-dir", str(skills_dir)]
    log = jobs_root.with_name(jobs_root.name + "-controller.log")
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(args, cwd=kit, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"Train ablation timed out after {timeout_seconds} seconds; see {log}") from exc
    if code:
        raise RuntimeError(f"Train ablation {benchmark} failed, exit {code}; see {log}")
    job = _job(jobs_root)
    if not job or not _healthy(job):
        raise RuntimeError(f"Train ablation has missing scores or model/execution errors: {log}")
    return job


def _rows(job: Path) -> dict[str, dict[str, Any]]:
    with (job / "summary.csv").open(newline="", encoding="utf-8") as source:
        rows = {row["qid"]: row for row in csv.DictReader(source)}
    if len(rows) != sum(1 for _ in (job / "summary.csv").open()) - 1:
        raise ValueError("duplicate task in ablation summary")
    for task_id, row in rows.items():
        trial = next(job.glob(f"{task_id}__*/result.json"), None)
        if not trial:
            raise ValueError(f"missing official result for Train task {task_id}")
        official = json.loads(trial.read_text())
        reward = (official.get("verifier_result") or {}).get("rewards", {}).get("reward")
        if reward is None or float(reward) != float(row["reward"]):
            raise ValueError(f"invalid or inconsistent Train score for {task_id}")
        trace = json.loads((trial.parent / "agent/trajectory.json").read_text())
        events = events_from_trace(trace)
        row["skill_reads"] = len(_skill_reads(events))
        row["tool_calls"] = sum(event["kind"] == "tool_call" for event in events)
        row["trial_path"] = str(trial.parent)
    return rows


def summarize(root: Path, benchmark: str, out_dir: Path, jobs: dict[str, Path], n: int,
              metrics_dir: Path, seed: int = 0) -> dict[str, Any]:
    all_rows = {arm: _rows(path) for arm, path in jobs.items()}
    ids = set(all_rows["A"])
    if len(ids) != n or any(set(rows) != ids for rows in all_rows.values()):
        raise ValueError("Train ablation arms do not share identical task IDs")
    with (metrics_dir / f"{benchmark}-baseline.csv").open(newline="", encoding="utf-8") as source:
        train_ids = {row["task_id"] for row in csv.DictReader(source) if row["split"] == "train"}
    if not ids.issubset(train_ids):
        raise ValueError(f"non-Train task leaked into ablation: {ids - train_ids}")
    comparison = []
    for task_id in sorted(ids):
        baseline = all_rows["A"][task_id]
        for arm in ARMS:
            row = all_rows[arm][task_id]
            comparison.append({"benchmark": benchmark, "task_id": task_id, "arm": arm,
                               "reward": float(row["reward"]), "baseline_reward": float(baseline["reward"]),
                               "input_tokens": int(row["input_tokens"]),
                               "output_tokens": int(row["output_tokens"]),
                               "cache_tokens": int(row.get("cache_tokens") or 0),
                               "duration_sec": float(row.get("duration_sec") or 0),
                               "skill_reads": row["skill_reads"],
                               "tool_calls": row["tool_calls"],
                               "model_error": row.get("model_error") == "True",
                               "execution_error": row.get("exception") == "True",
                               "trial_path": row["trial_path"]})
    with (out_dir / "paired_train.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    arms = {}
    base = all_rows["A"]
    for arm, rows in all_rows.items():
        fixes = sum(float(base[task]["reward"]) < .999 and float(rows[task]["reward"]) >= .999 for task in ids)
        regressions = sum(float(base[task]["reward"]) >= .999 and float(rows[task]["reward"]) < .999 for task in ids)
        arms[arm] = {"passed": sum(float(row["reward"]) >= .999 for row in rows.values()),
                     "fixes_vs_A": fixes, "regressions_vs_A": regressions,
                     "input_tokens": sum(int(row["input_tokens"]) for row in rows.values()),
                     "output_tokens": sum(int(row["output_tokens"]) for row in rows.values()),
                     "duration_sec_sum": sum(float(row.get("duration_sec") or 0) for row in rows.values()),
                     "explicit_skill_reads": sum(row["skill_reads"] > 0 for row in rows.values()),
                     "tool_calls": sum(row["tool_calls"] for row in rows.values()),
                     "model_errors": sum(row.get("model_error") == "True" for row in rows.values()),
                     "execution_errors": sum(row.get("exception") == "True" for row in rows.values()),
                     "job": str(jobs[arm])}
    result = {"benchmark": benchmark, "split": "train", "seed": seed, "task_ids": sorted(ids),
              "n": n, "arms": arms,
              "interpretation_limit": "small Train diagnostic subset; single replay per arm cannot establish causality"}
    write_json(out_dir / "summary.json", result)
    return result


def run_one(root: Path, benchmark: str, *, n: int = 3, seed: int = 0,
            dry_run: bool = False, resume: bool = True, stop_after: str | None = None) -> dict[str, Any]:
    out = root / "results/train_ablation" / benchmark
    out.mkdir(parents=True, exist_ok=True)
    arms = prepare_arms(root, benchmark, out)
    state_path = out / "stage_manifest.json"
    state = json.loads(state_path.read_text()) if resume and state_path.is_file() else {"benchmark": benchmark, "n": n, "seed": seed, "jobs": {}}
    if state["n"] != n or state["seed"] != seed:
        raise ValueError("resume configuration differs from prior Train ablation")
    if dry_run:
        return {"benchmark": benchmark, "mode": "dry_run", "arms": list(arms),
                "model_calls": 0, "official_scores": None}
    kit = (root.parent / "kits" / KIT_REL[benchmark]).resolve()
    config(kit, benchmark)  # same verified DeepSeek model; no secret is persisted
    for arm in ARMS:
        if arm in state["jobs"] and _healthy(Path(state["jobs"][arm])):
            continue
        jobs_root = kit / "workspace/runs" / (benchmark + "-train") / "r1-ablation" / arm
        started = time.monotonic()
        job = _run_harbor(kit, benchmark, jobs_root,
                          Path(arms[arm]["skills_dir"]) if arms[arm]["skills_dir"] else None,
                          n, seed)
        state["jobs"][arm] = str(job)
        state.setdefault("elapsed_sec", {})[arm] = round(time.monotonic() - started, 3)
        write_json(state_path, state)
        if stop_after == arm:
            return state
    result = summarize(root, benchmark, out, {arm: Path(path) for arm, path in state["jobs"].items()}, n,
                       root.parent / "metrics", seed)
    return {**state, "summary": result}


def render_report(root: Path, benchmarks: tuple[str, ...]) -> Path:
    """Summarize completed official Train runs without reading Val task feedback."""
    lines = ["# R1 Train A/B/C/D Diagnostic Replay", "",
             "Official Harbor scores on a small, identical Train subset for each arm.",
             "One execution per arm cannot establish Skill causality or statistical significance.", "",
             "| Benchmark | Arm | Pass/n | Fix/Regression vs A | Explicit reads | Tool calls | Input/Output tokens | Duration s |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for benchmark in benchmarks:
        summary_path = root / "results/train_ablation" / benchmark / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        for arm in ARMS:
            data = summary["arms"][arm]
            lines.append(f"| {benchmark} | {arm} | {data['passed']}/{summary['n']} | "
                         f"{data['fixes_vs_A']}/{data['regressions_vs_A']} | "
                         f"{data['explicit_skill_reads']} | {data.get('tool_calls', 'NA')} | "
                         f"{data['input_tokens']}/{data['output_tokens']} | "
                         f"{data['duration_sec_sum']:.1f} |")
    lines += ["", "D1/D2 are Train Insight linked rule fragments; the mapping is recorded in each arms_manifest.json.",
              "The Val regression audit does not enter this replay or the M3/M4 learning path.", ""]
    path = root / "results/train_ablation/TRAIN_ABLATION_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["all", *BENCHMARKS], default="all")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=ARMS)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    for benchmark in BENCHMARKS if args.benchmark == "all" else (args.benchmark,):
        result = run_one(args.root.resolve(), benchmark, n=args.n, seed=args.seed,
                         dry_run=args.dry_run, resume=args.resume, stop_after=args.stop_after)
        print(json.dumps({"benchmark": benchmark, "completed_arms": list(result.get("jobs", {})),
                          "mode": result.get("mode", "real")}, ensure_ascii=False), flush=True)
    if not args.dry_run:
        render_report(args.root.resolve(), tuple(BENCHMARKS if args.benchmark == "all" else (args.benchmark,)))


if __name__ == "__main__":
    main()
