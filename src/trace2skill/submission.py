"""Build a source-only, privacy-safe competition package."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json

PACKAGES = ("trace2skill", "skillrsi_m1", "skillrsi_m3", "skillrsi_m4")
BENCHMARKS = ("spreadsheetbench", "alfworld", "livemath")


def _usage_cache(path: Path) -> dict[str, int]:
    result = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_tokens": 0}
    for receipt in path.glob("*.json") if path.is_dir() else []:
        try:
            usage = json.loads(receipt.read_text()).get("usage") or {}
            for key in result:
                result[key] += int(usage.get(key) or 0)
        except (OSError, ValueError, TypeError):
            continue
    return result


def _harbor_manifest_totals(root: Path) -> dict[str, Any]:
    """Deduplicate official Job directories reachable from local run manifests."""
    paths: set[Path] = set()
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and "workspace/runs" in value:
            path = Path(value)
            if path.is_dir() and (path / "summary.csv").is_file():
                paths.add(path.resolve())
    for file in (root / "results").rglob("*.json"):
        if file.name not in {"runs_manifest.json", "holdout_runs_manifest.json", "stage_manifest.json",
                             "acceptance.json", "run_manifest.json", "interaction_runs_manifest.json"}:
            continue
        try:
            visit(json.loads(file.read_text()))
        except (OSError, ValueError, TypeError):
            continue
    totals = {"jobs": 0, "task_executions": 0, "input_tokens": 0, "output_tokens": 0,
              "cache_tokens": 0}
    for job in paths:
        try:
            with (job / "summary.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            totals["jobs"] += 1
            totals["task_executions"] += len(rows)
            for metric in ("input_tokens", "output_tokens", "cache_tokens"):
                totals[metric] += sum(int(float(row.get(metric) or 0)) for row in rows)
        except (OSError, ValueError, TypeError):
            continue
    totals["scope"] = "unique_completed_official_jobs_referenced_by_local_manifests"
    return totals


def build(root: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "src"
    source.mkdir(exist_ok=True)
    for name in PACKAGES:
        target = source / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(root / "src" / name, target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tests = destination / "tests"
    if tests.exists():
        shutil.rmtree(tests)
    shutil.copytree(root / "tests", tests, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Historical tests reference development modules. The package includes its
    # own portable tests, while the repository keeps the complete development suite.
    for item in tests.iterdir():
        if item.name != "test_submission_portable.py":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    skills_dir = destination / "skills"
    skills_dir.mkdir(exist_ok=True)
    for child in skills_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    best = {b: {"version": "no_skill", "accepted_skill": None} for b in BENCHMARKS}
    r18_acceptance = root / "results/r18/spreadsheetbench/val/acceptance.json"
    if r18_acceptance.is_file():
        accepted = json.loads(r18_acceptance.read_text())
        if accepted.get("accepted") is True and accepted.get("decision") == "accept":
            candidate = json.loads((root / "results/r17/spreadsheetbench/m4/candidate_manifest.json").read_text())
            source_skill = Path(candidate["skill_path"])
            import hashlib
            if hashlib.sha256(source_skill.read_bytes()).hexdigest() != candidate["skill_sha256"]:
                raise ValueError("accepted Skill SHA changed")
            target = skills_dir / "spreadsheetbench-r18" / "SKILL.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_skill.read_bytes())
            best["spreadsheetbench"] = {"version": "r18", "accepted_skill": "spreadsheetbench-r18/SKILL.md",
                                        "sha256": candidate["skill_sha256"]}
    write_json(destination / "skills/BEST_STATE.json", best)
    (destination / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "trace2skill-submission"\nversion = "0.1.0"\nrequires-python = ">=3.9"\n'
        'dependencies = []\n\n[project.scripts]\ntrace2skill = "trace2skill.pipeline:main"\n\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\n')
    readme = (root / "docs/SUBMISSION_README.md").read_text()
    if best["spreadsheetbench"]["version"] == "r18":
        readme = readme.replace("The package contains no accepted Skill because every tested public candidate was rejected.",
                                "The SpreadsheetBench R1.8 candidate was accepted after the frozen Train and official Val gates; the other two public Benchmarks retain no Skill.")
    (destination / "README.md").write_text(readme)
    design = (root / "docs/SUBMISSION_DESIGN.md").read_text()
    if best["spreadsheetbench"]["version"] == "r18":
        design = design.replace("The accepted best remains no Skill.",
                                "The R1.8 SpreadsheetBench candidate was accepted; other public best states remain no Skill.")
    (destination / "DESIGN.md").write_text(design)
    results = destination / "results"
    results.mkdir(exist_ok=True)
    rounds = []
    history = []
    harbor = []
    for b in BENCHMARKS:
        acceptance = json.loads((root / "results/r1" / b / "val/acceptance.json").read_text())
        rounds.append({"benchmark": b, "round": "r1", "score_source": "official_harbor_val",
                       "baseline_pass": acceptance["baseline_pass"], "candidate_pass": acceptance["candidate_pass"],
                       "task_count": acceptance["tasks"], "fixes": acceptance["fixes"],
                       "regressions": acceptance["regressions"], "decision": acceptance["decision"],
                       "candidate_count": 1, "best_version": "no_skill",
                       "stop_reason": "official_val_gate_rejected"})
        history.append({"benchmark": b, "round": "r1", "decision": acceptance["decision"],
                        "best_after_round": "no_skill", "score_source": "official_harbor_val"})
        history.append({"benchmark": b, "round": "r17", "decision": "not_evaluated",
                        "best_after_round": "no_skill", "score_source": None})
        harbor.append({"benchmark": b, "round": "r1", "split": "val", "tasks_per_arm": acceptance["tasks"],
                       "baseline": {key: acceptance.get("baseline_totals", {}).get(key)
                                    for key in ("input_tokens", "output_tokens", "duration_sec")},
                       "candidate": {key: acceptance.get("candidate_totals", {}).get(key)
                                     for key in ("input_tokens", "output_tokens", "duration_sec")}})
    r18 = root / "results/r18/spreadsheetbench/holdout_analysis.json"
    if r18.is_file():
        item = json.loads(r18.read_text())
        history.append({"benchmark": "spreadsheetbench", "round": "r18_train_holdout",
                        "decision": item["status"], "best_after_round": "no_skill",
                        "score_source": item["official_score_source"],
                        "completed_repeats": len(item["complete_repeats"])})
        harbor.append({"benchmark": "spreadsheetbench", "round": "r18", "split": "train_holdout",
                       "status": item["status"], "actual_task_executions": item["usage"]["task_executions"],
                       "input_tokens": item["usage"]["input_tokens"],
                       "output_tokens": item["usage"]["output_tokens"]})
    if r18_acceptance.is_file():
        item = json.loads(r18_acceptance.read_text())
        history.append({"benchmark": "spreadsheetbench", "round": "r18_val",
                        "decision": item["decision"], "best_after_round": best["spreadsheetbench"]["version"],
                        "score_source": "official_harbor_val"})
        harbor.append({"benchmark": "spreadsheetbench", "round": "r18", "split": "val",
                       "actual_task_executions": item["tasks"],
                       "candidate": {key: item["candidate_totals"].get(key)
                                     for key in ("input_tokens", "output_tokens", "duration_sec")}})
    write_json(results / "round_summary.json", {"schema_version": 1, "rounds": rounds,
                  "current_best": best, "r17_candidate_count": {"spreadsheetbench": 1, "alfworld": 0,
                                                                  "livemath": 0},
                  "r17_candidate_accepted": best["spreadsheetbench"]["version"] == "r18",
                  "independence_boundary": "R1.7 Source excludes its frozen Holdout; historical R1 saw all Train",
                  "r18_train_status": json.loads(r18.read_text())["status"] if r18.is_file() else "not_completed"})
    write_json(results / "acceptance_history.json", {"schema_version": 1, "history": history})
    generation = {"schema_version": 1, "service_usage_only": True,
                  "historical_exploration": {"r1": _usage_cache(root / "results/r1/cache"),
                     "r16": {b: _usage_cache(root / "results/r16" / b / "isolated_kimi_cache") for b in BENCHMARKS}},
                  "r17_incremental": {b: _usage_cache(root / "results/r17" / b / "isolated_kimi_cache")
                                       for b in BENCHMARKS},
                  "reproducible_submission_run": {"api_calls": 0, "prompt_tokens": 0,
                                                   "completion_tokens": 0, "cache_tokens": 0},
                  "harbor_tokens_separate_from_generation": True}
    legacy_path = root / "results/m1/summary.json"
    if legacy_path.is_file():
        legacy = json.loads(legacy_path.read_text())
        generation["legacy_m1_judge_reported_not_r18_incremental"] = {
            "prompt_tokens": int(legacy.get("judge_input_tokens") or 0),
            "completion_tokens": int(legacy.get("judge_output_tokens") or 0),
            "reused_cases": int(legacy.get("llm_reused") or 0),
            "provenance": "historical_M1_recorded_usage_not_current_service_receipts"}
    write_json(results / "generation_tokens.json", generation)
    write_json(results / "harbor_tokens.json", {"schema_version": 1, "token_source": "official_Harbor_summary",
                                                 "runs": harbor,
                                                 "all_known_manifest_jobs": _harbor_manifest_totals(root)})
    evaluation_text = (
        "# Official historical Val\n\n"
        "| Benchmark | No Skill | R1 | Decision |\n|---|---:|---:|---|\n" +
        "\n".join(f"| {x['benchmark']} | {x['baseline_pass']}/{x['task_count']} | "
                  f"{x['candidate_pass']}/{x['task_count']} | {x['decision']} |" for x in rounds) +
        "\n\nR1.7 generated hypotheses are not official scores.\n")
    if r18.is_file():
        item = json.loads(r18.read_text())
        evaluation_text += (f"\nR1.8 SpreadsheetBench Train Holdout: `{item['status']}`, "
                            f"{len(item['complete_repeats'])} complete paired repeats; "
                            f"official task executions recorded: {item['usage']['task_executions']}.\n")
    evaluation_text += "\nAccepted best versions are listed in `../skills/BEST_STATE.json`.\n"
    (results / "evaluation_summary.md").write_text(evaluation_text)
    return {"destination": str(destination), "packages": list(PACKAGES),
            "skills": sum(item["accepted_skill"] is not None for item in best.values()), "best": best}


def make_zip(destination: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                target.write(path, Path("submission") / path.relative_to(destination))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--path", type=Path, default=Path("submission"))
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args(argv)
    result = build(args.root.resolve(), args.path.resolve())
    if args.zip:
        make_zip(args.path.resolve(), args.zip.resolve())
        result["zip"] = str(args.zip.resolve())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
