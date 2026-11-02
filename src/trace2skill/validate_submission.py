"""Check package structure, state and accidental private-data inclusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from skillrsi_m4.core import validate_skill

REQUIRED = ("src/trace2skill/pipeline.py", "src/skillrsi_m1/reliability.py",
            "src/skillrsi_m1/m2_pipeline.py", "src/skillrsi_m3/core.py",
            "src/skillrsi_m4/core.py", "skills/BEST_STATE.json", "README.md", "DESIGN.md",
            "pyproject.toml", "results/round_summary.json", "results/generation_tokens.json",
            "results/acceptance_history.json", "results/harbor_tokens.json",
            "results/evaluation_summary.md", "tests/test_submission_portable.py")


def validate(path: Path) -> dict[str, Any]:
    errors, warnings, unverified = [], [], []
    for name in REQUIRED:
        if not (path / name).is_file():
            errors.append("missing:" + name)
    if errors:
        return {"status": "invalid", "error": errors, "warning": warnings,
                "unverified_external_requirement": unverified}
    best = json.loads((path / "skills/BEST_STATE.json").read_text())
    summary = json.loads((path / "results/round_summary.json").read_text())
    generation = json.loads((path / "results/generation_tokens.json").read_text())
    history = json.loads((path / "results/acceptance_history.json").read_text())
    for name, item in best.items():
        accepted = item.get("accepted_skill")
        if accepted is None:
            if item.get("version") != "no_skill":
                errors.append("best_state_does_not_match_accepted_history:" + name)
        else:
            skill = path / "skills" / accepted
            if not skill.is_file() or hashlib.sha256(skill.read_bytes()).hexdigest() != item.get("sha256"):
                errors.append("accepted_skill_sha_mismatch:" + name)
    if len(summary.get("rounds", [])) != 3 or not history.get("history"):
        errors.append("missing_round_or_acceptance_summary")
    if generation.get("service_usage_only") is not True:
        errors.append("generation_token_provenance_missing")
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        relative = str(file.relative_to(path))
        if any(part in file.parts for part in ("private_raw", "private_model_cache", "workspace", "kit.env")) or file.suffix in {".pyc", ".jsonl"}:
            errors.append("private_artifact:" + relative)
            continue
        if file.suffix not in {".py", ".md", ".toml", ".json"}:
            warnings.append("unscanned_extension:" + relative)
            continue
        text = file.read_text(errors="replace")
        if re.search(r"/Us" + r"ers/|/ho" + r"me/[^/]+/|/pri" + r"vate/var/|sk-[A-Za-z0-9]{15,}", text):
            errors.append("local_path_or_credential:" + relative)
        if relative.startswith("results/") and re.search(r"(?:task_id|case_id|source_trace_path|trial_path)\s*\"?:", text):
            errors.append("task_level_result:" + relative)
    skills = list((path / "skills").glob("*/SKILL.md"))
    if len(skills) != sum(item.get("accepted_skill") is not None for item in best.values()):
        errors.append("unlisted_or_missing_accepted_skill")
    if skills:
        for skill in skills:
            if validate_skill(skill.read_text(), "generic", [{"benchmark": "generic", "trigger_conditions": ["task"]}]):
                errors.append("invalid_skill:" + str(skill.relative_to(path)))
    else:
        warnings.append("no_accepted_skill; runtime may generate one when evidence and validation permit")
        unverified.append("official_rule_allowing_empty_skills_directory")
    readme = (path / "README.md").read_text()
    if "trace2skill evolve" not in readme or "--train-traces" not in readme:
        errors.append("README_command_missing")
    pyproject = (path / "pyproject.toml").read_text()
    if not re.search(r'(?m)^trace2skill\s*=\s*"trace2skill\.pipeline:main"', pyproject):
        errors.append("package_cli_entry_missing")
    if not (path / "src/trace2skill/__main__.py").is_file():
        errors.append("module_entry_missing")
    return {"status": "invalid" if errors else "valid_with_unverified_external_requirements" if unverified else "valid",
            "error": sorted(set(errors)), "warning": sorted(set(warnings)),
            "unverified_external_requirement": sorted(set(unverified)),
            "accepted_skill_count": len(skills), "file_count": sum(p.is_file() for p in path.rglob("*")),
            "official_rules_checked_at": "2026-11-02", "official_rules_sha256":
            "1d42a79001774e45d6e4ebaf61d4b7b3c06eec222caf7d5b05830483024345ed"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(args.path.resolve())
    write_json(args.path.resolve().parent / "submission_validation.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
