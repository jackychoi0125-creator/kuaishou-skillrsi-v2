"""Optional official Harbor Train A/B/C/D gate for a fresh Source-only Skill."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from skillrsi_m4.core import validate_skill


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _controls(candidate: dict[str, Any], out: Path) -> dict[str, Path | None]:
    skill = Path(candidate["skill_path"])
    if _sha(skill) != candidate["sha256"]:
        raise ValueError("candidate Skill SHA changed")
    text = skill.read_text()
    front = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not front:
        raise ValueError("invalid candidate Skill front matter")
    name = re.search(r"(?m)^name:\s*(\S+)", front.group(1))
    description = re.search(r"(?m)^description:\s*(.+)$", front.group(1))
    if not name or not description:
        raise ValueError("candidate Skill metadata missing")
    def neutral(n: str, d: str) -> str:
        return (f"---\nname: {n}\ndescription: {d}\n---\n\n# When to use\n\n"
                "- This Skill may appear in the list.\n\n# Procedure\n\n"
                "- Follow the ordinary task instructions.\n\n# Avoid\n\n"
                "- No extra operation is required.\n\n# Validation\n\n"
                "- Use the task's ordinary completion condition.\n")
    result: dict[str, Path | None] = {"A": None, "D": skill.parent.parent}
    for arm, content in (("B", neutral("neutral-control", "Neutral placeholder with no operation rule.")),
                         ("C", neutral(name.group(1), description.group(1)))):
        if validate_skill(content, candidate["benchmark"], [{"benchmark": candidate["benchmark"],
                                                                "trigger_conditions": ["ordinary task"]}]):
            raise ValueError("invalid neutral control")
        target = out / arm / (name.group(1) if arm == "C" else "neutral-control") / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        result[arm] = target.parent.parent
    return result


def validate_train(candidate: dict[str, Any], kit: Path, out: Path, config: dict[str, Any],
                   max_evaluation_tasks: int) -> dict[str, Any]:
    """Execute a frozen fixed-ID Train gate; no candidate is accepted here."""
    from .pipeline import _read_official
    task_ids = sorted(config["task_ids"])
    repeats = int(config.get("repeats", 2))
    if not task_ids or len(set(task_ids)) != len(task_ids) or repeats < 2:
        raise ValueError("Train holdout requires fixed distinct IDs and at least two repeats")
    if 4 * repeats * len(task_ids) > max_evaluation_tasks:
        return {"status": "skipped_budget", "eligible_for_val": False, "harbor_tasks": 0}
    skill_dirs = _controls(candidate, out / "controls")
    arm_shas = {a: _sha(path / next(x.name for x in path.iterdir() if (x / "SKILL.md").is_file()) / "SKILL.md")
                if path is not None else None for a, path in skill_dirs.items()}
    frozen = {"task_ids": task_ids, "task_ids_sha256": hashlib.sha256(json.dumps(task_ids).encode()).hexdigest(),
              "candidate_skill_sha256": candidate["sha256"], "arm_shas": arm_shas,
              "kit_run_sha256": _sha(kit / "run.sh"), "sampling_args": config["sampling_args"],
              "bench": config["bench"], "repeats": repeats, "max_evaluation_tasks": max_evaluation_tasks,
              "concurrent": int(config.get("concurrent", 1)), "reasoning": "off"}
    frozen["plan_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    plan_path = out / "train_gate_plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text()) != frozen:
        raise ValueError("frozen Train gate changed")
    write_json(plan_path, frozen)
    state_path = out / "train_gate_runs.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"plan_sha256": frozen["plan_sha256"], "runs": {}}
    if state["plan_sha256"] != frozen["plan_sha256"]:
        raise ValueError("Train gate resume fingerprint mismatch")
    order = [(r, a) for r in range(repeats) for a in (("A", "B", "C", "D") if r % 2 == 0 else ("D", "C", "B", "A"))]
    for repeat, arm in order:
        key = f"repeat{repeat}-{arm}"
        if key in state["runs"]:
            old = state["runs"][key]
            if old["skill_sha256"] != arm_shas[arm] or not Path(old["job"]).is_dir():
                raise ValueError("Train gate resume result or Skill changed")
            rows = _read_official(Path(old["job"]))
            if set(rows) != set(task_ids) or rows != old["rows"]:
                raise ValueError("official Train result changed")
            continue
        jobs = out / "jobs" / key
        if jobs.exists() and any(jobs.iterdir()):
            raise ValueError("unregistered Train Job directory")
        args = [str(kit / "run.sh"), "--bench", config["bench"], "--jobs-dir", str(jobs),
                "--reasoning", "off", "--concurrent", str(frozen["concurrent"]),
                *config["sampling_args"]]
        if skill_dirs[arm]:
            args += ["--skills-dir", str(skill_dirs[arm])]
        with (out / f"{key}.log").open("w") as target:
            try:
                completed = subprocess.run(args, cwd=kit, stdout=target, stderr=subprocess.STDOUT,
                                           timeout=int(config.get("timeout_seconds", 1800)), check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        summaries = list(jobs.glob("*/summary.csv"))
        if code or len(summaries) != 1:
            result = {"status": "infrastructure_or_scoring_error", "eligible_for_val": False,
                      "stopped_at": key, "exit_code": code, "harbor_tasks": len(state["runs"]) * len(task_ids)}
            write_json(out / "train_internal_validation.json", result)
            return result
        rows = _read_official(summaries[0].parent)
        if set(rows) != set(task_ids):
            raise ValueError("official Train task IDs differ from frozen holdout")
        state["runs"][key] = {"job": str(summaries[0].parent), "skill_sha256": arm_shas[arm], "rows": rows}
        write_json(state_path, state)
    paired = [{"task_id": t, "repeat": r, "arms": {a: state["runs"][f"repeat{r}-{a}"]["rows"][t]
                                               for a in ("A", "B", "C", "D")}}
              for r in range(repeats) for t in task_ids]
    passes = {a: sum(x["arms"][a]["official_reward"] >= .999 for x in paired) for a in ("A", "B", "C", "D")}
    improvements = {x["task_id"] for x in paired if x["arms"]["D"]["official_reward"] >= .999 and
                    all(x["arms"][a]["official_reward"] < .999 for a in ("B", "C"))}
    regressions = {x["task_id"] for x in paired if x["arms"]["D"]["official_reward"] < .999 and
                   any(x["arms"][a]["official_reward"] >= .999 for a in ("A", "B", "C"))}
    cost = {a: {metric: sum(x["arms"][a][metric] for x in paired)
                for metric in ("input_tokens", "output_tokens", "duration_sec")} for a in ("A", "B", "C", "D")}
    stable = all(sum(x["arms"]["D"]["official_reward"] >= .999 for x in paired if x["repeat"] == r) >
                 max(sum(x["arms"][a]["official_reward"] >= .999 for x in paired if x["repeat"] == r)
                     for a in ("A", "B", "C")) for r in range(repeats))
    expensive = cost["D"]["input_tokens"] > 1.5 * max(1, cost["B"]["input_tokens"]) or \
                cost["D"]["duration_sec"] > 1.5 * max(1, cost["B"]["duration_sec"])
    status = "regression_risk" if regressions else "high_cost" if expensive else \
             "needs_more_evidence" if not stable or len(improvements) < 2 else "eligible_for_val"
    result = {"status": status, "eligible_for_val": status == "eligible_for_val",
              "candidate_skill_sha256": candidate["sha256"], "plan_sha256": frozen["plan_sha256"],
              "official_score_source": "Harbor verifier_result.rewards.reward", "arm_passes": passes,
              "distinct_improved_tasks": len(improvements), "distinct_regressed_tasks": len(regressions),
              "cost": cost, "harbor_tasks": 4 * repeats * len(task_ids)}
    write_json(out / "train_internal_validation.json", result)
    return result
