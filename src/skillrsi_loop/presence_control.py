"""Build no-op Skill controls without changing the official Harbor agent."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from skillrsi_m3.core import write_json
from skillrsi_m4.core import validate_skill


def build(candidate_manifest: Path, out_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    candidate = json.loads(candidate_manifest.read_text())
    if candidate.get("status") != "candidate_for_internal_validation" or candidate.get("accepted"):
        return {"status": "no_candidate", "reason": "candidate_not_ready"}
    text = Path(candidate["skill_path"]).read_text()
    front = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not front:
        raise ValueError("candidate has no front matter")
    name = re.search(r"(?m)^name:\s*(\S+)", front.group(1)).group(1)
    description = re.search(r"(?m)^description:\s*(.+)$", front.group(1)).group(1)
    def neutral(label: str, meta_name: str, meta_description: str) -> str:
        return (f"---\nname: {meta_name}\ndescription: {meta_description}\n---\n\n"
                "# When to use\n\n- This document may appear in the Skill list.\n\n"
                "# Procedure\n\n- Follow the task's ordinary instructions.\n\n"
                "# Avoid\n\n- No additional operation is required by this document.\n\n"
                "# Validation\n\n- Use the task's ordinary completion condition.\n")
    contents = {"B": neutral("B", f"{candidate['benchmark']}-r16-neutral", "Neutral placeholder with no task procedure."),
                "C": neutral("C", name, description), "D": text}
    result = {"status": "prepared", "benchmark": candidate["benchmark"], "arms": {"A": {"skills_dir": None}}}
    for arm, markdown in contents.items():
        errors = validate_skill(markdown, candidate["benchmark"], [
            {"benchmark": candidate["benchmark"], "trigger_conditions": ["normal task"]}])
        if errors:
            raise ValueError(f"invalid {arm} control: {errors}")
        sha = hashlib.sha256(markdown.encode()).hexdigest()
        target = out_dir / arm / (name if arm == "C" else f"{candidate['benchmark']}-r16-{arm.lower()}") / "SKILL.md"
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(markdown)
        result["arms"][arm] = {"skills_dir": str(target.parent.parent), "skill_path": str(target),
                                "skill_sha256": sha, "body_kind": "single_rule" if arm == "D" else "no_operation",
                                "metadata_matches_D": arm == "C"}
    if not dry_run:
        write_json(out_dir / "controls_manifest.json", result)
    return result
