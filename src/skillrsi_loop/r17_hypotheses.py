"""Keep evidence-backed behavior hypotheses separate from accepted Skill utility."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from skillrsi_m3.core import EvidenceVerifier, TraceStore, write_json
from skillrsi_m4.core import compile_single_rule
from .r17_plan import freeze
from .r17_source import audit


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def run(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    plan = freeze(root)
    records = []
    m4 = {}
    for benchmark in ("spreadsheetbench", "livemath"):
        source_audit = audit(root, benchmark, plan)
        if source_audit["status"] != "clean":
            raise ValueError(f"{benchmark} Source audit not clean")
        base = root / "results/r17" / benchmark
        path = base / "m3/insights.jsonl"
        insights = _jsonl(path)
        store = TraceStore(base / "m2/trace_profiles.jsonl", benchmark)
        verified = EvidenceVerifier(store).verify(insights)
        if any(item["errors"] for item in verified["details"]):
            raise ValueError("hypothesis evidence recheck failed")
        if not dry_run:
            path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in insights))
            write_json(base / "m3/normalized_evidence_audit.json", verified)
        for item in insights:
            records.append({"benchmark": benchmark, "insight_id": item["insight_id"],
                            "hypothesis_type": "procedural_hypothesis" if benchmark == "livemath" else "behavior_hypothesis",
                            "observed_problem": item.get("observed_problem"),
                            "trigger_conditions": item.get("trigger_conditions", []),
                            "avoid_conditions": item.get("avoid_conditions", []),
                            "expected_behavior_change": item.get("expected_behavior_change"),
                            "expected_observation": item.get("expected_observation"),
                            "cost_bound": item.get("cost_bound"),
                            "supporting_refs": item.get("supporting_refs", []),
                            "evidence_validity": item.get("evidence_validity"),
                            "semantic_validation": item.get("semantic_validation"),
                            "utility_status": "unverified", "causal_status": "hypothesis_only",
                            "applicability_normalization": item.get("applicability_normalization"),
                            "compilation_status": "not_compiled"})
        if dry_run:
            with tempfile.TemporaryDirectory() as temporary:
                preview = Path(temporary) / "insights.jsonl"
                preview.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in insights))
                m4[benchmark] = compile_single_rule(benchmark, preview, base / "skills", base / "m4",
                                                     dry_run=True, version="r17")
        else:
            m4[benchmark] = compile_single_rule(benchmark, path, base / "skills", base / "m4",
                                                 version="r17")
        if m4[benchmark]["status"] == "candidate_for_internal_validation":
            for record in records:
                if record["benchmark"] == benchmark and record["insight_id"] == m4[benchmark]["rule"]["source_insight_ids"][0]:
                    record["compilation_status"] = "unaccepted_static_candidate"
    # Historical ALFWorld D is retained for diagnosis only. Its R1.6 Holdout was used to revise this phase.
    historical = json.loads((root / "results/r16/alfworld/m4/candidate_manifest.json").read_text())
    if historical.get("rule"):
        records.append({"benchmark": "alfworld", "insight_id": historical["rule"]["source_insight_ids"][0],
                        "hypothesis_type": "historical_r16_behavior_hypothesis",
                        "trigger_conditions": historical["rule"]["trigger"],
                        "expected_behavior_change": historical["rule"]["expected_behavior"],
                        "expected_observation": historical["rule"]["expected_observation"],
                        "cost_bound": None, "supporting_trace_ids": historical["rule"]["evidence_trace_ids"],
                        "utility_status": "unverified", "compilation_status": "do_not_reuse_as_independent_r17_candidate",
                        "reason": "R1.6 Holdout D tied B with more environment actions and task time"})
    if not dry_run:
        for benchmark in ("spreadsheetbench", "livemath"):
            if audit(root, benchmark, plan)["status"] != "clean":
                raise ValueError("M4 Source prompt leaked future Holdout")
        write_json(root / "results/r17/candidate_hypotheses.json", {"records": records,
            "m4_manifests": m4, "accepted": False, "new_harbor_tasks": 0,
            "independent_validation": "not_run", "r16_holdout_reuse": "forbidden"})
    return {"hypotheses": len(records), "m4_statuses": {b: item["status"] for b, item in m4.items()},
            "accepted": False, "new_harbor_tasks": 0}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root.resolve(), dry_run=args.dry_run), ensure_ascii=False))


if __name__ == "__main__":
    main()
