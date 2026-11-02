"""Re-read complete Source-Train math events and preserve unresolved reasoning hypotheses."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from skillrsi_m3.core import TraceStore, write_json
from skillrsi_m3.math_verification import classify_math_insight, verify_cited_equality
from .r17_plan import freeze
from .r17_source import audit


def run(root: Path) -> dict[str, Any]:
    plan = freeze(root)
    provenance = audit(root, "livemath", plan)
    if provenance["status"] != "clean":
        raise ValueError("math Source provenance is not clean")
    base = root / "results/r17/livemath"
    store = TraceStore(base / "m2/trace_profiles.jsonl", "livemath")
    insights = [json.loads(line) for line in (base / "m3/insights.jsonl").read_text().splitlines() if line.strip()]
    records = []
    for insight in insights:
        refs = []
        checks = []
        for ref in insight.get("supporting_refs", []):
            for event_id in ref["event_ids"]:
                event = store.get(ref["trace_id"], event_id)
                text = event["text"]
                refs.append({"trace_id": ref["trace_id"], "event_id": event_id,
                             "full_event_chars": len(text), "source_trace_sha256": ref["trace_sha256"],
                             "local_definition_mentioned": bool(re.search(r"(?i)definition|notation|defined|定义", text)),
                             "condition_mentioned": bool(re.search(r"(?i)assum|condition|except|provided|条件|假设", text)),
                             "visible_self_check_mentioned": bool(re.search(r"(?i)check|verify|recheck|核对|检查", text))})
                for equality in insight.get("arithmetic_equalities_to_verify", []):
                    checks.append(verify_cited_equality(store, ref["trace_id"], event_id, equality))
        level = classify_math_insight(insight, checks)
        records.append({"insight_id": insight["insight_id"], "level": level["level"],
                        "causal_status": "hypothesis_only", "full_original_events_reread": refs,
                        "independent_math_checks": checks,
                        "root_cause": "undetermined",
                        "limitations": "Text indicators are observations only; no final-answer correctness inference."})
    result = {"benchmark": "livemath", "scope": "R1.7 Source Train only", "records": records,
              "procedural_hypotheses": sum(r["level"] == "procedural_hypothesis" for r in records),
              "mathematically_verified_local_claims": sum(r["level"] == "mathematically_verified" for r in records),
              "full_events_reread": sum(len(r["full_original_events_reread"]) for r in records),
              "new_model_calls": 0, "harbor_tasks": 0,
              "conclusion": "No mathematical reasoning root was independently established from these traces."}
    write_json(base / "math_evidence_audit.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    value = run(args.root.resolve())
    print(json.dumps({key: value[key] for key in ("procedural_hypotheses", "mathematically_verified_local_claims",
                                                  "full_events_reread", "new_model_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
