"""M2 diagnostic adapter for the legacy constraint rules and judged records."""
from __future__ import annotations

from typing import Any

from .core import check_constraints, task_context


def candidate_hypotheses(benchmark: str, events: list[dict[str, Any]], trial_path: str) -> list[dict[str, Any]]:
    context = task_context({"trial_path": trial_path})
    candidates = check_constraints(benchmark, events, context, {"trial_path": trial_path})
    event_ids = {event["id"] for event in events}
    return [{"module": candidate["error_module"], "type": candidate["error_type"],
             "root_cause": candidate["root_cause"], "rule_id": candidate["rule_id"],
             "event_ids": [event_id for event_id in candidate["supporting_events"] if event_id in event_ids],
             "confidence": candidate["confidence"], "status": "hypothesis_only"}
            for candidate in sorted(candidates, key=lambda item: -item["confidence"])[:8]]
