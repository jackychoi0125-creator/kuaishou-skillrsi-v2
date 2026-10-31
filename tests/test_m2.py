import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillrsi_m1.core import sha256
from skillrsi_m1.m2 import (alfworld_sequence, build_l0, fetch, make_groups,
                            make_view, mechanism, select_representatives)


def sample_step(step_id, action, observation):
    call_id = f"c{step_id}"
    return {"step_id": step_id, "source": "agent", "message": "",
            "tool_calls": [{"tool_call_id": call_id, "function_name": "bash",
                            "arguments": {"command": f'bridge.sh send train:1 "{action}"'}}],
            "observation": {"results": [{"source_call_id": call_id,
                                         "content": json.dumps({"observation": observation,
                                                                "admissible_commands": ["go to kitchen"]})}]}}


class ProgressiveTraceTests(unittest.TestCase):
    def test_l0_build_does_not_open_atif(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "alfworld-baseline-original.csv"
            with metrics.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["split", "task_id", "task_type", "trace_chars"])
                writer.writeheader()
                writer.writerow({"split": "train", "task_id": "a", "task_type": "pick_and_place", "trace_chars": 400})
            record = {"benchmark": "alfworld", "task_id": "a", "outcome": "task_failure",
                      "reward": 0, "trace_validity": "valid", "reliability_flags": [],
                      "error_module": None, "error_type": None, "root_cause": "undetermined",
                      "rule_id": None, "diagnosis_status": "needs_review", "confidence": 0,
                      "critical_steps": [], "supporting_events": [],
                      "source_trace_path": str(root / "absent.json"), "trace_sha256": "abc"}
            records = root / "records.jsonl"
            records.write_text(json.dumps(record) + "\n")
            with patch("skillrsi_m1.m2.BENCHMARKS", ("alfworld",)), \
                 patch("skillrsi_m1.m2.read_verified_trace", side_effect=AssertionError("ATIF read")):
                rows = build_l0(root, records)
            self.assertEqual(rows[0]["estimated_full_trace_tokens"], 100)

    def test_alfworld_ir_keeps_event_links_and_detects_rejection(self):
        trace = {"steps": [sample_step(1, "open fridge", "Nothing happens."),
                           sample_step(2, "open fridge", "Nothing happens.")]}
        sequence = alfworld_sequence(trace)
        self.assertEqual(sequence[0]["next_action"], "open fridge")
        self.assertTrue(sequence[1]["repeated_action"])
        self.assertEqual(sequence[1]["state_change"], "unchanged")
        self.assertTrue(sequence[1]["invalid_return"])
        self.assertEqual(sequence[1]["observation_event_id"], "s2.obs1")

    def test_unknown_failure_remains_observed_signal_only(self):
        row = {"outcome": "task_failure", "diagnosis_status": "needs_review",
               "root_cause": "undetermined", "rule_id": None,
               "task_family": "simple_pick_place", "trace_signals": {"repeated_actions": 2}}
        self.assertEqual(mechanism(row), ("signal:no_progress", "observed_signal_only"))

    def test_terminal_mode_precedes_rejected_rule_candidate(self):
        row = {"outcome": "task_failure", "diagnosis_status": "needs_review",
               "root_cause": "undetermined", "rule_id": None,
               "candidate_rule_ids": ["tool_error"],
               "observed_failure_modes": [{"code": "action_budget_exhausted", "event_id": "s5.obs1"}]}
        self.assertEqual(mechanism(row), ("signal:action_budget_exhausted", "observed_signal_only"))

    def test_supported_event_is_in_view_and_fetch_returns_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = {"steps": [sample_step(1, "look", "Kitchen"),
                               sample_step(2, "open fridge", "Nothing happens.")]}
            source = root / "trace.json"
            source.write_text(json.dumps(trace), encoding="utf-8")
            row = {"trace_id": "alfworld:t1", "benchmark": "alfworld", "task_id": "t1",
                   "source_trace_path": str(source), "trace_sha256": sha256(source),
                   "supporting_events": ["s2.obs1"], "critical_steps": [2],
                   "signal_events": [], "task_family": "simple_pick_place",
                   "outcome": "task_failure", "diagnosis_status": "judge_supported",
                   "root_cause": "rejected_environment_action", "error_module": "action"}
            view = make_view(row, window=0, excerpt_chars=15)
            self.assertIn("s2.obs1", [event["event_id"] for event in view["events"]])
            records = root / "records.jsonl"
            records.write_text(json.dumps(row) + "\n")
            original = fetch(records, "alfworld", "t1", "s2.obs1")
            self.assertIn("Nothing happens", original["event"]["text"])
            self.assertEqual(original["raw_atif_step"], trace["steps"][1])

    def test_selection_preserves_supported_and_covers_benchmarks(self):
        rows = []
        for benchmark, task_id, status in [("alfworld", "a", "judge_supported"),
                                            ("spreadsheetbench", "s", "needs_review"),
                                            ("livemath", "m", "needs_review")]:
            rows.append({"trace_id": f"{benchmark}:{task_id}", "benchmark": benchmark,
                         "task_id": task_id, "task_family": "family", "complexity_bucket": "short",
                         "outcome": "task_failure", "diagnosis_status": status,
                         "root_cause": "rejected_environment_action" if status == "judge_supported" else "undetermined",
                         "error_module": "action" if status == "judge_supported" else None,
                         "rule_id": None, "confidence": 0.8 if status == "judge_supported" else 0,
                         "supporting_events": ["s1.obs1"] if status == "judge_supported" else [],
                         "estimated_full_trace_tokens": 100, "trace_signals": {}})
        groups = make_groups(rows)
        selected = select_representatives(rows, groups, 3, 2)
        self.assertEqual({row["benchmark"] for row in selected},
                         {"alfworld", "spreadsheetbench", "livemath"})
        self.assertEqual(selected[0]["trace_id"], "alfworld:a")


if __name__ == "__main__":
    unittest.main()
