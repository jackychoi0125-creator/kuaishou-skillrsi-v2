import unittest
import json
import tempfile
import io
from pathlib import Path
from unittest.mock import patch

from skillrsi_m1.core import (check_constraints, choose_candidate, diagnostic_gap,
                              events_from_trace, observed_failure_modes, outcome,
                              sha256, trace_validity)
from skillrsi_m1.cli import validate_records
from skillrsi_m1.judge import ChatClient, _json_object


def step(step_id, command, observation, admissible=None):
    import json
    call_id = f"call-{step_id}"
    content = json.dumps({"observation": observation,
                          "admissible_commands": admissible or []})
    return {"step_id": step_id, "source": "agent", "message": "",
            "tool_calls": [{"tool_call_id": call_id, "function_name": "bash",
                            "arguments": {"command": command}}],
            "observation": {"results": [{"source_call_id": call_id, "content": content,
                                         "extra": {"is_error": False}}]}}


class ReliabilityTests(unittest.TestCase):
    def test_terminal_alf_state_is_mode_not_inferred_root(self):
        trace = {"steps": [step(1, 'bridge.sh send train:1 "look"',
                                "room", ["go to kitchen"])]}
        trace["steps"][0]["observation"]["results"][0]["content"] = json.dumps({
            "observation": "room", "steps": 50, "max_steps": 50,
            "done": True, "won": False})
        modes = observed_failure_modes("alfworld", "task_failure", events_from_trace(trace), {})
        self.assertEqual(modes[0]["code"], "action_budget_exhausted")
        self.assertEqual(modes[0]["event_id"], "s1.obs1")
        self.assertEqual(diagnostic_gap({"benchmark": "alfworld", "outcome": "task_failure",
                                         "diagnosis_status": "needs_review", "candidate_count": 0}),
                         "causal_action_evidence_missing")

    def test_terminal_429_is_infrastructure_failure(self):
        row = {"task_id": "train__0000", "reward": "0.0", "model_error": "True",
               "agent_429_errors": "4", "execution_error": "False", "scoring_error": "False"}
        item = {"case_id": "train__0000", "trace": {"steps": [{"step_id": 1, "source": "user", "message": "task"}]}}
        validity, flags = trace_validity(item, row)
        self.assertEqual(validity, "valid")
        self.assertIn("observed_429", flags)
        self.assertEqual(outcome(row, validity), "infrastructure_failure")

    def test_embedded_trace_mismatch_is_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            path.write_text(json.dumps({"steps": [{"step_id": 1, "message": "original"}]}))
            item = {"case_id": "x", "trace": {"steps": [{"step_id": 1, "message": "changed"}]},
                    "source_trace_path": str(path), "trace_sha256": sha256(path)}
            validity, flags = trace_validity(item, {"task_id": "x"})
            self.assertEqual(validity, "mismatch")
            self.assertIn("embedded_trace_mismatch", flags)

    def test_alfworld_successful_action_is_not_inadmissible_failure(self):
        trace = {"steps": [
            step(1, 'bridge.sh send train:0000 "look"', "You are in a kitchen.", ["go to cabinet 1"]),
            step(2, 'bridge.sh send train:0000 "open cabinet 1"', "You open the cabinet 1.", ["close cabinet 1"]),
        ]}
        candidates = check_constraints("alfworld", events_from_trace(trace), {}, {"trial_path": "/nonexistent"})
        self.assertFalse(any(c["rule_id"] == "inadmissible_action" for c in candidates))

    def test_alfworld_rejected_action_has_real_event_ids(self):
        trace = {"steps": [
            step(1, 'bridge.sh send train:0000 "look"', "You are in a kitchen.", ["go to cabinet 1"]),
            step(2, 'bridge.sh send train:0000 "open cabinet 1"', "Nothing happens.", ["go to cabinet 1"]),
        ]}
        events = events_from_trace(trace)
        candidates = check_constraints("alfworld", events, {}, {"trial_path": "/nonexistent"})
        chosen = choose_candidate(candidates)
        self.assertIsNotNone(chosen)
        self.assertTrue(set(chosen["supporting_events"]).issubset({e["id"] for e in events}))

    def test_livemath_wrong_answer_does_not_imply_root_cause(self):
        trace = {"steps": [{"step_id": 1, "source": "user", "message": "Choose A–E"},
                           {"step_id": 2, "source": "agent", "message": "Therefore \\boxed{B}."}]}
        candidates = check_constraints("livemath", events_from_trace(trace), {}, {"trial_path": "/nonexistent"})
        self.assertEqual(candidates, [])

    def test_spreadsheet_repaired_tool_error_is_low_confidence(self):
        trace = {"steps": [
            {"step_id": 1, "source": "agent", "message": "", "tool_calls": [
                {"tool_call_id": "a", "function_name": "bash", "arguments": {"command": "python x.py"}}],
             "observation": {"results": [{"source_call_id": "a", "content": "Traceback (most recent call last):\nSyntaxError: bad", "extra": {"is_error": True}}]}},
            {"step_id": 2, "source": "agent", "message": "", "tool_calls": [
                {"tool_call_id": "b", "function_name": "bash", "arguments": {"command": "python fixed.py"}}],
             "observation": {"results": [{"source_call_id": "b", "content": "saved", "extra": {"is_error": False}}]}},
        ]}
        candidates = check_constraints("spreadsheetbench", events_from_trace(trace), {"instruction": ""}, {"trial_path": "/nonexistent"})
        self.assertEqual(candidates[0]["confidence"], 0.45)

    def test_unverified_math_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trace.json"
            source.write_text(json.dumps({"steps": [{"step_id": 2, "source": "agent", "message": "answer"}]}))
            record = {"benchmark": "livemath", "task_id": "lm1", "outcome": "task_failure",
                      "error_module": "planning", "error_type": "inefficient_plan",
                      "root_cause": "math_mistake", "supporting_events": ["s2.message"],
                      "critical_steps": [2], "evidence": [{"event_id": "s2.message", "step_id": 2, "excerpt": "answer"}],
                      "rule_id": "other", "source_trace_path": str(source)}
            with self.assertRaises(ValueError):
                validate_records([record])

    def test_judge_parser_accepts_first_complete_json_object(self):
        self.assertEqual(_json_object('{"module":"action"}\n{"unused":true}'), {"module": "action"})

    def test_judge_retries_malformed_response(self):
        client = ChatClient.__new__(ChatClient)
        client.api_key, client.model, client.url = "test-key", "test-model", "https://example.invalid"
        client.max_retries, client.timeout, client.thinking = 1, 1, False
        def response(content):
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}], "usage": {}}).encode())
        with patch("skillrsi_m1.judge.urllib.request.urlopen", side_effect=[response("not JSON"), response('{"ok": true}')]) as call, \
             patch("skillrsi_m1.judge.time.sleep"):
            value, _ = client.ask_json("system", "user")
        self.assertEqual(value, {"ok": True})
        self.assertEqual(call.call_count, 2)

    def test_evidence_verifier_checks_original_atif_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trace.json"
            source.write_text(json.dumps({"steps": [{"step_id": 2, "source": "agent", "message": "observed failure"}]}))
            record = {"benchmark": "alfworld", "task_id": "train__0000", "outcome": "task_failure",
                      "error_module": "action", "error_type": "invalid_action", "root_cause": "rejected_environment_action",
                      "supporting_events": ["s2.message"], "critical_steps": [2],
                      "evidence": [{"event_id": "s2.message", "step_id": 2, "excerpt": "observed failure"}],
                      "source_trace_path": str(source)}
            validate_records([record])
            record["evidence"][0]["excerpt"] = "fabricated"
            with self.assertRaises(ValueError):
                validate_records([record])


if __name__ == "__main__":
    unittest.main()
