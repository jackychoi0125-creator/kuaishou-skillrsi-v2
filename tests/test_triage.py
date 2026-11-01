import tempfile
import unittest
import json
from pathlib import Path

from skillrsi_m3.core import EvidenceVerifier, InsightConsolidator
from skillrsi_m4.core import rule_evidence_map, validate_skill
from skillrsi_loop.ablation import _healthy, prepare_arms
from skillrsi_loop.triage import (_alf_summary, _script_diff, _skill_reads,
                                  first_behavior_divergence, first_divergence, milestones)


class TriageTests(unittest.TestCase):
    def test_skill_read_requires_actual_tool_call(self):
        events = [
            {"id": "s1.message", "kind": "message", "text": "Consider SKILL.md", "step": 1},
            {"id": "s2.call1", "kind": "tool_call", "function": "read",
             "text": '{"path":"/skills/test/SKILL.md"}', "step": 2},
        ]
        self.assertEqual(_skill_reads(events), ["s2.call1"])

    def test_first_difference_aligns_milestones(self):
        before = [{"label": "workbook_inspect", "detail": "input", "event_id": "s2.call1", "step": 2},
                  {"label": "script_execute", "detail": "run", "event_id": "s3.call1", "step": 3}]
        after = [{"label": "skill_read", "detail": "SKILL.md", "event_id": "s2.call1", "step": 2},
                 {"label": "workbook_inspect", "detail": "input", "event_id": "s4.call1", "step": 4},
                 {"label": "script_execute", "detail": "run", "event_id": "s7.call1", "step": 7}]
        difference = first_divergence(before, after)
        self.assertEqual(difference["change"], "insert")
        self.assertEqual(difference["r1"][0]["label"], "skill_read")
        self.assertEqual(difference["r1"][0]["event_id"], "s2.call1")

    def test_script_change_is_first_aligned_behavior_difference(self):
        a = [{"label": "script_write", "detail": "sha_a", "event_id": "s3.call1", "step": 3}]
        b = [{"label": "script_write", "detail": "sha_b", "event_id": "s8.call1", "step": 8}]
        result = first_divergence(a, b)
        self.assertEqual(result["change"], "different_action_or_result")
        self.assertEqual(result["baseline"][0]["event_id"], "s3.call1")

    def test_skill_read_does_not_hide_first_task_behavior_difference(self):
        before = [{"label": "final_option", "detail": "A", "event_id": "s3.message", "step": 3}]
        after = [{"label": "skill_read", "detail": "SKILL.md", "event_id": "s1.call1", "step": 1},
                 {"label": "final_option", "detail": "B", "event_id": "s4.message", "step": 4}]
        result = first_behavior_divergence(before, after)
        self.assertEqual(result["change"], "different_action_or_result")
        self.assertEqual(result["r1"][0]["event_id"], "s4.message")

    def test_alf_actions_and_no_progress_are_observed(self):
        events = [
            {"id": "s1.call1", "kind": "tool_call", "step": 1,
             "arguments": {"command": "bash bridge.sh send game 'go to table 1'"}, "text": ""},
            {"id": "s1.obs1", "kind": "observation", "step": 1,
             "text": '{"observation":"Nothing happens."}'},
            {"id": "s2.call1", "kind": "tool_call", "step": 2,
             "arguments": {"command": "bash bridge.sh send game 'go to table 1'"}, "text": ""},
            {"id": "s2.obs1", "kind": "observation", "step": 2,
             "text": '{"observation":"Nothing happens."}'},
        ]
        summary = _alf_summary(events)
        self.assertEqual(summary["actions"], 2)
        self.assertEqual(len(summary["repeated_action_event_ids"]), 1)
        self.assertEqual(len(summary["no_progress_event_ids"]), 2)

    def test_script_diff_uses_artifacts_not_hidden_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            for root, content in ((a, "x=1\n"), (b, "x=2\n")):
                path = root / "artifacts/bench/answers/1/solution.py"
                path.parent.mkdir(parents=True)
                path.write_text(content)
            difference = _script_diff(a, b)
            self.assertFalse(difference["same_script"])
            self.assertIn("+x=2", difference["changed_scripts"][0]["diff_excerpt"])

    def test_conflicting_insight_conditions_are_not_merged(self):
        common = {"benchmark": "alfworld", "capability_scope": "interaction",
                  "principle": "Read the state after each action", "supporting_refs": [],
                  "source_group_ids": ["g1"], "evidence_grade": "single_trace_hypothesis"}
        first = {**common, "insight_id": "a", "trigger_conditions": ["state changed"],
                 "avoid_conditions": ["action rejected"]}
        second = {**common, "insight_id": "b", "trigger_conditions": ["action rejected"],
                  "avoid_conditions": []}
        pool, operations = InsightConsolidator().consolidate([first, second])
        self.assertEqual(len(pool), 2)
        self.assertIn("CONFLICT", [item["operation"] for item in operations])
        self.assertEqual(pool[0]["risk_level"], "high")

    def test_singleton_observation_is_hypothesis_with_medium_risk(self):
        class Store:
            profiles = {"train-a": {"trace_sha256": "hash"}}
            def get(self, trace_id, event_id):
                return {"id": event_id}
        insight = {"insight_id": "x", "benchmark": "alfworld",
                   "supporting_refs": [{"trace_id": "train-a", "trace_sha256": "hash", "event_ids": ["s1"]}],
                   "counterexample_refs": [], "trigger_conditions": ["state changed"],
                   "avoid_conditions": [], "causal_status": "hypothesis_only", "principle": "read state"}
        report = EvidenceVerifier(Store()).verify([insight])
        self.assertEqual(report["verified"], 1)
        self.assertEqual(insight["semantic_support"], "single_observation")
        self.assertEqual(insight["risk_level"], "medium")

    def test_rule_map_preserves_source_insight_and_ordinary_words(self):
        insight = {"insight_id": "train_1", "principle": "Check observed state",
                   "recommended_behavior": "Read observation before next command",
                   "trigger_conditions": ["state changed"], "avoid_conditions": []}
        skill = ("---\nname: alfworld-check\ndescription: State check\n---\n"
                 "# When to use\n- State changed\n# Procedure\n- Read observation before next command\n"
                 "# Avoid\n- Avoid repeated action\n# Validation\n- Check observed state\n")
        self.assertNotIn("prohibited_content", validate_skill(skill, "alfworld", [{**insight, "benchmark": "alfworld"}]))
        self.assertEqual(rule_evidence_map(skill, [insight])[0]["insight_ids"], ["train_1"])
        other = {**insight, "insight_id": "train_2", "principle": "Inspect inventory",
                 "recommended_behavior": "Count held objects"}
        mapped = rule_evidence_map(skill, [insight, other])
        self.assertTrue(any(rule["mapping_status"] == "unmapped" for rule in mapped))

    def test_ablation_preparation_requires_frozen_train_insight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r1 = root / "results/r1/alfworld"
            (r1 / "m3").mkdir(parents=True)
            (r1 / "m4").mkdir()
            skill = root / "SKILL.md"
            skill.write_text("---\nname: abc\ndescription: abc\n---\n")
            (r1 / "m4/candidate_manifest.json").write_text(json.dumps({
                "skill_path": str(skill), "sha256": "invalid", "insight_ids": ["x"]}))
            (r1 / "m3/insights.jsonl").write_text(json.dumps({"insight_id": "x", "status": "verified_observation"}))
            with self.assertRaisesRegex(ValueError, "SHA"):
                prepare_arms(root, "alfworld", root / "ablation")

    def test_model_error_run_is_not_usable_ablation_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            (job / "summary.csv").write_text(
                "qid,reward,model_error,exception\nx,1.0,True,False\n")
            self.assertFalse(_healthy(job))
            (job / "summary.csv").write_text(
                "qid,reward,model_error,exception\nx,1.0,False,False\n")
            self.assertTrue(_healthy(job))


if __name__ == "__main__":
    unittest.main()
