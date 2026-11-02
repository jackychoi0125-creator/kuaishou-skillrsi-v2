import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from skillrsi_loop.r18_validation import _formula_trigger_check, analyze, run
from trace2skill.adapters import BenchmarkAdapter
from trace2skill.evaluation import _controls
from trace2skill.validate_submission import validate


class R18ValidationTests(unittest.TestCase):
    def test_paired_gate_requires_both_repeats_and_no_regression(self):
        tasks = ["one", "two", "three", "four"]
        plan = {"holdout_task_ids": tasks, "policy": {"min_complete_repeats": 2,
                "max_input_ratio_d_over_b": 1.5, "max_duration_ratio_d_over_b": 1.5,
                "min_distinct_improved_tasks_vs_b_and_c": 2}}
        def row(task, reward):
            return {"task_id": task, "reward": reward, "input_tokens": 10,
                    "output_tokens": 2, "duration_sec": 1, "pi_tool_calls": 1}
        runs = {}
        for repeat in (0, 1):
            for arm in ("A", "B", "C", "D"):
                rows = [row(task, int(task in {"one", "two"} if arm == "D" else task == "one")) for task in tasks]
                runs[f"repeat{repeat}-{arm}"] = {"status": "valid", "attempts": [{"rows": rows, "elapsed_sec": 3}]}
        result = analyze(plan, {"runs": runs})
        self.assertEqual(result["status"], "needs_more_evidence")
        self.assertEqual(result["per_task_success_frequency"]["two"]["D"]["passes"], 2)
        self.assertEqual(result["mean_reward_delta_d_minus_b"], .25)
        runs["repeat1-D"]["attempts"][0]["rows"][0]["reward"] = 0
        self.assertEqual(analyze(plan, {"runs": runs})["status"], "regression_risk")
        del runs["repeat1-A"]
        partial = analyze(plan, {"runs": runs, "stopped_reason": "prespecified_budget"})
        self.assertEqual(partial["status"], "incomplete")
        self.assertFalse(partial["eligible_for_val"])

    def test_unknown_adapter_command_is_configurable(self):
        adapter = BenchmarkAdapter("custom_domain")
        command = adapter.get_evaluation_command(Path("/tmp/kit"), "val", skills_dir=None,
                                                  jobs_dir=Path("/tmp/jobs"), max_tasks=2)
        self.assertIn("custom_domain-val", command)
        self.assertNotIn("--skills-dir", command)

    def test_submission_validation_fails_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = validate(Path(directory))
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any(x.startswith("missing:") for x in result["error"]))

    def test_r18_resume_rejects_plan_hash_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "results/r18/spreadsheetbench/runs_manifest.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"plan_sha256": "old", "runs": {}}))
            with patch("skillrsi_loop.r18_validation.freeze", return_value={"plan_sha256": "new"}), \
                 patch("skillrsi_loop.r18_validation.freeze_val", return_value={}):
                with self.assertRaisesRegex(ValueError, "resume plan mismatch"):
                    run(root)

    def test_control_rejects_changed_candidate_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = root / "SKILL.md"
            skill.write_text("---\nname: example\ndescription: Test\n---\n")
            with self.assertRaisesRegex(ValueError, "SHA changed"):
                _controls({"benchmark": "other", "skill_path": str(skill), "sha256": "0" * 64}, root / "controls")

    def test_rule_trigger_and_readback_have_event_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            agent = trial / "agent"
            agent.mkdir()
            trace = {"steps": [{"step_id": 1, "source": "agent", "message": "",
                                "tool_calls": [{"tool_call_id": "c1", "function_name": "bash",
                                                "arguments": {"command": "formula = f\"=A1+B1\""}}]},
                               {"step_id": 2, "source": "agent", "message": "",
                                "tool_calls": [{"tool_call_id": "c2", "function_name": "bash",
                                                "arguments": {"command": "load_workbook(path, data_only=False); print(cell.value)"}}]}]}
            (agent / "trajectory.json").write_text(json.dumps(trace))
            item = _formula_trigger_check({"task_id": "x", "trial_path": str(trial)})
            self.assertEqual(item["trigger_event_ids"], ["s1.call1"])
            self.assertEqual(item["readback_event_ids"], ["s2.call1"])
            self.assertTrue(item["behavior_consistent"])


if __name__ == "__main__":
    unittest.main()
