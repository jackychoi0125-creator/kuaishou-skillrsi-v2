import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillrsi_loop.rule_selection import crossfit_audit, freeze_policy, select_one
from skillrsi_loop.stability import (_attempt_status, _budget_allows, _cluster_interval,
                                      _manifest, _recover_pending, _trial_rows, alfworld_counts,
                                      digest, run_benchmark, run_fingerprint, summarize_benchmark)


class StabilityTests(unittest.TestCase):
    def test_resume_plan_hash_and_skill_sha_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "runs_manifest.json"
            path.write_text(json.dumps({"plan_sha256": "original", "runs": {}}))
            with self.assertRaisesRegex(ValueError, "plan hash"):
                _manifest(path, "changed")
            out = root / "results/stability/alfworld"
            out.mkdir(parents=True)
            skill = out / "SKILL.md"
            skill.write_text("old")
            (out / "arms_manifest.json").write_text(json.dumps({"arms": {
                "A": {"skill_path": None, "skill_sha256": None},
                "B": {"skill_path": str(skill), "skill_sha256": "wrong"},
                "D1": {"skill_path": str(skill), "skill_sha256": "wrong"}}}))
            with self.assertRaisesRegex(ValueError, "Skill SHA"):
                run_benchmark(root, "alfworld", {"studies": {"alfworld": {}}}, dry_run=True)

    def test_resume_run_config_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "results/stability/livemath"
            out.mkdir(parents=True)
            (out / "arms_manifest.json").write_text(json.dumps({"arms": {
                "A": {"skill_path": None, "skill_sha256": None, "skills_dir": None},
                "B": {"skill_path": None, "skill_sha256": "b", "skills_dir": None},
                "D1": {"skill_path": None, "skill_sha256": "d", "skills_dir": None}}}))
            (out / "runs_manifest.json").write_text(json.dumps({"plan_sha256": "plan", "runs": {
                "repeat0-A": {"config": {"config_sha256": "old", "sampling_n": 1, "arm": "A"}, "status": "valid",
                              "attempts": [], "chosen_attempt": 0}}}))
            plan = {"plan_sha256": "plan", "budget": {
                "max_harbor_task_executions": 100, "per_benchmark_task_executions": 100,
                "max_input_tokens": 1000, "per_benchmark_input_tokens": 1000,
                "max_output_tokens": 1000, "max_wall_seconds": 1000,
                "per_arm_task_executions": 100},
                "studies": {"livemath": {"benchmark": "livemath", "task_ids": ["t"],
                "historical_strata": {"t": {"input_tokens": 1}}, "id_file": None}}}
            with patch("skillrsi_loop.stability.run_fingerprint", return_value={"config_sha256": "new"}):
                with self.assertRaisesRegex(ValueError, "config hash mismatch"):
                    run_benchmark(root, "livemath", plan, dry_run=False)

    def test_livemath_repeats_can_vary_seed_with_fixed_task_set(self):
        study = {"benchmark": "livemath", "task_ids": ["t1", "t2"],
                 "task_ids_sha256": digest(["t1", "t2"]), "sample_seed": 0}
        plan = {"plan_sha256": "plan", "budget": {"max_infra_retries_per_run": 1,
                "timeout_seconds": {"livemath": 900}}}
        fake_model = {"model_id": "model", "api": "openai", "base_url": "endpoint"}
        with patch("skillrsi_loop.stability.config", return_value=fake_model), \
             patch("skillrsi_loop.stability.file_sha", return_value="kit"), \
             patch("skillrsi_loop.stability._scorer_sha", return_value="scorer"):
            first = run_fingerprint(Path("/tmp/repo"), study, "A", 0, None, plan)
            second = run_fingerprint(Path("/tmp/repo"), study, "A", 1, None, plan)
        self.assertEqual(first["task_ids_sha256"], second["task_ids_sha256"])
        self.assertEqual((first["sample_seed"], second["sample_seed"]), (0, 1))

    def test_failure_reward_is_valid_and_infrastructure_error_only_retries(self):
        self.assertEqual(_attempt_status({"timed_out": False, "exit_code": 0,
                                          "rows": [{"task_id": "t", "reward": 0.0}], "errors": []}, {"t"}), "valid")
        self.assertEqual(_attempt_status({"timed_out": False, "exit_code": 4,
                                          "rows": [{"task_id": "t"}],
                                          "errors": [{"classification": "infrastructure"}]}, {"t"}), "infrastructure_error")
        self.assertEqual(_attempt_status({"timed_out": False, "exit_code": 4,
                                          "rows": [{"task_id": "t"}],
                                          "errors": [{"classification": "unclassified_run_error"}]}, {"t"}), "unclassified_run_error")

    def test_official_reward_and_trace_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            (job / "summary.csv").write_text(
                "qid,reward,input_tokens,output_tokens,duration_sec,model_error,exception,note\n"
                "t,0.0,10,2,1,False,False,\n")
            trial = job / "t__abc"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}}))
            (trial / "agent/trajectory.json").write_text(json.dumps({"steps": [{"step_id": 1, "message": "done"}]}))
            rows, errors = _trial_rows(job, "livemath")
            self.assertEqual(rows[0]["reward"], 0)
            self.assertEqual(errors, [])
            (job / "summary.csv").write_text(
                "qid,reward,input_tokens,output_tokens,duration_sec,model_error,exception,note\n"
                "t,0.0,10,2,1,True,False,429 rate limit\n")
            _, errors = _trial_rows(job, "livemath")
            self.assertEqual(errors[0]["classification"], "infrastructure")
            self.assertEqual(errors[0]["task_id"], "t")
            (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
            with self.assertRaisesRegex(ValueError, "reward mismatch"):
                _trial_rows(job, "livemath")

    def test_interrupted_completed_harbor_run_is_recovered_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            kit = root.parent / "kits/livemath/pi-harbor-kit"
            job = kit / "workspace/runs/livemath-train/r1-stability-pilot/repeat0-A-attempt0/job"
            (job / "t__abc/agent").mkdir(parents=True)
            (job / "summary.csv").write_text(
                "qid,reward,input_tokens,output_tokens,duration_sec,model_error,exception,note\n"
                "t,1.0,10,2,1,False,False,\n")
            (job / "t__abc/result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
            (job / "t__abc/agent/trajectory.json").write_text(json.dumps({"steps": [{"step_id": 1, "message": "done"}]}))
            record = {"config": {"repeat": 0, "arm": "A"}, "status": "running",
                      "pending_attempt": 0, "attempts": [], "chosen_attempt": None}
            _recover_pending(root, "livemath", record, {"t"})
            self.assertEqual(record["status"], "valid")
            self.assertEqual(len(record["attempts"]), 1)
            self.assertTrue(record["attempts"][0]["recovered_after_controller_interruption"])

    def test_alfworld_counters_keep_bridge_and_official_steps_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            transcript = trial / "artifacts/bench/answers/x/trajectory.jsonl"
            transcript.parent.mkdir(parents=True)
            transcript.write_text('{"command":"look"}\n{"command":"look"}\n')
            verifier = trial / "verifier/metrics.json"
            verifier.parent.mkdir(parents=True)
            verifier.write_text(json.dumps({"n_commands": 2, "executed_steps": 1,
                                            "max_steps": 50, "won": False,
                                            "grading_revision": "official"}))
            events = [{"id": "a", "kind": "tool_call", "arguments": {"command": "bridge.sh start x"}, "text": ""},
                      {"id": "b", "kind": "tool_call", "arguments": {"command": "bridge.sh send x look"}, "text": ""},
                      {"id": "c", "kind": "tool_call", "arguments": {"command": "bridge.sh status x"}, "text": ""},
                      {"id": "d", "kind": "tool_call", "arguments": {"command": "poll"}, "text": ""},
                      {"id": "e", "kind": "observation", "text": "Nothing happens. Not a valid action."}]
            result = alfworld_counts(trial, events)
            self.assertEqual(result["pi_tool_calls"], 4)
            self.assertEqual(result["bridge_send_tool_calls"], 1)
            self.assertEqual(result["other_pi_tool_calls"], 1)
            self.assertEqual(result["submitted_commands"], 2)
            self.assertEqual(result["official_executed_steps"], 1)
            self.assertIsNone(result["accepted_admissible_actions"])
            self.assertEqual(result["observed_invalid_action_event_ids"], ["e"])

    def test_pairing_aa_presence_rule_effect_and_cluster_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "results/stability/livemath"
            out.mkdir(parents=True)
            tasks = ["t1", "t2"]
            plan = {"plan_sha256": "plan", "studies": {"livemath": {
                "task_ids": tasks, "task_ids_sha256": digest(tasks), "sample_seed": 0,
                "historical_strata": {"t1": {"historical_reward": 1}, "t2": {"historical_reward": 0}}}}}
            # A itself changes on t1; B improves t2; D1 offers no additional lift.
            rewards = {"A": [[1, 0], [0, 0]], "B": [[1, 1], [0, 1]],
                       "D1": [[1, 1], [0, 1]]}
            manifest = {"runs": {}, "stopped_reason": None}
            for repeat in (0, 1):
                for arm in ("A", "B", "D1"):
                    rows = [{"task_id": task, "reward": rewards[arm][repeat][index],
                             "input_tokens": 10, "output_tokens": 2, "duration_sec": 1,
                             "pi_tool_calls": 0, "skill_read_event_ids": [],
                             "skill_discovery_event_ids": [], "trial_path": f"/{task}"}
                            for index, task in enumerate(tasks)]
                    manifest["runs"][f"repeat{repeat}-{arm}"] = {
                        "config": {"sampling_n": 2, "arm": arm, "config_sha256": f"{arm}{repeat}"},
                        "status": "valid", "chosen_attempt": 0, "attempts": [{"rows": rows, "attempt": 0,
                        "status": "valid", "elapsed_sec": 1}]}
            summary = summarize_benchmark(root, "livemath", plan, manifest)
            self.assertTrue(summary["a_a_variation"][0]["changed"])
            self.assertEqual(summary["effects"]["rule_effect"]["paired_mean_reward_delta"], 0)
            self.assertGreater(summary["effects"]["presence_effect"]["paired_mean_reward_delta"], 0)
            self.assertEqual(summary["effects"]["presence_effect"]["n_distinct_tasks"], 2)
            with (out / "paired_repeats.csv").open() as source:
                self.assertIsNotNone(_cluster_interval(list(csv.DictReader(source)), "B", "A"))

    def test_budget_stops_before_incomplete_block(self):
        plan = {"budget": {"max_harbor_task_executions": 4,
                           "per_benchmark_task_executions": 4,
                           "max_input_tokens": 100, "per_benchmark_input_tokens": 100,
                           "max_output_tokens": 100, "max_wall_seconds": 100}}
        usage = {"harbor_task_executions": 2, "input_tokens": 0, "output_tokens": 0,
                 "wall_seconds": 0, "per_benchmark": {}}
        self.assertEqual(_budget_allows(plan, usage, "alfworld", 3, 0), (False, "max_harbor_task_executions"))

    def test_crossfit_and_no_candidate_skip_val(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m2 = root / "results/m2"
            m2.mkdir(parents=True)
            (m2 / "trace_profiles.jsonl").write_text(json.dumps({"trace_id": "x", "task_id": "heldout",
                "benchmark": "livemath", "split": "train", "eligible_for_learning": True}) + "\n")
            audit = crossfit_audit(root, "livemath", ["heldout"], ["insight"])
            self.assertEqual(audit["status"], "contaminated")
            base = root / "results/stability/livemath"
            base.mkdir(parents=True)
            (base / "arms_manifest.json").write_text(json.dumps({"arms": {"D1": {"insight_ids": ["insight"]}}}))
            (base / "stability_summary.json").write_text(json.dumps({"status": "incomplete",
                "a_a_variation": [], "effects": {"rule_effect": {"paired_mean_reward_delta": None,
                "task_cluster_bootstrap_95pct_descriptive": None},
                "presence_effect": {"paired_mean_reward_delta": None}}}))
            result = select_one(root, "livemath", {"min_complete_repeats": 2,
                "min_distinct_improved_tasks_vs_B": 2, "min_mean_reward_delta_vs_B": .05,
                "max_input_token_ratio_vs_B": 1.5, "max_duration_ratio_vs_B": 1.5,
                "high_cost_input_ratio": 2.0, "policy_sha256": "policy"})
            self.assertEqual(result["status"], "needs_more_evidence")
            self.assertFalse((root / "results/r2/livemath/val").exists())

    def test_clean_crossfit_bundle_requires_isolated_train_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "repo"
            base = root / "results/stability/crossfit/livemath"
            base.mkdir(parents=True)
            (parent / "metrics").mkdir()
            (parent / "metrics/livemath-baseline.csv").write_text(
                "task_id,split\nsource,train\nholdout,train\nval,val\n")
            profiles = base / "trace_profiles.jsonl"
            profiles.write_text(json.dumps({"trace_id": "source-trace", "benchmark": "livemath", "task_id": "source",
                                            "split": "train"}) + "\n")
            insights = base / "insights.jsonl"
            insights.write_text(json.dumps({"insight_id": "i", "status": "verified_observation",
                                            "supporting_refs": [{"trace_id": "source-trace"}],
                                            "counterexample_refs": []}) + "\n")
            cache = base / "fresh_cache"
            cache.mkdir()
            from skillrsi_loop.stability import file_sha
            meta = {"source_train_task_ids": ["source"], "holdout_task_ids": ["holdout"],
                    "m2_profiles_path": str(profiles), "m3_insights_path": str(insights),
                    "m2_profiles_sha256": file_sha(profiles), "m3_insights_sha256": file_sha(insights),
                    "cache_root": str(cache), "cache_initially_empty": True}
            (base / "provenance.json").write_text(json.dumps(meta))
            self.assertEqual(crossfit_audit(root, "livemath", ["holdout"], ["i"])["status"], "clean")
            profiles.write_text(profiles.read_text() + json.dumps({"trace_id": "leak", "benchmark": "livemath", "task_id": "holdout",
                                                               "split": "train"}) + "\n")
            self.assertEqual(crossfit_audit(root, "livemath", ["holdout"], ["i"])["status"], "unverifiable")


if __name__ == "__main__":
    unittest.main()
