import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillrsi_loop.crossfit import audit_bundle
from skillrsi_loop.exposure import _advertisement, audit_trial
from skillrsi_loop.r16_plan import _stratified_holdout
from skillrsi_loop.rule_selection import paired_frequency_signal, select_one
from skillrsi_loop.stability import digest, file_sha
from skillrsi_m3.core import EvidenceVerifier, TraceStore, write_json
from skillrsi_m4.core import compile_single_rule
from skillrsi_loop.presence_control import build as build_presence_controls
from skillrsi_loop.holdout import find_sampling_seed, run as run_holdout
from skillrsi_loop.holdout_analysis import analyze as analyze_holdout


class R16Tests(unittest.TestCase):
    def test_frozen_holdout_is_stratified_and_disjoint(self):
        rows = {f"t{i}": {"reward": "1" if i < 4 else "0", "task_type": "family" + str(i % 2),
                           "input_tokens": str(i * 100), "tool_calls": str(i)} for i in range(8)}
        holdout = _stratified_holdout(rows, "spreadsheetbench")
        self.assertEqual(len(holdout), 4)
        self.assertEqual(sum(float(rows[t]["reward"]) for t in holdout), 2)
        self.assertFalse(set(holdout) & (set(rows) - set(holdout)))

    def test_exact_kit_sample_seed_targets_frozen_ids(self):
        ordered = [f"t{i}" for i in range(9)]
        import random
        target = set(random.Random(7).sample(ordered, 4))
        seed = find_sampling_seed(ordered, target, limit=10000)
        self.assertIsNotNone(seed)
        self.assertEqual(set(random.Random(seed).sample(ordered, 4)), target)

    def test_frequency_signal_reports_ceiling_without_impossible_paired_wins(self):
        pairs = [{"task_id": task, "repeat": repeat, "A_reward": 0, "B_reward": 1, "D1_reward": 1}
                 for task in ("a", "b") for repeat in (0, 1)]
        signal = paired_frequency_signal(pairs)
        self.assertTrue(signal["structural_ceiling"])
        self.assertEqual(signal["tasks_with_b_headroom"], 0)
        self.assertEqual(signal["effective_task_count"], 2)
        self.assertEqual(signal["mean_rule_delta_by_task"], 0)
        pairs[0]["B_reward"] = 0
        signal = paired_frequency_signal(pairs)
        self.assertIn("a", signal["improved_task_ids"])
        self.assertEqual(signal["effective_task_count"], 2)

    def test_selector_uses_frozen_holdout_not_aa_changed_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "results/stability/alfworld"
            base.mkdir(parents=True)
            (base / "arms_manifest.json").write_text(json.dumps({"arms": {"D1": {"insight_ids": ["i"]}}}))
            (base / "stability_summary.json").write_text(json.dumps({"status": "incomplete",
                "a_a_variation": [{"task_id": "aa_changed", "changed": True}],
                "effects": {"rule_effect": {"paired_mean_reward_delta": None,
                    "task_cluster_bootstrap_95pct_descriptive": None},
                    "presence_effect": {"paired_mean_reward_delta": None}}}))
            plan_path = root / "results/r16/EXPERIMENT_PLAN.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps({"studies": {"alfworld": {"holdout_task_ids": ["frozen_holdout"]}}}))
            policy = {"min_complete_repeats": 2, "min_distinct_improved_tasks_vs_B": 2,
                      "min_mean_reward_delta_vs_B": .05, "max_input_token_ratio_vs_B": 1.5,
                      "max_duration_ratio_vs_B": 1.5, "high_cost_input_ratio": 2.0,
                      "policy_sha256": "p"}
            with patch("skillrsi_loop.rule_selection.crossfit_audit", return_value={"status": "contaminated"}) as audit:
                select_one(root, "alfworld", policy)
            self.assertEqual(audit.call_args.args[2], ["frozen_holdout"])

    def test_one_repeat_cannot_promote_even_with_apparent_gain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "results/r16/alfworld"
            base.mkdir(parents=True)
            write_json(base / "train_internal_validation.json", {"status": "incomplete", "complete_repeats": [0]})
            rows = []
            for task in ("t1", "t2"):
                rows.append({"task_id": task, "repeat": 0,
                             **{arm: {"reward": 1 if arm == "D" else 0, "input_tokens": 10,
                                      "output_tokens": 1, "duration_sec": 1,
                                      "trial_path": str(base / "missing")} for arm in "ABCD"}})
            write_json(base / "holdout_paired.json", rows)
            write_json(base / "controls/controls_manifest.json", {"arms": {arm: {} for arm in "ABCD"}})
            write_json(base / "holdout_runs_manifest.json", {"runs": {
                f"repeat0-{arm}": {"attempts": [{"rows": []}]} for arm in "ABCD"}})
            policy = {"min_repeats_for_promotion": 2, "min_distinct_improved_tasks": 2,
                      "max_distinct_regressed_tasks": 0, "max_input_ratio_D_over_B": 1.5,
                      "max_duration_ratio_D_over_B": 1.5}
            with patch("skillrsi_loop.holdout_analysis.freeze", return_value={"policy": policy}), \
                 patch("skillrsi_loop.holdout_analysis.audit_bundle", return_value={"status": "clean"}), \
                 patch("skillrsi_loop.holdout_analysis.audit_trial", return_value={
                     "configured": True, "advertised": "unknown", "explicitly_read": False,
                     "read_event_ids": [], "behavior_consistent": {"state": "unknown"}}):
                result = analyze_holdout(root, "alfworld")
            self.assertFalse(result["candidate_eligible_for_val"])
            self.assertIn("fewer_than_prespecified_complete_repeats", result["reason_codes"])
            self.assertEqual(result["descriptive_observed_signal"]["improved_task_ids"], ["t1", "t2"])
            self.assertEqual(result["promotion_signal"]["effective_task_count"], 0)

    def test_no_candidate_skips_harbor_without_kit_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "results/r16/livemath/m4/candidate_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "no_candidate"}))
            with patch("skillrsi_loop.holdout.freeze_run_plan", side_effect=AssertionError("must not run")):
                result = run_holdout(root, "livemath")
            self.assertEqual(result["status"], "skipped_no_candidate")
            self.assertEqual(result["harbor_tasks"], 0)

    def test_exposure_levels_and_no_read_does_not_mean_no_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "t__run"
            (trial / "agent").mkdir(parents=True)
            trace = {"steps": [
                {"step_id": 1, "source": "agent", "message": "Available skill: sample-skill"},
                {"step_id": 2, "source": "agent", "message": "Check the result", "tool_calls": [
                    {"function_name": "terminal.exec", "arguments": {"command": "python check.py"}}]}]}
            (trial / "agent/trajectory.json").write_text(json.dumps(trace))
            (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1}}}))
            record = audit_trial(trial, "spreadsheetbench", configured=True, arm="D", skill_name="sample-skill")
            self.assertTrue(record["configured"])
            self.assertEqual(record["advertised"], "observed")
            self.assertFalse(record["explicitly_read"])
            self.assertEqual(record["causal_application"], "unknown")
            self.assertEqual(_advertisement([], "sample-skill"), [])

    def test_insight_evidence_validity_is_distinct_from_utility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = root / "trace.json"
            trace.write_text(json.dumps({"steps": [{"step_id": 1, "message": "checked feedback"}]}))
            profile = {"trace_id": "alfworld:t", "benchmark": "alfworld", "split": "train",
                       "eligible_for_learning": True, "source_trace_path": str(trace), "trace_sha256": file_sha(trace)}
            (root / "profiles.jsonl").write_text(json.dumps(profile) + "\n")
            insight = {"insight_id": "i", "benchmark": "alfworld", "principle": "Check feedback",
                       "recommended_behavior": "Check feedback before repeating", "trigger_conditions": ["No progress"],
                       "avoid_conditions": [], "causal_status": "hypothesis_only",
                       "supporting_refs": [{"trace_id": "alfworld:t", "trace_sha256": file_sha(trace),
                                            "event_ids": ["s1.message"]}], "counterexample_refs": []}
            EvidenceVerifier(TraceStore(root / "profiles.jsonl", "alfworld")).verify([insight])
            self.assertEqual(insight["evidence_validity"], "verified")
            self.assertEqual(insight["utility_status"], "unverified")

    def test_single_rule_compiler_requires_measurable_conditions_and_direct_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "insights.jsonl"
            insight = {"insight_id": "ins_x", "benchmark": "alfworld", "status": "verified_observation",
                       "evidence_validity": "verified", "utility_status": "unverified", "measurement_specificity": True,
                       "estimated_cost_risk": "low", "trigger_conditions": ["An action returns no state change"],
                       "avoid_conditions": ["The action changed state"],
                       "expected_behavior_change": "Choose a different admissible action after no-progress feedback",
                       "expected_observation": "The next submitted action differs from the rejected action",
                       "supporting_refs": [{"trace_id": "alfworld:source", "event_ids": ["e"], "trace_sha256": "sha"}],
                       "counterexample_refs": [], "risk_level": "low"}
            path.write_text(json.dumps(insight) + "\n")
            dry = compile_single_rule("alfworld", path, root / "skills", root / "out", dry_run=True)
            self.assertEqual(dry["model_calls"], 0)
            self.assertFalse((root / "skills").exists())
            result = compile_single_rule("alfworld", path, root / "skills", root / "out")
            self.assertEqual(result["status"], "candidate_for_internal_validation")
            self.assertEqual(result["rule"]["mapping_status"], "direct_structured_insight")
            self.assertFalse(result["accepted"])
            insight["measurement_specificity"] = False
            path.write_text(json.dumps(insight) + "\n")
            self.assertEqual(compile_single_rule("alfworld", path, root / "skills", root / "out",
                                                 dry_run=True)["status"], "no_candidate")

    def test_matched_metadata_control_has_no_operation_instruction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            skill = root / "candidate/SKILL.md"
            skill.parent.mkdir()
            skill.write_text("---\nname: alfworld-r16\ndescription: Read feedback when action fails.\n---\n"
                             "# When to use\n- No progress\n# Procedure\n- Choose a different action\n"
                             "# Avoid\n- Do not repeat\n# Validation\n- Command differs\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"status": "candidate_for_internal_validation", "accepted": False,
                                            "benchmark": "alfworld", "skill_path": str(skill)}))
            controls = build_presence_controls(manifest, root / "controls")
            c = Path(controls["arms"]["C"]["skill_path"]).read_text()
            d = Path(controls["arms"]["D"]["skill_path"]).read_text()
            self.assertEqual(c.split("---", 2)[:2], d.split("---", 2)[:2])
            self.assertNotIn("Choose a different action", c)
            self.assertIn("Choose a different action", d)

    def test_crossfit_audit_detects_group_insight_prompt_and_cache_leaks(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "repo"
            base = root / "results/r16/alfworld"
            m2, m3, m4, cache = (base / name for name in ("m2", "m3", "m4", "isolated_kimi_cache"))
            for path in (m2, m3, m4, cache, parent / "metrics"):
                path.mkdir(parents=True)
            (parent / "metrics/alfworld-baseline.csv").write_text("task_id,split\nsource,train\nholdout,train\n")
            trace = base / "trace.json"
            trace.write_text(json.dumps({"steps": [{"step_id": 1, "message": "checked"}]}))
            source_input = base / "source_validated.jsonl"
            source_input.write_text(json.dumps({"task_id": "source"}) + "\n")
            profile = {"trace_id": "alfworld:source", "benchmark": "alfworld", "task_id": "source",
                       "split": "train", "source_trace_path": str(trace), "trace_sha256": file_sha(trace)}
            (m2 / "trace_profiles.jsonl").write_text(json.dumps(profile) + "\n")
            for filename in ("positive_strategy_groups.json", "failure_pattern_groups.json", "contrast_groups.json"):
                (m2 / filename).write_text("[]")
            insight = {"insight_id": "i", "benchmark": "alfworld", "supporting_refs": [
                {"trace_id": "alfworld:source", "trace_sha256": file_sha(trace), "event_ids": ["s1.message"]}],
                "counterexample_refs": []}
            (m3 / "insights.jsonl").write_text(json.dumps(insight) + "\n")
            plan = {"plan_sha256": "p", "studies": {"alfworld": {"source_train_task_ids": ["source"],
                    "holdout_task_ids": ["holdout"], "source_sha256": digest(["source"]),
                    "holdout_sha256": digest(["holdout"])}}}
            write_json(base / "provenance.json", {"plan_sha256": "p", "source_train_task_ids": ["source"],
                      "holdout_task_ids": ["holdout"], "source_validated_path": str(source_input),
                      "source_validated_sha256": file_sha(source_input), "m2_dir": str(m2),
                      "m3_dir": str(m3), "m4_dir": str(m4), "cache_dir": str(cache),
                      "cache_initially_empty": True})
            self.assertEqual(audit_bundle(root, "alfworld", plan)["status"], "clean")
            (m2 / "failure_pattern_groups.json").write_text(json.dumps([{"trace_ids": ["alfworld:holdout"]}]))
            self.assertIn("m2_group_holdout_or_unknown_trace", audit_bundle(root, "alfworld", plan)["errors"])
            (m2 / "failure_pattern_groups.json").write_text("[]")
            insight["supporting_refs"][0]["trace_id"] = "alfworld:holdout"
            (m3 / "insights.jsonl").write_text(json.dumps(insight) + "\n")
            self.assertIn("m3_ref_holdout_or_unknown_trace", audit_bundle(root, "alfworld", plan)["errors"])
            insight["supporting_refs"][0]["trace_id"] = "alfworld:source"
            (m3 / "insights.jsonl").write_text(json.dumps(insight) + "\n")
            (m4 / "source_prompt.json").write_text("holdout")
            (cache / "request.json").write_text("holdout")
            errors = audit_bundle(root, "alfworld", plan)["errors"]
            self.assertIn("m4_prompt_or_manifest_contains_holdout_id", errors)
            self.assertIn("cache_contains_holdout_id", errors)


if __name__ == "__main__":
    unittest.main()
