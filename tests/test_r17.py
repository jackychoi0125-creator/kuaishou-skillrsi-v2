import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillrsi_loop.r17_behavior import align_trial, normalize_command
from skillrsi_loop.r17_generate import run_one
from skillrsi_loop.r17_funnel import _fresh_generation
from skillrsi_loop.r17_refine import apply_updates
from skillrsi_loop.r17_plan import build as build_r17_plan
from skillrsi_loop.r17_source import audit as audit_source, contains_task_id
from skillrsi_loop.stability import file_sha
from skillrsi_m3.core import EvidenceVerifier, InsightExtractor, write_json
from skillrsi_m3.math_verification import classify_math_insight, verify_cited_equality
from skillrsi_m4.core import compile_single_rule, validate_skill


class _Store:
    profiles = {"t": {"trace_sha256": "sha"}}

    def get(self, trace_id, event_id):
        if (trace_id, event_id) != ("t", "s1.call1"):
            raise KeyError(event_id)
        return {"id": event_id, "kind": "tool_call", "text": "checked workbook"}

    def ref(self, trace_id, event_ids):
        return {"trace_id": trace_id, "event_ids": event_ids, "trace_sha256": "sha"}


class _Replay:
    def ask(self, *_args, **_kwargs):
        return {"insights": [{"source_group_id": "g", "principle": "Inspect the target range",
                "recommended_behavior": "Check the target range before editing", "trigger_conditions": ["A cell range is specified"],
                "avoid_conditions": [], "expected_observation": "The inspection precedes the edit",
                "supporting_refs": [{"trace_id": "t", "event_id": "s1.call1"}]}]}, "cache-replay"


class R17Tests(unittest.TestCase):
    def test_singular_event_id_is_normalized_without_fake_evidence(self):
        group = {"group_id": "g", "kind": "positive", "benchmark": "spreadsheetbench",
                 "task_family": "cell", "trace_ids": ["t"], "evidence_events": ["t:s1.call1"],
                 "quality": {"reuse_readiness": "needs_review"}}
        diagnostics = []
        with patch("skillrsi_m3.core._evidence_payload", return_value={"events": [{"trace_id": "t", "event_id": "s1.call1"}]}):
            insights = InsightExtractor(_Store(), _Replay()).extract([group], diagnostics)
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0]["supporting_refs"][0]["event_ids"], ["s1.call1"])
        self.assertEqual(diagnostics[0]["normalized_event_id_refs"], 1)

    def test_official_step_alignment_deduplicates_poll_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            agent = trial / "agent"
            agent.mkdir()
            answers = trial / "artifacts/bench/answers/train:0001"
            answers.mkdir(parents=True)
            transcript = answers / "trajectory.jsonl"
            transcript.write_text('{"command":"look"}\n{"command":"go to shelf 1"}\n')
            def step(index, command, observed, count):
                return {"step_id": index, "source": "agent", "tool_calls": [{"tool_call_id": f"c{index}",
                    "function_name": "terminal.exec", "arguments": {"command": f'bridge.sh send train:0001 "{command}"'}}],
                    "observation": {"results": [{"source_call_id": f"c{index}", "content": json.dumps({
                        "observation": observed, "admissible_commands": ["go to shelf 1"], "steps": count})}]}}
            trace = {"steps": [step(1, "look", "Nothing happens.", 1),
                               step(2, "go to shelf 1", "At shelf.", 2),
                               {"step_id": 3, "source": "agent", "tool_calls": [{"tool_call_id": "poll",
                                 "function_name": "terminal.exec", "arguments": {"command": "bridge.sh status train:0001"}}],
                                 "observation": {"results": [{"source_call_id": "poll", "content": json.dumps({
                                     "observation": "Nothing happens.", "steps": 2})}]}}]}
            path = agent / "trajectory.json"
            path.write_text(json.dumps(trace))
            row = {"task_id": "train__0001", "trial_path": str(trial), "reward": 1,
                   "trace_sha256": file_sha(path), "input_tokens": 100, "output_tokens": 10,
                   "duration_sec": 1, "pi_tool_calls": 3, "skill_read_event_ids": [],
                   "alfworld": {"transcript_sha256": file_sha(transcript), "official_executed_steps": 2,
                                "bridge_status_tool_calls": 1, "observed_no_progress_event_ids": ["s1.obs1", "s3.obs1"]}}
            result = align_trial(row)
            self.assertEqual(result["unique_no_progress_commands"], 1)
            self.assertEqual(result["old_no_progress_observation_events"], 2)
            self.assertEqual(len(result["triggers"]), 1)
            self.assertTrue(result["triggers"][0]["rule_behavior_observed"])
            self.assertEqual(result["triggers"][0]["trigger_event_ids"], ["s1.call1", "s1.obs1"])
            self.assertEqual(normalize_command(" GO   to  Shelf 1 "), "go to shelf 1")

    def test_math_verifier_requires_literal_original_event(self):
        class Store:
            def get(self, trace_id, event_id):
                return {"text": "Check: 2+3=5, then continue."}
        verified = verify_cited_equality(Store(), "t", "s1.message", "2+3=5")
        self.assertEqual(verified["status"], "mathematically_verified")
        self.assertEqual(classify_math_insight({"insight_id": "i"}, [verified])["level"], "mathematically_verified")
        self.assertEqual(verify_cited_equality(Store(), "t", "s1.message", "2+3=6")["status"], "unverifiable")
        self.assertEqual(verify_cited_equality(Store(), "t", "s1.message", "__import__('os')=0")["status"], "unverifiable")

    def test_cost_bound_and_overbroad_trigger_are_static_gates(self):
        text = ("---\nname: tiny-rule\ndescription: conditional\n---\n# When to use\n- On every task\n"
                "# Procedure\n- Repeat until done\n# Avoid\n- Stop otherwise\n# Validation\n- Observe result\n")
        errors = validate_skill(text, "spreadsheetbench", [{"benchmark": "spreadsheetbench"}])
        self.assertIn("unbounded_repeat_check", errors)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            insight = {"insight_id": "i", "benchmark": "spreadsheetbench", "status": "verified_observation",
                       "evidence_validity": "verified", "utility_status": "unverified", "measurement_specificity": True,
                       "estimated_cost_risk": "low", "trigger_conditions": ["A target formula is specified"],
                       "avoid_conditions": [], "expected_behavior_change": "Inspect the target formula",
                       "expected_observation": "The target cell is inspected", "supporting_refs": [{"trace_id": "t"}],
                       "counterexample_refs": [], "risk_level": "low"}
            path = root / "insights.jsonl"
            path.write_text(json.dumps(insight) + "\n")
            self.assertEqual(compile_single_rule("spreadsheetbench", path, root / "skills", root / "out",
                                                 dry_run=True)["status"], "no_candidate")

    def test_frozen_budget_stops_before_kimi_client_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "results/r17/spreadsheetbench/m3"
            base.mkdir(parents=True)
            write_json(base / "selected_evidence.json", [{"events": [{"text": "x" * 5000}]}])
            plan = {"policy": {"max_new_kimi_total_tokens": 100, "max_new_kimi_calls_total": 4,
                               "max_new_kimi_calls_per_benchmark": 2}}
            with patch("skillrsi_loop.r17_generate.freeze", return_value=plan), \
                 patch("skillrsi_loop.r17_generate.audit", return_value={"status": "clean"}), \
                 patch("skillrsi_loop.r17_generate.run_m3", return_value={}), \
                 patch("skillrsi_loop.r17_generate.KimiClient", side_effect=AssertionError("no call")):
                result = run_one(root, "spreadsheetbench")
            self.assertEqual(result["status"], "budget_stopped")
            self.assertEqual(result["new_model_calls"], 0)

    def test_source_audit_rejects_holdout_group_and_cache_leak(self):
        self.assertFalse(contains_task_id('"livemath:lm202602-27"', 'lm202602-2'))
        self.assertTrue(contains_task_id('"livemath:lm202602-2"', 'lm202602-2'))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "results/r17/alfworld"
            m2 = base / "m2"
            m2.mkdir(parents=True)
            trace = base / "trace.json"
            trace.write_text('{"steps":[]}')
            source_file = base / "source_validated.jsonl"
            source_file.write_text(json.dumps({"task_id": "source", "benchmark": "alfworld", "split": "train"}) + "\n")
            plan = {"plan_sha256": "p", "studies": {"alfworld": {
                "source_train_task_ids": ["source"], "future_holdout_task_ids": ["holdout"],
                "consumed_r16_holdout_task_ids": ["old"], "source_sha256": "s", "future_holdout_sha256": "h"}}}
            write_json(base / "source_provenance.json", {"plan_sha256": "p",
                       "source_validated_sha256": file_sha(source_file), "cache_initially_empty": True})
            (m2 / "trace_profiles.jsonl").write_text(json.dumps({"trace_id": "alfworld:source", "task_id": "source",
                "split": "train", "source_trace_path": str(trace), "trace_sha256": file_sha(trace)}) + "\n")
            for name in ("positive_strategy_groups.json", "failure_pattern_groups.json", "contrast_groups.json"):
                write_json(m2 / name, [])
            with patch("skillrsi_loop.r17_source.train_metrics", return_value={x: {} for x in ("source", "holdout", "old")}):
                self.assertEqual(audit_source(root, "alfworld", plan)["status"], "clean")
                write_json(m2 / "failure_pattern_groups.json", [{"trace_ids": ["alfworld:holdout"]}])
                self.assertIn("group_holdout_or_unknown_trace", audit_source(root, "alfworld", plan)["errors"])
                write_json(m2 / "failure_pattern_groups.json", [])
                cache = base / "isolated_kimi_cache"
                cache.mkdir()
                (cache / "request.json").write_text('"holdout"')
                self.assertIn("cache_contains_holdout_id", audit_source(root, "alfworld", plan)["errors"])
                (cache / "request.json").write_text('{}')
                m3 = base / "m3"
                m3.mkdir()
                (m3 / "insights.jsonl").write_text(json.dumps({"supporting_refs": [{"trace_id": "alfworld:source",
                    "trace_sha256": file_sha(trace), "event_ids": ["missing.event"]}], "counterexample_refs": []}) + "\n")
                self.assertIn("m3_reference_event_missing", audit_source(root, "alfworld", plan)["errors"])

    def test_future_holdout_is_frozen_from_train_metadata_and_excludes_used_r16_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = {"studies": {b: {"holdout_task_ids": ["t0", "t1", "t2", "t3"]}
                               for b in ("spreadsheetbench", "alfworld", "livemath")}}
            write_json(root / "results/r16/EXPERIMENT_PLAN.json", old)
            train = {f"t{i}": {"reward": "1" if i % 2 == 0 else "0", "task_type": f"type{i%3}"}
                     for i in range(20)}
            with patch("skillrsi_loop.r17_plan.train_metrics", return_value=train):
                plan = build_r17_plan(root)
            for study in plan["studies"].values():
                source = set(study["source_train_task_ids"])
                future = set(study["future_holdout_task_ids"])
                used = set(study["consumed_r16_holdout_task_ids"])
                self.assertEqual(len(future), 4)
                self.assertFalse(source & future or source & used or future & used)
                self.assertEqual(source | future | used, set(train))

    def test_refinement_requires_per_insight_measurement_and_bound(self):
        source = [{"insight_id": "i", "benchmark": "spreadsheetbench", "utility_status": "unverified"}]
        malformed = {"updates": [{"insight_id": "i"}], "expected_observation": "wrong top-level location"}
        _, rejected = apply_updates(source, malformed)
        self.assertEqual(rejected[0]["reasons"], ["missing_per_insight_fields"])
        response = {"updates": [{"insight_id": "i", "observed_problem": "A tool error is visible",
            "expected_behavior_change": "Inspect the failed command output once",
            "expected_observation": "One error-reading tool call precedes the next edit",
            "estimated_cost_risk": "low", "max_extra_tool_actions": 1,
            "max_additional_checks": 1, "stop_condition": "the local error is identified"}]}
        updated, accepted = apply_updates(source, response)
        self.assertEqual(accepted[0]["status"], "operationalized_hypothesis")
        self.assertTrue(updated[0]["measurement_specificity"])
        self.assertEqual(updated[0]["cost_bound"]["max_extra_tool_actions"], 1)
        self.assertEqual(updated[0]["utility_status"], "unverified")

    def test_inverted_avoid_is_moved_to_prohibited_behavior_with_provenance(self):
        insight = {"insight_id": "i", "benchmark": "spreadsheetbench", "supporting_refs": [],
                   "counterexample_refs": [], "trigger_conditions": ["Formula task"],
                   "avoid_conditions": ["Agent writes formulas without checking output"],
                   "causal_status": "hypothesis_only", "status": "candidate"}
        EvidenceVerifier(_Store()).verify([insight])
        self.assertEqual(insight["avoid_conditions"][0], "The stated trigger does not apply")
        self.assertEqual(insight["prohibited_behavior"], ["Agent writes formulas without checking output"])
        self.assertEqual(insight["applicability_normalization"], "bad_behavior_moved_out_of_exclusions")

    def test_funnel_explains_raw_candidate_removed_by_cost_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "results/r17/spreadsheetbench"
            for directory in (base / "m2", base / "m3", base / "isolated_kimi_cache"):
                directory.mkdir(parents=True)
            for name in ("positive_strategy_groups.json", "failure_pattern_groups.json", "contrast_groups.json"):
                write_json(base / "m2" / name, [])
            write_json(base / "m3/selected_evidence.json", [{"kind": "positive", "events": [{"event_id": "e"}]}])
            write_json(base / "m3/extraction_diagnostics.json", [{"raw_index": 0,
                       "stage": "extractor_accepted", "insight_id": "i", "reasons": []}])
            (base / "m3/insights.jsonl").write_text(json.dumps({"insight_id": "i", "benchmark": "spreadsheetbench",
                "status": "verified_observation", "trigger_conditions": ["Formula task"], "avoid_conditions": [],
                "risk_level": "low", "measurement_specificity": True}) + "\n")
            write_json(base / "m3/insight_audit.json", {"verified": 1, "pending": 0,
                "details": [{"insight_id": "i", "errors": [], "semantic_validation": "not_independently_checked"}]})
            (base / "m3/insight_operations.jsonl").write_text('{"operation":"ADD"}\n')
            write_json(base / "m3/usage.json", {"api_calls": 1})
            write_json(base / "isolated_kimi_cache/extract.json", {"usage": {"stage": "m3_extract"},
                "response": {"insights": [{"source_group_id": "g", "principle": "p",
                  "recommended_behavior": "b", "supporting_refs": []}]}})
            value, rejected = _fresh_generation(root, "spreadsheetbench")
            self.assertEqual(value["stages"]["llm_raw_insights"], 1)
            self.assertEqual(value["stages"]["independently_semantic_validated"], 0)
            self.assertIn("numeric_cost_bound_missing", rejected[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
