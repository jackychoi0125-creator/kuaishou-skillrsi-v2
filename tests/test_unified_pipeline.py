import json
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillrsi_m1.core import events_from_trace, sha256
from skillrsi_m1.m2_pipeline import deep_contrast, make_contrasts, match_pair, run, select_l2
from skillrsi_m1.profiles import DIMENSIONS, profile_one
from skillrsi_m1.quality import audit_groups
from skillrsi_m1.evidence_audit import audit_views
from skillrsi_m1.reliability import run as validate_run, validate_one


def metric(task_id="task", reward="1.0", **extra):
    return {"task_id": task_id, "split": "train", "attempt": "baseline-v3", "reward": reward,
            "trace_path": "", "trial_path": "/missing", "model_error": "False",
            "agent_429_errors": "0", "execution_error": "False", "scoring_error": "False",
            "task_type": "pick_clean_then_place_in_recep", **extra}


def trace(message="First I will check the room, then move the object."):
    return {"steps": [{"step_id": 1, "source": "user", "message": "Clean the object."},
                      {"step_id": 2, "source": "agent", "message": message,
                       "tool_calls": [{"tool_call_id": "call-1", "function_name": "bash",
                                       "arguments": {"command": "look"}}],
                       "observation": {"results": [{"source_call_id": "call-1", "content": "Kitchen"}]}}]}


def profile(benchmark, trace_id, task_id, outcome, family, operations, goal="clean the object"):
    return {"trace_id": trace_id, "benchmark": benchmark, "task_id": task_id,
            "eligible_for_learning": True, "outcome": {"status": outcome, "metrics": {"trace_chars": "100"}},
            "task_signature": {"task_family": family, "task_subtype": "clean", "task_complexity": "small",
                               "required_operations": operations, "task_constraints": ["max_50_actions"]},
            "task_goal_excerpt": goal,
            "behavior_signature": {dimension: [{"label": "explicit_step_plan", "status": "observed",
                                                 "event_ids": ["s2.message"]}] for dimension in DIMENSIONS},
            "diagnosis": {"observed_symptoms": [], "confirmed_root_cause": "undetermined"}}


class UnifiedPipelineTests(unittest.TestCase):
    def test_success_with_model_error_and_reward_one_remains_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            source = trace()
            path.write_text(json.dumps(source))
            row = metric(trace_path=str(path), model_error="True", agent_429_errors="2")
            entry = {"case_id": "task", "source_trace_path": str(path),
                     "trace_sha256": sha256(path), "trace": source}
            record = validate_one("alfworld", row, entry)
            self.assertEqual(record["trace_validity"], "valid")
            self.assertEqual(record["task_outcome"], "success")
            self.assertIn("terminal_model_error", record["reliability_flags"])
            self.assertIn("observed_429", record["reliability_flags"])

    def test_missing_and_malformed_trace_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text("{}")
            row = metric(trace_path=str(path), reward="0")
            missing = validate_one("alfworld", row, None)
            self.assertEqual((missing["trace_validity"], missing["task_outcome"]), ("invalid", "uncertain"))
            malformed = validate_one("alfworld", row, {"case_id": "task", "source_trace_path": str(path),
                                                    "trace_sha256": sha256(path), "trace": {"steps": []}})
            self.assertEqual(malformed["trace_validity"], "invalid")

    def test_missing_reward_keeps_trace_valid_but_outcome_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            source = trace()
            path.write_text(json.dumps(source))
            row = metric(trace_path=str(path), reward="")
            entry = {"case_id": "task", "source_trace_path": str(path), "trace_sha256": sha256(path),
                     "trace": source}
            record = validate_one("alfworld", row, entry)
            self.assertEqual(record["trace_validity"], "valid")
            self.assertEqual(record["task_outcome"], "uncertain")
            self.assertIn("missing_or_invalid_reward", record["reliability_flags"])

    def test_success_and_failure_share_schema_and_event_grounded_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(trace()))
            base = {"benchmark": "alfworld", "task_id": "task", "trace_id": "alfworld:task",
                    "split": "train", "attempt": "baseline-v3", "source_trace_path": str(path),
                    "trace_sha256": sha256(path), "trace_validity": "valid", "reliability_flags": [],
                    "metric_trial_path": "/missing", "metrics": {"task_type": "pick_clean_then_place_in_recep"}}
            success, events = profile_one({**base, "task_outcome": "success", "reward": 1})
            failure, _ = profile_one({**base, "task_outcome": "failure", "reward": 0})
            self.assertEqual(set(success["task_signature"]), set(failure["task_signature"]))
            self.assertEqual(set(success["behavior_signature"]), set(DIMENSIONS))
            self.assertTrue(success["eligible_for_learning"])
            observed = [label for values in success["behavior_signature"].values() for label in values
                        if label["status"] == "observed"]
            self.assertTrue(observed)
            self.assertTrue(all(set(label["event_ids"]).issubset({event["id"] for event in events}) for label in observed))
            self.assertEqual(failure["diagnosis"]["confirmed_root_cause"], "undetermined")

    def test_matching_grades_and_rejection(self):
        positive = profile("alfworld", "alfworld:p", "p", "success", "multi_stage_processing", ["pick", "clean", "place"])
        negative = profile("alfworld", "alfworld:n", "n", "failure", "multi_stage_processing", ["pick", "clean", "place"])
        self.assertEqual(match_pair(positive, negative)["grade"], "B")
        same_task = {**negative, "task_id": "p"}
        self.assertEqual(match_pair(positive, same_task)["grade"], "A")
        unrelated = profile("alfworld", "alfworld:u", "u", "failure", "simple_pick_place", ["look"], "heat the kettle")
        self.assertIsNone(match_pair(positive, unrelated)["grade"])
        math_failure = profile("livemath", "livemath:m", "m", "failure", "topology", ["derive_answer"], "homology of spaces")
        pairs, unmatched = make_contrasts([positive, math_failure])
        self.assertEqual(pairs, [])
        self.assertEqual(unmatched[0]["status"], "no_matched_positive")
        math_success = profile("livemath", "livemath:s", "s", "success", "combinatorics",
                               ["derive_answer", "select_option"], "complete graph red blue coloring Ramsey theorem")
        different_graph_problem = profile("livemath", "livemath:f", "f", "failure", "combinatorics",
                                          ["derive_answer", "select_option"], "maximal graph minor free clique sum structure")
        self.assertEqual(match_pair(math_success, different_graph_problem)["reason"],
                         "math_problem_content_not_comparable")

    def test_singleton_group_cannot_be_promoted_as_general_rule(self):
        p = profile("alfworld", "alfworld:p", "p", "success", "multi_stage_processing", ["pick", "clean"])
        group = {"group_id": "positive_test", "benchmark": "alfworld", "task_family": "multi_stage_processing",
                 "behavior_pattern": ["action:tool_calls_executed"], "trace_ids": [p["trace_id"]],
                 "evidence_events": ["alfworld:p:s2.message"]}
        result = audit_groups([p], [group], [], set())
        self.assertEqual(result["groups"][0]["reuse_readiness"], "single_case_hypothesis")
        self.assertEqual(result["groups"][0]["causal_status"], "hypothesis_only")

    def test_evidence_audit_detects_missing_contrast_event(self):
        p = profile("alfworld", "alfworld:p", "p", "success", "multi_stage_processing", ["pick", "clean"])
        n = profile("alfworld", "alfworld:n", "n", "failure", "multi_stage_processing", ["pick", "clean"])
        for item in (p, n):
            item["diagnosis"].update({"critical_events": [], "root_cause_hypotheses": []})
        events = {p["trace_id"]: [{"id": "s1.message", "step": 1, "kind": "message", "text": "act"}],
                  n["trace_id"]: [{"id": "s1.message", "step": 1, "kind": "message", "text": "stop"}]}
        views = [{"trace_id": p["trace_id"], "events": [{"event_id": "s1.message", "step_id": 1,
                  "kind": "message", "content": "act", "compressed": False, "full_chars": 3}]},
                 {"trace_id": n["trace_id"], "events": []}]
        comparison = {"positive_trace_id": p["trace_id"], "negative_trace_id": n["trace_id"],
                      "contrast_id": "pair", "differences": [{"positive_event_ids": ["s1.message"],
                      "negative_event_ids": ["s1.message"]}]}
        result = audit_views([p, n], events, views, [comparison])
        self.assertEqual(result["summary"]["contrast_difference_event_retention"], 0)

    def test_l2_selects_positive_negative_and_pair(self):
        p = profile("alfworld", "alfworld:p", "p", "success", "multi_stage_processing", ["pick", "clean"])
        n = profile("alfworld", "alfworld:n", "n", "failure", "multi_stage_processing", ["pick", "clean"])
        pair = {"contrast_id": "pair", "positive_trace_id": p["trace_id"],
                "negative_trace_id": n["trace_id"], "matching_grade": "B", "match_score": .9}
        selected, pair_ids = select_l2([p, n], [pair], 2, 1, 1, 1, 1000)
        self.assertEqual(set(selected), {p["trace_id"], n["trace_id"]})
        self.assertEqual(pair_ids, ["pair"])

    def test_l2_keeps_math_success_and_failure_without_forced_pair(self):
        sheet_p = profile("spreadsheetbench", "spreadsheetbench:p", "p", "success", "cell_level", ["formula"])
        sheet_n = profile("spreadsheetbench", "spreadsheetbench:n", "n", "failure", "cell_level", ["formula"])
        math_p = profile("livemath", "livemath:p", "p", "success", "combinatorics", ["derive_answer"],
                         "Ramsey complete graph coloring")
        math_n = profile("livemath", "livemath:n", "n", "failure", "combinatorics", ["derive_answer"],
                         "maximal minor free graph structure")
        pair = {"contrast_id": "sheet-pair", "positive_trace_id": sheet_p["trace_id"],
                "negative_trace_id": sheet_n["trace_id"], "matching_grade": "C", "match_score": .6}
        selected, pair_ids = select_l2([sheet_p, sheet_n, math_p, math_n], [pair], 4, 2, 2, 2, 5000)
        self.assertEqual(set(selected), {item["trace_id"] for item in (sheet_p, sheet_n, math_p, math_n)})
        self.assertEqual(pair_ids, ["sheet-pair"])

    def test_deep_contrast_cites_divergent_action_events(self):
        p = profile("alfworld", "alfworld:p", "p", "success", "multi_stage_processing", ["pick", "clean"])
        n = profile("alfworld", "alfworld:n", "n", "failure", "multi_stage_processing", ["pick", "clean"])
        def action_trace(action):
            return {"steps": [{"step_id": 1, "source": "agent", "message": "",
                               "tool_calls": [{"tool_call_id": "c1", "function_name": "bash",
                                               "arguments": {"command": f'bridge.sh send train:1 "{action}"'}}],
                               "observation": {"results": [{"source_call_id": "c1", "content": json.dumps({
                                   "observation": "room", "steps": 1, "done": False, "won": False})}]}}]}
        pair = {"contrast_id": "pair", "matching_grade": "B", "shared_constraints": ["max_50_actions"],
                "common_behaviors": [], "behavior_differences": []}
        comparison = deep_contrast(pair, p, n, events_from_trace(action_trace("look")),
                                   events_from_trace(action_trace("open cabinet 1")))
        difference = next(item for item in comparison["differences"] if item["behavior"] == "action_sequence_diverged")
        self.assertEqual(difference["positive_event_ids"], ["s1.call1"])
        self.assertEqual(difference["negative_event_ids"], ["s1.call1"])
        self.assertEqual(difference["interpretation"], "hypothesis_only")

    def test_val_input_is_rejected_before_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validated.jsonl"
            path.write_text(json.dumps({"split": "val", "trace_id": "x"}) + "\n")
            with self.assertRaises(ValueError):
                run(path, None, Path(directory) / "m2", dry_run=True)

    def test_retry_source_replaces_original_without_losing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_dir, input_dir, output_dir = root / "metrics", root / "train", root / "out"
            metrics_dir.mkdir()
            input_dir.mkdir()
            original = root / "old.json"
            retry = root / "retry.json"
            original.write_text(json.dumps(trace("Old incomplete run")))
            retry.write_text(json.dumps(trace("First check, then finish.")))
            fields = ["benchmark", "task_id", "split", "attempt", "success", "reward", "trace_path", "trial_path",
                      "model_error", "agent_429_errors", "execution_error", "scoring_error", "original_error_type", "task_type"]
            base = {"benchmark": "alfworld", "task_id": "task", "split": "train", "attempt": "baseline-v3",
                    "success": "False", "reward": "0.0", "trace_path": str(original), "trial_path": str(root / "no_trial"),
                    "model_error": "True", "agent_429_errors": "4", "execution_error": "False",
                    "scoring_error": "False", "original_error_type": "", "task_type": "pick_clean_then_place_in_recep"}
            final = {**base, "attempt": "baseline-retry-v1", "success": "True", "reward": "1.0",
                     "trace_path": str(retry), "model_error": "False", "agent_429_errors": "0", "original_error_type": "model"}
            for suffix, row in (("baseline-original", base), ("baseline", final)):
                with (metrics_dir / f"alfworld-{suffix}.csv").open("w", newline="") as target:
                    writer = csv.DictWriter(target, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(row)
            (input_dir / "alfworld-train.atif.jsonl").write_text(json.dumps({
                "case_id": "task", "source_trace_path": str(original), "trace_sha256": sha256(original),
                "trace": json.loads(original.read_text())}) + "\n")
            with patch("skillrsi_m1.reliability.BENCHMARKS", ("alfworld",)), \
                 patch("skillrsi_m1.reliability.EXPECTED_TRAIN", {"alfworld": 1}):
                audit = validate_run(metrics_dir, input_dir, output_dir)
            self.assertEqual(audit["processed_count"], 1)
            self.assertEqual(audit["task_outcome"], {"success": 1})
            self.assertEqual(audit["source_replacements"][0]["trace_id"], "alfworld:task")
            record = json.loads((output_dir / "validated.jsonl").read_text())
            self.assertEqual(record["source_trace_path"], str(retry))
            self.assertEqual(record["trace_validity"], "valid")

    def test_small_integration_writes_all_new_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trace.json"
            path.write_text(json.dumps(trace()))
            rows = []
            for task_id, outcome, reward in (("p", "success", 1), ("n", "failure", 0)):
                rows.append({"benchmark": "alfworld", "task_id": task_id,
                             "trace_id": f"alfworld:{task_id}", "split": "train", "attempt": "baseline-v3",
                             "source_trace_path": str(path), "trace_sha256": sha256(path),
                             "trace_validity": "valid", "reliability_flags": [], "task_outcome": outcome,
                             "reward": reward, "metric_trial_path": "/missing",
                             "metrics": {"task_type": "pick_clean_then_place_in_recep", "trace_chars": "200"}})
            validated = root / "validated.jsonl"
            validated.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "m2"
            summary = run(validated, None, output, max_deep_read_traces=2, max_positive_traces=1,
                          max_negative_traces=1, max_contrast_pairs=1, max_llm_input_tokens=1000)
            self.assertEqual((summary["eligible_success"], summary["eligible_failure"]), (1, 1))
            for name in ("trace_profiles.jsonl", "positive_strategy_groups.json", "failure_pattern_groups.json",
                         "contrast_groups.json", "m2_summary.json", "l0.jsonl", "groups.jsonl", "evaluation.json"):
                self.assertTrue((output / name).is_file(), name)
            dry_output = root / "dry"
            dry_summary = run(validated, None, dry_output, dry_run=True)
            self.assertTrue(dry_summary["dry_run"])
            self.assertFalse(dry_output.exists())
