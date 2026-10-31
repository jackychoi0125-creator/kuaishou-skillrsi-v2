import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from skillrsi_m3.core import InsightConsolidator, TraceStore, observed_math_definition_check, run as run_m3
from skillrsi_m4.core import validate_skill, run as run_m4
from skillrsi_loop.gate import compare


class R1Tests(unittest.TestCase):
    def test_trace_ref_sha_and_train_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = {"steps": [{"step_id": 1, "message": "check state"}]}
            source = root / "trace.json"
            source.write_text(json.dumps(trace))
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            profile = {"trace_id": "alfworld:a", "benchmark": "alfworld", "split": "train",
                       "eligible_for_learning": True, "source_trace_path": str(source), "trace_sha256": sha}
            profiles = root / "profiles.jsonl"
            profiles.write_text(json.dumps(profile) + "\n")
            store = TraceStore(profiles, "alfworld")
            self.assertEqual(store.get("alfworld:a", "s1.message")["text"], "check state")
            self.assertEqual(store.ref("alfworld:a", ["s1.message"])["trace_sha256"], sha)
            source.write_text('{"steps":[]}')
            with self.assertRaises(ValueError):
                TraceStore(profiles, "alfworld").get("alfworld:a", "s1.message")
            profile["split"] = "val"
            profiles.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):
                TraceStore(profiles, "alfworld")

    def test_single_trace_not_upgraded_and_duplicate_vote(self):
        insight = {"insight_id": "x", "benchmark": "alfworld", "principle": "Check action feedback",
                   "supporting_refs": [{"trace_id": "t", "event_ids": ["e"], "trace_sha256": "sha"}],
                   "source_group_ids": ["g1"], "evidence_grade": "single_trace_hypothesis"}
        duplicate = {**insight, "insight_id": "y", "source_group_ids": ["g2"]}
        pool, ops = InsightConsolidator().consolidate([insight, duplicate])
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]["evidence_grade"], "single_trace_hypothesis")
        self.assertEqual(ops[-1]["new_independent_traces"], 0)

    def test_math_definition_observation_is_hypothesis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "trace.json"
            source.write_text(json.dumps({"steps": [{"step_id": 1, "message":
                "R means this local relation. Wait, let me re-read the definition: R means this local relation."}]}))
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            profile = {"trace_id": "livemath:x", "benchmark": "livemath", "split": "train",
                       "eligible_for_learning": True, "source_trace_path": str(source), "trace_sha256": sha,
                       "outcome": {"status": "success"}, "behavior_signature":
                       {"verification": [{"status": "observed", "event_ids": ["s1.message"]}]}}
            (root / "trace_profiles.jsonl").write_text(json.dumps(profile))
            (root / "positive_strategy_groups.json").write_text(json.dumps([{"group_id": "g1", "trace_ids": ["livemath:x"]}]))
            candidate = observed_math_definition_check(TraceStore(root / "trace_profiles.jsonl", "livemath"), root)
            self.assertEqual(candidate[0]["causal_status"], "hypothesis_only")
            self.assertEqual(candidate[0]["supporting_refs"][0]["event_ids"], ["s1.message"])

    def test_skill_static_checks(self):
        content = "---\nname: alfworld-r1\ndescription: Act on state feedback\n---\n# When to use\n# Procedure\n# Avoid\n# Validation\n"
        self.assertEqual(validate_skill(content, "alfworld", [{"benchmark": "alfworld"}]), [])
        self.assertIn("prohibited_content", validate_skill(content + "Go to countertop 1\n", "alfworld", [{"benchmark": "alfworld"}]))

    def test_official_paired_gate_and_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, cand = root / "baseline", root / "candidate"
            for folder, changed in ((base, False), (cand, True)):
                folder.mkdir()
                with (folder / "summary.csv").open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=["qid", "reward", "input_tokens", "cache_tokens", "output_tokens", "duration_sec", "exception", "model_error"])
                    writer.writeheader()
                    for n in range(17):
                        qid = f"lm{n}"
                        reward = 1.0 if n == 0 or (changed and n == 1) else 0.0
                        writer.writerow({"qid": qid, "reward": reward, "input_tokens": 1, "cache_tokens": 0,
                                         "output_tokens": 1, "duration_sec": 1, "exception": False, "model_error": False})
                        trial = folder / f"{qid}__abc"
                        trial.mkdir()
                        (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))
            conf = {"model_id": "x", "api": "a", "base_url": "u", "reasoning": "off", "seed": 0,
                    "kit_root": "k", "bench": "livemath-val"}
            outcome = compare("livemath", base, cand, root / "out", conf, conf)
            self.assertEqual(outcome["decision"], "accept")
            self.assertEqual(outcome["transitions"]["fail_to_pass"], 1)
            self.assertTrue((root / "out/diff.csv").is_file())
            with self.assertRaises(ValueError):
                compare("livemath", base, cand, root / "out", conf, {**conf, "model_id": "different"})

    def test_m3_m4_mock_integration(self):
        class FakeClient:
            def ask(self, stage, system, user, **kwargs):
                if stage == "m3_extract":
                    return {"insights": [{"source_group_id": "g1", "principle": "Check observed state",
                        "trigger_conditions": ["A command changes state"], "avoid_conditions": [],
                        "recommended_behavior": "Read the observation before the next command",
                        "supporting_refs": [{"trace_id": "alfworld:x", "event_ids": ["s1.message"]}],
                        "counterexample_refs": [], "source_channel": "positive"}]}, "fake-m3"
                return {"skill_markdown": "---\nname: alfworld-r1\ndescription: Check state\n---\n# When to use\n- A command changes state\n# Procedure\n- Read observation\n# Avoid\n- Repeating without feedback\n# Validation\n- State checked\n"}, "fake-m4"
            def usage(self):
                return {"model": "mock", "api_calls": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "cache_tokens": 0, "calls": []}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m2 = root / "m2"
            m2.mkdir()
            source = root / "trace.json"
            source.write_text(json.dumps({"steps": [{"step_id": 1, "message": "I checked the changed state."}]}))
            profile = {"trace_id": "alfworld:x", "benchmark": "alfworld", "split": "train",
                       "eligible_for_learning": True, "source_trace_path": str(source),
                       "trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "outcome": {"status": "success"}, "behavior_signature":
                       {"verification": [{"status": "observed", "event_ids": ["s1.message"]}]}}
            (m2 / "trace_profiles.jsonl").write_text(json.dumps(profile))
            group = {"group_id": "g1", "benchmark": "alfworld", "task_family": "interaction",
                     "trace_ids": ["alfworld:x"], "evidence_events": ["alfworld:x:s1.message"],
                     "behavior_pattern": ["verification:checked"],
                     "quality": {"reuse_readiness": "single_case_hypothesis",
                                 "adverse_behavior_present": False, "evidence_event_count": 1}}
            (m2 / "positive_strategy_groups.json").write_text(json.dumps([group]))
            for name in ("failure_pattern_groups.json", "contrast_groups.json", "l2_contrasts.json"):
                (m2 / name).write_text("[]")
            (m2 / "evidence_audit.json").write_text(json.dumps({"summary": {"event_ref_integrity": 1.0}}))
            client = FakeClient()
            result = run_m3(m2, "alfworld", root / "m3", client)
            self.assertEqual(result["verified"], 1)
            manifest = run_m4("alfworld", root / "m3", root / "skills", root / "m4", client)
            self.assertEqual(manifest["status"], "candidate")
            self.assertTrue(Path(manifest["skill_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
