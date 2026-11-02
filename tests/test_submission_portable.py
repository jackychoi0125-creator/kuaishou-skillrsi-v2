"""A tiny synthetic hidden-domain ATIF test; no model endpoint or Harbor Kit."""
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from trace2skill.pipeline import evolve
from trace2skill.adapters import BenchmarkAdapter
from skillrsi_m1.judge import read_model_config


class PortablePipelineTest(unittest.TestCase):
    def test_generation_endpoint_is_injected(self):
        with patch.dict("os.environ", {"TRACE2SKILL_GENERATION_API_KEY": "synthetic-key",
                                    "TRACE2SKILL_MODEL_ID": "synthetic-model",
                                    "TRACE2SKILL_GENERATION_URL": "https://example.invalid/chat/completions"}):
            key, model, url = read_model_config(None)
            self.assertEqual((key, model, url), ("synthetic-key", "synthetic-model",
                                                 "https://example.invalid/chat/completions"))

    def test_public_and_generic_adapters(self):
        from skillrsi_m1.adapters import BaseAdapter
        for name in ("spreadsheetbench", "alfworld", "livemath"):
            self.assertIsNot(type(BenchmarkAdapter(name).specialized), BaseAdapter)
        self.assertIs(type(BenchmarkAdapter("unseen_domain").specialized), BaseAdapter)

    def test_unknown_benchmark_real_offline_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces, metrics, output = root / "train.jsonl", root / "metrics.csv", root / "out"
            trace = {"steps": [{"step_id": 1, "source": "agent", "message": "First inspect the state",
                                 "tool_calls": [{"tool_call_id": "c1", "function_name": "orbit_shift",
                                                 "arguments": {"target": "blue"}}],
                                 "observation": {"results": [{"source_call_id": "c1", "content": "state changed"}]}},
                                {"step_id": 2, "source": "agent", "message": "Done"}]}
            traces.write_text(json.dumps({"case_id": "synthetic-1", "trace": trace}) + "\n")
            with metrics.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["task_id", "split", "reward", "task_type"])
                writer.writeheader()
                writer.writerow({"task_id": "synthetic-1", "split": "train", "reward": "1",
                                 "task_type": "orbital_control"})
            kwargs = dict(benchmark="new_hidden_domain", train_traces=traces, train_metrics=metrics,
                          output_dir=output, mode="real", skip_val=True, max_generation_tokens=0)
            dry = evolve(**{**kwargs, "mode": "dry-run"})
            self.assertEqual(dry["model_calls"], 0)
            self.assertEqual(dry["harbor_tasks"], 0)
            result = evolve(**kwargs)
            self.assertEqual(result["best"]["version"], "no_skill")
            self.assertEqual(result["candidate"]["status"], "no_candidate")
            profile = json.loads((output / "m2/trace_profiles.jsonl").read_text().splitlines()[0])
            self.assertEqual(profile["task_signature"]["task_family"], "orbital_control")
            self.assertEqual(evolve(**kwargs, resume=True)["fingerprint"], result["fingerprint"])
            self.assertEqual(result["harbor_tasks"], 0)
            state = json.loads((output / "run_manifest.json").read_text())
            skill = output / "candidate_skills/SKILL.md"
            skill.parent.mkdir(parents=True, exist_ok=True)
            skill.write_text("first")
            state["candidate"] = {"status": "candidate", "skill_path": str(skill), "sha256": "0" * 64}
            (output / "run_manifest.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "Skill SHA changed"):
                evolve(**kwargs, resume=True)

    def test_reject_non_train_input_and_changed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces, metrics = root / "train.jsonl", root / "metrics.csv"
            traces.write_text(json.dumps({"case_id": "x", "trace": {"steps": []}}) + "\n")
            with metrics.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["task_id", "split", "reward"])
                writer.writeheader(); writer.writerow({"task_id": "x", "split": "val", "reward": "1"})
            with self.assertRaises(ValueError):
                evolve(benchmark="other", train_traces=traces, train_metrics=metrics,
                       output_dir=root / "out", mode="real")
            metrics.write_text("task_id,split,reward\nx,train,0\n")
            result = evolve(benchmark="other", train_traces=traces, train_metrics=metrics,
                            output_dir=root / "valid_out", mode="real", max_generation_tokens=0)
            self.assertEqual(result["stop_reason"], "no_valid_train_trace")
            self.assertEqual(result["best"]["version"], "no_skill")

    def test_frozen_holdout_must_not_enter_source_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces, metrics, config = root / "train.jsonl", root / "metrics.csv", root / "config.json"
            traces.write_text(json.dumps({"case_id": "held", "trace": {"steps": [{"step_id": 1,
                                      "source": "agent", "message": "done"}]}}) + "\n")
            metrics.write_text("task_id,split,reward\nheld,train,1\n")
            config.write_text(json.dumps({"train_holdout": {"task_ids": ["held"]}}))
            kwargs = dict(benchmark="unseen", train_traces=traces, train_metrics=metrics,
                          output_dir=root / "out", config=config, mode="real", max_generation_tokens=0)
            with self.assertRaisesRegex(ValueError, "Holdout overlaps Source"):
                evolve(**kwargs)
            with self.assertRaisesRegex(ValueError, "exactly one evolution round"):
                evolve(**{**kwargs, "max_rounds": 2})


if __name__ == "__main__":
    unittest.main()
