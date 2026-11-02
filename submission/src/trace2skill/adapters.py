"""Benchmark-neutral ATIF adapter contract and public specializations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from skillrsi_m1.adapters import ADAPTERS, BaseAdapter
from skillrsi_m1.core import events_from_trace


class BenchmarkAdapter:
    def __init__(self, name: str):
        self.name = name
        self.specialized = ADAPTERS.get(name, BaseAdapter())

    def load_trace(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def load_metrics(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))

    def extract_task_signature(self, record: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
        return self.specialized.task_signature(record, public)

    def extract_behavior_signature(self, trace: dict[str, Any]) -> dict[str, Any]:
        from skillrsi_m1.profiles import behavior_signature
        events = events_from_trace(trace)
        return behavior_signature(events, self.specialized.signals({"task_outcome": "uncertain", "metrics": {},
                                                                     "metric_trial_path": ""}, events)[1])

    def extract_outcome(self, reward: Any) -> str:
        try:
            return "success" if float(reward) >= .999 else "failure"
        except (ValueError, TypeError):
            return "uncertain"

    def collect_evidence(self, trace: dict[str, Any]) -> list[str]:
        return [event["id"] for event in events_from_trace(trace)]

    def get_evaluation_command(self, kit_dir: Path, split: str, *, skills_dir: Path | None,
                               jobs_dir: Path, max_tasks: int | None = None,
                               bench_name: str | None = None, extra_args: list[str] | None = None) -> list[str]:
        command = [str(kit_dir / "run.sh"), "--bench", bench_name or f"{self.name}-{split}",
                   "--jobs-dir", str(jobs_dir), "--reasoning", "off"]
        if max_tasks is not None:
            command += ["--n", str(max_tasks)]
        if skills_dir is not None:
            command += ["--skills-dir", str(skills_dir)]
        command += extra_args or []
        return command
