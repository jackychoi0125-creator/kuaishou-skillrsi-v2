"""Run a resumable first Train-insight -> Skill -> official Val -> gate cycle."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from skillrsi_m1.core import BENCHMARKS
from skillrsi_m3.client import KimiClient
from skillrsi_m3.core import run as run_m3, write_json
from skillrsi_m4.core import run as run_m4
from .gate import compare

KIT_REL = {"spreadsheetbench": "spreadsheet/spreadsheetbench-harbor-kit-v3.3-layout2",
           "alfworld": "alfworld/pi-harbor-kit", "livemath": "livemath/pi-harbor-kit"}
CONCURRENT = {"spreadsheetbench": 5, "alfworld": 4, "livemath": 5}


def config(kit: Path, benchmark: str) -> dict[str, Any]:
    content = (kit / "kit.env").read_text()
    values = {}
    for name in ("KIT_MODEL_ID", "KIT_API", "KIT_BASE_URL"):
        match = re.search(rf"(?m)^(?:export\s+)?{name}=(.*)$", content)
        if not match:
            raise ValueError(f"kit.env missing {name}")
        values[name] = match.group(1).strip().strip("'\"")
    if not re.search(r"(?m)^(?:export\s+)?API_KEY=\S+", content):
        raise ValueError("kit.env missing API_KEY")
    if values["KIT_MODEL_ID"] != "ep-0lh2e2-1791798733636675306":
        raise ValueError("Kit model differs from verified DeepSeek V4 Flash")
    return {"model_id": values["KIT_MODEL_ID"], "api": values["KIT_API"],
            "base_url": values["KIT_BASE_URL"], "reasoning": "off", "seed": 0,
            "kit_root": str(kit.resolve()), "bench": benchmark + "-val",
            "kit_script_sha256": hashlib.sha256((kit / "run.sh").read_bytes()).hexdigest()}


def _job(root: Path) -> Path | None:
    jobs = sorted(p for p in root.glob("*/summary.csv")) if root.is_dir() else []
    return jobs[-1].parent if jobs else None


def harbor_run(kit: Path, benchmark: str, jobs_root: Path, skills_dir: Path | None) -> Path:
    existing = _job(jobs_root)
    if existing:
        return existing
    args = [str(kit / "run.sh"), "--bench", benchmark + "-val", "--concurrent",
            str(CONCURRENT[benchmark]), "--reasoning", "off", "--jobs-dir", str(jobs_root)]
    if skills_dir:
        args += ["--skills-dir", str(skills_dir)]
    jobs_root.mkdir(parents=True, exist_ok=True)
    log = jobs_root / "controller.log"
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(args, cwd=kit, stdout=output, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Harbor {benchmark} run failed (exit {completed.returncode}); see {log}")
    job = _job(jobs_root)
    if not job:
        raise RuntimeError(f"Harbor run produced no official summary: {log}")
    return job


def run_one(root: Path, benchmark: str, *, mode: str, resume: bool, stop_after: str | None,
            baseline_job: Path | None = None) -> dict[str, Any]:
    result_root = root / "results/r1" / benchmark
    result_root.mkdir(parents=True, exist_ok=True)
    state_path = result_root / "stage_manifest.json"
    state = json.loads(state_path.read_text()) if resume and state_path.is_file() else {"benchmark": benchmark, "stages": {}}
    if resume:
        audit_path = result_root / "m3/insight_audit.json"
        insight_path = result_root / "m3/insights.jsonl"
        if "m3" not in state["stages"] and audit_path.is_file() and insight_path.is_file():
            audit = json.loads(audit_path.read_text())
            state["stages"]["m3"] = {"insights": sum(bool(line.strip()) for line in insight_path.read_text().splitlines()),
                                     **audit, "resumed_from_verified_artifacts": True}
        manifest_path = result_root / "m4/candidate_manifest.json"
        if "m4" not in state["stages"] and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            skill = Path(manifest.get("skill_path", ""))
            if manifest["status"] != "candidate" or (skill.is_file() and hashlib.sha256(skill.read_bytes()).hexdigest() == manifest["sha256"]):
                state["stages"]["m4"] = manifest
    def done(stage: str, value: Any) -> None:
        state["stages"][stage] = value
        write_json(state_path, state)
    def stopped(stage: str) -> bool:
        return stop_after == stage
    model_path = root.parent / "model.txt"
    m2 = root / "results/m2"
    m3_dir = result_root / "m3"
    m4_dir = result_root / "m4"
    client = KimiClient(model_path, root / "results/r1/cache") if mode == "real" else None
    m3_ready = (resume and "m3" in state["stages"] and
                (mode == "dry-run" or (not state["stages"]["m3"].get("dry_run") and
                                       (m3_dir / "insights.jsonl").is_file())))
    if not m3_ready:
        start = time.monotonic()
        value = run_m3(m2, benchmark, m3_dir, client, dry_run=mode == "dry-run")
        done("m3", {**value, "elapsed_sec": round(time.monotonic() - start, 3)})
    if stopped("m3"):
        return state
    if mode == "dry-run":
        done("dry_run", {"model_calls": 0, "val_tasks": 0, "candidate_scores": None})
        return state
    if not (resume and "m4" in state["stages"]):
        start = time.monotonic()
        value = run_m4(benchmark, m3_dir, root / "workspace/skills", m4_dir, client)
        done("m4", {**value, "elapsed_sec": round(time.monotonic() - start, 3)})
    if stopped("m4") or state["stages"]["m4"]["status"] != "candidate":
        return state
    kit = (root.parent / "kits" / KIT_REL[benchmark]).resolve()
    conf = config(kit, benchmark)
    val_dir = result_root / "val"
    if not (resume and "baseline" in state["stages"]):
        start = time.monotonic()
        base = baseline_job or harbor_run(kit, benchmark, kit / "workspace/runs" / (benchmark + "-val") / "r1-baseline-recheck", None)
        done("baseline", {"job": str(base), "config": conf, "elapsed_sec": round(time.monotonic() - start, 3)})
    if stopped("baseline"):
        return state
    if not (resume and "candidate" in state["stages"]):
        start = time.monotonic()
        skills_dir = root / "workspace/skills" / benchmark / "r1"
        candidate = harbor_run(kit, benchmark, kit / "workspace/runs" / (benchmark + "-val") / "r1-candidate", skills_dir)
        done("candidate", {"job": str(candidate), "config": conf,
                           "skill_sha256": state["stages"]["m4"]["sha256"],
                           "elapsed_sec": round(time.monotonic() - start, 3)})
    if stopped("candidate"):
        return state
    if not (resume and "gate" in state["stages"]):
        acceptance = compare(benchmark, Path(state["stages"]["baseline"]["job"]),
                             Path(state["stages"]["candidate"]["job"]), val_dir,
                             state["stages"]["baseline"]["config"], state["stages"]["candidate"]["config"])
        write_json(val_dir / "official_results.json", {"baseline_job": acceptance["baseline_job"],
                                                  "candidate_job": acceptance["candidate_job"],
                                                  "baseline_pass": acceptance["baseline_pass"],
                                                  "candidate_pass": acceptance["candidate_pass"]})
        write_json(val_dir / "run_manifest.json", {"baseline": state["stages"]["baseline"],
                                                "candidate": state["stages"]["candidate"]})
        write_json(result_root / "version_state.json", {
            "benchmark": benchmark, "round": "r1", "decision": acceptance["decision"],
            "active_skill_path": state["stages"]["m4"]["skill_path"] if acceptance["decision"] == "accept" else None,
            "active_skill_sha256": state["stages"]["m4"]["sha256"] if acceptance["decision"] == "accept" else None,
            "best_baseline_job": state["stages"]["baseline"]["job"],
            "candidate_retained_for_audit": state["stages"]["m4"]["skill_path"]})
        done("gate", acceptance)
    if "gate" in state["stages"] and not (result_root / "version_state.json").is_file():
        decision = state["stages"]["gate"]["decision"]
        write_json(result_root / "version_state.json", {
            "benchmark": benchmark, "round": "r1", "decision": decision,
            "active_skill_path": state["stages"]["m4"]["skill_path"] if decision == "accept" else None,
            "active_skill_sha256": state["stages"]["m4"]["sha256"] if decision == "accept" else None,
            "best_baseline_job": state["stages"]["baseline"]["job"],
            "candidate_retained_for_audit": state["stages"]["m4"]["skill_path"]})
    return state


def report(root: Path, states: list[dict[str, Any]]) -> Path:
    lines = ["# R1 官方 Val 结果", "", "M3/M4 只读取 Train。以下分数来自 Harbor 官方评分器。", "",
             "| Benchmark | Baseline | R1 | 修复 | 回退 | 决定 |", "|---|---:|---:|---:|---:|---|"]
    for state in states:
        gate = state["stages"].get("gate")
        if gate:
            t = gate["transitions"]
            lines.append(f"| {state['benchmark']} | {gate['baseline_pass']}/{gate['tasks']} | {gate['candidate_pass']}/{gate['tasks']} | {t['fail_to_pass']} | {t['pass_to_fail']} | {gate['decision']} |")
        else:
            lines.append(f"| {state['benchmark']} | — | — | — | — | 阶段未完成 |")
    lines += ["", "历史参考 Val 分数为 SpreadsheetBench 30/40、ALFWorld 12/18、LiveMath 7/17；本表以当前 Kit 同配置复测结果作为配对基线。",
              "仅有一题净增益时为低置信观察，不表示统计显著。详细运行目录和调用记录见各 Benchmark 的 manifest 与 usage。"]
    lines += ["", "## 生成与调用审计", "",
              "| Benchmark | M3 Kimi Prompt / Completion / Cache | M4 Kimi Prompt / Completion / Cache | M3/M4 调用与耗时 | Skill bytes |", "|---|---:|---:|---:|---:|"]
    for state in states:
        bench = state["benchmark"]
        base = root / "results/r1" / bench
        def usage(stage: str) -> str:
            path = base / stage / "usage.json"
            if not path.is_file():
                return "—"
            item = json.loads(path.read_text())
            totals = {"prompt_tokens": 0, "completion_tokens": 0, "cache_tokens": 0}
            seen = set()
            for call in item.get("calls", []):
                call_id = call["call_id"]
                if call_id in seen:
                    continue
                seen.add(call_id)
                digest = call_id.rsplit("-", 1)[-1]
                receipt = next((root / "results/r1/cache").glob(digest + "*.json"), None)
                if not receipt:
                    continue
                original = json.loads(receipt.read_text()).get("usage", {})
                for key in totals:
                    totals[key] += original.get(key, 0)
            return f"{totals['prompt_tokens']} / {totals['completion_tokens']} / {totals['cache_tokens']}"
        manifest = base / "m4/candidate_manifest.json"
        size = json.loads(manifest.read_text()).get("bytes", "—") if manifest.is_file() else "—"
        def effort(stage: str) -> tuple[int, float]:
            path = base / stage / "usage.json"
            if not path.is_file():
                return 0, 0.0
            calls = json.loads(path.read_text()).get("calls", [])
            receipts = []
            for call in calls:
                digest = call["call_id"].rsplit("-", 1)[-1]
                receipt = next((root / "results/r1/cache").glob(digest + "*.json"), None)
                if receipt:
                    receipts.append(json.loads(receipt.read_text())["usage"])
            return len(receipts), round(sum(r.get("elapsed_sec", 0) for r in receipts), 1)
        m3_count, m3_time = effort("m3")
        m4_count, m4_time = effort("m4")
        lines.append(f"| {bench} | {usage('m3')} | {usage('m4')} | {m3_count}/{m4_count} calls; {m3_time}/{m4_time}s | {size} |")
    lines += ["", "表中 Token 为产生当前版本的去重 API 收据用量；缓存命中的本次增量调用为零。模型完整请求、响应及服务端 usage 保存在 results/r1/cache；各阶段调用次数、重试和耗时见 usage.json。",
              "Val 逐题反馈只用于 diff 与 Gate，未反馈给 M3/M4。"]
    lines += ["", "## Val 耗时与错误", "",
              "| Benchmark | Baseline wall sec | Candidate wall sec | Baseline / Candidate errors | Baseline / Candidate input tokens |",
              "|---|---:|---:|---:|---:|"]
    for state in states:
        gate = state["stages"].get("gate")
        if not gate:
            continue
        base = json.loads((Path(gate["baseline_job"]) / "summary.json").read_text())
        cand = json.loads((Path(gate["candidate_job"]) / "summary.json").read_text())
        lines.append(f"| {state['benchmark']} | {base.get('wall_sec', '—')} | {cand.get('wall_sec', '—')} | "
                     f"{base.get('errors', '—')} / {cand.get('errors', '—')} | "
                     f"{base.get('input_tokens', '—')} / {cand.get('input_tokens', '—')} |")
    lines += ["", "M1/M2 使用已有 Train 产物，本轮未新增 M1/M2 模型调用。拒绝的候选保留审计文件，当前版本指向无 Skill 基线。"]
    path = root / "results/r1/RUN_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("round", choices=["r1"])
    parser.add_argument("--benchmark", choices=["all", *BENCHMARKS], default="all")
    parser.add_argument("--mode", choices=["real", "dry-run"], default="dry-run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=["m3", "m4", "baseline", "candidate"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    benches = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    states = []
    for benchmark in benches:
        started = time.monotonic()
        state = run_one(args.root.resolve(), benchmark, mode=args.mode, resume=args.resume, stop_after=args.stop_after)
        state["last_elapsed_sec"] = round(time.monotonic() - started, 3)
        write_json(args.root / "results/r1" / benchmark / "stage_manifest.json", state)
        states.append(state)
        print(json.dumps({"benchmark": benchmark, "stages": list(state["stages"]),
                          "elapsed_sec": state["last_elapsed_sec"]}, ensure_ascii=False), flush=True)
    print(report(args.root, states))


if __name__ == "__main__":
    main()
