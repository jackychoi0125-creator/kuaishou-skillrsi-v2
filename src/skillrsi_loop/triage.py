"""Read-only, paired Val regression audit. Never feeds M3/M4."""
from __future__ import annotations

import csv
import difflib
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from skillrsi_m1.core import _alf_action, events_from_trace
from skillrsi_m3.core import write_json

BENCHMARKS = ("spreadsheetbench", "alfworld", "livemath")


def _trial(job: Path, task_id: str) -> Path:
    matches = list(job.glob(f"{task_id}__*/result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one official trial for {task_id} under {job}, got {len(matches)}")
    return matches[0].parent


def _events(trial: Path) -> list[dict[str, Any]]:
    return events_from_trace(json.loads((trial / "agent/trajectory.json").read_text()))


def _skill_reads(events: list[dict[str, Any]]) -> list[str]:
    return [event["id"] for event in events if event["kind"] == "tool_call"
            and re.search(r"(?i)SKILL\.md", event["text"])
            and re.search(r"(?i)\b(?:cat|sed|head|tail|read|open|less|grep)\b",
                          event["text"] + " " + str(event.get("function") or ""))]


def _skill_discoveries(events: list[dict[str, Any]]) -> list[str]:
    return [event["id"] for event in events if event["kind"] == "tool_call"
            and re.search(r"(?i)(?:skills?/|\bskills?\b)", event["text"])
            and re.search(r"(?i)\b(?:ls|find|list|search|glob)\b", event["text"])
            and event["id"] not in _skill_reads([event])]


def _command(event: dict[str, Any]) -> str:
    args = event.get("arguments") or {}
    return str(args.get("command") or args.get("cmd") or event["text"]) if isinstance(args, dict) else event["text"]


def _milestone(benchmark: str, event: dict[str, Any]) -> tuple[str, str] | None:
    kind, text = event["kind"], event["text"]
    lower = text.lower()
    if kind == "tool_call" and event["id"] in _skill_reads([event]):
        return "skill_read", "SKILL.md"
    if benchmark == "spreadsheetbench":
        if kind == "tool_call":
            cmd = _command(event)
            if re.search(r"(?i)(?:write_file|solution\.py|cat\s+>|tee\s+)", cmd) and re.search(r"(?i)(?:write|cat\s+>|tee\s+|content)", text):
                return "script_write", hashlib.sha256(cmd.encode()).hexdigest()[:12]
            if re.search(r"(?i)(?:python\d?\s+[^\n]*solution\.py|libreoffice|soffice)", cmd):
                return "script_execute", re.sub(r"\s+", " ", cmd)[:150]
            if re.search(r"(?i)(?:openpyxl|load_workbook|sheetnames|ls\b|find\b)", cmd):
                return "workbook_inspect", re.sub(r"\s+", " ", cmd)[:150]
            if re.search(r"(?i)(?:assert|verify|check|compare|diff)", cmd):
                return "result_check", re.sub(r"\s+", " ", cmd)[:150]
            return "tool_action", str(event.get("function") or "tool")
        if kind == "observation" and re.search(r"(?i)(?:traceback|error|no such file|failed)", text):
            return "tool_error", text[:150]
    elif benchmark == "alfworld":
        if kind == "tool_call":
            action = _alf_action(_command(event))
            if action:
                verb = action.split(" ", 1)[0].lower()
                return "environment_action:" + verb, action
            return "tool_action", str(event.get("function") or "tool")
        if kind == "observation":
            if re.search(r"(?i)nothing happens|not a valid action|can't do that|cannot do that", text):
                return "environment_no_progress", text[:150]
            if re.search(r'"won"\s*:\s*true', text, re.I):
                return "environment_success", "won=true"
    else:
        if kind == "message":
            if re.search(r"\\boxed\s*\{\s*[A-E]\s*\}", text):
                return "final_option", _option(text) or "unknown"
            if len(text.strip()) >= 30:
                return "reasoning", hashlib.sha256(text.encode()).hexdigest()[:12]
        if kind == "tool_call":
            return "tool_action", str(event.get("function") or "tool")
    return None


def milestones(benchmark: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for event in events:
        value = _milestone(benchmark, event)
        if value:
            label, detail = value
            result.append({"label": label, "detail": detail, "event_id": event["id"],
                           "step": event["step"]})
    return result


def first_divergence(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    """Align semantic milestones first; event IDs remain run-local evidence pointers."""
    labels_left = [m["label"] for m in left]
    labels_right = [m["label"] for m in right]
    matcher = difflib.SequenceMatcher(a=labels_left, b=labels_right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            return {"alignment": "task_milestone", "change": tag,
                    "baseline": left[i1:i2][:3], "r1": right[j1:j2][:3],
                    "baseline_prior": left[i1 - 1] if i1 else None,
                    "r1_prior": right[j1 - 1] if j1 else None}
        for offset in range(i2 - i1):
            a, b = left[i1 + offset], right[j1 + offset]
            if a["detail"] != b["detail"] and a["label"] not in {"reasoning", "script_write"}:
                return {"alignment": "task_milestone", "change": "different_action_or_result",
                        "baseline": [a], "r1": [b]}
    return {"alignment": "task_milestone", "change": "no_detected_milestone_difference",
            "baseline": [], "r1": []}


def _option(text: str) -> str | None:
    matches = re.findall(r"\\boxed\s*\{\s*([A-E])\s*\}", text, re.I)
    return matches[-1].upper() if matches else None


def _final_option(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event["kind"] == "message":
            value = _option(event["text"])
            if value:
                return value
    return None


def _alf_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [(event["id"], _alf_action(_command(event))) for event in events if event["kind"] == "tool_call"]
    actions = [(event_id, action) for event_id, action in actions if action]
    repeated = [actions[i][0] for i in range(1, len(actions)) if actions[i][1] == actions[i - 1][1]]
    no_progress = [event["id"] for event in events if event["kind"] == "observation" and
                   re.search(r"(?i)nothing happens|not a valid action|can't do that|cannot do that", event["text"])]
    won = any(event["kind"] == "observation" and re.search(r'"won"\s*:\s*true', event["text"], re.I) for event in events)
    observations = {event["id"]: event for event in events if event["kind"] == "observation"}
    sequence = []
    prior_state = None
    for event_id, action in actions:
        observation_id = event_id.replace(".call", ".obs")
        observation = observations.get(observation_id)
        raw = observation["text"] if observation else ""
        try:
            parsed = json.loads(raw)
            feedback = str(parsed.get("observation") or raw) if isinstance(parsed, dict) else raw
            state = {key: parsed.get(key) for key in ("observation", "inventory", "done", "won")}
        except (ValueError, TypeError):
            feedback, state = raw, {"observation": raw}
        state_digest = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:12]
        sequence.append({"action_event_id": event_id, "action": action, "observation_event_id": observation_id if observation else None,
                         "feedback_excerpt": feedback[:250], "observation_digest": state_digest,
                         "observation_signature_changed": None if prior_state is None else state_digest != prior_state,
                         "explicit_no_progress": bool(re.search(r"(?i)nothing happens|not a valid action|can't do that", feedback))})
        prior_state = state_digest
    return {"actions": len(actions), "repeated_action_event_ids": repeated,
            "no_progress_event_ids": no_progress, "terminal_won_observed": won,
            "action_observation_sequence": sequence}


def _spreadsheet_summary(trial: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    answer_root = trial / "artifacts/bench/answers"
    scripts = []
    for path in sorted(answer_root.rglob("*.py")) if answer_root.is_dir() else []:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        scripts.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                        "cell_refs": sorted(set(re.findall(r"\b[A-Z]{1,3}\d{1,6}\b", text)))[:30],
                        "formula_assignments": len(re.findall(r"(?i)(?:formula|=IF\(|=VLOOKUP\(|=SUM\(|=INDEX\()", text))})
    calls = [(_command(event), event["id"]) for event in events if event["kind"] == "tool_call"]
    output_paths = sorted({match for command, _ in calls for match in re.findall(r"[\w./-]+(?:\.xlsx|\.xlsm)", command)})[:25]
    errors = [event["id"] for event in events if event["kind"] == "observation" and
              re.search(r"(?i)traceback|syntaxerror|filenotfounderror|permission denied|#REF!|#NAME\?", event["text"])]
    return {"scripts": scripts, "workbook_paths_in_calls": output_paths, "tool_error_event_ids": errors}


def _script_diff(before: Path, after: Path) -> dict[str, Any]:
    left = {str(path.relative_to(before / "artifacts/bench/answers")): path for path in
            (before / "artifacts/bench/answers").rglob("*.py")}
    right = {str(path.relative_to(after / "artifacts/bench/answers")): path for path in
             (after / "artifacts/bench/answers").rglob("*.py")}
    changed = []
    for relative in sorted(set(left) | set(right)):
        a = left.get(relative)
        b = right.get(relative)
        old_lines = a.read_text(errors="replace").splitlines() if a else []
        new_lines = b.read_text(errors="replace").splitlines() if b else []
        if old_lines == new_lines:
            continue
        diff = list(difflib.unified_diff(old_lines, new_lines, fromfile="baseline/" + relative,
                                         tofile="r1/" + relative, lineterm=""))
        changed.append({"relative_path": relative, "baseline_exists": bool(a), "r1_exists": bool(b),
                        "changed_lines": len(diff), "diff_excerpt": "\n".join(diff)[:3000]})
    return {"changed_scripts": changed, "same_script": not changed}


def _verifier_note(trial: Path) -> str:
    path = trial / "verifier/metrics.json"
    if not path.is_file():
        return ""
    try:
        value = json.loads(path.read_text())
        return str((value.get("notes") or [""])[0])[:500]
    except (OSError, ValueError, TypeError):
        return ""


def _math_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    messages = [event for event in events if event["kind"] == "message"]
    checks = [event["id"] for event in messages if re.search(
        r"(?i)\b(?:check|verify|recheck|double.check|reconsider|sanity.check)\b|检查|验证|核对|复查",
        event["text"])]
    return {"final_option": _final_option(events), "visible_reasoning_chars": sum(len(x["text"]) for x in messages),
            "boxed_option_event_ids": [x["id"] for x in messages if _option(x["text"])],
            "answer_format_present": bool(_final_option(events)),
            "self_check_event_ids": checks,
            "answer_parse_status": "boxed_option_observed" if _final_option(events) else "no_boxed_option_observed"}


def _rules(skill_path: Path) -> list[dict[str, str]]:
    rules = []
    section = ""
    for line in skill_path.read_text().splitlines():
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
        elif re.match(r"\s*(?:[-*]|\d+\.)\s+", line):
            text = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
            if text:
                rules.append({"rule_id": "rule_" + hashlib.sha256((section + text).encode()).hexdigest()[:10],
                              "section": section, "text": text})
    return rules


def _rule_match(rules: list[dict[str, str]], events: list[dict[str, Any]], read_ids: list[str]) -> dict[str, Any]:
    if not read_ids:
        return {"rule_id": None, "match_status": "not_explicitly_read", "evidence_event_ids": []}
    index = next((i for i, e in enumerate(events) if e["id"] == read_ids[0]), len(events))
    after = " ".join(e["text"][:300] for e in events[index + 1:index + 12] if e["kind"] in {"message", "tool_call"}).lower()
    words = set(re.findall(r"[a-z]{5,}", after))
    candidates = []
    for rule in rules:
        tokens = set(re.findall(r"[a-z]{5,}", rule["text"].lower()))
        overlap = len(words & tokens) / len(tokens) if tokens else 0
        candidates.append((overlap, rule))
    if not candidates:
        return {"rule_id": None, "match_status": "no_rules_parsed", "evidence_event_ids": read_ids}
    score, rule = max(candidates, key=lambda item: item[0])
    if score < .35:
        return {"rule_id": None, "match_status": "no_supported_rule_match", "evidence_event_ids": read_ids}
    return {"rule_id": rule["rule_id"], "rule_excerpt": rule["text"][:180],
            "lexical_overlap": round(score, 3), "match_status": "candidate_only", "evidence_event_ids": read_ids}


def _task_metadata(benchmark: str, task_id: str, metrics_dir: Path) -> dict[str, Any]:
    path = metrics_dir / f"{benchmark}-baseline.csv"
    with path.open(newline="", encoding="utf-8") as source:
        row = next((r for r in csv.DictReader(source) if r["split"] == "val" and r["task_id"] == task_id), None)
    if not row:
        return {"task_type": "unknown", "task_complexity": "unknown", "complexity_basis": "missing_metadata"}
    if benchmark == "spreadsheetbench":
        cells = int(row.get("spreadsheet_cell_count") or 0)
        complexity = "large" if cells > 10000 else "medium" if cells > 500 else "small"
        basis = f"cell_count={cells}"
    elif benchmark == "alfworld":
        actions = int(row.get("alfworld_actions") or 0)
        complexity = "multi_stage" if any(x in row["task_type"].lower() for x in ("clean", "heat", "cool", "look")) else "single_stage"
        basis = f"task_family={row['task_type']}; prior_run_actions={actions}"
    else:
        chars = int(row.get("livemath_question_chars") or 0)
        complexity = "long" if chars > 4000 else "medium" if chars > 1800 else "short"
        basis = f"question_chars={chars}"
    return {"task_type": row.get("task_type") or "unknown", "task_complexity": complexity,
            "complexity_basis": basis}


def audit_case(benchmark: str, diff: dict[str, str], jobs: dict[str, Path],
               rules: list[dict[str, str]], metrics_dir: Path) -> dict[str, Any]:
    task_id = diff["task_id"]
    trials = {stage: _trial(path, task_id) for stage, path in jobs.items()}
    events = {stage: _events(trial) for stage, trial in trials.items()}
    milestones_by_stage = {stage: milestones(benchmark, value) for stage, value in events.items()}
    read_ids = _skill_reads(events["r1"])
    discovery_ids = _skill_discoveries(events["r1"])
    metadata = _task_metadata(benchmark, task_id, metrics_dir)
    extras = {}
    if benchmark == "spreadsheetbench":
        extras = {stage: _spreadsheet_summary(trial, events[stage]) for stage, trial in trials.items()}
        extras["paired_script_diff"] = _script_diff(trials["baseline"], trials["r1"])
        extras["verifier_note"] = {stage: _verifier_note(trial) for stage, trial in trials.items()}
    elif benchmark == "alfworld":
        extras = {stage: _alf_summary(events[stage]) for stage in trials}
    else:
        extras = {stage: _math_summary(events[stage]) for stage in trials}
    rule = _rule_match(rules, events["r1"], read_ids)
    errors = {stage: str(diff[f"{stage}_error"]).lower() in {"true", "1"} for stage in trials}
    if any(errors.values()):
        diagnosis = "execution_or_model_error"
        confidence = "medium"
    elif (benchmark == "alfworld" and diff["transition"] == "pass_to_fail" and
          len(extras["r1"]["no_progress_event_ids"]) > len(extras["baseline"]["no_progress_event_ids"])):
        diagnosis = "observed_action_repetition_or_no_progress"
        confidence = "medium_for_behavior_low_for_skill_causality"
    elif read_ids and rule["rule_id"]:
        diagnosis = "skill_direct_use_hypothesis"
        confidence = "low"
    elif read_ids:
        diagnosis = "skill_read_but_rule_undetermined"
        confidence = "low"
    else:
        diagnosis = "metadata_effect_or_sampling_undetermined"
        confidence = "low"
    result = {"benchmark": benchmark, "task_id": task_id, **metadata,
              "transition": diff["transition"],
              "official_reward": {stage: float(diff[f"{stage}_reward"]) for stage in trials},
              "input_tokens": {stage: int(diff[f"{stage}_input_tokens"]) for stage in trials},
              "output_tokens": {stage: int(diff[f"{stage}_output_tokens"]) for stage in trials},
              "duration_sec": {stage: float(diff[f"{stage}_duration_sec"] or 0) for stage in trials},
              "tool_calls": {stage: sum(e["kind"] == "tool_call" for e in value) for stage, value in events.items()},
              "skill_presence": {"metadata_available": True, "explicit_discovery_event_ids": discovery_ids,
                                 "explicit_read_event_ids": read_ids,
                                 "status": "explicit_read" if read_ids else "metadata_only"},
              "first_key_behavior_difference": first_divergence(milestones_by_stage["baseline"], milestones_by_stage["r1"]),
              "candidate_skill_rule": rule, "diagnosis_type": diagnosis,
              "diagnosis_evidence": {"baseline_event_ids": [x["event_id"] for x in milestones_by_stage["baseline"][:3]],
                                     "r1_event_ids": [x["event_id"] for x in milestones_by_stage["r1"][:3]],
                                     "skill_read_event_ids": read_ids,
                                     "benchmark_observations": extras},
              "attribution_confidence": confidence,
              "attribution_limit": "single paired run; natural sampling variation remains possible",
              "source_trials": {stage: str(trial) for stage, trial in trials.items()}}
    result["delta"] = {"input_tokens": result["input_tokens"]["r1"] - result["input_tokens"]["baseline"],
                       "output_tokens": result["output_tokens"]["r1"] - result["output_tokens"]["baseline"],
                       "tool_calls": result["tool_calls"]["r1"] - result["tool_calls"]["baseline"],
                       "duration_sec": round(result["duration_sec"]["r1"] - result["duration_sec"]["baseline"], 3)}
    return result


def run(result_root: Path, metrics_dir: Path) -> list[dict[str, Any]]:
    records = []
    for benchmark in BENCHMARKS:
        folder = result_root / benchmark
        stages = json.loads((folder / "stage_manifest.json").read_text())["stages"]
        jobs = {stage: Path(stages[stage]["job"]) for stage in ("baseline", "candidate")}
        skill = Path(json.loads((folder / "m4/candidate_manifest.json").read_text())["skill_path"])
        rules = _rules(skill)
        with (folder / "val/diff.csv").open(newline="", encoding="utf-8") as source:
            for diff in csv.DictReader(source):
                if diff["transition"] in {"pass_to_fail", "fail_to_pass"}:
                    normalized = {**diff, "r1_reward": diff["candidate_reward"],
                                  "r1_input_tokens": diff["candidate_input_tokens"],
                                  "r1_output_tokens": diff["candidate_output_tokens"],
                                  "r1_duration_sec": diff["candidate_duration_sec"],
                                  "r1_error": diff["candidate_error"]}
                    records.append(audit_case(benchmark, normalized, {"baseline": jobs["baseline"], "r1": jobs["candidate"]},
                                              rules, metrics_dir))
    if len(records) != 14 or Counter(r["transition"] for r in records) != {"pass_to_fail": 10, "fail_to_pass": 4}:
        raise ValueError("R1 changed-task set is not the expected 10 regressions and 4 fixes")
    path = result_root / "REGRESSION_AUDIT.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
    with (result_root / "REGRESSION_AUDIT.csv").open("w", newline="", encoding="utf-8") as target:
        fields = ["benchmark", "task_id", "task_type", "task_complexity", "transition",
                  "baseline_reward", "r1_reward", "delta_input_tokens", "delta_output_tokens",
                  "delta_tool_calls", "delta_duration_sec", "skill_read", "diagnosis_type", "attribution_confidence"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow({"benchmark": item["benchmark"], "task_id": item["task_id"],
                             "task_type": item["task_type"], "task_complexity": item["task_complexity"],
                             "transition": item["transition"],
                             "baseline_reward": item["official_reward"]["baseline"],
                             "r1_reward": item["official_reward"]["r1"],
                             "delta_input_tokens": item["delta"]["input_tokens"],
                             "delta_output_tokens": item["delta"]["output_tokens"],
                             "delta_tool_calls": item["delta"]["tool_calls"],
                             "delta_duration_sec": item["delta"]["duration_sec"],
                             "skill_read": bool(item["skill_presence"]["explicit_read_event_ids"]),
                             "diagnosis_type": item["diagnosis_type"],
                             "attribution_confidence": item["attribution_confidence"]})
    write_json(result_root / "REGRESSION_AUDIT_SUMMARY.json", {
        "changed_tasks": len(records), "transitions": dict(Counter(r["transition"] for r in records)),
        "by_benchmark": dict(Counter(r["benchmark"] for r in records)),
        "diagnoses": dict(Counter(r["diagnosis_type"] for r in records)),
        "skill_explicit_reads": sum(bool(r["skill_presence"]["explicit_read_event_ids"]) for r in records),
        "val_data_policy": "read-only regression diagnosis; excluded from M3/M4 and Train ablation selection"})
    (result_root / "REGRESSION_AUDIT.md").write_text(render_markdown(records), encoding="utf-8")
    return records


def render_markdown(records: list[dict[str, Any]]) -> str:
    lines = ["# R1 Regression Audit", "",
             "同一 Val task_id 的无 Skill 与 R1 官方运行配对。只诊断本轮评测；此文件与 JSONL 不作为 M3/M4 或 Train 消融的输入。",
             "里程碑先按任务动作类型对齐，再定位首个不同事件。单次随机执行无法证明 Skill 因果效应。",
             "参考方法：[SkillTriage 原论文](https://arxiv.org/abs/2608.11888)。", "",
             "| Benchmark | task_id | 类型/复杂度 | Reward B→R1 | Δ输入/输出 Token | Δ工具/秒 | 读取 Skill | 诊断/置信度 |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for item in records:
        d = item["delta"]
        read = "是" if item["skill_presence"]["explicit_read_event_ids"] else "仅元数据"
        lines.append(f"| {item['benchmark']} | {item['task_id']} | {item['task_type']} / {item['task_complexity']} | "
                     f"{item['official_reward']['baseline']:g}→{item['official_reward']['r1']:g} | "
                     f"{d['input_tokens']:+d}/{d['output_tokens']:+d} | {d['tool_calls']:+d}/{d['duration_sec']:+g} | "
                     f"{read} | {item['diagnosis_type']} / {item['attribution_confidence']} |")
    for item in records:
        lines += ["", f"## {item['benchmark']} / {item['task_id']}", "",
                  f"- 结果：{item['transition']}；官方 Reward {item['official_reward']['baseline']:g} → {item['official_reward']['r1']:g}。",
                  f"- 成本：输入 Token {item['input_tokens']['baseline']} → {item['input_tokens']['r1']}；"
                  f"输出 Token {item['output_tokens']['baseline']} → {item['output_tokens']['r1']}；"
                  f"工具调用 {item['tool_calls']['baseline']} → {item['tool_calls']['r1']}；"
                  f"耗时 {item['duration_sec']['baseline']:g}s → {item['duration_sec']['r1']:g}s。",
                  f"- Skill：{item['skill_presence']['status']}，读取事件 {item['skill_presence']['explicit_read_event_ids']}；"
                  f"关联规则 {item['candidate_skill_rule'].get('rule_id') or '未确定'}（{item['candidate_skill_rule']['match_status']}）。",
                  f"- 首次里程碑差异：{item['first_key_behavior_difference']['change']}；"
                  f"Baseline {[(x['label'], x['event_id']) for x in item['first_key_behavior_difference']['baseline']]}；"
                  f"R1 {[(x['label'], x['event_id']) for x in item['first_key_behavior_difference']['r1']]}。",
                  f"- 诊断：{item['diagnosis_type']}；置信度 {item['attribution_confidence']}。"
                  "该置信度不表示已确认 Skill 因果；单次配对可能受采样波动影响。"]
        extra = item["diagnosis_evidence"]["benchmark_observations"]
        if item["benchmark"] == "spreadsheetbench":
            diff = extra["paired_script_diff"]
            lines.append(f"- 表格证据：脚本是否相同 {diff['same_script']}；"
                         f"Baseline/R1 脚本数 {len(extra['baseline']['scripts'])}/{len(extra['r1']['scripts'])}；"
                         f"工具错误事件 {extra['baseline']['tool_error_event_ids']} / {extra['r1']['tool_error_event_ids']}。")
            for script in diff["changed_scripts"][:2]:
                lines.append(f"  - 变更脚本 `{script['relative_path']}`，差分行 {script['changed_lines']}；完整代码差分见 JSONL。")
            lines.append(f"- 输出文件路径（命令中）：Baseline {extra['baseline']['workbook_paths_in_calls'][:6]}；"
                         f"R1 {extra['r1']['workbook_paths_in_calls'][:6]}。评分器记录见 JSONL 的 `verifier_note`。")
        elif item["benchmark"] == "alfworld":
            a, b = extra["baseline"], extra["r1"]
            lines.append(f"- 环境动作：{a['actions']} → {b['actions']}；连续重复 {len(a['repeated_action_event_ids'])} → "
                         f"{len(b['repeated_action_event_ids'])}；无进展反馈 {len(a['no_progress_event_ids'])} → "
                         f"{len(b['no_progress_event_ids'])}；终态 won {a['terminal_won_observed']} → {b['terminal_won_observed']}。")
        else:
            a, b = extra["baseline"], extra["r1"]
            lines.append(f"- 数学可见结果：最终选项 {a['final_option']} → {b['final_option']}；"
                         f"可见消息字符 {a['visible_reasoning_chars']} → {b['visible_reasoning_chars']}；"
                         f"答案格式 {a['answer_format_present']} → {b['answer_format_present']}。"
                         "没有独立数学证明时，不把选项变化归因为某一步推导错误。")
    lines += ["", "## 使用边界", "", "Val 轨迹、评分器备注、代码差分只保存在忽略提交的审计产物中。"
              "规则修订与 A/B/C/D 消融只能使用 Train 轨迹和独立 Train 重放。"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/r1"))
    parser.add_argument("--metrics", type=Path, default=Path("../metrics"))
    args = parser.parse_args(argv)
    records = run(args.results.resolve(), args.metrics.resolve())
    print(json.dumps({"changed_tasks": len(records),
                      "regressions": sum(r["transition"] == "pass_to_fail" for r in records),
                      "fixes": sum(r["transition"] == "fail_to_pass" for r in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
