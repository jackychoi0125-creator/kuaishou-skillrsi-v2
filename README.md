# SkillRSI v2 — 统一正反轨迹画像与对照分析

## R1 自进化闭环

`skillrsi-loop r1 --benchmark all --mode real --resume` 顺序执行每个 Benchmark 的 M3 经验提取、M4 单候选 Skill 生成、官方 Harbor Val 基线复测、候选评测和逐题门控。可用 `--stop-after m3|m4|baseline|candidate` 中断，用 `--resume` 幂等恢复。`--mode dry-run` 只检查可选证据，不产生模型调用或虚构分数。

M3 只读取 `results/m2/` 中的 Train 画像、分组和原始 Trace 引用。它验证源文件 SHA-256 与 event ID，单例保留为假设；M3/M4 的 Kimi 原始请求、响应、服务端 usage、重试和耗时存放在忽略提交的 `results/r1/cache/`。M4 只消费已核验 Insight，产生 `workspace/skills/<benchmark>/r1/<skill-name>/SKILL.md` 与版本快照。无可信经验时写 `no_candidate` 并停止该 Benchmark 的候选评测。

Val 运行只使用官方 Harbor 评分。`results/r1/<benchmark>/val/diff.csv` 与 `acceptance.json` 对齐同一 task_id；缺题、配置不一致、评分异常会阻止接纳。门控要求净提升大于零且无回退题。旧参考分数不直接当作当前配对基线。`results/r1/RUN_REPORT.md` 汇总实际运行结果。生成产物和原始 Trace 均被 `.gitignore` 排除。

本仓库实现 Trace2Skill 的 M1 和 M2：M1 对三个 Benchmark 的 Train Trace 做确定性可靠性与结果审计；M2 对成功和失败轨迹采用同一任务、行为、结果、诊断 Schema，生成正向策略候选、失败模式和分级正反对照。正式 Train Baseline 的 154 条任务包含 98 成功、56 失败；M2 不把 reward=0 自动解释为某个推理错误。

## 输入与运行

默认假设本仓库位于 `rsiV2/kuaishou-skillrsi-v2/`，上一层已有正式 `metrics/*-baseline.csv`、原始 `*-baseline-original.csv` 和 Kit trial。M1 使用正式逐题 Metrics、原始 ATIF JSONL 与补跑源轨迹；M2 只读取 Train 任务的公开输入和轨迹。准备命令把原始 Train 轨迹原样包入 ATIF JSONL，并记录源文件 SHA-256：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip wheel setuptools
.venv/bin/python -m pip install -e .
.venv/bin/skillrsi-m1 prepare
.venv/bin/skillrsi-m1 validate
.venv/bin/skillrsi-m2 build
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

新 M1/M2 默认不调用模型，支持 `--dry-run`。M1 默认读取正式合并的 `*-baseline.csv`，用 `--metrics-variant baseline-original` 可复现原始运行口径。历史命令 `skillrsi-m1 diagnose --llm --resume` 仍保留，但不属于新的 M1 职责；旧 Judge 的受支持根因由 M2 校验 Trace SHA 与事件引用后导入。`skillrsi-m2 build --legacy-build` 可运行旧版 M2 聚类。

新输出在 `results/m1/validated.jsonl`、`audit.json` 和 `results/m2/trace_profiles.jsonl`、`positive_strategy_groups.json`、`failure_pattern_groups.json`、`contrast_groups.json`、`l2_contrasts.json`、`m2_summary.json`、`deep_views/`。已有 `records.jsonl` 等历史文件不删除；M2 还生成旧字段的转换视图。完整数据流、Schema 和兼容性见 [DESIGN.md](DESIGN.md)。

M3 前置质量审计输出 `results/m2/group_quality.json` 与 `evidence_audit.json`：统计单例、组内一致性、近似重复组、代表轨迹及复用准备状态，并把 L2 事件逐条与原始 ATIF 比对。单例组只保留为假设；成功轨迹中的重复动作也不会自动成为有效策略。LiveMath 弱相关题目不强制组成正反例。

L2 预算用 `--max-deep-read-traces`、`--max-positive-traces`、`--max-negative-traces`、`--max-contrast-pairs`、`--max-llm-input-tokens` 调整；`--dry-run` 只统计不写文件。M2 默认不调用 LLM，`max_llm_input_tokens` 是准备给后续语义分析的视图长度代理预算，不是实际 API 消耗。原始事件可用 `skillrsi-m2 fetch --benchmark alfworld --task-id train__0017 --event-id s19.call4` 回查。

## 历史 M1 诊断（兼容）

旧 `skillrsi-m1 diagnose` 的输出仍在 `results/m1/`：

- `records.jsonl`：全部 154 条 Train 任务，含成功和基础设施过滤结果。
- `failures.jsonl`：有效任务失败的诊断记录。
- `filtered.jsonl`：成功、模型/执行/评分异常或无效 Trace。
- `summary.json`：各 Benchmark 结果数、诊断状态和 Judge Token 用量。

原始 Train ATIF JSONL 放在 `data/train/`。两处生成数据默认被 `.gitignore` 排除，因为轨迹与证据摘录可能包含任务内容。代码不会读取 `solution/`、`tests/` 或参考答案；任务上下文只读取 `instruction.md`、`task.toml` 和 `environment/public/`。

## 三层处理

1. **Reliability Filter**：检查 Trace 存在性、结构、任务 ID 和 SHA-256；隔离终态模型错误、429、执行与评分异常。已获得 reward 1 的任务仍按成功处理，并保留异常标记。
2. **Error Classifier**：使用 [AgentDebug 的五模块错误分类](https://github.com/ulab-uiuc/AgentDebug/blob/main/README.md#-error-taxonomy)。规则候选先列出可观察的错误；可选 Kimi 分类器只能在证据支持的模块/类型中选择。
3. **Failure Localizer**：借鉴 [AgentRx 的轨迹 IR → 约束检查 → Judge](https://github.com/microsoft/AgentRx#pipeline-stages) 流程。约束检查会生成稳定的 ATIF event ID；Judge 只能选择真实存在的候选事件。证据不足时输出 `root_cause: "undetermined"`。

`confidence` 是待校准的诊断分值，并非客观概率。低于 0.65 的记录标为 `needs_review`。LiveMath 的错误选项不会自动触发数学根因诊断；当前仅将可直接核实的最终答案格式问题作为候选。

## FailureRecord 主要字段

```json
{
  "benchmark": "alfworld",
  "task_id": "train__0013",
  "outcome": "task_failure",
  "trace_validity": "valid",
  "error_module": "action",
  "error_type": "invalid_action",
  "critical_steps": [17],
  "supporting_events": ["s17.call2", "s17.obs2"],
  "root_cause": "rejected_environment_action",
  "confidence": 0.8,
  "diagnosis_status": "judge_supported"
}
```

每条记录还包含 `evidence` 摘录、原始 Trace 路径与 SHA-256、原 trial 路径和可靠性标记。上例展示结构，不代表 `train__0013` 最终一定获得该诊断。

失败记录还包含 `observed_failure_modes`、`candidate_count`、`candidate_rule_ids` 和 `diagnostic_gap`。这些字段把可直接观察的失败现象与可归因的根因分开。例如 ALFWorld 的 `action_budget_exhausted` 来自环境返回的 `steps/max_steps/done/won`；它说明终态，不自动证明哪一步规划造成耗尽。LiveMath 最终消息含有方框选项但得分为零的情况只标记 `boxed_option_present_reward_zero`，根因仍待独立数学验证。

## 边界

这是受 AgentDebug 和 AgentRx 思路启发的轻量实现，不是两套项目的完整复刻。规则发现的局部错误不等于最终失败根因；LLM Judge 可以拒绝候选。没有独立验证证据的 LiveMath 失败保留为待验证。下一阶段可以加入人工标注集，对置信度与关键步骤准确率做校准。

## 历史 M2 视图（兼容）

`skillrsi-m2 build --legacy-build` 读取 `results/m1/records.jsonl` 和上一层原始 Metrics，输出旧版 `results/m2/`：

- `l0.jsonl`：全部 Train 任务的 Metadata、Metrics、M1 摘要与成本估计。L0 不打开 ATIF；LiveMath 题型仅从公开问题文本用关键词粗标。
- `groups.jsonl`：按已支持根因、规则候选、可观察信号、未确定根因等机制分组的 `EvidenceGroup`。`mechanism_status` 区分已支持根因与仅观察到的现象；成功案例按任务家族分组，并可作为同类失败的正例指针。
- `selection.jsonl` 和 `views/*.json`：默认最多 12 条 L2 案例，每组最多 2 条。优先覆盖所有 M1 Judge 支持的案例、三个 Benchmark，再按频率、覆盖代理、不确定性、成本和家族/规模多样性挑选。视图只截取关键步骤附近事件；ALFWorld 另带动作、观察、状态文本变化、下一动作的轻量序列。
- `evaluation.json`：全量读取与渐进加载的上下文 Token **估算值**，以及 M1 已支持关键事件在 L1、L2 的保留率。

原始 ATIF 不修改。压缩视图中每个事件均保留 `trace_id`、`event_id`、原文件 SHA-256 和摘录哈希；可以回取原事件与完整 ATIF 步骤：

```bash
.venv/bin/skillrsi-m2 fetch --benchmark alfworld --task-id train__0000 --event-id s11.obs1
```

示例 ID 仅展示命令格式。可用 `--max-l2`、`--per-group`、`--window`、`--excerpt-chars` 调整读取预算。M2 不调用模型，因此 `evaluation.json` 的 Token 是上下文长度代理值，不是实际 API 计费；保留率也仅衡量 M1 已支持的事件指针，不保证未知根因被正确解释。设计参考 [SkillOpt-Lite 论文](https://www.microsoft.com/en-us/research/publication/skillopt-lite-better-and-faster-agent-self-evolution-via-one-line-of-vibe/) 的文件式轨迹探索和跨轨迹共性挖掘；独立验证门控留给后续阶段。

