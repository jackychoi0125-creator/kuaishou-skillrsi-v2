# SkillRSI v2 — M1 数据可靠性过滤与 M2 Trace 分类压缩

本仓库实现 Trace2Skill 的 M1 和 M2：从三个 Benchmark 的 **Train 原始 ATIF Trace**、原始逐题 Metrics 和公开任务约束生成可追溯的 `FailureRecord`，再生成按错误机制聚类的 `EvidenceGroup` 和少量代表轨迹视图。它尚不生成 Skill，也不把 reward=0 自动解释为某个推理错误。

## 输入与运行

默认假设本仓库位于 `rsiV2/kuaishou-skillrsi-v2/`，上一层已有 `metrics/*-baseline-original.csv`、Kit 原始 trial 和 `model.txt`。原始 CSV 仅用于关联任务、分数和基础设施状态；诊断步骤来自逐题 `agent/trajectory.json`。准备命令把 Train 轨迹原样包入 Kit 风格的 ATIF JSONL，并记录原文件 SHA-256：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip wheel setuptools
.venv/bin/python -m pip install -e .
.venv/bin/skillrsi-m1 prepare
.venv/bin/skillrsi-m1 diagnose --llm --workers 2
.venv/bin/skillrsi-m2 build
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

无需模型的规则检查可运行 `skillrsi-m1 diagnose`。模型阶段默认读取上一层的 `model.txt`，调用 Kimi K2.6，默认关闭 thinking 以减少长请求；可加 `--kimi-thinking` 使用网关默认 thinking 模式。`--model-config` 可指向另一配置文件。HTTP 429、读取超时和连接重置均会重试；失败的 Judge 记录可用 `--resume` 补跑。`--metrics-dir`、`--input-dir`、`--output-dir` 可覆盖默认路径。

输出在 `results/m1/`：

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

## 边界

这是受 AgentDebug 和 AgentRx 思路启发的轻量实现，不是两套项目的完整复刻。规则发现的局部错误不等于最终失败根因；LLM Judge 可以拒绝候选。没有独立验证证据的 LiveMath 失败保留为待验证。下一阶段可以加入人工标注集，对置信度与关键步骤准确率做校准。

## M2：渐进式 Trace 视图

`skillrsi-m2 build` 默认读取 `results/m1/records.jsonl` 和上一层原始 Metrics，输出 `results/m2/`：

- `l0.jsonl`：全部 Train 任务的 Metadata、Metrics、M1 摘要与成本估计。L0 不打开 ATIF；LiveMath 题型仅从公开问题文本用关键词粗标。
- `groups.jsonl`：按已支持根因、规则候选、可观察信号、未确定根因等机制分组的 `EvidenceGroup`。`mechanism_status` 区分已支持根因与仅观察到的现象；成功案例按任务家族分组，并可作为同类失败的正例指针。
- `selection.jsonl` 和 `views/*.json`：默认最多 12 条 L2 案例，每组最多 2 条。优先覆盖所有 M1 Judge 支持的案例、三个 Benchmark，再按频率、覆盖代理、不确定性、成本和家族/规模多样性挑选。视图只截取关键步骤附近事件；ALFWorld 另带动作、观察、状态文本变化、下一动作的轻量序列。
- `evaluation.json`：全量读取与渐进加载的上下文 Token **估算值**，以及 M1 已支持关键事件在 L1、L2 的保留率。

原始 ATIF 不修改。压缩视图中每个事件均保留 `trace_id`、`event_id`、原文件 SHA-256 和摘录哈希；可以回取原事件与完整 ATIF 步骤：

```bash
.venv/bin/skillrsi-m2 fetch --benchmark alfworld --task-id train__0000 --event-id s11.obs1
```

示例 ID 仅展示命令格式。可用 `--max-l2`、`--per-group`、`--window`、`--excerpt-chars` 调整读取预算。M2 不调用模型，因此 `evaluation.json` 的 Token 是上下文长度代理值，不是实际 API 计费；保留率也仅衡量 M1 已支持的事件指针，不保证未知根因被正确解释。设计参考 [SkillOpt-Lite 论文](https://www.microsoft.com/en-us/research/publication/skillopt-lite-better-and-faster-agent-self-evolution-via-one-line-of-vibe/) 的文件式轨迹探索和跨轨迹共性挖掘；独立验证门控留给后续阶段。

