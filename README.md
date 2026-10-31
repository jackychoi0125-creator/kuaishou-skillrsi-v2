# SkillRSI v2 — M1 数据可靠性过滤

本仓库当前实现 Trace2Skill 的第一阶段：从三个 Benchmark 的 **Train 原始 ATIF Trace**、原始逐题 Metrics 和公开任务约束生成可追溯的 `FailureRecord`。它不生成 Skill，也不把 reward=0 自动解释为某个推理错误。

## 输入与运行

默认假设本仓库位于 `rsiV2/kuaishou-skillrsi-v2/`，上一层已有 `metrics/*-baseline-original.csv`、Kit 原始 trial 和 `model.txt`。原始 CSV 仅用于关联任务、分数和基础设施状态；诊断步骤来自逐题 `agent/trajectory.json`。准备命令把 Train 轨迹原样包入 Kit 风格的 ATIF JSONL，并记录原文件 SHA-256：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip wheel setuptools
.venv/bin/python -m pip install -e .
.venv/bin/skillrsi-m1 prepare
.venv/bin/skillrsi-m1 diagnose --llm --workers 2
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

