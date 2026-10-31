# M1 设计要点

## 数据边界

- 以原始 ATIF `steps` 为定位依据；Metrics CSV 只提供结果和基础设施状态。
- 使用原始无 Skill Train 运行，不使用补跑合并后的分数替代原始轨迹。
- `task_id` 关联 Metrics 与 JSONL；`source_trace_path` 和 SHA-256 支持回查。
- 仅读取 Agent 可见的 `instruction.md` 与公开环境文件、任务运行约束。参考答案与评分脚本不进入诊断提示。

## 证据门控

约束检查器输出候选 `rule_id`、ATIF event ID 和步骤号。分类器只能选择候选支持的 AgentDebug 模块与类型；Judge 只能引用候选已有的事件 ID。低置信度候选和无法排除后续修复的脚本错误维持 `needs_review`。所有根因字段必须能回指 `evidence`。

## Benchmark 差异

- SpreadsheetBench：识别脚本/工具明确报错及无输出文件；早期报错后仍有工具调用时降级，避免把已修复的错误认作根因。
- ALFWorld：读取 `bridge.sh send` 动作、环境观察和 admissible commands；仅在环境也明确拒绝时，才把不在命令列表中的动作标为无效。重复且无状态变化、处理失败后放置也可成为候选。
- LiveMath：无独立数学验证时不判断推导根因；仅检测最终答案格式的可观察违约。

## 待校准事项

AgentDebug 的 17 个错误名称按模块使用；`hallucination` 在 Memory/Reflection 两模块均可出现。当前置信度由规则分值与 Judge 分值取较小值，不能解释为概率。后续需要人工核查样本估计 precision、关键步骤准确率，并建立稳定的接受阈值。
