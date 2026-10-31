# M1 Train 诊断结果

当前正式合并版请看 `results/m1/audit.json` 和 `validated.jsonl`：154 条 Train 全部处理，154 条 Trace 有效，98 条成功、56 条失败。四条模型错误补跑的原始与最终 Trace 路径均在审计里。旧 `results/m1/records.jsonl` 和 `summary.json` 是原始运行的历史诊断，口径为 96 成功、54 有效任务失败、4 基础设施失败。

新 M2 结果见 `results/m2/m2_summary.json`：98 条成功和 56 条失败都进入统一画像；当前生成 36 个正向策略候选组、30 个失败模式组、20 个分级正反配对，另有 36 条失败明确无合适正例。L2 在 30,000 Token 代理预算内深读 4 条成功、6 条失败，含 3 对正反对照；具体局部比较在 `l2_contrasts.json`。所有生成数据仍默认不纳入 Git。

同一份原始 Train 输入上，旧 M2 的渐进上下文代理值为 99,953 Token，新流程的旧字段兼容视图连同 L2 对照为 80,939 Token，减少约 19.0%；正式合并版为 78,885 Token。此处均为字符长度估算，不是 API 计费。新流程 M1/M2 均未发起模型调用；旧 Judge 的 98,827 Prompt Token 与 33,874 Completion Token 仅作为历史证据导入。

本地运行产物位于 `results/m1/`，默认不纳入 Git，因为 `evidence.excerpt` 会引用原始任务轨迹。运行 `skillrsi-m1 prepare` 和 `skillrsi-m1 diagnose --llm --workers 2` 可重新生成。

本次三套 Train 数据共 154 题：96 题成功、4 题因基础设施错误隔离、54 题为有效任务失败。有效失败中，4 题有可回查的关键步骤和证据事件，50 题保留 `root_cause: "undetermined"`。其中 LiveMath 没有把仅凭最终答错的题目强行解释成数学推理错误。

本地 `results/m1/summary.json` 汇总完整计数和 Judge Token；`records.jsonl`、`failures.jsonl` 中的每个受支持根因均引用原始 ATIF 的事件 ID、步骤号和证据摘录。

M2 运行产物位于 `results/m2/`，同样默认不纳入 Git。当前 154 条 Train 任务生成 27 个 EvidenceGroup 和 12 条代表视图；4 条 Judge 支持的根因共有 9 个关键事件，L1 和 L2 均保留全部事件引用。按估算的分析上下文长度，渐进加载比读取全量 Trace 少约 94.76%；此数字是 Token 代理值，不是实际 API 用量。重新生成：`skillrsi-m2 build`。

M1 新增诊断缺口统计：10 条缺少单元格级输出差异证据，29 条缺少独立数学验证，11 条虽有规则候选但未获 Judge 因果支持。54 条任务失败均有可观察的失败现象标记；现象标记不等于根因分类。
