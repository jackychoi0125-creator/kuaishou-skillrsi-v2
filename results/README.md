# M1 Train 诊断结果

本地运行产物位于 `results/m1/`，默认不纳入 Git，因为 `evidence.excerpt` 会引用原始任务轨迹。运行 `skillrsi-m1 prepare` 和 `skillrsi-m1 diagnose --llm --workers 2` 可重新生成。

本次三套 Train 数据共 154 题：96 题成功、4 题因基础设施错误隔离、54 题为有效任务失败。有效失败中，4 题有可回查的关键步骤和证据事件，50 题保留 `root_cause: "undetermined"`。其中 LiveMath 没有把仅凭最终答错的题目强行解释成数学推理错误。

本地 `results/m1/summary.json` 汇总完整计数和 Judge Token；`records.jsonl`、`failures.jsonl` 中的每个受支持根因均引用原始 ATIF 的事件 ID、步骤号和证据摘录。
