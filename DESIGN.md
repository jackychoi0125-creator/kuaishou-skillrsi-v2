# 统一 Trace 分析设计（M1 / M2）

## 实际调用链

`skillrsi-m1 prepare` 将原始 `baseline-original.csv` 的 Train ATIF 包装为 JSONL，保留源文件路径与 SHA-256。`skillrsi-m1 validate` 默认读取正式合并的 `baseline.csv`；四条补跑任务读取正式补跑 Trace，同时记录原始 Trace 路径。输出 `ValidatedTrace` 到 `results/m1/validated.jsonl` 和完整 `audit.json`。

`skillrsi-m2 build` 读取 `ValidatedTrace`，用 `adapters.py` 提取公开任务特征、从原始 ATIF 事件提取五维行为标签，并在 `m2_diagnostics.py` 中封装旧约束规则作为失败根因**假设**。只有旧 Judge 记录与当前 Trace SHA-256、事件 ID 完全匹配时才导入既有确认根因。随后生成正向策略候选组、失败模式组、分级正反对照组与 L2 深读视图。

```
Train Metrics + original/retry ATIF
    → M1 validate: trace_validity × task_outcome × reliability_flags
    → M2 profile: task_signature × behavior_signature × outcome × diagnosis
    → positive / failure / contrast groups
    → selected L2 deep views (trace_id + event_id → original ATIF)
```

M1 不调用模型，也不判定策略或根因。旧 `skillrsi-m1 diagnose` 保留为历史输出兼容入口；新流程不调用它。旧 M2 `--legacy-build` 仍可重现原有聚类，默认 `build` 写新结构并输出旧字段转换视图。

## 统一 Schema 与证据边界

M1 的 `trace_validity` 为 `valid / partial / invalid`，`task_outcome` 为 `success / failure / uncertain`，两者独立。奖励为 1 的任务即使曾出现可恢复的 429，仍保留成功与异常标记。未成功完成的终态模型错误在原始版本可为 `partial + uncertain`；正式补跑使用新的完整轨迹，保留补跑来源。

M2 的每条 `trace_profile` 使用相同的 `task_signature`、五维 `behavior_signature`、`outcome` 和 `diagnosis` 字段。行为标签必须指向真实 `event_id`，状态只允许 `observed / not_observed / not_observable / unknown`。成功不自动等于高效；失败不自动等于某行为造成结果。诊断区分观察现象、未证实假设和已确认根因。LiveMath 答错本身不构成数学错误证据。

正反匹配按 A（同 Train 任务不同执行）→ B（操作、约束、复杂度、实际题意相近）→ C（同家族且仍有一定操作与题意相似）判断。无足够相似度输出 `no_matched_positive`，尤其不会用不同数学题型强配 LiveMath。对照中的行为差异标为 `hypothesis_only`。

## 兼容性与数据来源

正式 Train 154 条由 `baseline.csv` 提供：原始 CSV 为 96 成功、58 失败；四条模型错误补跑替换后，2 条由失败变成功，形成 98 成功、56 失败。`audit.json` 逐题保存四条替换的原始/补跑路径。旧 M1 只分析原始轨迹得到 96/54/4，未丢题，但将四条基础设施失败从有效任务失败中隔离。

新文件为 `validated.jsonl`、`audit.json`、`trace_profiles.jsonl`、`positive_strategy_groups.json`、`failure_pattern_groups.json`、`contrast_groups.json`、`l2_contrasts.json`、`m2_summary.json` 和 `deep_views/*.json`。旧 `records.jsonl`、`summary.json` 原样保留；M2 仍提供 `l0.jsonl`、`groups.jsonl`、`selection.jsonl`、`evaluation.json`、`views/*.json` 的字段转换视图。原始 ATIF 不修改。

## 成本与局限

新 M1/M2 默认均为确定性程序，新增模型调用和计费 Token 都为 0；M2 记录各阶段耗时、深读数量、视图长度比例和估算上下文 Token。旧 Judge 的 15 个案例、98,827 Prompt Token 和 33,874 Completion Token 被标为历史导入，不重复计费。L2 视图可按 `trace_id`、`event_id` 回取原文；超长 Agent 消息只保留分析相关的首尾内容，原始事件不删除。

行为标签仍是规则观测，不等于有效策略；成功组是可复用策略**候选**。没有独立验证的数学根因继续保持 `undetermined`。匹配分数是排序启发式，B/C 级配对仍需后续独立验证，不能作为因果结论。

## M3 前置质量门控

`group_quality.json` 为每组报告成员数、代表轨迹、任务与行为一致性、最近邻组相似度、证据事件数和 `reuse_readiness`。单例固定为 `single_case_hypothesis`；即使多例一致，也仅可成为 `candidate_experience`，因果状态保持 `hypothesis_only`。

`evidence_audit.json` 将本次入选 L2 的事件逐条与原始 ATIF 比对，统计已确认根因、正反差异和 ALFWorld 成功终态反馈的保留情况。三个 Benchmark 均尽量纳入成功、失败样本；LiveMath 不满足题意相似阈值时保留两类样本但不建立正反配对。长数学推导的视图会标记需要回读全文。
# R1 M3/M4 contract

M3 只读取 M2 的 Train profiles/groups、质量审计和由 `trace_id:event_id` 指向的原始事件。`TraceStore` 在每次读取时重新验证 SHA-256，绝不执行轨迹中的文本。`EvidenceSelector` 优先多轨迹组及可核验对照；`InsightExtractor` 用 Kimi 输出候选，`InsightConsolidator` 记录 ADD/EDIT/AGREE/REMOVE，`EvidenceVerifier` 再核对每个事件和条件。每条 Insight 的 `causal_status` 仍为 `hypothesis_only`，证据支持的是观察事实，不是因果证明。LiveMath 的领域数学结论不进入 R1 Skill；只有轨迹中直接可观察的程序性检查可作为单例假设。

M4 只接收已验证的 Insight，生成一份有触发条件、步骤、禁忌和验证条件的 Harbor `SKILL.md`。静态审计拒绝题目 ID、私有答案路径、具体动作路线和超长规则；候选快照与 Insight ID 保留在 manifest。每个 Benchmark 独立从无 Skill 基线起步。

`skillrsi-loop` 保留阶段 manifest 和官方 Harbor Run 目录。Gate 核对同一 Val task_id、官方 result.json 与 summary.csv 一致、模型和 Kit 配置一致，然后按 Fail→Pass/Pass→Fail 等四类统计。净增益不大于零或有回退时，`version_state.json` 指向无 Skill 基线；候选仍保留供审计。Val 的题目内容、轨迹或评分细节不会传回 M3/M4。

