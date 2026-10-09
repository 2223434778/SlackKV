# 2026-10-09 自主研究证据

详见 [研究报告](../../docs/AUTONOMOUS_RESEARCH_REPORT_20261009.md) 和 [决策日志](../../docs/AUTONOMOUS_RESEARCH_LOG.md)。

8 次完整生命周期运行，各 40 Agent 请求、32 resume、4×2048 后台输出；一次模型加载前的 import failure 单独保留。所有 server/launcher 原始日志以 `.log.gz` 保存，长请求与 oracle JSON 以 `.json.gz` 无损保存；服务器原始文件未删除。`SHA256SUMS.json` 记录服务器导出文件的原始字节 hash：对于仅保留 gzip 的 JSON，先解压再校验对应 `.json` 的 hash。本目录关闭 Git 文本换行转换；原始 patch/Prometheus 格式保留，不清除证据中的空白。

- `execution.json`：命令/开关、状态、运行时指纹和保守 GPU 成本；后期 runner 还含累计预算检查、runner hash。
- `runs/*/run.json.gz`：未经改写的请求时间、哈希、trace、控制器与后台记录。
- `analysis.json`：独立分析；未采样运行的 allocator 数据为 UNKNOWN。
- `output_tokens.jsonl`、`cold_oracle.json.gz`：后续运行的逐 token 输出、两个敏感输入以及冷/并发复核。
- `runtime_base.patch`：vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` 上既有工作区差异，SHA256 `fe7250bce3813e53ebd881c60956a2d4c65df28aa73d9a4ac488c0b4d00ccbb8`；未跟踪的 dev API 在 `runtime_untracked/`，必须一并恢复。新增 delta 位于 `experiments/autonomous_admission/*.patch`，路径按实际隔离副本调整。
- `runtime_regression.txt`：既有底层测试 15 passed，包含非致命启动警告。
- `round_summary.json`：累计成本、逐 run 固定路径机会和匹配审计。首轮执行对照有输入漂移，只归入 structure-only，不能与反序输入匹配对照混合声称重复验证。

离线复算（不加载模型）：

```bash
python -m experiments.autonomous_admission.analyze results/autonomous_admission/diagnostic-k2
python -m experiments.autonomous_admission.compare results/autonomous_admission/eager-repeat results/autonomous_admission/default-repeat --purpose execution --output /tmp/execution_reverse_check.json
python -m experiments.autonomous_admission.compare results/autonomous_admission/default-repeat results/autonomous_admission/full-default --purpose full --output /tmp/full_check.json
python -m experiments.autonomous_admission.round_summary results/autonomous_admission
python -m pytest tests/test_autonomous_admission.py tests/test_research_reassessment.py -q
```

运行环境为现有 Ubuntu 服务器、GPU0 RTX A6000、Qwen2.5-1.5B-Instruct/FP16、TP1、2 GiB GPU KV、8 GiB CPU KV；绝对 pause，K2。GPU1 未用于本轮。

复现实测先恢复记录的 vLLM 基础差异与 dev API，再用已存在的可执行 `apply_patch` 应用两个 delta；在当前隔离环境中运行 `python -m experiments.autonomous_admission.run --name <全新名称> ...`。名称不得重复覆盖历史目录。单次 watchdog 留 90 秒清理，总上限 30 分钟；累计达到预算时 runner 拒绝新运行。invariant 仅作正确性对照，不与非 invariant 数字混比。

文件中早期 `purpose` 的默认文字未随开关更新，实际实验用途以冻结协议、`allocator_trace`、命令和研究日志为准。`valid` 是原生命周期完整性判据，不意味着输出逐 token 等价或研究目标已经达成。
