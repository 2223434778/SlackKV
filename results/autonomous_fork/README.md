# 第三轮证据与复现

报告：[AUTONOMOUS_RESEARCH_ROUND3.md](../../docs/AUTONOMOUS_RESEARCH_ROUND3.md)。本目录为 synthetic fork 诊断、独立内容 oracle 和历史 hybrid workload 回归；没有新的 serving candidate。

- `execution.json`：每个GPU运行的参数、源码指纹、状态、含启动/编译/退出成本；七个运行共0.20048675264813934 GPU·小时，失败计入。`COMPLETE`不等于内容/质量全面正确。
- `inputs.jsonl.gz`、`requests.json.gz`：实际prompt与输出。五个完整fork运行的51输入严格一致，固定输出128/1024；invariant run单独处理。failed run输出缺失保持UNKNOWN。
- `episodes.json`：客户端时间与reset响应；retry旧版source_delta派生零值为INVALID，不可使用。实际恢复由raw source和counter共同核对。
- `metrics_after.prom`：五个forkrun的**计时负载后、oracle前**真实counter。H2D2016MiB、D2H2310MiB；missing series不是0。
- `server_logs/*.log.gz`：完整日志，不是筛选后的事件摘要。`native_offloading_scheduler.py.gz`与`runtime_scheduler_final.py.gz`保存完整运行时源码，`native_inflight_audit.json`给出已有去重位置。
- `cold_oracle.json.gz`：16个不同cache_salt的serial cold Prefill；raw trace与参考输出交叉验证。只证明被测输入在invariant路径的逐token身份。
- `legacy-regression-2/`：原40/32/BG8192工作量，Round2精确实际输入。它与fork不属于同一性能条件。
- `SHA256SUMS.json`：101个服务器文件的**原始字节**hash；仅gzip存档的文件先解压再核对。已是`.gz`的运行时源码按其压缩字节核对。服务器原文件完整保留。
- `analysis.json`、`cost_model.json`：服务器原派生结果，保持byte-exact；`*_local_recomputed.json`为本机独立复算。Python3.13浮点sum变化使成本差约3e-17；Phase0CSV的Git换行转换只改变byte SHA，不改变带宽值。最终成本复算额外列出第二轮footprint，不回写原结果。
- `independent_audit.json`、`analysis_failures.txt`、两次runtime test记录、GPU退出状态：审计与失败证据。本机archive解包曾拒绝已存在但相同的metadata，比较hash后跳过相同文件，无历史文件覆盖。

离线复算不加载GPU：

```bash
python -m experiments.autonomous_fork.audit results/autonomous_fork
python -m experiments.autonomous_fork.analyze results/autonomous_fork --output /tmp/round3-recomputed.json
python -m experiments.autonomous_fork.costs results/autonomous_fork --output /tmp/round3-costs.json
python -m pytest -q tests/test_autonomous_fork.py tests/test_autonomous_admission.py tests/test_autonomous_capacity.py tests/test_research_reassessment.py --basetemp=.pytest_cache/round3-reproduce
```

实测需要原有服务器、venv与cached模型。完整旧runtime基底记录在`results/autonomous_admission/runtime_base.patch`（vLLM commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`）及`runtime_untracked/`，再加两轮已保存deltas；本轮从已有Round2 runtime复制隔离目录，只有额外allocation诊断。也可在**新隔离副本**恢复本目录完整scheduler；先核对allocator/native connector指纹与`environment.json`一致。不要覆盖历史运行时。

以下使用尚不存在的新名称，避免覆盖历史；旧名称已存在，runner会拒绝：

```bash
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
cd /media/a6000/D/ZhaoBinHe/SlackKV
$PY -m experiments.autonomous_fork.run --name new-fork-cold --kv-gib 4 --gate 8 --background-phase cold
$PY -m experiments.autonomous_fork.run --name new-fork-decode --kv-gib 4 --gate 8 --background-phase decode
$PY -m experiments.autonomous_fork.run --name new-fork-8 --kv-gib 8 --gate 8 --background-phase decode
$PY -m experiments.autonomous_fork.run --name new-fork-k2 --kv-gib 4 --gate 2 --background-phase decode
$PY -m experiments.autonomous_fork.run --name new-fork-oracle --kv-gib 4 --gate 8 --background-phase decode --batch-invariant --oracle
$PY -m experiments.autonomous_admission.run --round-directory autonomous_fork --name new-legacy-regression --execution default --kv-gib 2 --trace --runtime-source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round3-source --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl.gz
```

新运行仍累计进入Round3同一ledger、上限4GPU·小时；不能删旧execution重置预算。名字和run角色改变时，批量audit里的oracle固定名称也应以新的独立目录/明确协议适配，不能默默把新run与旧run作候选配对。模型下载或环境迁移未验证；7B只有配置模型估计，无实测结果。
