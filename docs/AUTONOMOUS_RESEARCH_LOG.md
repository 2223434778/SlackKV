# 自主研究日志

## 2026-10-09：需求准入诊断

- 问题：历史 2 GiB KV 配置下 P95 Resume TTFT 约 6.4 秒，直接异步 KV 恢复仅约 60–84 ms。剩余时间的可优化部分尚未识别。
- 决策：先观察真实 allocator 检查，避免根据非原子的控制器快照推断容量因果。保持既有 BGv2 workload、K2 和 OnDemand，使用独立运行时副本，不改历史环境。
- 预期收益：本轮诊断本身不声称加速；确定容量失败、门控和请求竞争是否值得改变架构。
- 实现：可选 allocator 采样，结果变化即记录，否则每 gate 每请求至多每 100 ms 一次；保留完整日志、异常和保守 GPU 时间。默认关闭，不改变分配结果。
- 协议：40 个 Agent 请求、32 个续接边界、固定 4×2048 后台输出，绝对时钟等待，GPU KV 2 GiB；单项上限 30 分钟，本轮上限 4 GPU·小时。诊断与性能验证区分，未被调用的 allocator 检查为 UNKNOWN，不能由头部失败外推整条队列。
- 运行命令：`python -m experiments.autonomous_admission.run --name diagnostic-k2 --trace`，在服务器项目根目录，指定现有 venv Python。
- 启动失败：首次因绝对时钟函数名引用错误，在加载模型前退出；已修正并保存失败日志，未消耗 GPU 推理时间。
- 诊断有效：40 请求/32 边界完成，P95 Resume TTFT 6459.75 ms；28/32 续接在首次准入前观测到容量失败，无缺失 allocator 观测。253 个失败样本全部 FULL_SEQUENCE，失败时 reservation 均为 0；无 preemption、OOM、lease 泄漏。后台 8192 tokens 完成，wall 133372.65 ms。观测支持真实容量阻塞，不能据此量化 HOL 的因果贡献。
- 成本：含启动/退出的单 GPU 保守计时 169.52 秒（0.04709 GPU·小时）。

## 2026-10-09：有限容量回填，预先冻结筛选协议

- 假设：某些队头因完整 KV 容量不足而无法准入时，后面较小请求可能仍然能够实际分配。有限回填或可减少闲置容量；与已有 backfilling 调度相近，暂不主张新颖性。
- 实现：仅在真实 `allocate_slots` 返回 None 后跳过；同一容量失败请求最多消耗 2 次额外成功准入，每个 scheduler tick 最多一个回填；计数跨 tick 保留，头部准入/取消即清理。保留正常 full-sequence/reservation、K2、remote 状态和实际请求所有权检查；有 lease 时关闭额外绕过，不用于 FullSlackKV。有限回填不等于普遍的 liveness 证明。
- 预注册：先执行未采样的 OnDemand 和回填各一次，固定 K2/2 GiB/BGv2。只有 P95 至少改善 5%、session median 不恶化超过 2%、后台 wall 不恶化超过 3%，且所有现有正确性检查通过，才扩展反序复验；否则停止该方向，不调阈值或负载挑结果。两次为筛选，不构成稳定显著性证明。
- 文件：`backfill.py`、`backfill.patch`、`run.py`；新增跨 tick 上限、无效扫描和取消清理测试。19 项独立分析/状态测试通过。
- 运行命令：`python -m experiments.autonomous_admission.run --name screen-ondemand --backfill 0`，随后同条件 `--name screen-backfill --backfill 2`。诊断补丁在两者均关闭采样；调度补丁默认 limit 0，只改变候选的回填开关。
- 结果：真实系统筛选失败，停止扩展。P95 6541.02 → 7650.72 ms（+16.97%），session median +3.17%，后台 wall +2.87%，Agent external H2D +3.22%（19276.25 → 19897.50 MiB），混合输出吞吐 -2.79%。40/32 完成、无 OOM；E00:turn:1 在首 token 后发生二次远程恢复。4 次回填分配（2 次是异步恢复后的模型准入，不应解释为 4 个不同请求），计数不超 2。静态 allocator fit 并不能保障未来 decode 增长和免 preemption。
- 输出校验：候选与基线一个请求输出 hash 不同（E02:turn:2），但全部 prompt hash 相同；先前诊断与独立 OnDemand 基线也有两个输出 hash 不同（E02:turn:2、E04:turn:1）。因此不能直接归因于回填/KV 损坏，也不能声称精确输出等价；记录 UNKNOWN。性能本身已失败，不调整门槛重试。将为后续运行保存输出 token 和两个可疑请求的输入，以便独立排查。
- 历史底层回归：隔离运行时的 native prefetch、connector-first 和 first-token gate capacity 单元测试全部通过（15 tests）。不代表 Lease/HOL 全矩阵或无限期公平性已证明。
- 成本：screen baseline 168.32 秒、candidate 172.12 秒；累计 0.14165 GPU·小时（保守含启动/退出）。

## 2026-10-09：执行基线复核

- 新证据：全部历史生命周期测试显式 `--enforce-eager`，同时禁用 torch.compile/CUDA Graph。KV 槽位周转随 decode 完成而发生，执行时长会放大准入队列。必须建立更强的执行基线，避免围绕弱基线优化 KV。
- 方法：同一个 OnDemand、K2、2 GiB KV、BGv2、FP16 模型，只移除 `--enforce-eager`，让当前 vLLM 自行选择默认编译/Graph 路径。记录真实最终模式；若失败保存失败，不改系统依赖。启动仍受现有 360 秒 readiness 和 1800 秒总上限约束。
- 这不是 SlackKV 算法收益或创新性证明，也不与不同执行模式的 FullSlackKV 作横向性能比较。先测机会大小，再决定是否值得在强基线上继续研究。
- 验证：完成所有固定请求与后台工作、K2/容量/OOM/活性检查；检查输出身份，若不同则保留质量 UNKNOWN 并诊断，不以数值加速替代正确性。
- 运行命令：`python -m experiments.autonomous_admission.run --name default-execution --execution default --backfill 0`。
- 首轮有效：P95 -17.14%，session median -32.73%，后台 wall -49.00%，混合输出吞吐 +96.08%，Agent H2D 相同。确认实际模式是 VLLM_COMPILE 与 FULL_AND_PIECEWISE；联合改变编译和 Graph，不能拆分二者各自贡献。输出 hash 不全相同，正确性仍 UNKNOWN。
- 下一步预注册：再运行 default→eager 的反序配对，各一轮，使用同一新版 runner；不调模型/负载/KV/gate。保存逐 token 输出和两个历史不稳定请求的输入。性能只作两对描述性结果，不宣称统计显著、生产泛化或新颖算法。
- 在反序两轮计时负载结束后，用当前 OpenAI completion 协议已确认支持的唯一 `cache_salt` 对两个相同输入各执行两次冷 Prefill，`max_offload_tokens=0`，保存 token、logprobs 和 usage。不把这些复核请求计入性能指标，但 GPU 预算包含它们。先确认 cached_tokens=0，再评估是否一致；不同不能直接推断 KV 损坏。
- 新版 runner 会累计读取本轮成本，在剩余预算不足 0.5 GPU·小时或仍有 RUNNING 记录时拒绝启动；watchdog 为清理留 90 秒，只终止自己启动的 server。
