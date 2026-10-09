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
- 正确性复核进展：default-repeat 的两个输入分别在 token 7 和 41 与冷 Prefill 分歧；各自两个冷重复完全一致。completion usage 未提供 cached_tokens，不能用缺失字段当 0。后续 oracle 增加 SOURCE_TRACE，直接确认 local/external matched_tokens 都为 0，并添加每输入 3 个并发冷请求；不同模式的数字不拼接成性能样本。
- 核查当前 vLLM 源码：`batch_invariant.py` 有 SM80 的 Triton matmul overrides，文档支持 compute capability >=8.0；`envs.py` 的 >=9.0 注释与实际实现不一致，不能仅据注释拒绝 A6000。若默认路径仍出现不一致，将尝试已存在的 `VLLM_BATCH_INVARIANT=1`，仅用于正确性对照，绝不与非 invariant 性能混比。
- eager-repeat 冷对照：10/10 SOURCE_TRACE 确认 local/external hit 均为 0。E02:turn:2 的三个并发冷请求出现两种不同输出，E04:turn:1 的串行/并发冷请求输出不同。**不复用 KV 也能复现输出差异**，支持批处理/数值路径影响，否定“hash 不同就必然 KV 损坏”的判据；仍不能排除另外存在 KV 错误。
- 决策：增加一轮 `--name invariant-validation --execution default --batch-invariant --oracle`，相同 40/32 负载和 10 个冷复核。预注册成功标准为所有冷请求被真实 source trace 确认为冷，两个敏感输入的续接/串行/并发输出全部逐 token 一致；否则保留失败和 UNKNOWN，不对非 invariant 吞吐做错误归因。
- 反序性能对照完成：P95 -15.81%，session median -29.94%，后台 wall -48.92%，混合输出吞吐 +95.77%；两对的八个 session 全部 E2E 改善。两对 P95 改善为 15.81–17.14%，不是统计显著性结论。编译基线上的固定路径完全删除远程等待仅改善 P95 0.870% 和 1.051%；这仍不是重排调度后的因果上界。

## 2026-10-09：强基线上的原架构兼容性筛选（预注册）

- 问题：仅改变执行栈已有更大机会，原 JIT/PSA1 在强基线上是否仍有明显价值？
- 方法：一轮 `--name full-default --execution default --policy D`，backfill 0、batch-invariant 0、K2、2 GiB、同一 BGv2。沿用当前完整 SlackKV 控制器/PSA1，不修改门槛；对照两个已完成的同配置 compiled OnDemand。仅筛选兼容性与收益空间，不宣称单轮稳定性或非 invariant 输出等价。若无明确改善，不投入 FullSlackKV 正式矩阵。
- 预算：独立上限 30 分钟，运行完保存所有拒绝、promotions、lease、安全/活性记录及末尾迁移计数快照；不比较 eager Full 与 compiled OnDemand。
- 结果：40/32 与后台任务完成，无 OOM、lease/logical 泄漏；10 Promotion attempts 全部拒绝（9 capacity、1 progress），没有有效 Promotion。Full P95 6099.82 ms，相对输入完全相同的 compiled OnDemand-repeat 5254.99 ms 恶化 16.08%；session median +0.40%，后台 wall +0.55%。这是一轮筛选，不能声称稳定的负百分比，但没有值得扩展正式矩阵的正证据。

## 2026-10-09：独立审计修正与本轮决策

- 修正：首轮 default-execution 的 E07:turn:3/4 输入因前轮输出数值差异而漂移。此前“两对复验”只在 ISL/OSL/共享前缀长度结构上成立，不能当作两个精确输入匹配的对照。保留原始结果，将首轮标记为 structure-only，排除精确输入匹配汇总。反序 eager-repeat/default-repeat 全部 40 prompt hash 相同；主性能观察为 P95 -15.81%、session median -29.94%、BG wall -48.92%。Full-default 也与该 matched default-repeat 全部输入相同；与 default-execution 的对照同样只作结构匹配观察，不用于主结论。不能宣称两次精确输入复现或统计显著。
- 正确性控制完成：batch-invariant 的 40/32 生命周期有效，10/10 冷对照均实际 local/external hit 0，全部与相应续接逐 token 相同；两个敏感输入在串行和并发下均一致。这支持本次被测恢复路径的正确性，不证明所有模型、边界、Lease 或 Native Promotion 都无误。invariant 模式明显变慢，只用于 debug，不混入性能对照。
- 全部 8 次完整运行有效，另有一次模型加载前 import failure；累计保守成本 0.40033 GPU·小时（含启动、编译、oracle 和退出），最大单次 389.35 秒，未使用 GPU1。
- 决策：本轮达到“当前方向收益不足，需要战略调整”的阶段停止条件。停止扩大有限回填和当前 JIT/Frontier 矩阵；保留已有底层 native/offload/保护机制，采用 compiled OnDemand 作为下一轮强执行基线，正确性另设 invariant oracle。下一项算法研究应围绕实际 demand 的 KV 驻留时间、decode 增长与并发能力，先建立可优化排队成本的因果分解，再设计机制；不把编译开关视为创新。
- 仍 UNKNOWN：真实生产负载泛化、非 invariant 任务质量、完整 GPU KV memory-time、计算与可避免队列延迟的完整因果拆分、普遍公平性/活性，以及成功 Native Promotion 在 CUDA Graph 下的安全性（本轮成功数 0）。不再为凑齐既定 Phase 消耗剩余预算。
- 归档：长 run/oracle JSON 和原日志无损 gzip，服务器与本机原文件保留；85 个逻辑原始文件 SHA256 校验通过。离线读取兼容 gzip 并避免双计数；最终 22 tests passed。原始 patch 的 diff 空白与 Prometheus 原生 HELP 空白保留，限这些证据文件不做 whitespace lint，未修改科学数据。
