# 自主研究第 1 轮：容量、执行基线与输出一致性

本轮完成真实系统研究闭环：加入 allocator 诊断 → 实现有限回填 → 实测否定 → 核查执行栈 → 验证更强 OnDemand → 排查输出差异 → 筛选 compiled FullSlackKV。没有获得可保留的新 SlackKV 加速算法；获得了停止继续扩展当前 JIT/Frontier 矩阵、重建研究基线的实测依据。

**主性能观察：**一个全部 40 个输入 hash 匹配的真实配对中，compiled OnDemand 相对 eager OnDemand 的 P95 Resume TTFT 降低 **15.81%**，session E2E median 降低 **29.94%**，后台固定工作耗时降低 **48.92%**。这是 vLLM 已有编译/Graph 的收益，不是 SlackKV 创新，也尚不是多次精确输入复现或统计显著性证明。

**负结果：**有限回填 P95 恶化 **16.97%**；同输入强基线下 FullSlackKV P95 恶化 **16.08%**，10 次 Promotion 全部拒绝。保留所有数据，停止扩展这两个方向。

## 1. 最大瓶颈与可优化空间

在当前限制为 2 GiB 的 GPU KV 池中，长上下文请求需要等待实际 KV 容量释放。首次诊断的 32 个续接中，28 个在首次准入前出现真实容量失败；253 个失败采样全部来自 FULL_SEQUENCE 检查，失败样本的 in-flight reservation 均为 0。没有观察到该诊断中的 preemption。这排除了“本次被观测失败主要是虚假 reservation 拒绝”的解释，但不能排除其他时间/场景中的 reservation 问题。

采样是实际执行的检查，不是整条队列的原子观察。28/32 是观测到的请求数量，不能将失败样本数当作等待时长，不能把队头失败外推给尚未检查的请求。剩余尾延迟中，容量排队、首次 Prefill、既有 decode、运行时开销各自的完整因果份额仍 UNKNOWN。

在 compiled OnDemand 中 P95 仍有 5.25 秒，而 pre-first-token 远程等待 median 约 49 ms。固定观测路径上删除全部远程等待再重算 P95，两个 compiled run 分别只改善 **0.870% / 1.051%**。这是局部机会估计，**不是改变调度/资源占用后的因果上界**。单纯提前 H2D 不值得继续投入同类大矩阵，除非先证明它能改变并发/排队，而不是只消除这几十毫秒。

首次输入 Prefill 和固定输出 token 的模型计算必须发生，规定的工具 pause 也保留；这些不等于其观测耗时全部不可优化。执行栈实验已证明模型执行周期可以缩短。请求排队不是已证明的不可避免计算成本，下一轮应直接识别哪些资源等待可在公平约束下消除。

## 2. 环境与冻结协议

- Ubuntu 服务器，GPU0 RTX A6000，GPU1 未使用；driver 550.54.14，Python 3.11.16，PyTorch 2.11.0+cu126 / CUDA 12.6。
- Qwen2.5-1.5B-Instruct，snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，FP16 / TP1，block size 16；GPU KV 2 GiB、4681 blocks，CPU KV 8 GiB。
- 8 个并发 session × 5 turns，32 个续接边界；后台 4×2048 输出，全部完成；K2、绝对 CLOCK_MONOTONIC pause，原 ISL/OSL/共享前缀结构不变。
- frozen source SHA256：`9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`。
- vLLM upstream `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，保存现有完整 tracked diff、未跟踪的 dev API 和新增 delta。隔离副本 `/media/a6000/D/ZhaoBinHe/slackkv-autonomous-20261009-source`；没有修改或删除历史运行时。
- `default` 仅移除 `--enforce-eager`，实际启用 VLLM_COMPILE 与 FULL_AND_PIECEWISE。编译和 Graph 联合改变，不能拆分各自贡献。模型启动/编译不计入 workload 延迟，但计入 GPU 预算。

本负载使用真实 trace 的长度/共享前缀结构与代理 pause、合成 token 输入，具有受控资源竞争；不是生产流量或语义 Agent 成功率测试。2 GiB 是实验预算，不是 A6000 物理容量。不能由此泛化所有模型、显存配置或真实 Agent。

## 3. 实现与淘汰有限回填

allocator 诊断仅在显式启用时记录：结果变化立即输出，否则每请求每检查点至多每 100 ms 一次。未启用时不输出新采样，不改变分配结果。

有限回填在真实 `allocate_slots` 返回 None 后，允许扫描后续请求并实际尝试分配。每个被越过的容量失败请求跨 tick 最多记 2 次成功分配，每 tick 至多一次回填；头部准入/取消即清理。有 Lease 时不绕过，原有完整序列、reservation、remote 状态、K2 与请求所有权检查仍执行。异步恢复和随后模型准入可能各占一次计数，不能解释为两个不同请求。

筛选门槛在实验前冻结：P95 至少改善 5%、session median 不恶化超过 2%、后台 wall 不恶化超过 3%、所有正确性检查通过才反序复验。实际结果未满足，未调门槛/负载重试。

| 运行 | Resume P95 ms | Session E2E median ms | BG wall ms | Agent H2D MiB |
|---|---:|---:|---:|---:|
| eager OnDemand screen | 6541.02 | 35582.49 | 132324.77 | 19276.25 |
| eager 有限回填 | 7650.72 | 36712.14 | 136126.09 | 19897.50 |
| 变化 | +16.97% | +3.17% | +2.87% | +3.22% |

回填实际触发 4 次成功分配，E00:turn:1 在首 token 后再次恢复 KV，产生约 621.25 MiB 额外 H2D。静态“现在装得下”不足以保证后续 decode 增长不引发 preemption。有限回填与既有 backfilling 思路相近，未主张新颖性；本实现不应作为优化默认启用，也没有普遍活性证明。

## 4. 强执行基线与匹配审计

| 严格输入匹配的反序配对 | eager OnDemand | compiled OnDemand | 变化 |
|---|---:|---:|---:|
| Resume P95 ms | 6241.82 | 5254.99 | -15.81% |
| Resume median ms | 3589.46 | 1824.40 | -49.17% |
| Session E2E median ms | 34114.18 | 23901.11 | -29.94% |
| BG wall ms | 132209.51 | 67534.75 | -48.92% |
| BG weighted TPOT ms | 16.1183 | 8.2242 | -48.98% |
| 混合生成吞吐 tokens/s | 89.20 | 174.62 | +95.77% |
| Agent H2D MiB | 19290.69 | 19281.50 | -0.048% |

40 个 prompt hash 完全匹配，ISL/OSL/工具 pause 相同；八个 session 的 E2E 全部改善，没有用牺牲某个 session 或后台固定工作制造这一观察。混合吞吐按全部 Agent 输出与 8192 个后台输出、BG 首 token 至全部请求完成的窗口计算；不含初始 BG Prefill，不是独立稳态吞吐基准。

首轮 eager/default 观察曾得到 P95 -17.14%、session -32.73%、BG -49.00%，但独立审计发现 default-execution 的 E07:turn:3/4 输入随前轮输出漂移。该轮只保留为**结构匹配观察**，排除精确输入匹配汇总。此前日志里的“两对复验”已追加明确修正，不掩盖问题；当前仅一个精确输入匹配配对，不能称两次精确复现。

## 5. 输出差异与正确性闭环

非 invariant 下，诊断和独立 OnDemand 之间也出现 output hash 差异，候选回填的不一致不能直接归因于 KV 损坏。保存完整输出 token、两个敏感输入、冷 oracle 的 logprobs 与实际 source 命中记录。

eager 冷 oracle 的 10/10 请求确认为 local/external hit 0。E02:turn:2 的并发冷请求出现不同输出；E04:turn:1 的串行/并发冷输出也不同。因此，不复用 KV 本身就能出现差异，exact output hash 不是当前默认批处理数值模式下充分的 KV 损坏判据。

启用既有 `VLLM_BATCH_INVARIANT=1` 后，完整 40/32 生命周期完成；两个敏感输入各 2 次串行冷 Prefill 与 3 次并发冷 Prefill共 **10/10**，全部确认没有缓存命中，并与对应续接**逐 token 一致**。该控制仅支持被测输入和 OnDemand 恢复路径，不证明所有边界/模型或成功 Native Promotion 的正确性。invariant 较慢（该轮 BG wall 269585.69 ms），只用于 debug，不混入性能对照。

22 个分析/状态/归档与既有复盘测试通过；隔离 vLLM 的 native prefetch、connector-first、first-token gate capacity 15 个底层回归通过。8 个完整运行都完成 40/32 和后台固定工作，没有 OOM、记录中的 lease/logical 泄漏；这不等于重跑所有历史 Lease/HOL adversarial tests 或证明普遍公平性/活性。

## 6. compiled FullSlackKV 筛选

`full-default` 与 `default-repeat` 的全部 40 个输入 hash 匹配，执行模式、KV 预算、K2、BGv2 相同。Full P95 **6099.82 ms**，相对 compiled OnDemand **+16.08%**；session median **+0.40%**、BG wall **+0.55%**、混合吞吐 **-0.55%**。这是单轮筛选，不能宣称稳定的负百分比；没有正证据支持扩展正式矩阵。

32 个边界中 22 次 source-late skip；10 次 Promotion attempts 中 9 capacity rejected、1 progress rejected、**0 success**；没有 Lease acquire。因而这轮验证了当前架构在编译执行栈下拒绝/回退路径的完成能力，**没有验证成功的零-forward Promotion 与 CUDA Graph 协作**。

末尾 Prometheus 快照记录 CPU→GPU **20212613120 bytes（19276.25 MiB）**，与 Agent source trace 复算一致；GPU→CPU **5423824896 bytes（5172.5625 MiB）**。这是该轮迁移快照，缺少其他所有性能 run 的相同端点快照，不能宣称 D2H 配对改善。传输时间计数也不能当作端到端阻塞时间或相加成关键路径。

## 7. 资源与复现

8 个完整运行加冷复核，保守累计 **0.400327 GPU·小时**，包含服务器启动、编译和退出；最长运行 **389.35 秒**。另一次 import 名称错误在模型启动前失败，保存错误日志并修正；没有 GPU 推理成本。GPU1 未使用，所有本轮 server 已退出，两个 GPU 恢复 19 MiB / 0% 利用率。没有删除历史代码、原始数据或调整服务器依赖。

新增代码、运行时 delta、原有运行时快照、85 个文件的导出 SHA256、原始 `.log.gz`、请求/后台数据、逐 token oracle 和复算命令均在 [results/autonomous_admission](../results/autonomous_admission/README.md)。决策及预注册协议见 [自主研究日志](AUTONOMOUS_RESEARCH_LOG.md)。Windows/Ubuntu 离线复算的诊断和主配对数值一致，压缩证据 hash 验证通过。

## 8. 战略决定、下一阶段与不确定性

本轮达到授权中的“现方向收益不足，需要重要战略调整”阶段停止条件。**停止扩大当前 JIT/Frontier 和有限回填实验矩阵。**保留已经有效的 CPU offload、native promotion 基础能力及 ownership/reservation/PSA 保护，不不可逆删除机制。

下一轮以 compiled OnDemand 为强性能基线，正确性用独立 invariant oracle；先针对实际 demand 的 KV 驻留时间、decode 增长、完成时间与准入协作建立因果干预，再设计改变有效并发的机制。候选至少要说明未来资源需求如何计价、如何避免再次 preempt/重复迁移、如何保持后台与 session 公平性，并与已有 InferCept/KVFlow/Continuum 等工作比较，不能把普通 backfilling、编译开关或资源管理协作直接称为创新（已有文献背景见 [Reassessment](RESEARCH_REASSESSMENT.md)）。

最重要的不确定性是：**扣除必须计算与固定 pause 后，剩余数秒尾延迟到底还有多少可通过公平资源管理消除？**本轮真实容量失败和执行干预给出方向，没有完成该因果分解。完整 GPU KV memory-time、非 invariant 的任务质量、生产 workload 泛化、成功 Promotion 的 Graph 正确性和广泛 liveness 都仍 UNKNOWN。后续先解决这些可识别性问题，再决定算法和必要的匹配资源场景，不为完成 Phase 消耗剩余预算。
