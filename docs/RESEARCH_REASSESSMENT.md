# SlackKV architecture reassessment

日期：2026-10-09。审查基线：`e2e904c2ebe344f1d4dd8cd347fb6fabd246a6f8`。

本轮按新的自主研究交接执行，覆盖仓库规则、`docs/experiment_log.md`、`docs/decisions.md`、指定阶段报告、控制器/分析代码、保存的 vLLM 补丁及关键 CSV。没有读取旧 Codex 会话，没有修改 serving 算法，没有 GPU 实验。新增分析是对已提交数据的复算，不能替代新架构的性能验证。

## 1. Current State

**当前最大问题是优化对象与实际端到端瓶颈未对齐。** 项目证明了 CPU KV 恢复、原生预取和受控 lease 保留的可行性，但没有证明移除恢复等待能显著缩短当前多轮工作流。完整策略又受到大 prefix 的容量约束，导致大部分预取根本不能执行。应该先回答“调度/容量联动是否有足够收益”，再决定是否保留 JIT。

当前链路：上一轮完成 → CPU source 可见性 → 客户端 EDF/blackout 预取调度 → logical reservation → 引擎物理容量与 PSA 检查 → connector-first 预取 → GPU physical lease → 真实 resume/K2 → 首 Token → 精确 lease handoff/release。客户端 deadline 来自已注入的工具等待代理，不是可部署的恢复时间预测。

近期运行使用 Qwen2.5-1.5B FP16、单张 RTX A6000、4681 个 16-token KV block、2 GiB GPU KV 和 8 GiB CPU KV；P_SAFE=102 ms、K2、3510-block logical budget。**2 GiB 是实验限制，不是 A6000 的物理显存上限，也不是已证明的生产最优分配。** 单卡原型没有证明第二张卡或其他模型收益。

Phase 4C 的八 session/40 turn/32 boundary 是固定 correctness 子集：合成 token ID 保持公开 ISL/OSL/KV 结构，pause 来自经过 100–3000 ms 筛选的 TraceToolProxy。不是代表性真实 Agent 语义/时间分布。长等待池之外的机会、真实 arrival correlations、生产预测误差尚未评估。[workload evidence](../results/phase4c/summary.md)

## 2. Validated Mechanisms

“有效”在本节只指已验证的特定机制，不是整体策略有效。

| 机制 | 核实证据 | 有效范围与限制 |
|---|---|---|
| CPU KV 避免完整重算 | Phase 2C-Fix `resume_metrics.csv`：OFFLOAD TTFT 中位 94.822 ms，KEEP 后被挤出重算 148.513 ms；恢复 external=4096 | 单 prefix 压力 A/B，3 次重复；不证明选择算法优越 |
| 提前恢复可隐藏部分恢复路径 | Phase 2D `metrics.csv`：OnDemand 98.806 ms，JIT-250 42.590 ms 有效恢复延迟 | 已知 deadline；普通生成请求预取代理，存在 warmup/调度行为，不能把差值全归于 PCIe |
| native promotion 不做 Transformer forward | Phase 3A/3B endpoint token/byte/forward 指标；后续 lease smoke | 原生路径正确；完成时间仍受引擎影响，不等于后台无干扰 |
| connector-first 改善预取完成耦合 | 3B 隔离诊断约 24–27 ms client E2E，copy 约 5 ms；正常 A TTFT 增加约 10–14 ms | 用普通推理延期换取预取推进；并非免费的异步重叠 |
| lease 保留并安全交接 | 3S unleased 0/3 local、leased 3/3 local，refcount 2→1→2→1 | 有保障的有限工作流；取消、失联、过长 pause 的资源回收不是自动保证 |
| lease/HOL 循环等待 | 4E-A-RCA `liveness_runs.csv` 5 次 NONPROGRESS；9 个规定对照 PROGRESS。无 X 三 lease 后 free=750，E04 不能分配；队首 `break` 阻断后方 lease 接管 | 强受控因果证据，范围为 pinned scheduler；不要泛化为所有 vLLM 死锁 |
| PSA-1 恢复上述进度 | 4F-A 正式补丁与报告：free=2073、candidate=1323、head progress=1278，拒绝后只剩 750 的危险分配；规定 no-X/X/自身保护对照完成 | 特定集合和队列假设的 liveness；没有一般最优性、动态多轮证明 |
| K2 生命周期修复 | `results/k2_lifecycle_fix/vllm_k2_lifecycle_delta.patch` 及 4G-A-FIX ownership 审计 | 首输出后重入不再重新占 slot；修复与选择 K=2 是否最优是不同问题 |
| guarded prefetch allocation failure 回退 | 4G-B-FIX 补丁：真实 allocate_slots 返回 None 时零传输/零 lease 拒绝；新矩阵实际触发 1 次 | 避免 connector-only 自己阻止释放 in-flight reservation 的 normal prefill；没有正常请求调度改进 |

早期 Phase 3C/3D 在人工 deadline 下减少 miss/blackout，3D P95 124.47 ms 却高于 OnDemand 109.03 ms。3H 的三请求 K2 TTFT 改善 12.50%，decode throughput 仅保留 93.11%、makespan 增加 3.53%；3G-Full 的 K1 tradeoff 失败。任何 Result A 都必须按该阶段验收含义阅读，不能累积成整体加速证据。

本地无法执行 pinned Linux vLLM runtime tests：部分测试导入 `experiments/phase4d_a/runtime.py` 即加载服务器绝对路径下的 tokenizer，所需 serving 环境不在本机。这里对其做静态/保存数据审查，没有声称重新验证 CUDA 或全部历史单元测试。

## 3. Failed Hypotheses

1. **只要 KV 提前到达 GPU 就能降低整体延迟：未成立。** 3Q/3R 的 T6 成功后仍被驱逐；lease 修复缓存生存问题后，4F/4G 仍未达到 10% P95 目标。
2. **即时 free capacity 代表未来可推进：已被反例推翻。** 4E lease/HOL 循环等待、4G-B in-flight reservation 与 connector-first 相互等待是不同失败模式。
3. **固定 pre-first 并发限制普适最优：未成立。** 从 max_tokens=1 到多 token 再到大 prefix，证据和 tradeoff 改变。K2 在后续矩阵对所有政策共同固定，因此当前 A/D 比较不能估计 K2 自身收益或损失。
4. **logical reservation 必然改善整体性能：未成立。** 4F-B D 对 C 配对 P95 中位仅 -1.060%，H2D 相同，lease memory-time 增加；有 selectivity，不等于 E2E 利益。
5. **source recheck 单独足够：否。** 4G-C-FIX 40→127 selections，均无成功 promotion。候选可见性与安全执行机会分开。
6. **legacy guard false-reject 主导：原结论无效。** 123 observations 中 exact-known=54、missing=69；known agreement=54、mismatch=0。禁止把 missing allocator 值填为 0/0。
7. **旧 trace 的 pause 是真实 resume slack：不可沿用。** 4A/4B 语义审计未确立 request-start ground truth；Phase 1 的 oracle/proxy结果不能证明实际恢复预测器可用。

## 4. Root-Cause Assessment

### 已证实

- 完整 serving 策略没有稳定显著性能优势。4F-B D/A 配对 P95 -1.181%，descriptive bootstrap [-4.06%, +1.94%]；4G-A-FIX-K2repair 配对 P95 +1.382%，session E2E 配对中位 +1.29%。不是“尚未运行完整矩阵”，而是已完成的负性能证据。
- 4G-A-FIX 的 D 256 boundaries 中 80 个 native calls：62 capacity_rejected、17 progress_rejected、1 ok。拒绝是安全 fallback；不能把调用量当成迁移量。
- 4G-D eligible candidate selections 的真实 allocator 已知 46/46，其中 fit=11、no-fit=35（76.09%）。这个比例是选择条件下的频率，不是所有 pause 的潜在成功率。
- 消除观测到的 KV remote wait 的**直接**收益量级很小，见下一节。

### 值得检验，但尚未证实

- 队列长尾可能来自长期 active KV 占用、FCFS allocation break、prefill/chunk 服务、K2 或后台计算竞争的组合。现有边界事件不能将这些等待互斥地归因。
- 外部 controller 依据 deadline 选择 protected head，与引擎真实 waiting/running/owned 状态脱节，可能过度保守；但未完成首 Token 的终止预取决策请求仍是需要进度保障的真实请求。删掉它不能自动构成安全 frontier。
- 小模型、短传输、大前缀和固定小 KV pool 可能让排队代价远高于恢复代价。这是对当前 regime 的解释，需要机制测量，而不是推广到全部硬件的根因。

### 对 4G-D “dominant” 的独立审计

原报告 121/126 terminal-head selections、poison union/workload 80.61% 均是观察性指标；128 poison episodes 中 123 被标记 Mixed/unknown。`analysis.classify_result()` 优先按 4/7 exact-known F0-blocked shadow recoveries 判定 Frontier dominant，即使另有 35/46 eligible no-fit。该分类不是总体收益的因果排序。

复算区分 8 个 F0-blocked 选择与其中 7 个 exact-known；一个 UNKNOWN 不进入 4/7 分母。没有把 4/7 外推为所有 boundary 的 57% 可改善率。F1/F2 改 frontier 对同一四个选择产生重叠 recoveries，不是八个独立机会。

`shadow.py` 在 frontier existing lease 未知时仍有 `int(existing or 0)` 回退。当前保存的 F0–F4 有身份行全部有 lease 值，故未发现这次样本受该回退影响；未来数据必须传播 UNKNOWN。这里只记录风险，不改历史脚本。

selection→scheduler snapshot 中位 17.67 ms、最大 128.35 ms；snapshot RTT 中位 65.78 ms、最大 410.75 ms；source sample age 中位 3674.19 ms。快照不是 controller 选择时刻的原子状态，不能保证真正尝试时仍 fit/source-ready。probe CPU duty 约 5.3–5.7% 也使 4G-D 不适合作零开销性能基线。

## 5. Performance Upper Bound

### 可复算的最小分解

对 boundary i 定义目标 resume 时刻 d、引擎到达 a、首次模型 admission m、首 Token f：

`L_i = f-d = (a-d) + (m-a) + (f-m)`（本批数据各时刻合法，指标代码对 L 作非负截断）。

保存数据中的 `scheduler_wait=m-a` **包括 async remote KV load**。不能视为纯队列等待。把首 Token 前成对的 `WAITING_FOR_REMOTE_KVS→REMOTE_KV_READY` 区间并入 `R_i`；再把它与 `[a,m]` 的交集从 scheduler_wait 扣除，得到 `Q_i*`。`Q_i*` 是非 remote 状态的 pre-admission 时间，不是已经识别原因的“纯容量等待”。`R_i` 也包含 completion polling/scheduler delay，不是 CUDA copy 时长。

| 量，中位数的定义见下 | OnDemand-K2 | FullSlackKV+PSA1-K2 |
|---|---:|---:|
| 8 个 run 的 boundary P95 中位 ms | 6434.53 | 6413.48 |
| run 内 remote wait P50 的中位 ms | 58.87 | 58.49 |
| run 内 remote wait P95 的中位 ms | 83.96 | 106.59 |
| 各 run 实际 P95 boundary 的 Q* 中位 ms | 6316.89 | 6175.32 |
| 同一组 tail boundary 的 R 中位 ms | 57.83 | 59.12 |
| 同一组 tail boundary 的 f-m 中位 ms | 31.59 | 62.77 |

样本为 16 runs、512 boundaries；首 Token 前 remote 配对 UNKNOWN=0。首次 admission/首次 output 界定直接等待；首 Token 后 preemption/reload 保留在原数据中，**不被冒充为首 Token 前可节省时间**。三个 component 的分别中位数不能相加当成 P95。

### 条件上界：固定观测路径的删除实验

设所有未来 arrival、队列顺序、计算 batch、其他请求服务和容量演变保持观测路径；假定可零成本消除某请求全部 R，无 source/容量/干扰限制。计算 `L'_i=max(0,L_i-R_i)`，对每 run 的全部 32 值重新取 nearest-rank P95（第 31 个），而不是只修改旧 tail：

| 乐观删除场景 | OnDemand P95 改善中位（范围） | Full P95 改善中位（范围） |
|---|---:|---:|
| 删除全部 observed R | 0.905%（0.775–1.097%） | 0.994%（0.805–3.464%） |
| 删除 admission 前 R 加**全部** admission 后时间 | 1.393%（1.216–2.976%） | 2.723%（1.641–3.937%） |

第二行甚至免费删除新 suffix prefill 和首输出模型工作，是真正实现不了的固定路径乐观诊断。四个 boundary 的 R 总和/session E2E，在 64 session samples/policy 的中位为 OnDemand 0.738%、Full 0.756%。它仅描述四个恢复 boundary 的直接成本份额，不包括首轮、后续 decode 恢复或闭环 arrival shift。

**结论的边界：** 固定路径下，单纯改善 H2D/restoration timing 不接近 10% P95；它既不是新实验，也不是重调度系统的全局理论上界。真实系统移除某个等待可能推进后方请求，也可能提前注入新轮导致更高拥塞；R 与其他 compute 可重叠。不能把删除表解释成 causal speedup。固定路径“安全且可行”的预取收益上限不会超过免费删除全部 R 的场景，但真实安全集合未知，不能计算其精确值。

若要达到当前约 6.4 s P95 的 10% 改善，必须改变约数百毫秒（量级约 0.6 s）的尾部关键路径；远高于典型约 0.06 s remote wait。删除全部 pre-admit 时间的诊断可产生约 95.8% 改善，但这是不可实现的瞬时 admission，不是候选架构上界。当前只能给重调度收益**未知**：非干扰理想假设下由 0 到整个可改变的延迟，不能据此宣称存在 10% 机会。

### 资源/服务限制与未识别项

- 第一 boundary 前缀总需求 10501 blocks 对 4681 pool，不能同时驻留全部 prefix；单 prefix 1271–1444 blocks，加新 input/首 Token progress 更高。X 的逐 token 增长和正在 running 请求占用还要计入，不能把 3510 logical budget 当成 allocatable free。
- 首轮校准 native issue→T6 最大 84.49 ms、budget=102 ms，来自九个真实大小样本；不是严格服务上界。transfer counter、T6、HTTP return 是不同量。
- 4F 的额外重复 H2D（B 中位 275.41 MiB）与 lease 的保留作用已有机制证据；4G-A-FIX A/D H2D 均 19276.25 MiB、重复为零，所以此处再消灭 duplicate traffic 没有现成收益空间。
- prompt-only offload 的未命中完整 generated-tail blocks 不能凭 token 残差推成测得的重算时间。Prefill/recomputation、新 input prefill、PCIe device time、normal-forward suppression、allocation/gate/prefill budget 各自贡献缺少完整计时/阻塞原因，均不填为零。
- 任意新架构必须同时计算 GPU memory-time、CPU footprint、transfer/model work，以及固定 4×2048 background 完成时间。只优先 Agent 或减少 X 工作量会产生伪收益。

## 6. Candidate Architectures

### A. Engine-state progress certificate 与短期可撤销预取

**原理：** 从引擎实际 waiting queue/运行所有权建立 progress certificate；不以“client earliest unfinished deadline”单独代表需保护请求。若 head 已有真实所有权且下一步可执行，可保护已承诺的 incremental growth，向后推进 frontier；否则保留它的进度需求。无真实执行承诺时维持 PSA-1 fallback。预取 protection 必须可撤销，取消/超时不能留下硬 lease。

**可能解决什么：** 去掉已被引擎保障的重复 reserve，开放部分被 terminal decision 限制的机会；仍主要减少 R 或重复传输，不自动消除 Q*。

**安全要求：** 原子绑定 certificate、allocation 与 owner 身份；考虑 in-flight reservations、水位和真实共享 block ID；撤销期间先完成 DMA、再递减引用；每步存在一个可运行/可完成/可安全回收的推进路径，未来 unleased head 不能被后方 lease 锁死。历史 failed prefetch 不等于 finished real request。

**最小验证：** 先 exhaustive 小状态机探索 lease-HOL、first-token preemption、head 取消和新到达；再沿原状态记录 exact-known recoverable 集合，验证至少覆盖有明确 E2E 贡献的机会。未经安全验证不得执行 shadow 的四个例子。

**失败风险/创新：** 即便开放四个候选，直接省几十毫秒而 P95 不动；allocator race 或持续 background growth 破坏 certificate。潜在创新是可证明的 engine-state 进度证书；与 Banker-style safe-state、vLLM admission 及相关资源调度的关系需证明，不能只给 PSA 换名字。

### B. Selective retention / partial restoration，改变驻留对象

**原理：** 对 KV segments 选择保留量 x，使复用的边际收益与 block·time 成本相权衡；例如在可验证稳定前缀上选择 `benefit(x)-lambda*memory_time(x)`，预算包含活跃 decode 和 DMA。可以是转移之前保留部分 KV，或只提前恢复连续前缀，余下到 demand 时恢复/重算。

**可能解决什么：** 降低等待期 speculative footprint、提升少量可复用 state 存活率。普通 full attention 首输出仍需要完整历史 KV；**部分预取不减少完整执行 footprint**，因此不直接修复大 prefix active-request capacity queue。真正降低执行驻留需要 layer/block streaming、分段 attention 或压缩，工程与正确性要求大幅增加。

**安全要求：** bit-exact KV、prefix/block identity、position 和 source completeness；缺页不能被视为 hit，所有需要的 KV 必须在读取前可用；受限 staging buffer、DMA 引用和释放顺序；近似压缩必须另有质量门槛，不能偷换语义。

**最小验证：** 离线量化 retained/partial bytes、missing suffix 的 transfer/recompute cost 和 full-resume footprint；两 prefix 短 correctness test 对照全部 local 与 partial+restore，逐层数据与 logits 校验；若仅节省 R，不升级为 streaming 大改。

**失败风险/创新：** 小 copy 启动成本、重复迁移、suffix 重算、prefix 离散 hash 使收益负；streaming 在每个 decode token 重复 PCIe 成本。与 InferCept chunking、KVFlow node retention、Multi-Segment Attention 显著重叠，需进一步源码/全文对比。

### C. Joint demand admission and KV restoration（推荐验证）

**原理：** 用实际已到达请求的执行需求与真实 block owner 做联合 admission，不再把“未来 deadline 前把全 prefix 放进 GPU”作为主目标。CPU 保留仍用于避免重算；未到达 speculative state 默认可驱逐。引擎决定下一可服务集合时，对恢复、incremental input prefill、decode growth 一并做原子容量承诺；commit 后 state 归真实请求，不需要外部长时间 lease。

最初 PoC 仅实现“真实 demand 的进度保障＋恢复绑定”，先维持 FCFS/K2/背景优先级，以隔离内存安全效果。若证据证明主要是容量/队首等待，再考虑有年龄上限的 fit-aware 重排或 bounded background quantum；重排和后台配额属于独立变量，不能与移除 lease 混成一次胜利。

资源模型：对被服务集合 S，真实 block union 加新增承诺不超过 B；对 restore j，成本包含 H2D 与 delayed normal-forward，不能只用 bytes/bandwidth。目标是减少多轮关键路径中的 capacity/HOL 等待，使用实际到达和 session age，不依赖 oracle tool deadline。保持现有 OnDemand 是首要基线：它已经 demand-load，因此仅改预取 API 很可能与它等价。

**为何更有希望：** Q* 是秒级可观测组成，R 是几十毫秒；C 唯一直接提出改变请求执行集合/驻留生命周期而非增加候选。它也可能被现有 OnDemand 击败；当前没有证明 Q* 中可改变的部分足够大。

**安全要求：** atomic reserve→restore→ownership handoff，失败及时 rollback；禁止有 prefetch pending 就无限暂停 normal compute；每个 finite request 在有限 load 下有 bounded age/fairness 保证；至少一条 demand 进度路径；取消和 preemption 可释放/安全交换，真实 required KV 不丢失；后台获得正的服务配额。对于无限持续过载只能保证公平份额，不能承诺所有请求有限 latency。

**最小验证：** 先标记每次阻塞的首因和可替代服务集合，再作两个校准 replay：只免费删除 remote wait；允许在相同容量/服务成本下联合 admission。若第二个在合理成本扰动下仍无 10% 空间，停止。仅通过后才请求小 GPU safety/paired trial。

**失败风险/创新：** 正常 OnDemand 已接近最佳 capacity scheduling；队列等待其实由计算饱和决定；priority 只把延迟转给 X/其他 Agent；新 input prefill 与 decode 不能按固定成本线性外推。一般“KV 与调度联动”不新，可能的新点须是 Agent 多轮连续性、可撤销状态和进度证明结合，并相对强 baselines 而非 NaiveJIT 证明价值。

### 比较

| 方向 | 当前端到端收益证据 | 成功概率判断 | 改造/首次 GPU 成本 | 安全难度 | 创新判断 |
|---|---|---|---|---|---|
| A state-aware frontier | shadow 四个机会，直接收益小 | 机制可行性中；性能低 | 中/小 | 高，需要实际队列证书 | 窄，需要证明而非调参 |
| B partial/retention | 未证明 capacity execution 收益 | partial 低；streaming 未知 | 中到很高/中到高 | 数据正确性与 DMA 较高 | 已有紧邻工作 |
| C demand/serving 联合 admission | 有秒级 Q*，可改变份额未知 | 中低，但最匹配主要量级 | 先离线低；runtime 中到高 | 高，可分阶段撤销推测 lease | 条件成立才值得研究 |

这些是决策判断，不是实验成功概率的统计估计。

### 初步新颖性检索（只到摘要/作者材料，未完成全文源码审计）

- [InferCept](https://arxiv.org/abs/2402.01869)：augmented inference 的 intercept/KV 管理；作者[说明](https://mlsys.wuklab.io/posts/infercept/)讨论保留、交换和丢弃/重算权衡。排除了“工具等待 KV 管理”本身的新颖性。
- [FastServe](https://arxiv.org/abs/2305.05920)：token 级抢占、调度与 GPU/host state 管理。排除了“一般调度和搬运协同”本身的新颖性。
- [KVFlow](https://arxiv.org/abs/2507.07400)：workflow graph、KV node eviction 和 next-step prefetch。排除了“workflow-aware 预取/选择性保留”本身的新颖性。
- [Continuum v7](https://arxiv.org/abs/2511.02230v7)：TTL retention 与 program-level FCFS，考虑重载成本和排队；新方向 C 必须区别于其多轮连续性思路。
- [MORI](https://arxiv.org/abs/2606.00866)：按相对 idleness 划分 GPU/CPU 并做分层 admission；新方向需区别于其容量适配。
- [Multi-Segment Attention](https://arxiv.org/abs/2606.02964)：明确涉及更灵活的 KV 管理，并与 agent serving 结合；B 不能先声称 partial KV 是新思想。

不能将这些论文报告的 speedup 移植为 SlackKV 预期值；硬件、workload、baseline、目标指标均不同。后续需补充 ConServe、Apt-Serve、CacheWise 等原始论文/代码的具体可比性，当前未作完整 SOTA 比较。

## 7. Recommended Direction

选择 **C 的机会诊断与安全模型**，暂不直接实现完整 C、不继续 Phase 4G-E、不替换 live frontier。若该诊断失败，停止当前以 JIT deadline 为中心的优化目标，重新定义研究问题或硬件/workload regime；继续保留 CPU offload 这一有证据的基础机制。

优先理由是研究信息价值：A/B 的直接收益几乎只能触及低于 2% 的固定路径部分；C 先验证是否存在可改变的容量/队首等待，能快速否定“继续修 PSA 必然有整体收益”的前提。C 并非已有证据的性能赢家。

## 8. Minimal Validation Plan

下列是下一阶段待用户决定的计划，**本轮未执行 GPU 部分**。首先冻结所有假设、命令、样本选择、比较基线和资源预算，再运行。

1. **CPU/静态阶段（先做）：** 建立 allocator 所有权与等待队列有限状态模型。探索 3–5 request、不同 KV size、background growth、cancel、preemption、DMA pending、lease handoff；模型必须重现 4E U3 与 4G-B connector-first 失败，再验证 demand 原子 admission 的 resource conservation、no circular wait 和公平条件。状态模型通过不是实机安全证明。
2. **最低成本观测（需后续批准）：** 先一个 OnDemand-K2 短完整 8-session run，与原 workload、2/8 GiB budgets 相同；每次不能调度时记录 exact allocation need/available、in-flight/watermark、first blocked reason（capacity/K2/token budget/async source/connector-only）、running owner/growth、队列次序和完成。timestamp 使用 monotonic，记录 device copy 与 CPU polling分别计时。若一个 run 不支持 capacity/HOL 解释，停止，避免矩阵。
3. **条件收益模型：** 使用上述原始信息重放 baseline，并验证 per-request first/finish、order 与 block trajectories；预算/compute/transfer成本不变，仅替换 demand admission 规则。分别使用实际成本及 ±20% 成本敏感性，不调参数直到达标。输出区间；缺失依赖使重放不识别时，不宣称上界。理论免费删除只用于机会筛查，不当成测得 speedup。
4. **只有前述通过才做微型实机：** 独立实验分支；先 4E causal controls、4G-B allocation failure、cancel/首 Token 后 preemption 的小 correctness tests；再 OnDemand 与 C 三个交替配对完整 32-boundary run，BGv2 固定工作，P95/session E2E/背景wall/TPOT/H2D/lease和active KV memory-time都计入。三对仅筛查，不作显著性结论；预先 GPU wall 总预算 20 分钟（含 smoke/startup），到预算终止并保留全部尝试。不能为达标减少后台工作或更改 baseline。

初始 C 不同时改变 K2 或 CPU source 策略。若需 background quantum，必须单独对照相同 quantum 的 OnDemand，避免把通用 priority 收益归于 KV 架构。

## 9. Kill Criteria

- 阻塞原因测量表明大部分尾部 Q* 是不可避免的 compute service，或相同预算下经过校准的理想 admission 在成本敏感性范围内也达不到 10% P95 改善：停止 C 性能实现。
- 模型不能复现 baseline 或关键状态为 UNKNOWN：停止收益结论，补数据而不是调策略。
- 任何新的 circular wait、KV identity/refcount 错误、无界 starvation、background 无正服务配额：否决设计；不把超时样本丢掉后再算性能。
- 三对筛查中 P95 配对改善中位不足 10%，或 session E2E 变差，或背景 fixed-work wall/TPOT 比匹配 baseline 恶化 >5%，或 H2D 增加 >5%：不扩大矩阵。已有 4F/4G 原验收阈值保留，新阶段门槛需在执行前冻结。
- 如果唯一收益来源是降低 background priority 或放大 GPU KV pool，不能称为原 SlackKV JIT 成功。可重新讨论研究问题，但不偷偷改变实验。
- 若 C 与 Continuum/MORI/FastServe 等实现实质相同且无法指出可测试差异，停止创新主张，即使工程上有用。

## 10. Open Questions

- Q* 的 exact first-blocking-cause timeline；当前 capacity snapshots 与 boundary event 不足以归因每毫秒。
- 完整 pinned vLLM patched source/环境在服务器的可复现打包与测试方式。仓库保存 delta/fingerprint，不等于本机具备可运行完整 runtime。
- native DMA 与 normal-forward 是否能真实并行、SM/memory bandwidth影响，以及 connector-first 每次 normal-step 延后量。
- active KV、idle evictable KV、lease、logical budget、in-flight/watermark 的 block-ID union time series，及安全取消语义。
- Prefix prompt-only 残差与 measured prefill cost；source hits/算术 token residual 不能取代 GPU 工作计时。
- 无 oracle 的实际 tool start/end、LLM arrival/finish 和预测误差；生产 semantic correctness 与 batch-sensitive token变化的质量评估。
- 是否应该把优化目标改为给定 SLO 的并发容量/成本，而非本 workload 的恢复 P95；需要用户确定问题，不由本轮默认改变。
- 更合理 KV budget、模型大小和 pause 分布下的 crossover；任何后续实验应先预测 crossover，不能无目的大范围 sweep。

## Reproduction and delivered artifacts

在仓库根目录执行：

```powershell
python -m scripts.research_reassessment
python -m pytest -q tests/test_research_reassessment.py --basetemp=.pytest_cache/reassessment-validation-1
```

14 tests PASS，涵盖 UNKNOWN、配对 remote events、多次恢复、首输出后恢复排除、nearest-rank ranking 改变，以及完整提交数据回归。首次 direct-script 调用因 module path 失败，改为 `-m` 后完成；首次 integration fixture 因沙箱 temp 目录权限失败，改到已核实不存在的工作区临时目录后通过。未修改阈值或样本。Python 启动有 Anaconda location warning，但 analysis/tests 正常退出。

`results/research_reassessment/audit.json` 保存输入 SHA256、阶段复算和每 policy统计；`boundary_decomposition.csv`、`run_opportunity.csv`、`session_opportunity.csv` 保存逐行和逐 run 数字。原始阶段文件一律保留。历史 `paired_deltas.csv` 中 duplicate MiB 百分比在 0/0 时为 NaN，不是实际无穷/性能异常；新增 JSON 禁止 NaN，缺测保持 UNKNOWN，不使用该百分比形成结论。

本轮结束于报告和提交，等待研究方向决策。
