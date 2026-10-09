# SlackKV 第三轮自主研究：共享恢复机会验证与研究问题重新定位

日期：2026-10-09。起点：`7519bda64de6f2c48051b17f60f8e21c83be564e`。本机 main 与服务器 `research/capacity-round2-20261009` 起点一致，origin/main 无增量；本机干净，服务器历史未跟踪数据保留。阅读 AGENTS、两轮报告、复盘、源码与原始数据后开展工作，没有继承旧会话的推理作为证据。

**结果：本轮没有产生已验证有效的新 serving 算法，也没有相对 Compiled OnDemand 的算法加速。** 完成了一个真实机制验证闭环和两个成本/新颖性淘汰闭环。重要新负结论是：当前 pinned 原生 OffloadingConnector 已经抑制共享前缀的在途重复恢复，不能把普通 singleflight 包装成 SlackKV 创新。新的 synthetic fork workload 中，4/8 GiB 与 K2/K8 的实际 H2D 都只有一次 ancestor/episode。满足停止条件 B，范围限于这些机制与测试条件，不是 KV 优化普遍无效。

## 1. 选择研究问题的依据

前两轮指出低预算下容量失败真实存在，但局部调度收益与 session E2E 不一致。继续只提高 JIT 成功率没有充分依据。本轮先检查另一种可能改变资源关键路径的问题：**动态 fork 的多个实际请求是否把同一 CPU ancestor 同时物化为多份 GPU KV**。它可能同时放大 H2D、不可复用在途占用和准入阻塞，因而比几十毫秒的预取领先量更有潜在资源价值。

这是需要证伪的假设。原生 APC 只有完成的 GPU blocks 能安全共享，不能仅凭“启用 prefix cache”推断在途恢复也去重。[vLLM issue #44882](https://github.com/vllm-project/vllm/issues/44882) 描述了一般 connector 的类似问题；该 issue **不能证明本项目 pinned OffloadingConnector 有此缺陷**。

FP16 Qwen2.5-1.5B：28 layers、2 KV heads、128 head dimension，28672 bytes/token。八个分支共享 24576 tokens，各增加 256 private +128 output：首轮唯一工作集模型为 793182208 bytes（0.739 GiB，含一个 null block），假定八份 root 时为 5725683712 bytes（5.332 GiB），差额 4932501504 bytes。这只是反事实容量模型；**实测没有这八份 root**。第二轮最大 private 增长后，模型为 837222400 bytes（0.780 GiB），加独立 BG 也远低于 4 GiB；不能称本场景存在真实 4-GiB allocator 压力。

成本模型、环境配置与原生代码证据：[cost_model.json](../results/autonomous_fork/cost_model.json)、[second_turn_footprint.json](../results/autonomous_fork/second_turn_footprint.json)、[environment.json](../results/autonomous_fork/environment.json)、[native_inflight_audit.json](../results/autonomous_fork/native_inflight_audit.json)。模型估计不等于实际驻留 memory-time。

## 2. 最小真实实验与预注册

[protocol.md](../results/autonomous_fork/protocol.md) 保存原始问题、候选准入条件和两次显式诊断补充。只有观测到共享在途重复恢复，才实现候选；候选需要 episode E2E 至少改善 10%，P95/BG 不恶化超过 5%，H2D 不增加，工作量与生命周期完整，才扩展反序和容量验证。本轮未满足实现前提，没有改阈值或调参数寻求正结果。

- Ubuntu，GPU0 RTX A6000；GPU1 未用于推理。Qwen2.5-1.5B-Instruct，FP16/TP1，原生 CPU KV 8 GiB，APC，默认 VLLM_COMPILE/FULL_AND_PIECEWISE；回填、credit、epochs 均关闭。
- **Synthetic fork/join stress**，不是生产 Agent trace。每 run 三个 episode、八个 child、两轮，共 48 child calls +3 BG calls。每 child call 固定 128 outputs，BG 每 episode 8192-token prompt /1024 outputs；所有 BG work 完成。
- ancestor 24576 tokens；first prompt 24832 tokens；second prompt 25024 tokens，包含固定 canonical 128-token 历史和 64-token 新增，合计追加 192。实际生成内容不反馈，避免数值执行差异改变后续输入；这也意味着不能评价语义任务质量。固定工具间隔 0.5 秒，实际客户端间隔单独记录。
- CPU priming 后调用 GPU-only reset，保留 CPU cache。以真实 external hit 验证 intervention，实现的是受控冷 GPU 恢复，不是假定生产中总有这种重置。
- 初始 cold BG 同时释放；发现无重复后，显式增加 decode BG：相同 BG 先到首 token，再释放 child。保存原条件，BG E2E 包含其冷 Prefill。两者不能作为算法配对。
- 首次假设 cold Prefill 消耗 batch budget 掩盖重复；decode control 也无重复后，修正为原生在途 key 去重。K8/8 GiB、K2/4 GiB用于限定负结论，而非候选矩阵。
- 五个完整 fork run 的 **51 个实际输入全部严格匹配**。输出工作量、hash、source 身份、终止事件和独立 E2E 算术校验通过。不同 gate/预算/background/invariant 属于不同条件，禁止混为算法增益。

## 3. 实测结果与原因

每个条件只有一次 run，内含三个 episode；episode 共享服务进程，不能当作三次独立统计重复。表中 P95 是 48 个 child calls 的 nearest-rank，包含 first fork calls 和第二轮 continuation，**不是历史 32-boundary Resume P95**。

| Run | GPU KV / gate / BG | Workflow median ms | Child TTFT P95 ms | BG median ms | 实际 H2D MiB | 实际 D2H MiB | Remote-ready wait median ms |
|---|---|---:|---:|---:|---:|---:|---:|
| fork-control-4-retry | 4 GiB / K8 / cold | 4360.56 | 709.27 | 8574.71 | 2016 | 2310 | 222.54 |
| fork-decode-control-4 | 4 GiB / K8 / decode | 4134.64 | 480.42 | 8568.08 | 2016 | 2310 | 56.58 |
| fork-decode-control-8 | 8 GiB / K8 / decode | 4068.17 | 495.85 | 8541.77 | 2016 | 2310 | 58.79 |
| fork-decode-k2-4 | 4 GiB / K2 / decode | 4154.34 | 474.15 | 8615.56 | 2016 | 2310 | 56.52 |

四个非 invariant 控制的十二个 episode 都是 **1 个 external ancestor restore +7 个 local ancestor hits**；第二轮全为 local prefix hits。H2D 2016 MiB 恰为三份 24576-token root，逐请求 source 与实际 engine transfer counter 相符。allocation 时 sampled root block union 仅一份；这不是全程峰值或 owned memory-time 的证明。全部完整 fork run preemptions=0，观测 K 峰值不超过各自限制，最后 running/waiting=0。

原生代码已经包含完整链条：`_blocks_being_loaded` 在 APC 下建立；lookup 发现任何命中 key 正在加载时返回 None、延期 follower；实际 load 加入 key；完成后移除 key，完成的 GPU blocks 经 APC 被 follower 复用。原生 connector 未被修改；指纹 `bc883cf47d882d4406d7708a8d3520abe2752f545985705126f41fca66558bf0`，完整源码无损保存为 `native_offloading_scheduler.py.gz`。因此没有实现普通 coalescing，也没有移除这项保护。

`fork-decode-control-4` episode 1 的最后完成 branch：首次 TTFT 436.15 ms，首次首 token 后 service 1383.38 ms，工具/客户端间隔 553.84 ms，第二次 TTFT 284.86 ms，第二次首 token 后 service 1448.00 ms，初始 issue offset 28.41 ms，总计 4134.64 ms。两段首 token 后 service 合计 2831.38 ms（68.48%）；它包含计算、batch 调度与客户端观察成本，**尚不能分解为不可避免计算与可优化调度**。

cold 的 remote-ready wait 222.54 ms 明显高于 decode 的 56.58 ms。该区间是 scheduler 观察到的完整等待，包含完成通知/执行交互，不能解释为纯 DMA。DMA 实际完成到 engine 发布 ready 的间隔尚未测量，不能把两条件差异全部归因于 GPU 计算，也不能通过推迟 child 到达宣称系统加速。

独立实测与原始事件：[analysis.json](../results/autonomous_fork/analysis.json)，各 run 的 `requests.json.gz`、`inputs.jsonl.gz`、`episodes.json`、`metrics_after.prom`、完整 `server_logs/*.log.gz`。`GPU_KV_owned_memory_time` 保持 UNKNOWN。

## 4. 第二、第三个方向的淘汰

**完整 active-prefix streaming，保留精确 full attention。** 即使八个分支每个 attention iteration 只共同搬一次 root，不算私有 KV、staging、kernel 或干扰，两个 128-output 调用的 256 次 attention 仍需 180388626432 bytes H2D。使用 Phase0 1-GiB pinned H2D P50 26.279616 GB/s，DMA 服务时间估算 6.8642 秒，已经超过约 4.13 秒 fork E2E。当前 unique working set 又可以驻留，因此不开发此 PoC。这里的“下界”**条件于参考带宽**，不是现代环境或所有硬件的严格物理上界；Phase0 为历史 CUDA12.1 测量，未在 Torch2.11 重新测量。不能据此否定 CPU attention、压缩、稀疏 attention、不同 batch/模型或真正 active set 溢出的设计。

**peer GPU 作为显著更快的 nearline tier。** 相同历史 1-GiB 数据中，GPU1→GPU0 26.367456 GB/s，pinned CPU→GPU0 26.279616 GB/s，传输时间优势约 0.333%。在本硬件方向上没有支持“大幅缩短恢复”的带宽依据；可能的额外容量与 CPU 压力效应需要另一组匹配实验，不能由这个比值否定所有 peer-tier 价值。未开展新 GPU 矩阵。

**继续写 TTL/priority/host write-filter/公平尾部驱逐。** 完成论文机制与部分作者代码审查，发现与已有方案高度重叠；不再写一个缺少机制区别的 PoC。现有 host manager 本身也有 LRU/ARC 与 immutable hash 去重，不能把所有旧基线默认当作无策略缓存。

## 5. 相关工作与创新性边界

查阅方法：最终候选相关工作阅读算法/系统章节与可获取作者代码；其余按下面范围明确标注。未将论文其他硬件上的收益拿来当本项目结果，也未在本环境安装这些完整系统。

| 来源与审查范围 | 已有机制 | 对本轮选择的影响 |
|---|---|---|
| [InferCept v2](https://arxiv.org/html/2402.01869v2)，swap/调度章节 | interception 期间保留/丢弃/CPU swap 按 memory waste 排序；分层流水与 chunk swap budget 协调推理 | 普通 idle offload、transfer/compute overlap 和保留成本已有基础；duration 估计与预算假设应作为比较点 |
| [FastServe](https://arxiv.org/html/2305.05920v1)，skip-join MLFQ、swap 章节 | token 级抢占、首轮长度判断、饥饿保护、低优先级主动 offload | 普通 priority/token preemption 不是新算法；也不直接代表 Agent 长生命周期优化 |
| [KVFlow](https://arxiv.org/html/2507.07400v1)，图与 runtime 章节 | workflow step graph、依赖距离传播、GPU/CPU/loading/offloading 状态及 prefetch；已有 loading 节点去重 | fork 图/预取/普通恢复去重已有相近机制；静态固定 prompt graph 与动态实际输入是可能边界，尚非新颖性证明 |
| [Continuum v7](https://arxiv.org/html/2511.02230v7)，TTL utility 与调度章节 | reload benefit /乱序成本/内存机会成本确定 TTL，program FCFS 与 TTL-aware priority、终止释放 | 简单 TTL、租期 pin 或连续性优先级不能重新命名为创新 |
| [MORI](https://arxiv.org/html/2606.00866v1)，admission/cache 章节 | 工具与 inference 时间定义连续 idleness，GPU/CPU 队列分层准入、亲和路由 | history-aware multi-tier admission 已有；其真实 Claude Code 工作集更接近需要补足的验证域 |
| [Autellix](https://arxiv.org/html/2502.13965v1)，PLAS/ATLAS 与实现章节 | program attained service /多线程实际关键路径、MLFQ、饥饿提升、合并 swap kernel | 单请求优先级不足以代表程序关键路径；简单 session credit 与关键路径排序已高度重叠 |
| [EfficientAgent](https://arxiv.org/html/2609.33762v1)，working-set/admission 章节及[作者 release](https://github.com/KunmingSHAO/efficientagent_release) | prefix survival/stack distance 与 CPU pool、写入压力；request 首次确定 store admission，保存 hot-prefix 延伸 | host write filter 与工作集 admission 不能作为本轮新方案。作者栈为 vLLM0.13/LMCache0.3.12/TP8，不能直接与本栈混比；未获得其生产完整时间戳 trace |
| [RR-Evict](https://arxiv.org/html/2609.32278v1)，tail chunk /activity 章节 | 空闲会话轮转分块驱逐尾部，活动重新检查与 completion-awareness | 避免整条 trajectory 被 LRU 同步清空已有专门算法；普通公平 partial-prefix retention 不新 |
| [CacheScout](https://arxiv.org/html/2608.14624v1)，online transition 与实现章节 | 在线 Agent 转移概率、anchor reuse /重算成本评分 | 工作流预测型缓存已有成熟方向；没有预测数据和匹配基线时不写另一预测器 |
| [Multi-Segment Attention /AsymCache](https://arxiv.org/html/2606.02964v1)，attention/eviction 章节 | 精确利用非连续 KV segments，仅计算缺失 query segments；按边际计算价值驱逐 | “非前缀残留 KV 仍可使用”与按重算成本留块已有核心方案。它不等同每个 decode step 从 CPU 搬完整 active root |
| [KVSET](https://arxiv.org/abs/2609.27746)，**仅摘要**，HTML 未获取 | stack-based KV 容量规划 | 容量规划的新颖性需比较，未据摘要声称已审查算法细节 |
| [HiCache 文档](https://docs.sglang.io/docs/advanced_features/hicache_design)及[作者 v0.5.9 实现](https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/mem_cache/hiradix_cache.py) | 共享 radix node、load_back 物化 device indices、ongoing_load_back、锁引用与完成事件；分层搬运 | ordinary shared materialization 和生命周期保护已有实现。使用可获取 tag 审查，不冒充最新 main，也没有 SGLang 实测胜负结论 |

最初 candidate “共享 ancestor materialization”在**已有代码与实测**两方面均失去价值前提。论文中的算法与环境差异不能自动构成 SlackKV 的实质创新。

## 6. 正确性、旧负载回归与失败

独立 `fork-oracle-4` 采用 `VLLM_BATCH_INVARIANT=1`，其 timing 不参与性能对比。三 episode 完成后，对 episode-zero 的八个分支两轮共 **16 个相同输入**逐个赋不同 cache_salt，max_offload_tokens=0，冷 Prefill；raw source 确认 local/external hits 都为零，16/16 的128输出 tokens 逐 token 匹配。审计还将 oracle reference 与保存的请求输出及完整 server source trace交叉核对。这不是所有输入、取消、抢占、跨模型或生成任务质量的证明。

旧 hybrid workload 保留且实际重跑：`legacy-regression-2`，Compiled OnDemand，K2/2 GiB/CPU8 GiB，严格复用 Round2 的全部40个输入；40 Agent calls、32 resumes、8192 BG outputs 完成，valid=true，OOM=false，logical/lease leaks=0。P95 Resume 4853.24 ms，session E2E median 23878.57 ms，BG wall 67764.82 ms；28/32 resumes 有真实 pre-admission capacity-failure 样本，实际 H2D 20212613120 bytes。与 Round2 的运行时源码版本、进程起点及到达顺序不同，只作回归/描述性证据，**不能将时间差算作算法加速**。没有本轮新算法，所以也没有必要用指标下降冒充候选通过。

失败全部保留：

1. 首次上传缺少远端父目录，未启动 server；创建目录后检查每次上传 success。
2. `fork-control-4` 在 episode zero 后调用过时 source_delta helper 缺参数失败；保存 failure/log/metrics/inputs 与 **54.31秒 GPU 成本**。该版本尚未及时保存 requests.json，逐 token 输出 UNKNOWN；不能纳入完整性能统计。
3. retry 的 derived source_delta 使用错误 local/external 标签，产生零；**这些派生字段 INVALID**，原 `episodes.json` 不改写。raw source 与实际 counter 独立验证成功，最终 runner 删除该字段。
4. 分析时 oracle ledger 仍 RUNNING，未完成成本字段导致 KeyError；没有生成结果。最终 analyzer 明确拒绝活账本，不能补零成本。
5. 首次 CPU runtime test collection 缺少项目 PYTHONPATH，三个 import errors；保留 `runtime_regression.txt`。同时加入 runtime 和 repo 路径后，`runtime_regression_retry.txt` 的 15 tests 通过，CUDA_VISIBLE_DEVICES 为空。
6. 导出先拒绝已有 environment metadata 文件；检查为字节完全相同后允许跳过已有相同文件。101个原始 byte hashes 全部通过。Python3.13 浮点 sum 与服务器3.11的总成本差小于1e-15；本机 Phase0 CSV 的 CRLF 改变 byte SHA，但所有带宽数值一致。原始 server analysis/cost model 保持 byte-exact，本地复算单独存档。

52 项本地分析/模型/状态/复盘 tests、15 项 CPU-only native/connector-first/gate-capacity tests 通过。没有新 serving 状态机；只有默认关闭的 `SLACKKV_FORK_TRACE` allocation snapshot，不改变 allocator 决策。未执行完整 GPU Lease/HOL/cancellation adversarial 矩阵，不能据有限运行宣称无界公平性或全生命周期安全。后台工作量从未减少，但没有算法配对，无法声称公平性得到改善。

## 7. 本轮产出与资源

新增：固定 fork runner、exact footprint/streaming 成本模型、独立分析和冷 oracle、原生 in-flight 审查、lossless export/byte audit、12项新测试。原 shared runner 仅增加显式 Round3 目录选项和对应真实 pool 验证，默认旧路径行为保持；regression 与 fork 共用一张连续账本。

七个 GPU execution：四个默认模式完整 fork controls、一个独立 invariant correctness run、一个旧负载回归、一个部分运行后失败。合计 **0.20048675264813934 GPU·小时（约12.03 GPU·分钟）**，最长260.33秒；包括加载、编译、验证、失败和退出。仅GPU0，预算上限4GPU·小时，不因更换目录重置。CPU审查、模型与两个CPU-only unit-test尝试不占用GPU。最后两卡均19MiB/0%，compute-apps为空。

仅1.5B模型在本机缓存。7B理论 FP16 weights约14GiB，KV56KiB/token，单A6000资源估计可行；服务器HF API 20秒探测超时，未下载、未更改代理/依赖。未验证7B/生产trace/其他SLO regime。这个限制不等于物理硬件不支持7B，也不是当前停止的预算理由。

关键数据：[results/autonomous_fork/](../results/autonomous_fork/)，`SHA256SUMS.json`、`independent_audit.json`、`analysis.json`、`cost_model.json`、`native_inflight_audit.json`、各 execution/inputs/requests/metrics/full logs、cold_oracle、两次 runtime tests、final_gpu_state。服务器原始文件未删除。早期失败 runner 未作完整版本快照，保存其 hash 与 failure；最终源码、最终运行时完整 scheduler、每run配置/hash及实际输入可复现本轮成功实验，不能冒称早期失败版本也完整可重建。

复算与实测命令见 [commands.sh](../results/autonomous_fork/commands.sh) 和 [README.md](../results/autonomous_fork/README.md)。最终提交用 Git 历史定位，避免把预期提交 hash 写成已完成事实。

## 8. 下一研究判断

**值得继续的问题：长上下文、多分支 Agent 的精确 KV 工作集超出 GPU/CPU tier 时，真实缺失 KV/重复 Prefill/恢复依赖究竟消耗多少程序关键路径与 SLO 并发成本。** 这继续围绕 Agent Serving/KV，最终目标不变。相比 JIT，它可能改变秒级重算和可服务工作集；本轮仅证明当前已驻留、host-hot synthetic fork 不值得开发重复恢复消除，尚未证明生产中的这个更大机会。

保留一个**未通过机会验证的机制候选**：按实际消费者所需 prefix range 建立 ready dependency，把共享 root 的发布与 leader 私有尾部恢复分开，用已完成、不可变且受引用保护的 ranges 解阻短消费者；不预测未来 deadline、不保留无限 lease。与 native whole-job deferral 的具体区别是 ready 粒度，而非 ordinary singleflight。必须进一步与 HiCache 的分层 readiness、KVFlow loading node 和 Multi-Segment Attention比较；新颖性目前 UNKNOWN，未实现、未测速，不能称最有希望的已有效算法。

后续先做一次明确选择：获取可运行较大模型及可重放的真实长上下文/分支 trace，用同模型、执行栈、GPU/CPU预算、输入和输出工作量建立 OnDemand +已有适用强方法基线，测 actual DMA-ready→engine-ready 发布时间、异质私有尾部依赖、host prefix-hole 与重算 FLOPs、owned block residency trajectory。**只有恢复依赖或缺失 KV 的可删成本达到预注册目标，例如至少20% workflow critical path，或同SLO下有明确容量收益，才开发该候选；否则放弃，而不是再次提高 promotion hit rate。** 这是下一轮新协议建议，未回改本轮10% screen门槛。

安全设计尚未解决的核心是共享物化对象的真实所有权：leader cancel/preempt不能释放仍在DMA或被follower引用的block；只有完成且身份一致的range可发布；失败清理后回退OnDemand；使用真实allocator/in-flight reservation；给后台与老请求有界推进机会。需要CPU小状态验证和GPU取消/抢占压力测试，不可仅靠本轮成功完成认定安全。

当前最重要的技术决策：**停止在已被原生缓存保护、恢复成本小且工作集可驻留的条件下开发 JIT/普通 coalescing/完整 CPU streaming。先证明新的 host-miss / readiness critical-path regime，再决定算法。** 不改变项目最终目标，不申请耗尽剩余GPU预算来扩大一个失去前提的方向。
