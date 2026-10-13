# 自主研究日志

## 2026-10-09：第三轮，共享恢复机会证伪与研究问题重新定位

- 起点7519bda64de6f2c48051b17f60f8e21c83be564e，origin/main无增量；本机main干净，服务器历史未跟踪数据保留。单独Round3预算；从Round2 runtime复制隔离源，未删除旧实现。
- 假设：动态fork共享CPU ancestor在途重复物化可能把0.739GiB唯一首轮工作集放大到5.332GiB，改变真实准入/关键路径。容量模型只是反事实，必须先测重复，再决定是否开发coalescing；ordinary singleflight不主张创新。冻结protocol、工作量、screen门槛。
- 实测：synthetic fork3episodes×8branches×2turns，root24576、first24832、second25024tokens，固定128outputs/call及BG3×1024outputs。cold/decode、4/8GiB、K2/K8四个默认compiled controls共12episodes全部1次ancestor external restore+7local hits；实际H2D各2016MiB、D2H2310MiB、preemptions0，五个完整run全部51输入严格匹配。
- 机制：原生OffloadingConnector已有_blocks_being_loaded，APC在途keylookup延期、load完成释放key后复用GPU cache。初始cold-prefill遮蔽重复的解释不完整，decode control与源码审查纠正。停止ordinary coalescing开发，未写候选serving状态机。只新增默认关闭allocation trace，owned memory-time保持UNKNOWN。
- 代表性decode4GiB workflow median4134.64ms，TTFTP95480.42ms，BGmedian8568.08ms，remote-ready wait median56.58ms。关键分支两段首token后service共2831.38ms，占68.48%；不能拆分计算/调度或认为全不可优化。cold wait222.54ms不是纯DMA；更大模型完成通知延迟仍待测。
- 成本否定：完整active root每attention step streaming，即使8分支共享一次fetch，256steps需180.39GB，历史26.28GB/s参考下6.86sec，工作集能驻留，当前regime不开发。peer输入方向历史传输优势约0.333%，不支持把nearline带宽当大收益；容量价值不由此否定。
- 文献：审查InferCept/FastServe/KVFlow/Continuum/MORI/Autellix及EfficientAgent、RR-Evict、CacheScout、MSA/AsymCache；HiCache作者v0.5.9已共享节点物化与事件/引用保护。KVSET仅摘要明确标注。普通TTL/host写过滤/公平尾驱逐/去重与已有机制重叠，不制造无价值PoC。
- 内容验证：独立batch-invariant fork+16输入cold Prefill，raw hits0且16/16逐128tokens匹配；计时不混入默认模式。旧40input/32resume/BG8192 Compiled OnDemand回归valid，P954853.24ms，28/32真实容量失败；不能把run间差异算算法收益。未做完整GPU cancel/Lease/HOL矩阵。
- 失败：上传父目录、首run helper参数、retry派生source_delta错误零值、live-ledger分析、首次CPUtests缺repo PYTHONPATH全部记录；首run54.31sec计入，缺输出UNKNOWN。原raw不改写。导出101hash通过，Python版本sum/CSV换行差异分开记录。
- 检查/成本：52localtests、15runtimeCPUtests通过；7GPUexecutions累计0.20048675264813934GPUh，最长260.33sec，GPU0only，最后两卡空闲；CPU测试不分配GPU。7B资源模型可行但HF20sec探测超时、未下载/改环境。
- 决策：本轮停止B，未产生已验证新算法、无强基线算法加速。下一问题是长上下文多分支Agent的host-miss/recompute与恢复依赖critical path；range ready发布仅未验证候选，先查真实trace/较大模型与已有强方法的差异。最终Agent/KV目标不变。报告docs/AUTONOMOUS_RESEARCH_ROUND3.md；数据results/autonomous_fork/。

## 2026-10-09：第二轮，容量干预与需求连续性 PoC

- 起点：本机、服务器均为 e239088b3891396cde654669d74020e8fce82ecd，origin/main 无新增提交；本机干净，服务器历史未跟踪原始数据保留。GPU0/1 空闲。第二轮单独连续预算，不重置任何本轮失败成本。
- 先验问题：2 GiB 的真实容量失败是否能够通过资源干预改变秒级关键路径，而不只是减少几十毫秒 H2D。冻结 Compiled OnDemand、K2、Qwen FP16、CPU 8 GiB、40/32、BGv2；2/4/8 GiB 仅用于容量诊断，资源增加绝不包装为新算法收益。
- 实现：独立 round runner，保存全部实际输入，后续固定输入资源回放，维持 finish-relative 工具 pause；明确输出反馈/语义质量与资源性能不同。原运行时保持，候选使用另一个副本。
- 失败：4 GiB 首次在 workload 前被历史固定 4681-block 验证器拒绝；保存 failure.txt、原始日志及 60.07 秒成本。第二轮按 pinned layout 的 28672 bytes/token×16 验证实际预算，保留不匹配拒绝；未修改历史验证器或放宽其他门槛。
- 容量初测：2 GiB P95 5199.92 ms，31/32 续接有真实容量失败；4 GiB P95 1163.51 ms、9/32，Agent H2D 19280.19→9925.13 MiB，BG wall +0.79%。单次资源干预支持容量影响关键路径，不能等同全部等待因果分解或稳定加速。
- 候选：one-turn continuation credit，只给真实到达 WAITING 续接有界机会，每个较早冷队头至多一次成功分配，失败不回填、不 pin、不持 Lease；默认关闭，cancel 清理计数，保留 OnDemand/K2/allocator 语义。先测试 2 GiB matched control/candidate，P95 >=10%、session median 不恶化、BG wall/TPOT 与 H2D 不恶化 >5% 才扩展；否则停止，不改 credit 数调参。协议在 results/autonomous_capacity/protocol.md。
- 新颖性边界：Continuum 已有 TTL-aware priority/program FCFS，MORI 已有相对 idleness 分层准入，Autellix 已有 program attained-service。一般连续性/优先级不是创新；本 PoC 只测试无需 speculative pin 的有界需求重排是否值得进一步研究，不声称可发表创新。
- 检查：31 项分析/状态/复盘测试通过；新增预算布局、重复 allocation、跨 tick credit、取消、lease/default、remote/in-flight/preemption 排除测试。有限状态测试不等于整个引擎活性证明。后续实机结果将追加。

### 第二轮：容量转折与 credit 淘汰后的下一候选

- 8 GiB 同输入 OnDemand：32/32 无观测容量失败、Agent H2D 0，P95 445.32 ms；session median 18.84 秒，与 4 GiB 的 18.78 秒接近；BG wall 67.85 秒。容量解除不能消除全部计算/调度时间，也没有显著增加本固定 BG 尾部主导窗口的混合吞吐。实际 owned KV memory-time 尚 UNKNOWN。
- 新源码、精确输入 matched credit screen：control P95 6061.25 ms → candidate 3642.52 ms（-39.90%），但 resume median +20.64%、session median +6.08%，BG wall -0.16%、H2D -1.44%；四次 credit 分配。冻结 session 门槛失败，停止该候选，不反序扩展、不调 credit。不能把 P95 排名改善等同所有会话整体改善。
- 不因一次失败结束研究。第二候选为 two-call soft working-set epochs：真实到达请求的 prompt+declared max output 估计与实际 pool 大小结合，保留最多两个调用、最长一秒 idle 的逻辑工作集；不持物理 KV、不预取、不新增 allocator reservation，后台不受 cohort gate。已入组需求/空 cohort 保持原始 allocator 进度路径。所有初始请求已发送，队列等待完整进入 E2E。未来上下文增长的估计不等于真实可分配容量；实际 allocator 仍是唯一物理判定。
- 冻结顺序：2 GiB control/candidate；仍需 P95 >=10%、session median 不恶化、后台与 H2D 不恶化 >5%、完整生命周期通过，才反序和 4 GiB。失败即停止该 lifetime/call-count 方案，不挑调参最优。顺序 programs 是已声明适用域，并行 DAG 尚未证明。
- 38 项本地分析/状态/复盘测试通过；两个 runtime patch 阶段各 15 项 native/connector-first/gate-capacity 回归通过（CUDA_VISIBLE_DEVICES=''，CPU tests，0 GPU 推理消耗）。缺失续接 expiry、cancel、terminal、异步重复 allocation、已知 output 上限与增长进度已有状态测试；仍需真实系统验证。

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

### 第二轮结束：两个候选均未满足整体门槛

- Soft epochs matched screen：P95 4782.34→9889.70 ms（+106.80%），session median +14.37%，BG wall -0.08%，actual H2D -38.60%；27 epoch，最终 0，preemption 0。容量失败 28/32→2/32 不能视为压力消失，gate 在 allocator 前隐藏等待；E02:turn:2 到首次 allocator 已 9720.85 ms。停止，不调 lifetime/call-count。
- 两候选严格 40 输入匹配，输出 hash 非全等，本轮没有新 invariant oracle；内容/质量等价 UNKNOWN。不能借 Round1 两个输入校验证明新机制。
- 独立 audit 验证 87 个导出原始 hash 和全部 P95/session 算术。Credit candidate actual H2D 比 source 多 458752 bytes，原因 UNKNOWN，迁移门槛用 actual counter，source estimate 保留；8 GiB counter 缺失 UNKNOWN、source restore 0 单列。D2H 各 run 5172.5625 MiB。阈值及两个淘汰决定不变。
- 7 valid + 1 setup failure，0.2332685893 GPU·小时，最长 130.45 秒；GPU1 未用，两卡最后 19 MiB/0%，无 compute process。40 项本地测试通过、两阶段各 15 项 CPU-only runtime 单测通过。完整历史 GPU Lease/HOL 矩阵和 owned KV memory-time 未验证。
- 按 B 的有界负结论结束，两个 PoC 默认关闭，保留 OnDemand/安全机制；不推论全部调度算法无效。目标保持，后续面向明确 SLO 的有效并发/显存成本，先补真实稳态会话、更大模型及 owned-block/关键路径证据。报告：docs/AUTONOMOUS_RESEARCH_REPORT_ROUND2_20261009.md。

## 2026-10-09：外部研究审查与下一轮交接（无 GPU 执行）

- 审查快照 61139e310a5976cf4311e8bbc4badf82f56439af；用户要求审查 Codex 实验并提供解决思路。完整读取三份指定报告，补读 Round1、实现及公共作者材料。
- 独立复算 fork-decode-control-4 原始 requests/日志/episodes/counters：48 child +3 BG calls，H2D 2016 MiB，workflow median 4134.640602 ms，TTFT P95 480.419642 ms；两份 runtime 源码 SHA256 与报告匹配。不是全归档/完整 GPU 复验。
- 新诊断：该 run 首轮 client→engine 和 engine FIRST_TOKEN→client 的逐请求中位数为 90.79/115.24 ms，原因尚未归因；48 calls 只有3对 own remote-wait，lookup None 的 follower deferral 缺少专用计量。不能相加 followers 等待或中位数组件当 E2E 收益。
- 建议先补旧负载 K2/gate-off 同资源对照、分层计时及真实 residency，再选择一个适用已有方法作为机制对照；不直接实施 range-ready。所有候选收益与新颖性仍待证实，执行 Codex 可据证据调整。
- 交接：docs/RESEARCH_REVIEW_AND_NEXT_ROUND_20261009.md；复算 scripts/review_round3_evidence.py；结果 results/research_review_20261009/spotcheck.json。标准库 CPU 复算断言通过，未修改 serving、未运行 GPU、未消费服务器预算。

## 2026-10-10：Round4 基线与容量机会闭环开始

- 从 main d403714 审查交接与实际源码；本机 clean，服务器仅历史 untracked，全部保留。独立 runtime 副本，不修改前轮 source 或环境。
- 预注册 `docs/BASELINE_AND_OPPORTUNITY_ROUND4.md`：2/4 GiB 各三对 compiled K2/gate-off，固定交替顺序和40个实际 token 输入，完整 BGv2。主要指标 session E2E，>=10% 改善且 BG wall/TPOT <=+5%；不将开关本身算创新。首轮同时启动顺序作为实际协变量保存。
- 普通开关只改 gate 参数，保留 allocator/in-flight/身份/取消保护与恢复事件。29项本地及服务器 CPU 回归通过。CPU setup 尾部误执行目录返回126，但 runtime copy 成功，无 GPU 操作；没有隐去该命令失败。
- 另预注册独立诊断 on/off、真实物理引用释放 witness，以及 fixed-TTL/program-order 对照（不是完整 Continuum 或新算法）。固定1秒 TTL/age guard；优先保留普通 demand/decode/BG 的安全回退。小状态 CPU 测试先验证共享、部分复用、DMA引用、超时、取消与公平性保护，GPU invariant smoke 必须在性能筛查前通过。
- 预算连续记录在 `results/autonomous_round4/*/execution.json`，含加载/编译/失败/清理；不通过目录重置同轮额度。初始资源边界保持每run30分钟、总4 GPU·小时。

### Round4结束：排除大幅gate收益，真实物理保留仍失败

- 2/4 GiB各三对compiled K2/gate-off；中位配对session变化−0.37/−0.63%，P95−1.51/+12.63%，均未达到10%整体门槛。4 GiB尾部变异大，不选择最优run或宣称稳定性。gate-off仅研究开关，默认部署路径未改变。
- 独立共同runtime诊断OFF/ON：session+4.89%、P95+2.29%，n1观测扰动与运行变异不可分。175容量失败快照、30/32续接准入前失败；reservation全部0，median deficit390blocks；每次busy+free=4680且allocator available=free。已空闲cached block本就可分配，固定快照再idle eviction不能补缺口。173单/2双完成集合witness是静态物理可行性，非最快集合、因果时长或收益上界。
- 生命周期可回退的physical pin先通过invariant16token冷/热一致、idle timer、HTTP取消smoke；program-order/physical-pin各n1 matched screen，session+6.87/+20.09%、P95+4.42/+22.20%，H2D未减少。pin32次取得/32释放（28HOL压力回退、4TTL），最终0。停止该具体控制，不调TTL/age，不触发预注册的反序复验；不否定全部Continuum/retention。
- 首次resource-off因instrumentation缩进错误失败，完整51.60秒成本、日志、源码保留；修复后disabled AST等价，再独立v2 OFF/ON。CPU setup/fixture/temp-dir失败亦记录；最后44本地/15native CPU测试通过。
- 独立复算17完整性能run全部40FG输入/输出长度、4BG输入hash与8192工作量匹配，P95/每session E2E算术一致；271导出原始hash通过。19GPU执行18VALID/1FAILED，0.5462201082 GPUh，最长111.96秒，GPU0only，最终两卡19MiB/0%且无computeapp。
- 决策B：没有被接受的新算法。更合理的待识别问题是active KV释放/执行协作，但当前173/175单owner快照不支持共享释放集合的创新收益。下一轮优先真实外生到达/共享工作流与既有强算法对照，先验证差异机会；保持最终目标。完整owned memory-time、全40输入质量、remote DMA取消矩阵及开放负载公平性UNKNOWN。报告docs/AUTONOMOUS_RESEARCH_ROUND4.md。

## 2026-10-10：Round4外部复核与研究机会筛选建议（无GPU）

- 审查465c4d2240b34210e93d6d861e3ec0f9fc830774。原始OFF/order/pin三个run的逐session E2E复算与报告一致；六对gate CSV复算中位效果为2GiB -0.3678%、4GiB -0.6295%。不再保留K2主导排队的未支持假设。
- 原始pin日志32 acquire/32 release/最终0；28 HOL fallback、4 TTL expiry、0 REAL_REQUEST_ALLOCATED交接；持有时间中位274.26ms。不能等同零local cache hit，也不能用单次三元组宣称稳定20.09%因果退化。
- 阅读ownership/retention源码和诊断汇总；没有重新验证271文件或完整175快照原始审计。固定快照中idle可驱逐块已计入available；继续只调JIT/idle保留缺少正证据。
- 建议暂停旧负载局部策略循环，下一交付先是有来源数据支持的研究机会决策，最多一个具体可证伪缺口，不预设共享超图调度。原AGENTS预算/目标保留，具体见docs/RESEARCH_REVIEW_ROUND4_20261010.md。CPU复算完成，GPU0小时；未修改serving或启动新实验。
## 2026-10-10：交接后的研究机会判断（CPU，无新GPU轮次）

- main快进到1c9c7da，读取RESEARCH_REVIEW_ROUND4_20261010.md；接受暂停旧JIT/gate/pin/普通回填变体，独立筛选而非预设第五套算法。
- 公共SemiAnalysis Weka v7固定23f152f6f0f9399a85901b89a6458def0ef16729；预定前16完整行/64MiB上限，取得15条46,447,908bytes。Ubuntu公共API超时，改本机只读HTTPS成功；不是GPU失败。未获取完整1.85GB、未付费生成或提交大原始trace。
- CPU2661请求/47子group/892子请求，代理input median118208/P95624320，output median460/P955053；同模型完成输入历史prefix覆盖97.83%，其他显式容器3.46%。不同trace/model隔离，无序hash不冒充prefix；93.05%超过旧32K，不能直接复用旧模型设置或假定代理长度是真tokens。
- 4单元测试通过，独立CSV算术和92固定抽样直接LCP对照通过。真实host miss/重算、工具-only时长、owned-memory-time、准确DAG与A6000部署成本UNKNOWN；不把结构复用当cache hit或速度收益。
- 最近MORI、EfficientAgent、RR-Evict、Continuum及现成AIPerf loader覆盖了多种直观候选。唯一保留成本识别问题：自然预算下可复用prefix是否被host决策丢失；实际Z与区别未识别，不写新runtime、不启动GPU。建议暂停/归档原JIT论文主张，保留项目目标与工程。报告docs/RESEARCH_OPPORTUNITY_DECISION.md；结果results/opportunity_decision_20261010/，GPU成本0。
- 2026-10-10后续自主探索：按用户放宽后的创新标准，允许组合已有准入/尾部驱逐方法。新增host_prefix CPU7容量×4策略；cost-tail全部失败，不能只用长度平方代替复用判断。连续prefix参考未缓存量只改善约1%，写入模型大幅下降，转向真实host写入放大验证。
- 真实compiled1.5B/natural85%GPU、host256MiB/forcedGPU-cold单路径PoC：full1/frontier1、frontier2/full2两对，P95-17.86/-19.88%，BGmedian-8.08/-9.15%，D2H-94.40%、总迁移-67.54%，H2D增加8.01倍。实际source/inputs/outputs/容量核算通过；不宣称生产/整体Session/SOTA/统计显著。
- 19项测试、native320请求差分/引用取消检查、oracle2单输入16token独立cold比对通过。oracle1salt字段位置错误INVALID、calibration1接口体误判FAILED、CPU中断/分析动态counter错误均保留。全部8GPUrun含失败启动清理0.1388070682GPUh，GPU1未用，最终两卡空闲。
- 决定：保留可持续partial checkpoint +写入准入+公平尾部回收组合，停止直接cost-tail；固定单路径cap有双会话逐出反例，不能部署。下一步自然聚合压力/多会话adaptive额度与强已有准入对照。详见docs/AUTONOMOUS_RESEARCH_HOST_PREFIX_20261010.md、results/host_prefix_screen_20261010/与results/host_prefix_cost_20261010/。
- 预算连续性补充：本次新增0.1388070682GPUh，保守合并Round3/4共0.8855139290GPUh<4；budget.json按目录列小计，未来runner合并检查，不用新结果目录重置授权。

- 2026-10-10 外部审查18ad6b7：独立核对四run原始计时、输出量、记录输入哈希、source/transfer计数及oracle2，确认受控partial checkpoint正结果。新增GPU消耗0。接受继续该方向；先补NoHost与StaticFairPrefix，再决定adaptive复杂度，并验证自然回收/真实host预算。具体交接：docs/RESEARCH_REVIEW_HOST_PREFIX_20261010.md；复算results/research_review_host_prefix_20261010/spotcheck.json。
# 2026-10-10 — 执行侧接受 host-prefix 独立审查

同步main至`b363f16`，阅读`docs/RESEARCH_REVIEW_HOST_PREFIX_20261010.md`及spotcheck；对照checkpoint、成本评分源码与既有协议。接受先NoHost、后StaticFairPrefix及生命周期、再自然压力的顺序，暂不增加adaptive frontier复杂度。澄清cost-tail是累计而非边际评分，其失败不否定整个成本敏感类别；修正主报告H2D为原来的8.014倍（增加701.4%）。本次仅审查与文档修订，没有新GPU实验或性能结论，累计预算不变。未修改原始结果。

## 2026-10-10 — 三问题匹配验证完成

- 按host-prefix审查实施NoHost、静态均分和无GPU reset验证；冻结协议HOST_PREFIX_FOLLOWUP_PROTOCOL.md，定义提交158778f。16个GPU run全部COMPLETE/exit0，最终两卡空闲。
- 两个种子单会话partial vs NoHost P95−17.93/−20.09%，BG−7.91/−9.35%，通过机制门槛。双会话固定256MiB均分4672/4688tokens，逐次恢复正确；P95−8.40/−8.90%，BG−4.76%，未通过10%门槛。
- 无reset四闭环会话2GiB GPU/512MiB host：相对NoHost Session E2E median−8.8766%、BG makespan−5.8354%，四会话E2E皆改善，但一个P95+1.41%；相对Full E2E−8.8721%。未通过预注册10%主门槛，不调阈值。
- GPU-hot无host命中，E2E median+1.60%、一会话+8.52%；host充足时实际cap=Full，普通Offload比NoHost快75.02%，不是partial创新；相对等效Full中位+0.94%、一会话+5.46%。全部负对照保留，单次并发差异不能判定统计原因。
- 本机无损归档独立校验全部payload/source/bytes/pool；30测试、真实native metadata生命周期通过。pytest临时路径问题、tar沙箱权限失败及随后INVALID空摘要保留并修复，未改有效GPU数据。无新多会话GPU内容oracle。
- 新增0.3645630389GPUh，连续累计1.2500769679/4h；CPU处理不占GPU。报告docs/AUTONOMOUS_RESEARCH_HOST_PREFIX_FOLLOWUP_20261010.md，原始results/host_prefix_followup_20261010/。
- 决定：保留简单容量受限工程候选，停止本轮、不开发动态评分或正式矩阵；先证明真实部署下合理host预算确有压力，再决定机制深化。生产适用性/动态公平性/owned-memory-time/SOTA优势仍UNKNOWN。

- 2026-10-11 外部审查c62d9f6：独立复算全部16run原始requests与指标快照，确认单会话胜NoHost、多会话前缀保留及受控无reset约8.88%会话改善；未通过历史10%门槛，hot/ample单会话退化警示保留。新增GPU消耗0，未重跑payload哈希或完整GPU测试。下一自主目标为验证合理GPU/pinned-host预算下的部署价值，不追阈值调参、不规定细碎实验步骤。审查与范围docs/RESEARCH_REVIEW_FOLLOWUP_20261011.md，复算results/research_review_followup_20261011/spotcheck.json。


## 2026-10-12 — 部署机会 CPU 审计（10-11启动）

- 基线834ed5a，冻结DEPLOYMENT_OPPORTUNITY_PROTOCOL_20261011。固定15条Weka源SHA，27个trace/model独立事件审计：累计唯一proxy输入最大2760192，活跃峰值769344，活跃+离线future-needed保留峰值1332032；3/27的保留目标增加峰值，不能把累计工作集当并发显存。
- 官方配置核算1.5B FP16 KV28672B/token、32B262144B/token；32B TP2权重61.0275GiB，85%双卡扣权重仅剩至多20.5526GiB，未扣runtime、不是实测KV池。原生cpu_bytes_to_use已聚合world_size，不重复乘TP。闲置host约118.87GiB可用不能证明512MiB预算合理。
- 38项相关测试通过；独立区间集合参考检查共享/完成时序、namespace隔离。当前1.5B双重压力没有可信证据，32B长上下文只保留为资源可行性候选，部署价值UNKNOWN，不扩大任意小cap矩阵。
- 32GiB pinned probe已设计但NOT_STARTED_ENVIRONMENT：额度恢复后研究盘/media/a6000/D不存在，repo/runtime/账本均不可访问，upload和创建临时目录失败。仅只读诊断，未挂盘/sudo/改环境；请求原挂载点或新路径。无新GPU性能或正确性结论。
- 新增GPU0，连续1.2500769679056248/4h；两种后续runner预算纳入新family。报告docs/AUTONOMOUS_RESEARCH_DEPLOYMENT_OPPORTUNITY_20261012.md；原始事件、容量、官方metadata与环境证据results/deployment_opportunity_20261011/。

- 2026-10-12恢复：用户修复研究盘，原路径/runtime/账本可访问，保留dirty服务器工作树。GPU0先有RustDesk占用，选择GPU1；新增GPU选择和历史账本缺失拒绝启动，40 tests PASS。pin32_20261012实际32GiB pinned分配+zero27.01465s，整worker55.67864s；RSS+32.104GiB，del+GC仍缓存，进程退出后系统可用RAM恢复；VmPin/VmLck0不能否定实际is_pinned=True。无需256/512MiB物理cap，无新增模型性能证明。新增0.01574406685GPUh，累计1.2658210347572914/4。原始probe结果和报告恢复补充均已保存。

- 2026-10-12 外部审查b85027a：复算27 namespace汇总、32B KV/权重容量与pin32 RSS/系统内存差，支持报告边界；新增GPU0。32GiB分配成功不等于serving性能通过，27s是初始化成本而非逐请求成本。下一自主目标为明确部署下Full/NoHost的实际可避免成本，允许有依据的透明合成场景，不强制完整生产trace或自动转32B。交接docs/RESEARCH_REVIEW_DEPLOYMENT_20261012.md。

## 2026-10-12 — 32B 自然 GPU 容量与有限 host 对照完成

- 接受用户较大模型建议，但先验证资源与成本。固定官方 Qwen2.5-32B-Instruct revision、FP16/TP2/compiled，独立 checkout，保留原服务器 dirty 树。模型24文件官方hash通过；下载1中断转双路、下载2 deadline失败、下载3成功，原失败保留，下载GPU0。
- 自然85%配置实际70,000 GPU KV tokens=17.08984GiB。21×28,672初始历史目标147GiB大于全RAM+实测GPU pool；只是合成保留压力，不称生产或必需驻留量。性能冻结相同自然池、64GiB host、21×3 turns/128outputs、5s工具gap、背景3×8K、单seed、无GPU reset。
- 原生8K CPU restore可行；独立batch-invariant两敏感输入6请求，12,480-prefix恢复与独立cold逐16token一致，来源核算通过。随机性能输入不用于质量声明，完整取消/抢占/长期公平性仍UNKNOWN。
- 真实NoHost/Full/Fair各66请求有效匹配。Full写448.875GiB、CPU/GPU复用0；Fair仅写64GiB、恢复128GiB/524,288tokens，compute减少28.14%。Fair E2E median相对NoHost/Full−16.44/−18.40%，背景−13.16/−15.23%，21session全部改善，通过冻结10%整体/5%退化门槛。P95仅−6.39/−8.50%，不换指标宣称全部胜出。
- 吞吐+21.88/+24.76%，总迁移比Full少57.23%；Fair/Full仍同样实际host64GiB，未证明RSS或active GPU footprint下降。第三BG首token至响应结束平均间隔+16.94/+14.70%，代理Decode代价披露；总BG完成更快。启动123.68/155.75/280.26s的顺序差异单列，需反序复验。
- 首次退出后GPU释放未确认，额外保留0.4724857332GPUh未知区间上界；不是实测忙碌。后续验证ownedgroup为空且两卡19MiB/0%。用户批准连续4.5GPUh仅完成这一组，最终4.1919880197，新增family2.9261669850，余额不启动额外实验。
- 82原始文件84,782,502解压bytes SHA/长度与压缩-only本机复算通过。acquire/source版本、原始输入/metrics/日志、失败与commands保存；可行性初版runner只有SHA无完整快照的缺口明示。线程池超时清理修复未用于本次性能运行，独立测试；工程失败保留。
- 决策A：保留简单capacity-conditioned prefix admission候选，不宣称静态均分新颖/SOTA/统计显著。下一步反序与种子复验、非均匀历史和较长输出的少量匹配筛选，先确认regime和后台Decode代价，再决定动态额度。报告docs/AUTONOMOUS_RESEARCH_LARGE_MODEL_20261012.md；结果results/large_model_20261012/。项目最终目标不变。

- 2026-10-12 独立审查eead834：复算三方法原始请求计时/记录哈希配对和迁移来源，确认Fair相对NoHost/Full会话E2E−16.44%/−18.40%，接受初步整体正结果。保留单run、配置差异与后台输出阶段代价；未重算payload SHA/完整oracle或新增GPU。下一目标为自主反序复验和少量收益边界，遵守仅完成本组三方法的4.5GPUh授权范围。详见docs/RESEARCH_REVIEW_LARGE_MODEL_20261012.md及results/research_review_large_model_20261012/spotcheck.json。

- 2026-10-12 用户批准分阶段系统架构交接：当前优先实现动态前缀基本份额、弹性借用与原生安全尾部回收（M1），随后压力切换（M2），有明确机会再加可选预取（M3）。复用compiled/native offload，允许成熟组件，不把旧反序多种子矩阵作为开发前置条件。文档docs/SLACKKV_SYSTEM_ARCHITECTURE_20261012.md；AGENTS新增当前入口与优先级，旧证据保留。仅文档交接，未实现运行时或新增GPU；数字预算未扩大，代码/CPU验证可先自主推进。

- 2026-10-12执行侧审查719d856：整体接纳M1优先、成熟组件组合和预取关闭；核查实际native manager/LRU/BlockStatus/store/completion。需补清原子victim入口、冻结旧计划的未使用承诺、ref_cnt=-1写入保护/TP汇总、迟到通知物理结算、共享fallback统一计费及write游标。max-min公平不替代latency公平，M2/M3不为完整架构强行加入。审查文档docs/SLACKKV_SYSTEM_ARCHITECTURE_EXECUTION_REVIEW_20261012.md；本次无运行时实现或新实验，GPU账本不变，未扩旧预算。

- 2026-10-12 接纳599b89b执行侧架构审查，原子store/reclaim、逐批准入、物理/逻辑结算分离、共享降级计费和write游标约束合并至主架构§9。审查侧检查归档manager/LRU/TP scheduler，未执行新测试或GPU。M2可与M1已有全存行为合并，M3保持可选。下一实质交付为M1实现和CPU/native验证，不再开启纯文档审查循环；预算不扩展。


- 2026-10-12 system_m1_20261012：按 f2d6af0 主架构§9实现 M1，不再仅审查。新增 max-min 已知前缀分配、短需求借用、逐批 native 净新增准入、保护引用的连续尾部回收、跨 generation 物理结算；原生按需恢复，预取关闭。隔离 overlay 仅两处 metadata hook，disabled AST 相等，不修改旧 native tree。64项本地测试通过，native v1/v2分别8/12组PASS，每版含120次交错操作；实际 store-job builder/cursor/TP完成汇总验证通过。M1模型内容/性能/后台公平性 UNKNOWN，未把历史32B StaticFair收益记为M1成果。共享、分支、未知身份停止可选写入，不无计费切Full；显式end、tombstone GC未实现等限制完整披露。一次工具429未执行命令，未绕过审批，稍后重试通过。新增GPU0，连续4.191988019727527GPUh不变。报告docs/SLACKKV_SYSTEM_M1_REPORT.md，原始结果results/system_m1_20261012/。决定：交付CPU/native里程碑；下一步小模型内容/生命周期smoke需新数值授权，M2/M3不提前开展。


- 2026-10-12 M1修正交接（审查基线92e65cb）：独立执行原始控制器确认 end/finish/collect 后同代迟到请求重新获准（reason=None, ceiling=2, allowed=2）。指导执行侧保留独立结束状态、添加针对性控制器/native回归，并准备内容/生命周期与M1对StaticFair的小规模GPU验证；沿用架构，不扩M2/M3，不要求先做反序多种子。原64项及native审计未由审查侧重跑，本地pytest不可用；本次无代码修复/GPU运行。交接docs/SLACKKV_SYSTEM_M1_CORRECTION_20261012.md。新增0.5GPUh仍为待批准建议，先完成代码/CPU/脚本，若执行会话已有新授权则遵循之。


- 2026-10-13 M1修正（交接4d1da69）：先在旧92e65cb controller上跑新回归，1 failed/15 passed；同扩展native审计旧版FAILED，同代结束并物理collect后恢复store资格。新增latest终止bit独立于Member GC/reset，旧/未知/未来end no-op，更高代合法，普通finish不终止；未更改native有效DMA结算。修复后74项本地回归、15组native场景PASS（含120交错），共享system prefix不同后缀一次fallback并合法read。真实模型配置/tokenizer CPU准备smoke_v3：FP16 TP1 compiled、GPU512MiB、host224MiB/512块、模型limit18432；发现旧32K limit不适配，在GPU启动前修正，v1/v2/v3记录保留。runner/config/输入/预算锁/清理门/比较器已交付。BUG_FIXED与CPU_NATIVE_PASS=PASS，GPU_CONTENT=NOT_RUN，动态收益UNKNOWN；新增GPU0、历史4.191988019727527不变。未修改cutlass .pth警告的共享环境，GPU启动仍需真实验证。结果results/system_m1_fix_20261012/，报告M1_REPORT修正附录；唯一待决策项是新0.5GPUh授权，M2/M3关闭。


- 2026-10-13 system_m1_smoke：用户批准新增0.5GPUh，冻结efb74ae输入/模型/资源做A后B。三次FAILED依次为PATH缺已有ninja、隔离native旧scheduler依赖缺失、INFO trace被logger层级过滤；原数据保留，修复后content4真实13请求PASS，host5904token恢复/cold零命中/16IDs相同、目标448→512实际保存、终止代拒绝写入、最终512free。native16组PASS/本地75项。compiled7个匹配性能run每个108544输入/192输出，M1/Fair前序+2.83%/反序-6.01%median E2E，M1/NoHost-9.20%/-10.00%；M1repeat与单Full接近，不宣称稳定收益。首请求1124→463ms、NoHost1114→450ms，具体原因UNKNOWN，全部计入。全协议M1 D2H504MiB/H2D512.3125 vsFull336/485.1875；会话后terminal新代多写224MiB没重用，未来准入机会，本次未改策略。324原始文件hash/压缩-only复算及源/真实H2D校验PASS。新增0.13476591384806347GPUh/0.5，连续4.326753933575591；11run全部清理、两卡19MiB/0%，停止GPU且M2/M3关闭。报告SLACKKV_SYSTEM_M1_GPU_REPORT_20261013.md；下一步预注册等价warmup并研究无用terminal写入与共享prefix覆盖，不能把阶段完成替代整体收益。


- 2026-10-13 独立审查82bf363：CPU复算7个性能run原始请求/解压输入哈希及三处prom，确认M1/Fair median +2.83197%/-6.01401%、来源守恒和H2D=external×28672；核对content4原始16IDs/cold零命中/35状态快照，独立复现终止generation修复。接受M1基础实现及被测GPU路径，不接受稳定胜Full；long3 Full与M1均约42ms，末尾224MiB写入不在原会话E2E范围。下一步小幅terminal准入优化、等价warmup/不同marker oracle、单个多会话竞争筛选；共享覆盖先于无证据M3，不重写主架构、不要求大矩阵。已知新增0.5GPUh授权有效，已用0.13476591384806347，按原范围/账本执行，不重复申请已有许可。审查新增GPU0，未重跑服务器测试/native或完整manifest。详见docs/RESEARCH_REVIEW_M1_GPU_20261013.md，复算results/research_review_m1_20261013/spotcheck.json。


- 2026-10-13 system_m1_followup（418288c交接）：接纳terminal gate、双marker oracle、等价warmup与单个资源竞争筛选，主架构不变/M2M3关闭。M1 gate仅停止未发出可选store，Full/StaticFair共同使用独立terminal gate，读/在途结算/结束身份保留。62本地相关测试/19 native组PASS，content2两个真实host恢复与零命中cold各16IDs一致、CEDAR17/MAPLE42可区分、新terminal D2H0、最终512free。首次plain completion oracle FAILED，runner循环变量覆盖run名落入contention.py目录，两缺陷修复且失败原数据/源码SHA/预算保留。冻结chat输入四会话增长+17请求、130304输入/272输出、GPU512MiB/host224MiB、五形状独立warmup且不清GPU；一次M1/Fair/Full强对照median E2E5234.78/5330.23/5652.81ms（M1/Fair−1.79%、M1/Full−7.40%），10续接nearest-rank P95252.17/262.41/345.56ms（−3.90%/−27.03%）。M1少重算6432tokens但Fair→M1 D2H196→476MiB、H2D420→595.875；动态增量小且成本高，不默认替换Fair。首次正式请求均401–405ms，全部计入。297原文件SHA及压缩版复算/独立逐请求来源时间戳核验通过。准备元数据澄清19456为四配置会话合计、S退出后加入D实际18432；最终CPU准备输入SHA不变，未改旧数据。新增0.07534938563277440GPUh，原0.5授权累计0.21011529948083787、历史连续4.402103319208365；五run全部清理，两卡19MiB/0%。报告docs/SLACKKV_SYSTEM_M1_NET_BENEFIT_20261013.md，原数据results/system_m1_smoke新五目录，results/system_m1_followup复现/分析。保留M1容量竞争下可选路径，下一优先共享覆盖/最小多owner计费；本轮共享比例/新NoHost/并发后台/正式统计/memory-time未执行，不外推生产、不包装普通公平/terminal为创新，提交后停止本轮。


- 2026-10-13 独立审查375c28e：CPU复算三性能run输入哈希、warmup、原始计时、起止prom及双marker内容记录，M1/Full median−7.3951%、总DMA−35.7966%；M1/Fair median−1.7908%、总DMA+74.0057%，认可独立前缀M1工程里程碑和本筛选正结果，动态仍opt-in。独立controller复现terminal-only加入使A目标512→256→512；结合native先observe/reclaim后禁写的路径，要求补齐不新增未来保留需求语义，未执行该反例native/GPU。发现性能trace_states=true/M1有57详细快照，开销未测，下次性能测量调整，不为此单独大复验。下一实现增量优先有限共享system-prefix+私有后缀的单次物理计费/安全释放，M2/M3关闭，当前场景不再追参。审查新增GPU0，未重跑完整服务器测试或manifest；既有0.5授权累计0.21011529948083787不重置。详见docs/RESEARCH_REVIEW_M1_NET_20261013.md，results/research_review_m1_net_20261013/spotcheck.json。


- 2026-10-13 system_m1_shared（b242f2d交接）：接纳terminal需求修补与有界公共system-root/私有tail增量，不调旧序列/M2M3关闭。Member请求path/retention需求分开，新terminal不分份额，已有terminal增长不扩需求/转余数；原生读取和在途结算正常，真实完成才end。共享flag默认0，显式32块支持root，key_refs多逻辑引用+distinct目标并集收费+DMA祖先闭包尾回收，保留旧独立key_owner模式。86本地相关回归/28组CPU-native审计PASS；包括512块terminal反例、奇数预算terminal增长、公共块一次存储、A结束B仍读、深层load祖先保护、失败root孤立尾部、旧generation/重复ticket/native reset。三次GPU content失败保留：肯定回答不符startswith格式、priming错误数字、选择题cold B错码。恢复原输入后content4用独立敏感identity oracle，EOS前码或肯定句、全部16IDs不变，最终代码进一步拒绝否定/仅出现词等，实际cold三答案离线再验。content4三恢复含A结束后B与各零命中cold16IDs一致、私有marker不同、host2560+GPUlocal512、70状态守恒/公共32块/最终512free；不声称完整KV字节正确。性能先冻结更强root-aware StaticFair2432深度（root512一次+private1920×4），无详细状态日志，各方法共同两个checkpoint；一次19请求153088输入/304输出强对照M1/Fair/Full median5380.69/5428.11/5512.98ms（−0.87%/−2.40%），11-resume nearest-rank P95239.90/255.81/306.01ms（−6.22%/−21.60%）。DMA M1/Fair总1097.25/567MiB（+93.52%），少重算6528tokens但成本高；不默认动态，不扩大矩阵。433原文件SHA/四版content源SHA/实际执行核心SHA/压缩复算/独立来源timestamp/physical checkpoint校验PASS，预算浮点末位差说明，原数据未改。新增0.10946156016611189GPUh，原0.5授权累计0.3195768596469497、历史4.511564879374477；七run全部清理，两卡19MiB/0%。报告docs/SLACKKV_SYSTEM_M1_SHARED_20261013.md，原数据results/system_m1_smoke新七目录/审计准备分析results/system_m1_shared；保留动态与静态，下一重分配成本事件及共享覆盖再判断，不将工程进展包装SOTA，不宣称NoHost/生产/并发公平或memory-time。commit/push后停止本交接。


- 2026-10-13 system_m1_shared归档补充：content4固定机制路径可定位A初次512→256块尾部112MiB在首次恢复前回收，B末段192→396块补写89.25MiB后仅一次B_end全6336GPU-local/host0再结束；共201.25MiB未复用写入，原始状态/prom/来源证据在unused_write_evidence.json，不作性能全成本/生产上界/删除反事实归因。另修正共享deferred独占尾部诊断低报（含公共根target不能直接从独占数减），audit4新增回归后最终29组PASS，无分配/回收/DMA改变，旧GPU快照原样；原文件总435校验。分析器按cohort筛选，重复同policy不再隐式选最后一run；预算仍计所有family。最终86项相关测试PASS，额外GPU0，累计0.3195768596469497不变。


- 2026-10-13 独立审查c07976b：复算共享三性能run原始计时/输入哈希/起止prom/来源/checkpoint及content4三组16IDs与70状态守恒，确认M1/Fair median−0.87355%、DMA+93.51852%；M1/Full−2.39971%、DMA−13.87363%。输入/顺序/Fair规则改变，不能以旧百分比判断版本退化。接受有限共享与terminal功能，动态仅保留可选，root-aware StaticFair为工程参考。下一步只做一次有上限CPU/已有事件成本定位，核查shared重复祖先遍历，测得明显开销才做等价优化，有在线信号才提出最多一个候选；否则停止该动态变体性能优化，不扩矩阵/M2M3。201.25MiB事后未复用不是在线可免费消除的上界。审查新增GPU0，服务器完整测试/native/manifest未重跑；既有0.5授权已用0.3195768596469497。详见docs/RESEARCH_REVIEW_M1_SHARED_20261013.md及results/research_review_m1_shared_20261013/spotcheck.json。


- 2026-10-13 system_m1_cost（5495a36交接）：一次有界CPU/已有事件定位，未调参数/M2M3关闭。真实native元数据固定状态计时确认共享保护祖先重复遍历：单512保护块约9→0.375ms，空闲不变；needed与已有保护链祖先闭合，新增key-not-in-keep条件，保持分配/引用/free/event等价。原/候选29组审计完整结果一致、各75次shared回收；安装版本29组PASS；84项相关本地回归PASS。现有70状态定位1228写/回收块537.25MiB，70MiB为前缀深度推断的重写，与201.25MiB事后未复用分类重叠，不作key级/性能全归因。复用历史门控不能排除已有真实host复用的末段B扩额，未来GPU/end不可在线使用；未推出新策略。真实contentPASS，3cold/恢复完整16IDs一致、严格有限语法离线复核、共享生存与最终512free。冻结原M1/修正版/StaticFair各一次compiled同输入19请求153088/304：median5341.393/5365.992/5466.442ms；修正版/原median+0.4605%、序列+0.2862%，修正版/Fair median−1.8376%而DMA+93.5185%。CPU局部改善未转为可确认整体加速，停止当前动态变体性能调参，保留optional组件/StaticFair参考，无部署切换。CPU后分析2失败（缺报告证据、旧依赖）保存，不重跑GPU；268原文件哈希与2runner源码hash复核，跨Python求和末位公开。新GPU0.05931228920166985，原0.5账本累计0.37888914884861963、历史4.570877168576147，四新run全部COMPLETE/cleanup、两卡19MiB/0%。CPU真实GPU关键路径累计/阻塞迁移/并发后台公平性UNKNOWN。报告docs/SLACKKV_SYSTEM_M1_COST_20261013.md，复现/原数据results/system_m1_cost与原system_m1_smoke新四目录。提交并push后结束本交接。


- 2026-10-13 research_direction_after_m1（用户收尾M1后的新任务，基线2cdf515）：CPU机会判断，建议暂缓新系统，保留32BStaticFair正结果/生命周期工具/全部历史，不动默认、不开发M2M3/即时动态额度。独立重校82归档文件84782502原字节、三方法198请求/哈希/128输出及prom/时戳；固定私有初始前缀、只首次写入且两次复用的future-aware token分配参考上限524288=Fair实际external，额外0；仅该条件类的token界，不是rolling/compute/E2E全局上界。无限KV输入连续性参考1214976可复用、Fair仍重算690688；不推导可达加速。公共hash代理SHA d078503...重新统计2661调用/65container-model组，calls median20/P95220/max311，无完整>=3调用组满足旧32Kproxy输入+输出；不是Qwen实tokens/线性Agent，不能挑短片段制造新实测。读取EfficientAgent准入/实现条件、RR-Evict公平tail/结束回收、MORIsticky tier及KVFlow图预取；不把普通组件当主创新。唯一条件候选为异质长期会话持久部分前缀价值分配，缺强公平baseline损失/在线区分/成本曲线，未立项PoC；仅提出一个带事前token机会/在线信号停止门槛的CPU最小验证方案，当前不申请GPU。8项新CPU/归档回归PASS；GPU新增0，历史4.570877168576147及M1授权已用0.37888914884861963不变。报告docs/RESEARCH_DIRECTION_AFTER_M1.md，证据results/research_direction_after_m1/evidence_verified.json；无历史文件覆盖。提交push完成本阶段。
