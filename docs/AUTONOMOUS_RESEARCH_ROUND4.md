# Round4：强基线、真实容量依赖与物理保留对照

2026-10-10。起点为 main `d403714ea3ed91e585a2b9f80beb363023272de8`，先读 AGENTS.md、前三轮证据及外部交接，再冻结 [实验协议](BASELINE_AND_OPPORTUNITY_ROUND4.md)。历史原始数据和 runtime 均保留。本轮按有界负结论停止，不申请扩大矩阵，不改变 Agent Serving 的最终目标。

## 结论与技术决定

**本轮没有得到可接受的新加速算法。** 删除自定义 K2 gate 没有达到整体改善门槛；实际物理保留也失败。新增成果是匹配基线、真实物理 block 的释放依赖诊断、带安全回退的保留控制及可审计负结果，不能包装成论文算法。

当前受控负载中，显著等待发生在引擎准入前，并伴随 active KV 容量不足；没有证据将全部等待归因于容量或 K2。最值得进一步识别的是：**如何改变活跃请求执行进度，使真实占用及时释放，而不损害其他 Agent 和后台任务？** 它可以改变容量与计算的共同关键路径，潜力大于仅提前几十毫秒恢复。但通用优先级、SRPT 或普通回填已有大量先例，尚不能算 SlackKV 的创新。

停止本次 fixed-TTL/program-order 控制，不调 TTL 或 age guard。保留原来默认 OnDemand 和所有身份/DMA/allocator 保护；新运行脚本显式选择 gate-off，未把小收益开关推广为新的生产默认。后续优先建立有外生到达时间、真实工作流结构和共享关系的实验机会；只有发现现有方法无法处理的具体资源依赖，才实施新调度 PoC。

## 条件、验收与真实性

- GPU0 RTX A6000，GPU1 未使用；Qwen2.5-1.5B-Instruct FP16、TP1、block16；CPU KV8 GiB，GPU KV2/4 GiB。每 token28672 bytes，每 block458752 bytes，池分别4681/9362 blocks，含一个不可用 null block。
- 使用默认编译与 CUDA Graph；性能模式 batch-invariant OFF。真实运行命令、源码 hash、执行模式在每个 `execution.json` 与服务日志中。执行包 vLLM0.23.0+cu126、Torch2.11.0+cu126、Python3.11.16；这是共同的**历史 patched runtime**，不是 pristine upstream release。
- 移除 gate 是唯一基线干预。JIT、FullSlackKV、credit、epochs、backfill、完整 fork block-list tracing 都关闭；native offload、in-flight reservation、singleflight、真实 allocator 和身份/取消/DMA 保护保留。历史正常 step/resume/source 日志仍开启，不能称为无观测开销的生产基线。
- 固定旧 `capacity-2/inputs.jsonl.gz` 的40个实际 token 数组，SHA256 `ec8db1853d0465f07e38fc17ca0db32aa88f68488de53867f09627dd472e8cd2`。8 sessions×5 turns、32续接、原输出长度与 finish-relative 工具等待；BGv2为4×9500输入/2048输出，总8192输出。
- 这是 hybrid structural replay，未来输入不随本次生成内容改变。初始并发 issue 顺序保存但不强制相同；不是外生生产时间戳 trace，也不能评价真实任务质量。独立审计确认17个完整性能运行全部40前台输入 hash、输出长度以及4个后台输入 hash/工作量匹配。
- 预注册主要指标是 full session E2E median：至少改善10%，BG wall/TPOT 增幅都不超过5%。K2 各容量三对，以逐对百分比的中位数筛选；保留对照先一个三元组，只有通过才做反序复验。无显著性或长期稳定性声明。

## 真实结果

百分比负值表示降低。完整逐 run 指标见 [metrics.csv](../results/autonomous_round4/metrics.csv)，每个 session、P99、迁移及 preemption 在 [round_summary.json](../results/autonomous_round4/round_summary.json) 与 [control_summary.json](../results/autonomous_round4/control_summary.json)。

| 对照/干预 | 样本 | Session median变化 | Resume P95变化 | BG wall变化 | BG TPOT变化 | 通过门槛 |
|---|---:|---:|---:|---:|---:|---|
| 2 GiB，gate-off相对K2 | 3对，中位配对效果 | −0.37% | −1.51% | −0.16% | −0.16% | 否 |
| 4 GiB，gate-off相对K2 | 3对，中位配对效果 | −0.63% | +12.63% | −0.07% | −0.06% | 否 |
| 2 GiB，program-order相对同runtime OFF | 1对 | +6.87% | +4.42% | +0.90% | +0.91% | 否 |
| 2 GiB，order+physical-pin相对同runtime OFF | 1对 | +20.09% | +22.20% | +3.69% | −1.07% | 否 |
| physical-pin相对order-only | 1对，机制拆分 | +12.37% | +17.03% | +2.77% | −1.96% | 否 |

2 GiB三对 session变化为 +4.72/−0.37/−4.13%；4 GiB为−1.04/−0.63/−0.47%。4 GiB P95三对为+12.63/+96.46/−23.28%，尾部变化很大。不能选一个 run 宣称稳定改善，也不能把2→4 GiB资源效应归为新算法。

保留三元组绝对值：OFF/order/pin 的 session median 为23746.27/25377.64/28517.68 ms；P95为5365.76/5603.17/6557.21 ms。order-only 使5个 session变慢，E03 +20.54%；pin使全部8个 session变慢，范围+13.80～+29.92%。pin的最大 session +19.52%，混合输出吞吐−3.56%。后台完整完成，不以删减后台工作制造收益。

真实服务端 H2D计数为19276.25/19276.25/19281.50 MiB，D2H均5172.5625 MiB；pin没有减少迁移。它发生1次 preemption，另外两次为0。计数取自独立新进程结束时的 Prometheus 累计值；共同 warmup与BG设 `max_offload_tokens=0`，保留原计数，不把 source-ready 事件推算值替代真实字节。raw run 中沿用的 `OnDemand-custom-gate-off` 标签不能区分控制算法，应以 `control_config.json` 和上述目录名识别。

## 资源依赖诊断：支持什么、不支持什么

另设同一个修复后的诊断 runtime，OFF→ON一对，均 compiled/gate-off/2 GiB。观测 ON 相对OFF：session +4.89%、P95 +2.29%、BG wall +0.92%、TPOT +0.93%，并发生额外一次 preemption。n=1结果混合了运行变异与观测扰动，不等于精确 instrumentation 开销；不能将诊断 run 的性能拼入无额外 tracing 的主要对照。

[diagnosis.json](../results/autonomous_round4/resource-on-repaired/diagnosis.json) 和原始服务日志显示：

- 30/32续接在准入前有真实容量失败；总计175个失败快照、265个资源事件。失败采样限制为100 ms，真实 free事件不节流。这不是完整容量阻塞时间积分。
- 续接逐请求阶段的中位数：client issue→engine arrival24.13 ms，arrival→admit2081.63 ms，admit→engine FIRST_TOKEN34.20 ms，engine→client first26.78 ms；TTFT median2252.32 ms。中位组件不可相加，外部组件包含序列化/API/IPC/观察，不宜称为纯网络开销。
- 全部175个失败快照的 in-flight reservation为0；缺口中位390 blocks≈170.63 MiB。active unique中位3603，busy unique中位3628，idle cached且refcount0中位1052。每个快照都核对 `busy + pool_free = 4680`，并确认 `available = pool_free`；最后allocator check距快照最多229.83 μs。
- 已可分配的 idle cache 已计入 available。**固定这些快照的运行状态，再驱逐 idle cache不能填补缺口。** 这不否定更早保留/准入改变后续轨迹，也不是容量独占排队原因的证明。
- 枚举真实物理引用的最小“请求完成集合”：173个快照一请求完成即可补缺口，2个需要两个。外部/DMA引用不能被这些请求完成虚假释放。Witness冻结 decode增长、reservation及其他引用；它不是最快完成集合、延迟预测或反事实加速上界。单请求 witness 按ID选取，剩余输出数也不能当成最优调度成本。

所有175个快照由 [integration_audit.py](../results/autonomous_round4/integration_audit.py) 独立复核。[resources.py](../experiments/autonomous_round4/resources.py) 的 ownership CPU总160.75 ms、快照CPU P95约641.46 μs，不含JSON/日志；完整 owned GPU KV memory-time、lookup None follower等待原因、API边界完整因果贡献仍UNKNOWN。

## 最小实现与安全性

物理控制在真实请求 free之前 `touch` 完整 hashed blocks，直到固定1秒TTL、成功新请求分配、取消或压力回收。保留额外引用，不冒充已经服务请求的所有权。native allocator和DMA引用不变；取消、老turn延迟D2H free、新turn进入、部分共享复用均有明确处理。

program首次到达决定 waiting队列顺序，超过1秒的等待优先；并非完整 Continuum调度器。Decode增长/后台容量压力先撤销 idle pins；有排队 pin消费者时的实际分配失败立即撤销并重试，避免只等TTL的 Lease/HOL循环。无新请求时使用原engine probe唤醒完成超时释放。新控制默认关闭。

性能pin run取得32次真实额外引用并释放32次：28次 `QUEUED_CONSUMER_HOL_FALLBACK`、4次 `TTL_EXPIRED`，最终0。这证明保护和安全回退实际执行，也显示该实现没有创造可持续保留收益；不能直接推广为所有需求回收/TTL方案无效。

先行 batch-invariant GPU smoke 验证2048 local cached tokens，续接16输出与 salted冷Prefill逐token相同，冷请求实际local/external hit均0；无新HTTP时TTL释放；断开生成流触发引擎CANCEL，最终pin0。44项本地单元测试及15项native CPU回归通过。完整性能运行无OOM、历史lease/logical leak0、规定工作完成。

这些验证不是所有 interleaving 的证明：remote DMA leader/follower取消、完整GPU Lease/HOL压力矩阵、全部40输入的内容/质量等价、开放到达下无界饥饿尚未验证。age guard提供有限优先级保护，不能据此宣称普遍公平性。失败控制不投入生产。

## 文献与创新性筛选

重点读 [Continuum v7](https://arxiv.org/html/2511.02230v7) §§4–5 的program顺序、TTL及成本/排队收益估计；[作者实现](https://github.com/Hanchenli/vllm-continuum)明确缺少论文estimation。因此本轮是固定TTL、压力回收和age guard的机制对照，**不是完整论文复现，也没有推翻论文结果**。[MORI](https://arxiv.org/html/2606.00866v1) §4 已研究idleness、tier placement与admission，简单把这些组合不构成新颖性。

另筛选“共享物理释放集合驱动的decode执行”：使用所有权超图，衡量一个请求/请求集合完成后真实可分配的union，而非逐request KV大小相加。它有潜在区别，但本负载173/175 witness是单请求，未展示非加性依赖能带来价值；普通completion/priority调度又与已有工作重合。本轮不为凑算法写新PoC。该比例是快照比例，不能转为关键路径占比或总体收益上界。

本轮并未发现应改变最终研究目标的证据；发现的是当前独立长前缀hybrid workload不足以支持上述联合释放创新。下一轮应先找真实/公开工作流中共享KV、外生到达和分支完成资源依赖，复用现有diagnostic；事先定义SLO并发/owned memory-time与后台约束，加入适用已有算法强基线。若这类依赖或收益空间不存在，继续否定该候选；不靠合成极端压力或更大模型掩盖创新不足。

## 失败、归档与复现

首次 `resource-off` 失败：resource instrumentation补丁改变external-token分支缩进，触发 `num_external_computed_tokens > 0` assertion及EngineDeadError。这是实现错误，不是算法负结果。完整失败日志/部分数据/51.60秒预算与 [failed_source](../results/autonomous_round4/failed_source/) 保存。修复后先做去除disabled观测代码的AST等价检查，两native文件均通过，随后另建共同v2 runtime运行OFF/ON，不覆盖失败。

CPU失败亦记录：setup尾部误执行目录exit126、两个不合法AST测试fixture编辑、本机临时目录权限与一次缺失父目录；修正后44项通过。原`.pth`缺包警告保留，未修改服务器环境。

19次GPU执行：17次完整性能、1次有效correctness smoke、1次失败，累计 **0.5462201082 GPU·小时**，最长111.96秒，均GPU0。加载、编译、warmup、失败和清理计入。两卡末尾19 MiB/0%，无compute app；见 [budget_summary.json](../results/autonomous_round4/budget_summary.json)。有界负结论已充分支持停止该具体路线，无须消耗完4小时。

原始结果见 [results/autonomous_round4](../results/autonomous_round4/summary.md)，271个导出逻辑文件逐一SHA256校验；服务器原文件不删除，Git以无损gzip归档日志、JSONL及大run JSON。55份实际native/实验源码压缩快照及三个runtime Python树fingerprint在 `provenance/`，源基线、working diff、runner SHA和参数在execution ledger。旧runtime没有upstream Git身份，包版本不等于pristine source复现保证；保存patch与实际修改文件便于核对。

复现/离线审计命令见 [commands.sh](../results/autonomous_round4/commands.sh)。独立复算P95及每session E2E与主summary一致；缺失计量保持UNKNOWN。后续重大新增运行必须建立新的预注册与连续预算，不能拿本轮目录或最优run重置筛选。
