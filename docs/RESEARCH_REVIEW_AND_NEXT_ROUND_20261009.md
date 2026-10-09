# SlackKV 外部研究审查与下一轮交接

日期：2026-10-09。审查快照：`61139e310a5976cf4311e8bbc4badf82f56439af`。

这是一份给执行 Codex 的研究交接。用户要求外部审查实验、帮助解决困难，同时保留执行者的自主判断。以下建议可以被新证据推翻；不要把它们当成必须跑出正结果的命令。继续遵守 AGENTS.md 顶部的自主研究授权、单实验 30 分钟/单轮 4 GPU·小时、证据保存和服务隔离约束。

## 1. 审查结论

**赞同停止现有 JIT/Frontier、普通共享恢复去重和当前场景完整 active-KV streaming 的扩展。下一轮优先解决“有多少容量排队真正可避免”，不直接实现 Round3 末尾的 range-ready 候选。**

三个阶段揭示了不同问题，不能拼成“所有 KV 管理无效”：

| 证据 | 可支持的结论 | 不支持的外推 |
|---|---|---|
| Reassessment 固定路径删除 remote wait 后，OnDemand P95 改善中位约 0.905% | 当前旧负载的直接恢复时间优化空间小 | 所有改变驻留/排队的策略最多改善 1% |
| Round2 2→4 GiB：P95 5199.92→1163.51 ms，session median 23525.76→18780.22 ms | 容量干预影响端到端路径 | 同资源软件可以免费获得全部资源增益 |
| credit：P95 -39.90%，session median +6.08%；epoch：H2D -38.60%，session median +14.37% | 两个具体实现没有满足原冻结目标 | 所有程序调度/物理保留策略失败 |
| Round3 每 episode 一次 ancestor restore，七个 local root hits | 当前原生 connector 已去重，不应再开发 ordinary singleflight | 所有恢复依赖延迟都已消除 |

Round3 的两个 GPU KV 预算均足以容纳实测唯一工作集。人为 2-GiB 预算作为压力实验是有效的，但不能当作 A6000 的物理上限或代表性部署结论。新颖性尚未成立，不能用一般工程优化或重新命名已有机制填补。

## 2. 本次独立复核做了什么

完整阅读用户指定三份文档，并补读 Round1、fork runner/analyzer、epoch 实现、公共作者材料；读取 Round3 聚合数据。对 `fork-decode-control-4` 的保存请求、完整 server log、episodes、Prometheus counters 独立复算，同时核对两个 gzip 运行时源码的解压 SHA256。

- 48 个 child calls +3 个 BG calls；每 child 128 outputs，每 BG 1024 outputs。
- 三个 episode 的 workflow 分别为 4105.350553、4134.640602、4162.707304 ms，与保存结果完全一致。
- H2D = 2113929216 bytes = 2016 MiB = 三份 24576-token ancestor；source events 与实际 counter 一致。
- child TTFT nearest-rank P95 = 480.419642 ms。
- native scheduler 指纹 `bc883cf47d882d4406d7708a8d3520abe2752f545985705126f41fca66558bf0` 与报告相符。

复算命令（仅 Python 标准库、CPU）：

```sh
python scripts/review_round3_evidence.py --output /tmp/slackkv_round3_spotcheck.json
```

本次结果保存于 `results/research_review_20261009/spotcheck.json`。这不是重新验证全部 101 个归档文件，也没有执行 GPU 实验、完整内容 oracle 或全生命周期压力测试。

## 3. 比继续增加候选更优先的四个问题

### 3.1 历史 K2 不应未经对照成为永久研究前提

旧负载的 Compiled OnDemand 仍使用 patched runtime 和 first-token K2。源码中 gate 检查发生在 external lookup 之前，gate owner 到引擎产生首 token 才释放。Round3 的 K8/4 GiB 与 K2/4 GiB 是低压力共享 fork，不能回答旧独立大前缀/2 GiB regime 的 K2 影响。

保留历史 K2 作为回归，同时增加**同一 pinned compiled runtime、所有候选关闭、仅关闭自定义 first-token gate**的对照。原生 allocator、in-flight reservation、KV 身份和取消保护仍保留。先审查开关依赖，不能随手删安全检查。说明该对照与干净 upstream 还剩哪些差异；没有完成核验就不要命名为 pristine vLLM。

这是待测基线问题，不是认定 K2 导致全部排队。若关闭 gate 只是把排队变成 preemption/重复计算，结果也要保留。

### 3.2 客户端 TTFT 中存在尚未解释的明显非引擎区间

代表性 fork run 中，按同机 monotonic 时间戳复算：

| 分量的逐请求中位数（ms） | 第一轮 24 calls | 第二轮 24 calls |
|---|---:|---:|
| client issue → engine RESUME_ARRIVED | 90.79 | 45.33 |
| engine arrival → admission | 68.00 | 7.74 |
| admission → engine FIRST_TOKEN | 183.13 | 126.85 |
| engine FIRST_TOKEN → client first token | 115.24 | 71.92 |

这些中位数不能相加当某个请求的 TTFT。issue 时间在客户端 JSON 序列化前，FIRST_TOKEN 是 scheduler 处理首输出的位置；区间混合序列化、API/IPC、CPU 竞争、输出处理和客户端观察，不等于纯网络开销。还没定位到具体根因，也没有可直接宣称的加速比例。

记录 API receive/engine enqueue/engine output/API flush/client receive 边界；先做 instrumentation on/off 的短匹配检查，避免完整 block-list 日志本身扰动。性能结果保留用户端 E2E，同时单列 engine service。若修复公共 runner/API 开销，把修复应用于所有方法并重建基线；不能计为 SlackKV 算法贡献，也不能通过把计时起点后移获得收益。

### 3.3 当前 remote-wait 统计遗漏了 follower lookup deferral 的专门归因

保存的原生 `_lookup` 在命中 `_blocks_being_loaded` 时返回 None；主 scheduler 的 `ext_tokens is None` 分支将请求跳过，**没有进入该请求自己的 WAITING_FOR_REMOTE_KVS，也没有专用区间事件**。本次复核的 48 calls 只有 3 对 remote-wait 事件。因此 56.58-ms 指标描述三个实际加载者的观测等待，不能作为全部消费者恢复依赖等待的完整计量。

这不推翻去重结论，也不说明漏记了巨大收益。代表性日志中第一轮 followers 的 arrival→admission 约 64–74 ms，在 root ready 后约 1.8–11.6 ms 被准入；并未看到当前条件下秒级 follower HOL。没有 lookup 原因事件时，不把这整个区间都归于共享加载。

如继续研究 readiness，补记 `lookup_deferred` 的具体原因、关联 job/key/range、首次/末次观察及解除事件。把 CUDA transfer elapsed、CPU 观察完成、发布可复用状态、实际 admission 分开。CUDA event duration 不能直接与 CPU monotonic 绝对时间相减；轮询只能给出设备完成到主机观察的界限时，报告区间而非伪精确时间。并发 followers 的等待不能相加冒充 workflow 可节省时间，须在请求依赖 DAG 上重算最长路径。

### 3.4 一个没有物理保留的 epoch 失败，不等价于强保留基线已失败

`WorkingSetEpochs.allowed()` 使用逻辑 charge，idle entry 还会限制其他 program 准入，但它不保证相应 KV 在 GPU 中存活。它可能付出了准入等待，却没有买到物理保留的全部收益；这是一种机制解释，需通过 residency 证据确认。不要据这个失败否定 Continuum/其他保留策略，也不要简单把它改成永久 pin。

需要一个最接近目标机制的已有方法作为**正向对照**，验证本实验环境能否识别“驻留减少重排队”的效果。不要求一次安装所有 SOTA。选择一项可审计、可匹配实现的保留/准入策略；与共同 runtime、资源和保护条件对齐。简单移植只能标为 mechanism-matched control，不能声称完整复现论文。

特别注意：本次打开的 Continuum 作者仓库 README 明确说明公开版本不含论文中的 estimation。故不能把默认跑通该仓库等同完整 TTL 模型复现，也不能因代码不完整卡住整轮研究。其 README 同时提供固定并发和 Poisson program arrival 两种入口，可供负载协议参考。

## 4. 建议执行的下一轮：先完成一个可证伪的主问题

**主问题：在同 GPU/CPU 预算、相同完整工作量和后台约束下，旧负载中的容量排队究竟由自定义限制、必要 active working set，还是可改变的跨轮驱逐/准入顺序主导？**

### A. 小规模基线与观测闭环

1. 读取最新 main 并保留本地/服务器未提交工作。将本审查快照与新提交区分，继续冻结模型、实际 token 输入、输出工作量、编译模式和背景协议。
2. 在旧负载的 2/4 GiB 上比较 K2 与 gate-off Compiled OnDemand。这是有限机制对照，不是参数 sweep；已有 8 GiB 可作为容量充分的历史参考，若作当轮精确比较则重新匹配运行。
3. 对关键条件预先安排少量独立 run，交替/随机化方法顺序并保存 seed。建议三个配对作为方向筛查；不要把单 run 内三个 episode 当三个独立重复。若预算不支持，把结论写为不确定，不调样本直到显著。
4. 只在代表性短 run 加详细事件。记录 queue enter/exit、gate、token budget、lookup deferral、真实 allocation need/available、in-flight、preemption、完成释放与 active/idle/in-flight 的 block-ID union。避免给未检查的请求臆造 allocator 原因；允许多约束并存。按状态变化重建持续时间，不能把每 100 ms 或幂次采样点当连续因果时间。
5. 先解释已有 seconds-level wait。对实际释放事件建立“哪批 running requests 完成才能让队头装下”的依赖；区分不可调度和本轮没被检查。运行中真实拥有的、idle evictable 的、DMA 保持的 block 需要去重，不能把 pool reservation 当 owned memory-time。

这一阶段建议先以不超过约 1 GPU·小时安排，含启停/编译/失败；若新信息不足，可在原整轮 4 GPU·小时内自行调整。预算是上限，不是必须消耗量。没有 GPU 操作由本审查代替执行。

### B. 用已有强方法和有依据的工作负载定位机会

选一个最接近机制的已有 control，检查它相对正确基线是否改变 queue/residency/E2E。若原生合理配置已解决问题，保留工程结论，不能为制造新算法而恢复弱基线。

无需等待下载 7B 才做 A。已有 1.5B 足以诊断控制路径、归因错误和机会是否存在。更大模型是后续泛化检查，不能被当成“自动更容易获得加速”。新的压力点应来自可解释的 active/reuse working set 与 tier 容量比，报告 unique bytes、growth 和共享程度。

之后选择少量公开可重放 agent trajectories，保留真实上下文增长、共享关系和程序内依赖；新 program 到达可采用预先冻结的外生 arrival schedule，后续 turn 仍 finish-relative，以避免一个策略变慢就自然降低系统到达负载。固定并发闭环与外生到达的结论分别报告。缺工具时间戳就标注缺失或代理，不编造生产时间；结构回放也不等于语义任务质量。无需重新付费生成大量大模型 traces。

首轮评价以完整 program E2E 和背景服务为主，补报告完成数、队列是否持续增长、超时/未完成数及等待年龄。若改为 SLO goodput，必须在跑候选前冻结具体 SLO 和 offered load；不能运行后挑 SLO。正常压力点和容量充分点都保留，避免只选能让方法获胜的条件。

### C. 根据结果只实现一个最有依据的候选

以下是选择分支，不是三项必做任务，也不是已有创新主张：

| 若实测发现 | 可继续的具体设计问题 | 必须击败/区分的对象 |
|---|---|---|
| gate-off 解决多数等待 | 固化更合理执行基线，重新判断是否还有研究问题 | 工程基线自身；不包装为算法 |
| 跨轮被逐出导致反复重排队，且已有保留 control 有效 | 用真实共享 block 的边际占用、decode growth 和可撤销保留联动 demand admission；显式计入给其他程序增加的等待 | 普通 program priority、TTL/Continuum、MORI；仅加年龄/阈值不构成新意 |
| 完成的共享 range 受无关 private-tail job/发布周期拖累，并影响真正关键分支 | range readiness 与真实消费者依赖解耦，保持 singleflight、身份和引用保护 | 原生 whole-job deferral、KVFlow loading nodes、HiCache readiness；收益必须大于额外调度/搬运成本 |
| 真实 host miss/前缀洞导致大量重复计算 | 先定位 store admission、prefix completeness 和 eviction 的具体失败；只在现有方法无法覆盖时设计新机制 | EfficientAgent、现有 LRU/ARC、适用的 segment-cache 方法；普通 write filter 不新 |

对于保留/准入候选，可以先用一个可核验的设计账本，而不是直接拍 TTL：`保留收益 = 实测可避免的恢复/重算 + 可避免的重准入等待`，`成本 = 真实 marginal block-time + 其他 demand/BG 延迟 + 迁移干扰`。其中每项都须可观测或报告估计误差。共享块只按 union 的边际占用计费；已开始的 DMA 和 decode growth 要计入；不能把 UNKNOWN 计零。这个账本本身不是创新，只有实质不同的决策机制及优势才可能成立。

不要以“部分 range 能早发布”自动认定 fork 更快。同步 fork/join 的 E2E 由最晚分支决定，早完成一个本来非关键的分支可能没有收益；在 32K 上限/1.5B 下，几千 token 私有尾部的纯传输时间也可能仅数毫秒。必须测消费者需要何种 range、解除后能够执行什么、是否改变 join/后续链路。

## 5. 怎样避免再次得到一组难解释的正负数字

- **主要验收提前冻结。** 新一轮建议完整 program E2E 为主，辅以尾部、BG wall/TPOT、SLO goodput和最大等待年龄。初筛可沿用至少 10% 主指标改善、BG 不恶化超过 5%等量级，但不能运行后更改。旧轮验收结论不回改。
- **H2D/命中/失败次数是解释变量。** 未来一个明确以增加搬运换取降低重算的候选，可在执行前规定可接受资源代价；不应天然因为 H2D 增加就否决所有不同资源权衡。仍必须匹配总内存预算，报告成本与背景干扰，不事后豁免失败条件。
- **单次负性能不足以证明机制普遍无效。** 可以因低预期价值停止投入，但报告“该次筛查未通过”。Round2 不同 control 的 P95 变化还混合源码/到达差异，不能直接当作纯随机方差估计。新的微型配对应避免这种混淆，近门槛结果保留不确定性。
- **20% direct-path 份额不是普适进入门槛。** 可以作为某个 readiness 候选的预注册筛查，但排队放大/同 SLO 容量机会不一定等于观测直接等待份额。若主张间接收益，必须有可信受控干预或经 baseline 校准的模型，不凭相加等待宣称上界。
- **工程改善与研究贡献分别报告。** 公共编译、API/日志优化和 K 配置调整先归入强基线。与已有方案相似不代表不能做 control，也不代表已有实质创新；写清“现有方案做什么、缺口为何真实、改变哪条决策/依赖、消融如何证伪”。
- **安全验证和性能筛查分阶段完成。** 任何修改 allocator/readiness/保留生命周期的原型，上性能测试前先过最小安全 smoke；扩大实验前补 leader cancel、follower cancel、DMA pending、preemption、失败回退、长工具等待与后台推进。逐 token invariant oracle 单独计时；不能让安全测试和默认性能模式互相冒充。

## 6. 给执行 Codex 的交付要求与自主空间

你可以在上述主问题下自主选择更有信息价值的实验、修正 instrumentation、实现适用 control，并在证据足够后选择或否定候选。无需逐条等外部批准，不必凑齐三个候选；若发现本建议错误，用代码路径/原始数据反驳并记录调整理由。保持现有预算、历史数据、其他服务和最终 Agent Serving/KV 目标。

本轮至少交付：

1. `docs/BASELINE_AND_OPPORTUNITY_ROUND4.md`：冻结基线、原生/patch 差异、工作负载与主指标、机会与新颖性比较、停止条件。
2. `results/autonomous_round4/`：连续 GPU ledger、精确输入/配置/源码、失败记录、原始事件与紧凑复算数据。
3. `docs/AUTONOMOUS_RESEARCH_ROUND4.md`：主要结果和反例；到底改变了哪个资源关键路径；哪些结论仍 UNKNOWN；若停止，明确停止的是机制、条件还是整个研究问题。

出现以下情况可以结束本轮，不用耗尽预算：合理原生基线已消除机会；在匹配资源/工作量下候选不能改善主指标；缺少必要数据使关键结论不可识别且预算内无法补足；机制与已有方法等价且没有可测试差异。若只是某篇作者代码难安装，优先选择可审计的小对照，不让整轮变成环境迁移。

## 7. 本次核对的外部来源

以下只支持机制/基线选择，不将他人 speedup 移植成 SlackKV 预期，也未完成所有作者代码复现：

- [Continuum v7](https://arxiv.org/html/2511.02230v7)：TTL 与程序顺序共同考虑保留和重新排队成本；[作者仓库](https://github.com/Hanchenli/vllm-continuum) README 标明公开版本缺少论文 estimation。
- [MORI](https://arxiv.org/html/2606.00866v1)：相对 idleness 与分层 admission，排除“一般分层准入”本身的新颖性。
- [KVFlow](https://arxiv.org/html/2507.07400v1) §3.2：loading 状态依赖检查和 ready request 调度；range-ready 必须说明比已有 node readiness 多了什么。
- [EfficientAgent](https://arxiv.org/html/2609.33762v1)及[作者代码](https://github.com/KunmingSHAO/efficientagent_release)：host working set、写入准入与 dependency-preserving replay；公开代码存在不等于生产完整 trace 已可用。

本次只新增离线审查、复算脚本和结果，不改变 serving 行为。仓库文档是交接载体；它不会自动唤醒或向另一台机器的 Codex 会话注入消息。
