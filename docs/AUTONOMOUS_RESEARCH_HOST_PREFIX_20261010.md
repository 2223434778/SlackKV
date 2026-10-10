# 从 JIT 时机转向 host 写入放大与部分前缀保留

后续更新：NoHost、静态均分与无reset三问题已完成16个GPU run，见[三问题验证报告](AUTONOMOUS_RESEARCH_HOST_PREFIX_FOLLOWUP_20261010.md)。本文保留原轮次证据；后续自然驱逐压力下约8.9%会话收益未达到10%主门槛。

2026-10-10；本机起点 `685ac5ca728d063c6b99ad406862a65ec613130f`。用户放宽创新门槛：允许复用论文方法，要求整体有区别和实测价值。本轮完成 CPU 候选淘汰、真实 GPU 成本测量及一个最小机制 PoC，不改变历史负结果或最终 Agent Serving 目标。

**决定：保留“限制不可持续的尾部写入、保存可恢复前缀检查点”作为下一主候选。它在受控 host-overflow 条件下取得两对反序匹配的真实收益。当前固定单路径上限是已有接口的最小验证，不是已完成的新论文算法；不宣称生产负载或多会话公平性成立。停止按长度平方直接平衡缓存损失的候选。**

## 依据与本轮实际工作

旧 hybrid 的主要瓶颈是 active KV/执行进度；JIT 直接恢复领先量小，idle pin 又可能付容量成本。这些结论保留。新增公开轨迹有高历史前缀复用，但真实 host misses 未知，不能直接把97.83%结构覆盖解释为系统收益。本轮先研究写入/驱逐与可恢复前缀之间的关系，允许借用已有方法再形成组合。

1. 新增 host-only CPU 回放：同一15条轨迹、2661请求、27个trace/model namespace，7容量×4策略共74,508条逐请求结果。只在记录完成时写入、固定原到达/耗时，无GPU状态/实际DMA，不是 serving simulator。
2. 原生式 block LRU、前缀连续 leaf LRU、RR式尾部轮转及 cost-tail。原生式模型以32个64-token块分批写入，反向touch/原子拒绝；真实native manager320请求差分两次均通过。RR参考不是完整SGLang RR-Evict。
3. Qwen1.5B真实GPU前缀成本：3长度×4缓存状态×3输入，共36个测量调用，真实累计prefix hit379248与预期一致。
4. 单路径 host-checkpoint PoC：full1/frontier1、frontier2/full2两对；另做batch-invariant独立冷Prefill oracle。全部GPU成本含失败与清理0.1388070682GPU·小时（约499.71秒），GPU1未参与。

冻结协议：[CPU筛选](HOST_PREFIX_SCREEN_PROTOCOL.md)、[成本测量](HOST_PREFIX_COST_PROTOCOL.md)、[机制PoC](HOST_PREFIX_CHECKPOINT_PROTOCOL.md)。源码在`experiments/host_prefix/`，没有修改vLLM runtime或全局服务器环境。

## CPU筛选：否定成本公平，发现写入线索

cost-tail选最小`已观察长度²−驱逐后深度²`损失的叶子。这是累计重算损失代理，不是当前缓存状态再驱逐一个块的边际损失：此前已经丢失的尾部可能被重复计入，过度保护已截断的长路径。它也没有真实复用概率；全部7个容量点未通过预先门槛。524288代理tokens时未缓存输入比block LRU增加96.46%、P95增加33.67%；1048576时分别增加114.84%、208.33%。长历史可能过时，长度驱动保护会挤走更有用的分支。没有继续调参数或声称二次代理量是GPU时间；这个负结果不否定其他成本敏感策略。

另一条线索来自前缀连续LRU：在65536/131072/262144容量点，未缓存输入相对block LRU分别减少0.99/1.44/1.12%，写入代理量却减少95.80/94.56/93.13%。这不足以证明实际速度，但支持验证“拒绝存不下的后缀，避免边写边丢”的机制。全部容量与namespace保留，不选择单个最佳点。

数据：[CPU assessment](../results/host_prefix_screen_20261010/retry/assessment.json)、[逐请求CSV.gz](../results/host_prefix_screen_20261010/retry/requests.csv.gz)、[namespace metrics](../results/host_prefix_screen_20261010/retry/namespace_metrics.csv)。公开输入仍是固定Weka前缀，有作者筛选、代理长度过计数和非随机样本限制；各trace没有人为同时启动。

## 真实成本：不要把命中tokens直接换成线性加速

Compiled、串行、无后台、GPU自然85%分配；max_offload_tokens=0隔离GPU前缀复用。表为3个不同输入的TTFT中位ms，包含客户端/执行栈开销，不是纯GPU事件计时。

| 输入长度 | 冷 | 50%前缀 | 87.5%前缀 | 完整前缀 |
|---|---:|---:|---:|---:|
| 8192 |317.10|184.57|70.91|34.52|
| 16384 |757.02|461.36|154.51|53.35|
| 28672 |1696.23|1083.36|357.90|86.37|

完整前缀仍需重算最后16-token块；留前半不会严格把TTFT减半。支持后续使用实测`C(L,k)`，不能单凭tokens或`L²−k²`排序，也不外推到公开样本624K长度。证据：[calibration2](../results/host_prefix_cost_20261010/calibration2/requests.json.gz)。

## PoC设置与真实结果

Ubuntu GPU0 A6000、缓存Qwen2.5-1.5B-Instruct FP16/TP1、compiled默认/APC/native CPU connector；四个性能run的真实GPU池都为1,355,712tokens（日志36.2GiB），相同模型、执行栈、GPU资源、host256MiB及12个实际输入。关闭旧gate/pin/credit/epochs/backfill。**256MiB host是人为受控压力预算，GPU冷状态是明确干预，均不冒充自然生产条件。**

一条28672-token输入路径、6次调用、每轮增长64、每次16输出；每轮清空空闲GPU cache，启动8192输入/256输出后台，后台进入decode后发Agent调用，后台结束才进入下一轮。全部6个后台请求完成，共1536输出；Agent96输出。后续输入固定canonical tokens而非模型输出反馈，所以不评价任务成功率。这个显式join不是自然工具轨迹，不把其总时长命名为普通Session E2E。

Full OnDemand允许完整输入offload。Frontier通过原生`max_offload_tokens`把写入限在`floor(host_bytes/(16×28672))×16=9360`tokens；限制由物理CPU块容量给定，没有性能调参。没有增加KV身份转换、物理pin或reservation。它保留精确前缀，其余输入照常prefill。

| Run | Resume P95 ms | Resume median ms | BG median ms | D2H bytes | H2D bytes | external tokens |
|---|---:|---:|---:|---:|---:|---:|
| full1 |1727.73|1714.37|3488.64|4792582144|167444480|5840|
| frontier1 |1419.12|1400.79|3206.75|268369920|1341849600|46800|
| frontier2 |1414.79|1402.76|3199.39|268369920|1341849600|46800|
| full2 |1765.78|1739.82|3521.80|4792582144|167444480|5840|

Resume P95为每run5个续接的nearest-rank，实际上取最大值；冷第一轮保留在传输与后台核算中。第一/反序第二对P95降低17.86/19.88%，BG median降低8.08/9.15%；D2H均降低94.40%、总H2D+D2H降低67.54%。H2D为原来的8.014倍（增加约701.4%）：更多有效恢复换取H2D，避免大量重新计算与D2H。不是所有迁移方向都减少。

独立核算：全输入和输出工作量严格匹配，真实local-hit=0；H2D=external tokens×28672、全部token source守恒。Frontier的46800=5×9360，说明五次续接都恢复最大保留前缀。Full把167152个Agent计算输入tokens再次写入CPU；frontier只写9360一次，Agent计算输入降至126192。真实重复历史prefill分别138160/97200tokens；必需初始输入与增长均为28992，没有裁输入或输出。恢复与少写发生在同一PoC，尚未做能区分两者时间贡献的消融。

两个配对均通过预注册资源/退化门槛。仅两对同一输入序列，不宣称统计显著或稳定泛化；没有与完整EfficientAgent/LMCache或RR-Evict运行栈比较。原始输入、完整日志、指标与独立汇总：[GPU结果目录](../results/host_prefix_cost_20261010/)、[analysis.json](../results/host_prefix_cost_20261010/analysis.json)。

## 正确性、活性、失败与限制

- 19项相关测试通过；native CPU manager320请求差分、在途load引用保护、取消store回收通过。性能四run最终running/waiting均0、正常退出，后台没有被裁工作；单次BG吞吐代价未见恶化。
- 有效oracle2采用batch-invariant1：9360 external/0 local的部分恢复，与独立盐化0 external/0 local冷Prefill的16输出tokens逐一匹配。时延不混入性能。仅一个输入，未穷举取消/抢占、模型质量或所有KV身份。
- 固定单路径cap不是多会话公平算法。CPU反例测试：A/B都取满整个tier会交替清空对方。因此不能部署本cap；必须研究共享前缀计费、动态额度/准入与退化公平性。没有GPU owned-memory-time或总体SLO并发收益证明。
- CPU第一次实现因冗余全缓存扫描请求中断，退出机制无完整traceback，标UNKNOWN；原源码/失败记录保留，等价优化后native差分再通过。
- calibration1错误要求reset返回true，而实际API200空体，正式测量前失败；66.79秒计预算。SSH启动调用也超时，确认后台仍运行后没有重启第二份服务。
- 初次分析把新空tier尚未生成的transfer标签当UNKNOWN；保存分析失败，改用明示零迁移预热的已知起点，结束counter仍必须存在，source/bytes严格复算。
- oracle1把cache_salt错放kv_transfer_params，所谓冷参考实际有28720local hits；结果标INVALID，54.18秒计预算，不能用token相同作证明。修为顶层payload后oracle2通过；新增字段位置回归测试。
- 环境既有nvidia_cutlass_dsl `.pth`警告原样保留，未安装/修改环境；所有有效运行正常退出。最终两卡均19MiB/0%。失败、大原始文件服务器保留，提交紧凑gzip副本。

## 复用已有工作后，真正待验证的区别

[EfficientAgent论文](https://arxiv.org/html/2609.33762v1)与[作者实现](https://github.com/KunmingSHAO/efficientagent_release)已研究host写入压力，压力下保留小延伸、拒绝大refill；[RR-Evict](https://arxiv.org/html/2609.32278v1)已有连续前缀尾部回收与公平退化；[MORI](https://arxiv.org/html/2606.00866v1)已有分层准入/idleness。可以直接借用这些组件，不因重叠淘汰组合。

当前PoC区别只是把一次大写入变为容量可持续的部分前缀检查点，而非只全存/全拒；尚未证明这个区别未被已有完整系统覆盖，不能宣称新颖性。真正值得推进的组合问题是：**host不足时，联合决定每会话保留深度、允许写入量与公平尾部回收，使写入能够在下次真实续接兑现，而非用host流量制造几乎不命中的碎片。** 不需要每个组件原创，但要让新增深度决策在消融中带来实测价值。

2026-10-10审查后调整推进顺序，依据[独立审查](RESEARCH_REVIEW_HOST_PREFIX_20261010.md)：先补简单强基线，不优先开发adaptive frontier。第一步在当前诊断条件加入NoHost（仅max_offload_tokens=0，GPU APC仍开启），预先固定少量独立种子与交错顺序；若部分前缀仅胜full而不胜NoHost，收窄为避免无效offload的工程修正。第二步比较多会话full、NoHost、StaticFairPrefix，按实际host物理块均分额度，检查交替续接、退出/取消及在途引用保护；写入上限本身不等于物理配额或回收。通过后才冻结自然回收负载，移除每轮reset，包含GPU-hot/host充足负对照，以完整会话完成时间、每会话退化和固定后台工作量为主要判断。仅在简单方法存在可证实的不足时增加动态额度、准入或公平尾部回收。若收益仅存在于人为256MiB/强制冷状态，停止推广。更大模型/长上下文须先做硬件可行性估计，不直接扩大矩阵。

本轮满足“有初步真实机制证据，值得聚焦验证”的停止条件；资源未耗尽，不继续随机调参。最重要UNKNOWN仍是自然聚合压力下，相对强已有方法还剩多少可实现收益。

预算连续性：本次新增0.1388070682GPUh；保守合并此前Round3的0.2004867526、Round4的0.5462201082，共0.8855139290GPUh，仍低于4h。原execution字段保留本次8run小计，完整授权范围另存[budget.json](../results/host_prefix_cost_20261010/budget.json)，当前runner检查合并历史范围，不能通过新目录重置额度。

## 复现与版本

CPU：`python -m experiments.host_prefix.run --input .tmp/opportunity_trace_prefix.jsonl --output <new-directory>`；原输入获取与SHA见旧[机会数据manifest](../results/opportunity_decision_20261010/source_manifest.json)。`python -m experiments.host_prefix.assess --directory results/host_prefix_screen_20261010/retry`。

GPU命令见冻结协议；全部run的`execution.json`保存server argv、源码指纹、预算与退出码；`environment.json`保存模型snapshot、软件版本和真实资源。服务器checkout仍d403714且保留历史改动，实际runtime为隔离round4-source，不能把本机审查起点当服务器干净HEAD。runner旧版本按记录SHA256校验后归档，native manager/LRU原源码gzip保留。

本机独立重算：`python -m experiments.host_prefix.analyze_gpu --directory results/host_prefix_cost_20261010`；测试：`python -m pytest -q tests/test_host_prefix.py tests/test_opportunity_decision.py`。所有重大研究成果与失败随本次Git提交，原历史矩阵未修改。
