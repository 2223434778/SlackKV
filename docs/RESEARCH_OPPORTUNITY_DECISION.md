# 研究机会判断：暂不进入新算法实现

> 后续更新（2026-10-10）：用户允许复用已有组件，创新门槛已放宽。本页保留此前CPU机会判断的历史证据。新增host写入放大/部分前缀保留的真实PoC与失败见[后续研究报告](AUTONOMOUS_RESEARCH_HOST_PREFIX_20261010.md)；当前决定是保留该机制继续验证，原JIT负结论不变。

2026-10-10；审查起点 main `1c9c7da`，依据 [最新交接](RESEARCH_REVIEW_ROUND4_20261010.md)。本次只有CPU结构分析和文献核对，GPU消耗0，未改变运行时、历史数据或项目最终目标。

**决定：暂停并建议归档原 JIT 论文主张；保留工程基础和证据。新的长期 Agent KV 问题确实值得关注，但目前没有一个同时通过“实际可避免成本”和“相对已有方法实质区别”门槛的候选。不要现在启动第五套调度 PoC。** 归档建议针对当前论文路线，不是删除代码或认定该领域没有价值。

## 已停止什么

| 机制 | 已有证据与范围 | 判断 |
|---|---|---|
| 原JIT/PSA/Frontier整体路线 | 旧1.5B hybrid中远端恢复通常几十ms；固定路径删全部等待P95约1%；compiled Full无稳定整体优势 | 基础恢复/安全接口有效，算法优势未成立；不是重调度的全局上界 |
| 有限回填、credit、soft epochs | 各自匹配筛选未通过整体门槛；credit改善P95却拖慢session，epochs减迁移却显著恶化延迟 | 停止具体实现，不推广为全部调度/保留无效 |
| K2主导排队 | Round4两种容量各三对，去gate的session中位配对改善仅0.37/0.63% | 不继续调gate来追求大收益 |
| fixed-TTL physical pin/program-order | Round4 session+6.87/+20.09%；32pin均释放，零成功交接事件 | 停止具体控制；不能去掉HOL保护或冒充完整Continuum负结果 |
| 重复ancestor物化/完整root逐步streaming | Round3 native singleflight已恢复一次；受测条件的streaming成本不利 | 普通singleflight不新，负结论限已测host-hot小工作集 |

这些不是新实验。本轮没有重复旧矩阵。active ownership与执行进度确有相关证据，但长队列不等于大可优化空间；173/175单完成witness不是最快释放策略。完整因果分解、owned memory-time与生产泛化仍UNKNOWN。

## 先看新数据，再判断问题

选择一个公开负载族：[SemiAnalysis Claude Code Weka v7轨迹](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/blob/23f152f6f0f9399a85901b89a6458def0ef16729/README.md)，固定revision `23f152f6f0f9399a85901b89a6458def0ef16729`。事先限定文件前16条完整行、64MiB上限，取得15条/46,447,908 bytes；下一完整行未能在上限内取得，不按分析结果筛选。原数据生产者已有min请求数、版本、非图像及上限等筛选，本次文件前缀也非随机样本，不能估计全部生产分布。

只提交2661请求的紧凑结构CSV、汇总、获取manifest与脚本，不提交大原始文件/原session标识。结果入口：[summary.md](../results/opportunity_decision_20261010/summary.md)。原前缀SHA256为 `d078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994`，可用固定版本与边界重新取得。

| 观察 | 本次样本结果 | 可支持的解释 |
|---|---:|---|
| 请求/显式子Agent分组/子请求 | 2661 /47 /892 | 有长期与嵌套结构，不代表真实完整DAG |
| 代理输入长度median/P95 | 118208 /624320 | 与旧20K上下文族不同；非目标模型真实tokens |
| 输出长度median/P95/max | 460 /5053 /26571 | 不能继续沿用旧≤512输出假设 |
| 同模型已完成历史输入LCP覆盖 | 输入量加权97.83% | 历史输入具有高结构复用；不是实际cache hit |
| 包括已发起未完成的乐观LCP覆盖 | 97.84% | 两者很接近；不是对所有readiness开销的上界 |
| 来自其他显式容器的已完成前缀覆盖 | 3.46% | 该样本不支持把大规模共享分支作为默认主问题 |
| 同容器、同模型相邻输入增量median | 1664 | 只是容器邻接诊断，不能当真实单Agent增长率 |
| 相邻容器end gap负值 | 53次 | 存在交叠/扁平交错，不能强行线性化 |
| 非相邻历史前缀优于容器前驱 | 53次 | APC可搜历史，不能把这一点重新命名为创新 |
| 超旧32K /超131072代理tokens | 2476(93.05%) /1232(46.30%) | 旧配置无法覆盖此结构；不能筛短后声称完整复现 |

按`trace+model`隔离，使用有序prefix trie，禁止跨session或跨模型KV合并，也不以无序hash集合冒充连续可复用前缀。已完成历史的插入时间为记录的`t+api_time`，只保存输入prefix，假设不驱逐；不含尚不知hash的生成KV，不能当真实GPU可用状态。

字段语义尤其重要：最新作者卡说明`in`是64-token hash块计数，并记录长尾过计数；本次不是精确Anthropic或Qwen tokenization。旧同族[数据schema](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-051826/blob/main/README.md)说明内部`t`是root-absolute，本次全部验证不早于spawn；但准确父子依赖和扁平链条仍需重建。`api_time`包含原端点行为，不是A6000纯计算时间；原`think_time`仅保留为reported gap，不当纯工具延迟。prompt文本、真实token ID、工具分类/工具单独时间、实际GPU/CPU命中、DMA和所有权均缺失。

[AIPerf现有Weka loader](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/weka-trace.md)已经提供hash前缀重放、时间规范化和链条/分支重建；其依赖有推断和异常边界。重写一个同类loader不是创新，且不能把推断DAG当原始真值。真实对照若将不同Claude模型映射为一个开源模型，必须重新定义身份隔离、分词和执行条件，不能沿用原API耗时当新栈成本。

## 部署可行性：不靠2GiB制造问题

双A6000各48GB是物理资源，旧2GiB池只是压力配置。合理单卡预算应为`48GiB−权重−编译/activation/workspace−安全余量`，先按目标模型实测，而非先选有利2GiB。并发应取真实到达/闭环强度；本数据各trace相对时间都从0开始，不能自动同时启动15条来制造生产并发。跨租户共享仅在身份/权限允许时存在，CPU tier按实际可用RAM确定，8GiB同样不是硬件事实。

仅作单位敏感性：假定每代理token28672bytes，则118208/624320分别约3.16/16.67GiB；27个trace-model namespace的全历史唯一输入prefix tokens中位222720、P95 1454080，对应约5.95/38.83GiB。**这是模型投影，不是真实同时工作集、reuse工作集或可部署Qwen驻留量**，更不能把全部历史保留量等同所需CPU容量。它说明需要自然长上下文的可行性估计，而非继续2GiB短窗口。

已缓存1.5B/32K配置不能忠实覆盖。没有选择或下载新模型，也未确认完整族在单卡A6000上可行；模型大小、长上下文正确性/成本与实际RAM预算尚未确定。故本轮不以“换大模型就会好”推进GPU。

## 唯一保留的验证问题，而非已接受的新算法

**问题：长期会话中原本可复用的历史prefix，在自然GPU/CPU预算下是否因host存储/驱逐决策丢失，造成值得优化的重复prefill关键路径成本？** 新样本支持“历史可复用”X；尚不支持“现有方法因Y丢失并付出Z”。Z现在UNKNOWN。

成本模型只定义待测量：`Z = Σ实际丢失且可复用prefix的重算GPU时间 + 非重叠恢复等待`，须按程序关键路径计入，排除必需新token prefill/decode，不重复相加并行分支。结构97.83%不得替代Z。CPU LRU/无限缓存oracle也不能证明真实加速。

可证伪陈述：在保留代表性结构、合理单卡预算及相同模型/执行栈的匹配对照中，容量条件化host写入/驱逐仍使一个具体已观测依赖上的可复用prefix丢失，至少20%的程序关键路径来自该可避免成本；使用到达时已有信息改变该决策，存在减少损失且不把CPU写压力、迁移或等待转嫁给其他Agent/后台的机制。20%是将来机会门槛，不是本次测得值或回改旧验收。

最小反例：若host足以覆盖实际重用距离，或强已有方法已保留这些prefix，修改该决策无收益；若强压力下所有未来复用不可因果区分，优先保存某会话只会转嫁损失。当前数据不能排除任一反例。没有提出已具有新颖性的具体策略，因此不写runtime。

## 最近已有方法及停止门槛

| 已有方法 | 已覆盖 | 对本项目的后果 |
|---|---|---|
| [MORI §4.3](https://arxiv.org/html/2606.00866v1) | 相对idleness、sticky GPU/CPU placement、两tier准入 | busy/idle排序和分层保留不是缺口 |
| [EfficientAgent §3–4、Appendix D](https://arxiv.org/html/2609.33762v1) | reuse工作集/容量转折、首次lookup做容量条件化host write admission，preemption保持决策 | 简单host write filter、复用距离和反馈准入已覆盖，必须纳入强对照 |
| [RR-Evict §3](https://arxiv.org/html/2609.32278v1) | 保护active引用，idle轨迹轮转tail chunk驱逐、完成/活动检查 | 通用tail-prefix公平保留不是创新 |
| [Continuum §4–5](https://arxiv.org/html/2511.02230v7) | TTL收益/排队模型及program顺序 | 不以本项目简化控制失败否定完整算法 |

本次读取重点算法与条件，未安装/运行这些系统；不同栈不能直接用论文数字对照本项目。创新门槛没有通过，不能靠命名“生命周期协同”填补。

下一项若继续，应是**一个最小成本识别测量**，不是新策略矩阵：固定可行、可忠实表示的负载族和模型，观测实际host misses/重算/ready依赖，并与适用已有方法对照。只有Z达到门槛且定位到该方法尚未覆盖的一条具体决策，才冻结新PoC；否则停止。不得为此特意构造大共享分支、缩小显存、裁输出、重定义SLO或使用未来信息。当前没有足够依据冻结GPU性能协议，所以本任务结束于CPU机会判断。

**最重要不确定性：** 数据映射到可部署栈后，已有强方法还剩多少实际可避免成本，而非是否能做出更复杂的KV机制。若用户选择不再投入这项成本识别，建议将原论文路线停留为有界负结果/工程研究记录；未自动改变最终项目目标。
