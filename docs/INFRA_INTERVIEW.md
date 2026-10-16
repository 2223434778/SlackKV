# 面试讲解：按实际理解和贡献使用

以下是讲解模板；先完成动手清单，再按自己亲手完成的部分调整。项目由用户主导、AI辅助实现与验证，不能把上游vLLM allocator、scheduler或CUDA kernels说成本人从零实现。

## 30秒版本

SlackKV是一个基于vLLM原生CPU offload的多会话KV管理原型。它用固定会话槽位限制可写前缀，提供真实多轮历史、本地工具、失败原子提交、owned进程管理和可复算报告。重点是前缀是否能活到下次复用，以及异步生命周期和指标是否可信；不是假设offload总会更快。单卡真实聊天和受压恢复分别验证，NoHost/Full/StaticFair分别对照。

## 3分钟展开

1. **问题。** GPU存不下所有历史时可offload，但host也有限；全存可能反复写入随后被逐出的KV。需要区分源数据可用、实际分配、在途引用和未来复用，不能只看cache hit比例。
2. **最小设计。** 应用层固定槽位，每槽按16token块配置连续前缀上限；不暗中借空槽。vLLM原生接口执行lookup/store/load；薄terminal gate只抑制已知最后请求的新写入。session/generation salt隔离身份，失败不提交半轮。
3. **工程闭环。** 固定upstream与库hash，doctor先核资源；受管服务记录owner出生时间、ready超时和连续预算，清理只触及自有进程组。真实聊天保留模型输出，benchmark另用冻结输入与等输出工作量。
4. **验证。** 两会话三轮有真实请求重叠。E1受压marker位于实际host恢复区间，完整输出IDs与独立cold一致，并核对source和49MiB H2D，不把GPU热命中当host收益。数据见[工程E1](INFRA_E1_STATUS_20261016.md)。
5. **取舍。** 固定额度保守、可能留空；不是硬物理分区，多Pool不受全局准入约束。独立close是逻辑确认，native物理块仍待引用结算/LRU；当前是原型，不承诺生产隔离或任意模型。

## 常见追问

**为什么不把每个会话全部offload？** 先看预算和复用工作集。一次32B合成对照中，Full写448.875GiB却没有host恢复；StaticFair写64GiB，随后恢复128GiB，实际少算524288输入tokens，E2E median相对NoHost/Full改善16.44%/18.40%。单次合成筛选不等于稳定生产加速，后台输出间隔代理还有代价。[原报告](AUTONOMOUS_RESEARCH_LARGE_MODEL_20261012.md)

**为什么停止动态M1？** 它功能上支持借用、共享和安全回收，但新增写入不一定在结束/缩额前兑现。修复CPU保护祖先重复遍历后，局部调用约9→0.375ms，实际端到端却未确认改善；修正版相对StaticFair仅约1.84% median差异，DMA约多93.52%。这是局部热点和整体瓶颈不同的证据。[成本定位](SLACKKV_SYSTEM_M1_COST_20261013.md)

**怎样测异步复制？** 请求端只给端到端时间，不能把发起copy前后CPU时间直接叫设备复制耗时。原生worker用CUDA start/end events与完成关系计设备时间，跨worker聚合也不等于请求阻塞时间；新项目复用其字节/source计数，缺少关键路径分解就记UNKNOWN。

**取消后能不能立刻回收？** 客户端断开不等于native load/store完成。保护引用必须由原生完成/失败路径结算，逻辑会话不能直接free物理块。失败历史不提交，换generation/salt后正常Prefill；不通过假推理请求传close，不声称即时abort已确认。

**为什么自由生成demo不能公平排名？** 模型输出影响下一轮真实输入，不同策略/数值batch路径可能改变后续工作。demo用于端到端使用验证；benchmark冻结输入tokens和输出数量，保证模型工作量一致；二者的结论分开。

**有什么实际踩坑？** transformers5的BatchEncoding长度是字段数，不是token数；这会破坏上下文保护。显式return_dict=false加类型检查后修复。另有Windows默认GBK、SSE没有DONE、超时后线程池等待reader、PID重用、缺失metric补零等边界；每个都应从实际测试或失败记录讲起。

**项目新增与复用边界？** 新增的是会话/准入协议、固定配额计算、terminal适配、可重复运行与计费/进程管理、可信报告与验证；vLLM负责执行、allocator、cache索引、异步复制和CUDA优化。成熟方法可以复用，关键是能运行、能解释且不夸大贡献。

没有足够证据时可以直接说：尚未验证多机、多模型、全局多租户隔离、任意取消时序、生产SLO或完整memory-time；下一步要补哪项明确测试，而不是补一个未经验证的架构名词。
