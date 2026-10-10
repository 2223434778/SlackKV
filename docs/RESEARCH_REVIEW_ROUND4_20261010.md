# Round4 外部复核：结束当前局部优化循环，先重新确认研究机会

日期：2026-10-10。审查提交：`465c4d2240b34210e93d6d861e3ec0f9fc830774`。这是外部研究建议，不是新实验结果或已实现算法。

## 1. 决定

**接受 Round4 的限定负结论。暂停在旧 hybrid workload 上继续开发 JIT、gate、fixed-TTL、program-order 和普通回填变体。下一交付应是研究机会筛选，而不是预先指定的第五轮调度算法。**

上一轮外部审查提出的 K2 主导排队假设，没有获得支持；不应继续以“也许还需调参”保留它。此前 fork 中较大的客户端边界时间也不能外推为旧独立大前缀负载的主因。需要根据本轮证据更新判断。

项目保存的 CPU offload、原生恢复接口、ownership/lease 安全经验、工作负载回放和原始数据仍有工程价值。但截至本提交，**没有支持将原 JIT 路线作为已成立的新算法贡献的证据**。这是研究决策，不等于证明所有 Agent KV 管理不存在机会。

## 2. 核对结果

完整阅读 Round4 报告、冻结协议、retention 实现及两个 patch、ownership witness 实现、控制与诊断汇总。独立读取 OFF/order/pin 三个 run 的原始 `run.json.gz`，从40条 turn记录重算每个 session E2E；读取 pin 完整原始服务日志，配对所有 ACQUIRE/RELEASE；从 metrics.csv 重新计算六对 gate 对照。没有执行 GPU，也没有声称重验全部271个归档文件或全部内容正确性。

| 结果 | 复核值 | 判断 |
|---|---:|---|
| 2 GiB gate-off 会话耗时中位配对变化 | −0.3678% | 没有接近预注册10%目标 |
| 4 GiB gate-off 会话耗时中位配对变化 | −0.6295% | 没有接近目标；P95变化不稳定 |
| OFF / order / pin 会话耗时中位 | 23746.27 / 25377.64 / 28517.68 ms | 与报告一致 |
| pin 相对 OFF | +20.0933% | 单次三元组，全部8个session变慢 |
| pin 的额外引用生命周期 | 32 acquire / 32 release / 最终0 | 记录内守恒 |
| pin 释放原因 | 28 HOL fallback / 4 TTL expired | 无 REAL_REQUEST_ALLOCATED 交接事件 |
| pin 持有时间中位数 | 274.26 ms | 付出了保留占用，未形成预期交接 |

pin 中持有时间范围约1.63–1006.61 ms。这里“零成功交接事件”并不表示全程零 GPU cache hit：释放额外 pin 引用后，块仍可能作为普通缓存命中。原始 source 事件中确有一个包含 local hit，不能将二者混同。

单次三元组足以说明没有继续扩展的正证据，但不能把+20.09%当成稳定因果效应。运行时序、新增日志和策略本身的贡献没有完全隔离。报告保留这个边界是正确的。

## 3. 现在明确了什么

### 3.1 主问题是 active KV 与执行进度，不是未驱逐的 idle cache

Round4诊断报告给出的续接阶段中位数：进入引擎前24.13 ms，引擎到达→准入2081.63 ms，准入→引擎首token34.20 ms，引擎→客户端26.78 ms。它们不是可直接相加的同一请求路径，但支持优先分析准入前等待。

175个失败快照中，可驱逐 idle cache 已包含在 allocator available 中；固定当时 running 状态再清理 idle cache不能增加这部分可用容量。完成请求后释放真实引用才可能进一步填补缺口。这显著削弱了继续只改 idle eviction/JIT 的理由。

这仍不证明所有排队均不可优化。准入、批处理、执行进度和跨轮驻留会共同改变时间轨迹；但必须提出具体可避免工作或更好的资源使用证据，不能把“等待很长”直接当成“有很大可优化空间”。

### 3.2 当前 pin 控制付了成本，却没有实现目标中的续接收益

源码中，等待超过1秒的老请求优先于 retained consumer。实际 allocation 失败且队列中存在 retained consumer 时，`resolve_blocked` 会回收 idle pins；`reclaim` 回收的是全部 entries，而非按缺口选择最小集合。

这个保守策略有安全理由。实际28次 HOL fallback 和零成功交接表明：本控制在压力下没有持续实现“保留后由下一轮直接接管”的机制。因而本轮结果同时包含一个政策设计上的相互抵消：保留带来容量机会成本，老请求/进度保护又可能在消费者受益前撤销保留。

**不要据此删除 HOL 保护、无限提高 pin 优先级或调长 TTL。** 那会重开历史循环等待与公平性问题。也不要把这个机制对照当成完整 Continuum，或推论所有物理保留均失败。它给出的行动结论是停止本具体实现，不是再堆一组修补参数。

### 3.3 “173/175只需一个请求完成”不是下一算法已经成立

该 witness 最小化的是请求集合的数量，并冻结其他引用和增长。它不是最早释放时刻、最小剩余服务时间或最优调度集合。单个请求可以释放足够容量，不意味着该请求值得优先执行，也不意味着给它加速不会拖慢后台与其他会话。

当前证据还不能支持“共享释放超图调度”成为主线。不要为让这个想法成立，专门构造大量共享分支后再把它叫代表性工作负载。

## 4. 给执行 Codex 的下一任务

**先交付一份短的 `docs/RESEARCH_OPPORTUNITY_DECISION.md`，回答“是否还有值得投入的新问题”，暂不以实现新调度器为任务完成标准。**

具体要求：

1. 汇总已经停止的机制和适用范围，沿用原证据，不重复运行旧矩阵。区分工程能力、未成立的算法优势、仍缺数据的判断。
2. 从项目已有数据或少量可获得的公开 agent trajectories 中选择一个有出处的工作负载族，优先做CPU侧结构分析。先看数据，再选候选；不能先指定必须发现共享、多分支或host miss。记录上下文增长、实际共享前缀、依赖关系、输出长度、工具时间是否可用及缺失项。无需付费生成大量模型轨迹。
3. 明确部署约束为何合理：单卡模型与KV预算、并发、CPU tier容量、是否多租户；2 GiB只是压力实验条件。需要归一化压力时如实报告，不能把任意缩小显存解释为真实部署需求。更大模型只用于有理由的后续验证，不是保证收益的办法。
4. 最多选择一个具体缺口，并给出一条可证伪陈述：真实负载中发生X，已有方法因Y付出Z成本，候选改变某个具体决策/依赖，预计主指标如何变化及哪些请求承担代价。Z须有测量或清楚标注假设的成本模型；没有可靠数据就结论UNKNOWN。
5. 对照最接近的已有方法，写清其已解决什么、尚未解决什么。作者代码不完整时如实标注，不能把简化失败版本当作SOTA。不要把一般priority、TTL、singleflight、共享block计费或SRPT重新命名成创新。
6. 在写新运行时之前，给出最小反例/可行性依据和明确停止条件；若确需GPU确认机会，先冻结小规模协议并沿用AGENTS预算，仍不得把oracle未来信息放入可部署策略。没有可信机会就停止，并明确建议归档当前论文路线或由用户选择新的研究问题。

执行者可以反驳本审查、选择不同的有证据缺口，不需要迎合预设候选。最终目标仍是代表性 Agent Serving 的安全、可重复端到端改善。新问题筛选失败也是有效结论；不能以“还有四小时预算”作为继续随机试策略的理由。

本建议不要求重新完成所有仪表化、完整SOTA安装或大型trace采集后才允许思考。先使用已经足够的证据；只有一个具体决策确实依赖缺失数据时，才补最小必要测量。

## 5. 原始证据入口及快速复核

- [Round4报告](AUTONOMOUS_RESEARCH_ROUND4.md)与[协议](BASELINE_AND_OPPORTUNITY_ROUND4.md)。
- [metrics.csv](../results/autonomous_round4/metrics.csv)、[control_summary.json](../results/autonomous_round4/control_summary.json)。
- [diagnosis.json](../results/autonomous_round4/resource-on-repaired/diagnosis.json)：本次读取汇总及算法，未重新加载约5.7MB压缩完整资源日志重做175快照审计。
- [retention.py](../experiments/autonomous_round4/retention.py)：`reorder`、`resolve_blocked`、`reclaim`。
- 三个 control 目录下的 `runs/measured_rep1_a/run.json.gz`；pin 原始日志位于相同子目录的 `server_logs/measured_rep1_a.log.gz`。

以下标准库命令重算本次新增的pin生命周期观察：

```sh
python - <<'PY'
import collections, gzip, json, statistics
from pathlib import Path
p = Path("results/autonomous_round4/control-pin/runs/measured_rep1_a/server_logs/measured_rep1_a.log.gz")
events = [json.loads(s.split("SLACKKV_ROUND4_RETENTION ", 1)[1])
          for s in gzip.decompress(p.read_bytes()).decode().splitlines()
          if "SLACKKV_ROUND4_RETENTION " in s]
active, held, reasons = {}, [], collections.Counter()
for e in events:
    if e["event"] == "ACQUIRE":
        assert e["program"] not in active
        active[e["program"]] = e["timestamp_monotonic"]
    elif e["event"] == "RELEASE":
        held.append(1000 * (e["timestamp_monotonic"] - active.pop(e["program"])))
        reasons[e["reason"]] += 1
assert len(held) == 32 and not active
print(dict(reasons))
print({"min_ms": min(held), "median_ms": statistics.median(held), "max_ms": max(held)})
PY
```

本次仅离线复核和研究交接，GPU消耗为0，没有改变服务运行时，也没有启动下一轮实验。
