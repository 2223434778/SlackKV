# 32B 双重容量压力：匹配三方法筛选

基线 `b9685aed2da6eec7675245e730732fc51b2d0b00`。本报告接续部署机会审查及用户提出的较大模型验证；最终目标保持 Agent KV 资源管理。**静态前缀准入在这一组三方法筛选中通过整体门槛：E2E median 比 NoHost 降低 16.44%，比 Full 降低 18.40%。** 这是值得复验的正结果，不是稳定性或生产泛化证明。

## 问题和可行性

研究假设是：全量 host 写入在长期会话的聚合工作集超过可用缓存时，可能淘汰即将复用的前缀；在同样 host 预算下按会话限制写入，能否保住一部分前缀并减少真实重算，而无需预测工具完成时间或 JIT promotion？这与旧实验中仅几十毫秒的恢复等待优化不同。复用已有准入和公平前缀组件，不预先宣称新颖算法。

Qwen2.5-32B-Instruct 的官方固定版本为 `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`。FP16、TP2、双 RTX A6000、默认 compiled/CUDA Graph；卡间实测拓扑 SYS，无 NVLink。权重 payload 61.0275 GiB，实际自然 85% 配置得到 70,000-token GPU KV 池，即全模型 17.08984 GiB，而非此前不计 runtime 的 20.55 GiB 算术上界。

全模型 KV 为 262,144 bytes/token，TP2 每 rank 131,072 bytes/token。21 个独立 28,672-token 初始历史合计 147 GiB，超过本机约 125.65 GiB 物理 RAM 加实际 GPU KV 池。这说明这一**合成保留目标**无法整体放入两层，不等于全部历史必须同时驻留，也不证明生产需求或最小容量下界。64 GiB 是明确的 host-serving 预算，非机器物理上限；同时需给 OS、runtime、模型加载及其他服务留空间。

初步可行性真实观测：28,672-token 冷请求 TTFT 13.35 s；同 GPU 前缀复用 TTFT 169 ms；8K CPU 恢复 TTFT 134 ms、来源计数确认 8,192 external tokens。上述不同输入的数值不是公平算法对比；16-token 校准只用于估算时间，不能替代 128-token 混合吞吐。

## 冻结配置与验收

协议见 `LARGE_MODEL_PROTOCOL_20261012.md`。一个种子、21 sessions × 3 turns，输入 28,672 + turn × 512，输出每次 128；固定历史扩展不拼接实际生成输出。5 s 合成工具等待、首个背景 Token 后每 session 间隔 20 ms 首次释放。背景为三个顺序 8K-input/128-output 请求。每方法 66 请求、1,863,168 input tokens、8,448 output tokens。

这是长历史、短输出、闭环合成压力，不是生产 trace，也不能评价真实工具行为或 Agent 成功率。公开负载分析支持探索较长历史，但其 proxy tokens、原端点时间和输出分布不等同这一 Qwen 配置。旧 1.5B workload 和全部负对照保留。本轮没有 GPU reset。

三方法为真正无 connector 的 NoHost、64 GiB 原生 Full、同样 64 GiB 的 StaticFairPrefix。Fair 的准入上限是前 20 sessions 各 12,480 tokens、最后一个 12,544 tokens，共 262,144 tokens；这不是物理硬分区或动态公平性证明。GPU 固定为自然池实测对应的每 rank 9,175,040,000 bytes，各 run 验证实际 70,000 tokens。NoHost 不预分配虚构 host 池；原生 connector 的 cross-layer 布局与 NoHost 的默认布局有差别，因此 NoHost 对比是系统配置比较。Full vs Fair 只改变 host 写入上限。

预注册主要门槛：Fair session E2E median 相对 **NoHost 和 Full 都改善至少 10%**；每 session E2E 和背景 makespan 相对两者均不恶化超过 5%。完整报告 P95、吞吐、来源和迁移；不替换主指标、不降低门槛。一组三方法单次筛选不能宣称统计显著。

## 正确性与活性边界

独立 batch-invariant oracle 用两个 28,672-token 文本，将不同 CEDAR17/MAPLE42 标记放在缓存前缀，分别检查 store、12,480-token CPU 部分恢复与不同 top-level cache_salt 的独立 cold Prefill。六请求各 16 个输出，两个标记被正确区分，恢复与冷请求逐 Token 一致。来源复算：resume external 12,480、compute 16,192、GPU 0；独立 cold compute 28,672、GPU/external 0。

这是两个敏感输入的内容与身份证据，不能覆盖全部取消、抢占、远端 DMA、生成质量和生命周期状态。oracle 使用 batch-invariant=1，性能测试为 0，不混合计时。随机 Token 性能负载仍只验证规定工作量，不声称质量一致。旧 Lease/HOL/引用生命周期实现未更改，NoHost 为安全配置回退；三方法的完成性和回收需在最终结果中单独核验。

## 性能结果

| 指标 | Compiled NoHost | Compiled Full | StaticFairPrefix |
|---|---:|---:|---:|
| Session E2E median，s | 969.20 | 992.50 | 809.89 |
| Resume P95，s | 360.74 | 369.05 | 337.67 |
| 背景 makespan，s | 753.97 | 772.34 | 654.72 |
| 混合 output tokens/s | 7.412 | 7.241 | 9.034 |
| local compute prompt tokens | 1,863,168 | 1,863,168 | 1,338,880 |
| external restored tokens | 0 | 0 | 524,288 |
| GPU local hit tokens | 0 | 0 | 0 |
| H2D，GiB | 0 | 0 | 128 |
| D2H，GiB | 0 | 448.875 | 64 |
| 实际 host pool reservation，GiB | 0 | 64 | 64 |
| 启动至 ready，s | 123.68 | 155.75 | 280.26 |

全部 198 个性能请求输入身份、长度、输出数量、背景工作量、执行配置（除预定义 connector 差别）和 scheduler SHA 匹配；每方法实际 GPU pool 为 70,000 tokens。压缩副本独立复算与服务器相同。Fair 相对 NoHost/Full：E2E −16.44%/−18.40%，P95 −6.39%/−8.50%，背景 −13.16%/−15.23%，吞吐 +21.88%/+24.76%。21 个会话全部改善：相对 NoHost 的范围 −13.85% 至 −18.93%，相对 Full −15.90% 至 −20.82%。两个冻结主门槛都通过，但 P95 本身未达到 10% 改善。

NoHost 的 GPU 起始内存为321/19MiB，Full/Fair为19/19MiB；GPU0此前约298MiB远程桌面由外部释放，本实验未终止它。起始utilization均0%，实际KV池相同，但不能声称所有外部GPU活动和文件缓存状态严格相同。Full vs Fair起始GPU状态一致，是更干净的准入策略对照。启动时间差和这一外部状态限制均需反序复验。

Full 写入 448.875 GiB 却没有后续读取，Fair 用同样池保留每个 session 的初始前缀，两次续接各恢复总计 262,144 tokens。Fair 的 compute prompt tokens 少 28.14%，D2H 少 85.74%，总迁移 192 GiB，比 Full 少 57.23%。source 总和与全部逻辑输入一致，没有省略输入或减少规定生成。观测支持“host 写入准入保住可重用前缀，从而避免实际 Prefill”这一机制。没有逐出事件级 trace，不能进一步将 Full 的所有未命中或时长归因于某一具体 LRU 操作；NoHost/Full 的约 2.4% E2E 差异也不能宣称显著因果退化。

背景三个请求 E2E，NoHost 为 37.60/355.25/360.68 s，Full 为 38.45/364.59/368.83 s，Fair 为 38.21/355.64/260.40 s。相对 NoHost，前两次约 +1.62%/+0.11%，第三次明显更快；相对 Full 三次皆更快。公平性门槛通过不等于没有干扰：第三个背景请求的“首 Token 到响应结束平均间隔”相对 NoHost/Full **+16.94%/+14.70%**。这是客户端区间除以 127 的代理量，没有逐 Token 时戳，不能当真实 TPOT 分布或纯 GPU Decode 时间。保留这一 tradeoff，不用总吞吐掩盖。

GPU pool 没有增加；Fair 和 Full 仍分配相同 64 GiB host 池，**没有证明减少物理 host RSS 或 active GPU KV footprint**。NoHost 不使用该池，Fair 相对它额外消耗 CPU 内存并迁移 192 GiB，换取这一负载的延迟改善。Full/Fair ready 的系统 MemAvailable 约 49 GiB；SwapFree相对各自启动前减少约5.75/89MiB，退出后未立刻恢复，原快照保留，不能归因到具体进程。完整 owned GPU/CPU memory-time、开放到达的 SLO 并发、CPU/PCIe 对其他服务的影响仍 UNKNOWN。

三方法全部 COMPLETE，owned live group 最终为空，GPU 回到每卡 19 MiB、0% utilization；没有可观测挂起。这仅证明有限运行的完成性，不代替完整 adversarial cancellation/preemption/公平性证明。新机制只用原生 `max_offload_tokens`，未修改 KV 数据、allocator、scheduler、Lease 或 DMA 生命周期；保留真实无 connector 的配置回退。

回答用户三个问题：**部分前缀胜过 NoHost；21 会话静态均分保住整体收益；不清空 GPU 时收益仍然存在**，均限于本次 32B/自然 GPU pool/64 GiB host/固定长历史合成对照。旧 1.5B 多会话不足 10% 及 hot/ample 负对照仍有效，不被此次结果覆盖。

## 失败、资源和追溯

模型 acquisition1 从单路转为双路，保留中断记录；acquisition2 到 30 分钟下载 deadline，保留失败和部分文件；acquisition3 成功。24 个文件全部对照官方 LFS SHA256 或 Git blob SHA1 核验，镜像仅为传输渠道。约 61.04 GiB 模型文件留在研究盘、不提交 Git。下载约 48.43 分钟，未预留 GPU，GPU 成本为 0。

第一次可行性进程退出后 GPU 内存释放没有及时独立确认。保留原快照，并按原退出到首次确认空闲的整个未观测区间额外计 0.4724857332 GPUh **上界**，不是实测 GPU 忙碌时间。后续 runner 等待设备内存释放并验证 owned group 无存活进程；不终止 RustDesk 或其他服务。

本机解包曾因 Windows realpath 权限失败，后用拒绝绝对路径/`..`/非普通文件的复制方式完成；紧随失败的分析缺失文件错误为工程问题，不当 GPU 负结论。一次测试插入引起 NameError，修正后重测，不弱化断言。provenance 批量复制曾因 acquisition 脚本实际在 `.tmp` 而失败，核查实际路径后 acquire3 SHA 正确匹配。超时单测首次因 Windows 无 `os.killpg` 失败，使用 POSIX 模拟后通过。已有 `nvidia_cutlass_dsl` `.pth` 警告及首轮 semaphore/shared-memory 清理警告保留，没有改包或驱动。

线程池可能在超时异常后等待 HTTP reader，最终 runner 修复为先终止 owned server group 再抛异常，并有单测。本次没有触发超时，Fair 使用改动前冻结版本，执行中由本会话监督；未把修复版伪称性能源码。

归档验证首次使用通用 reader 时，旧本机中途 controller.log 遮蔽最终 `.gz`，正确拒绝了哈希；改为读取 manifest 指定的副本并新增旧 raw 遮蔽测试后，82文件复核通过。原始服务器文件没有修改，纯压缩复算已提前通过。最终52项CPU测试通过。

matched NoHost/Full runner SHA256 为 `bd3fe75ccd7a93b5891201363577c2c39151ef1015e7292e865cef6d10121419`；Fair 为 `61d282d94f91f1654d2a287c6710fcc3d0cc2779e0eb11db4557591f51ee36e8`，仅新增显式 4.5h 授权参数和记录字段。可行性初版 runner 记录了 SHA，但改动前未保存完整源码快照；这一校准运行的追溯缺口保留，不能假称全部版本归档。主三方法保存 runner/client/native hashes 与实际配置。

用户批准连续上限从 4 扩至 **4.5 GPUh，仅完成这组三方法对照**。最终连续累计 **4.1919880197 GPUh**，新增 large-model family **2.9261669850 GPUh**，其中清理不确定性上界 0.4724857332；扣除此上界的新增进程计费为 2.4536812518 GPUh，均不退账。NoHost/Full/Fair 各为 0.7084516344/0.7534938823/0.6908339341 GPUh，最长 Full 1355.29 s < 30 分钟。预算含以前各 family、加载/编译/请求/退出、两卡及不确定性；余额 0.3080119803h 不用于额外实验。见 `budget.json`。

## 复现

原始结果：`results/large_model_20261012/`。输入 `.xz`、请求/日志 `.gz` 为无损副本，原文件留在服务器。分析器直接支持这些格式；`raw_manifest.json` 用于逐文件解压后 SHA256 复核。

```bash
python -m experiments.large_model.analyze --directory results/large_model_20261012/feasibility1
python -m experiments.large_model.verify_archive --directory results/large_model_20261012
python -m experiments.large_model.analyze_oracle --directory results/large_model_20261012/oracle1
python -m experiments.large_model.analyze_screen --directory results/large_model_20261012/screen_nohost1
python -m experiments.large_model.analyze_screen --directory results/large_model_20261012/screen_full1
python -m experiments.large_model.analyze_screen --directory results/large_model_20261012/screen_fair1
python -m experiments.large_model.analyze_screen --compare results/large_model_20261012/screen_nohost1 results/large_model_20261012/screen_full1 results/large_model_20261012/screen_fair1
```

GPU 启动使用固定模型目录及已有 conda/native source，实际 argv 和时限见各 `execution.json`，完整命令见 `commands.sh`。复跑消耗新预算，须确认额度并使用新 run 名。82 个原始文件、84,782,502 解压 bytes 的 SHA/长度全部独立通过；纯压缩副本的分析和 oracle 重算通过，见 `independent_recheck.json`。CPU 回归命令：`python -m pytest tests/test_large_model.py tests/test_host_prefix_followup.py tests/test_deployment_opportunity.py tests/test_host_prefix.py tests/test_opportunity_decision.py -q --basetemp .tmp/pytest_large_model_repro`。

## 研究判断

选择停止条件 A：找到有明确机会、真实实现和初步匹配证据的候选。最有价值的当前问题是**在 host 也有限时，如何控制写入以保住跨会话可复用的初始前缀**，不是继续优化旧 JIT 的几十毫秒恢复等待。此次写入减少和重算减少都有真实计数；静态均分足以取得整体初步收益，不急于设计复杂评分。

与 EfficientAgent、RR-Evict、MORI 的准入/前缀回收/分层管理有组件重叠，比较沿用 `RESEARCH_OPPORTUNITY_DECISION.md` 与 `AUTONOMOUS_RESEARCH_HOST_PREFIX_20261010.md`；没有重做完整论文系统的匹配 SOTA 实验，不宣称静态均分本身新颖。可保留的实质成果是：这一自然 GPU 容量、较合理 host 预算的 TP2 长历史 regime 中，全存并未取得复用，而容量约束下跨会话前缀准入有端到端证据。论文价值还需强已有方法和更现实需求支持。

不改变项目最终目标。下一优先项是在新授权范围内做反序/多种子复验，随后少量非均匀上下文、较长输出或不同并发的预注册匹配筛选；保留 Full/NoHost、相同资源和真实工作量。先确认收益转折和后台 Decode 代价，再决定是否需要动态额度、借用与会话失效回收，而非马上开发新的 Frontier。

最大不确定性：一个固定闭环 seed、均匀长历史/短输出能否泛化；顺序与启动/文件缓存差异；open-loop 到达、结束/变长 session、公平性和内容生命周期；更强已发表算法是否同样取得收益。本轮不给生产适用性、统计显著、完整质量正确或长期活性的结论。
