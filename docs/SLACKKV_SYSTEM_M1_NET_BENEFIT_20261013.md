# M1 净收益筛选：terminal gate、内容 oracle 与四会话竞争

日期：2026-10-13。交接基线 `418288c`，实现与结果以本报告所在提交为准。按 Astra 审查执行，保留主架构，M2/M3 关闭。

## 一页结论

**保留 M1 为容量竞争下的可选机制，不默认替代简单 StaticFair，不扩展 JIT。** 本轮在一个预先冻结的合成场景中看到相比 native Full 的整体改善，但相对 StaticFair 的增量较小且迁移成本明显更高；尚无稳定性能优势、生产适用性或创新性证明。

| 正式测量（各方法一次，compiled） | M1 + terminal gate | StaticFair + terminal gate | Full + terminal gate |
|---|---:|---:|---:|
| A 会话 E2E / ms | 8547.92 | 8694.04 | 9139.91 |
| B 会话 E2E / ms | 7677.74 | 7829.91 | 8275.17 |
| 短会话 S E2E / ms | 2291.64 | 2371.08 | 2485.89 |
| 新加入 D E2E / ms | 2791.83 | 2830.55 | 3030.45 |
| 四会话 E2E median / ms | 5234.78 | 5330.23 | 5652.81 |
| 续接 TTFT P95 / ms | 252.17 | 262.41 | 345.56 |
| 重算输入 tokens | 95968 | 102400 | 116736 |
| GPU 本地命中 tokens | 12544 | 12544 | 12544 |
| host 恢复 tokens | 21792 | 15360 | 1024 |
| D2H / MiB | 476 | 196 | 1641.5 |
| H2D / MiB | 595.875 | 420 | 28 |

M1 相对 Full 的 median E2E −7.40%，四个会话均降低 6.48–7.87%，续接 P95 −27.03%；相比 StaticFair 分别 −1.79%、四会话 −1.37–3.35%、P95 −3.90%。P95 为 **10 个续接样本的 nearest-rank（此处即最大值）**，不是稳定尾延迟估计。全序列/A 会话完成时间改善分别 6.48% 与 1.68%。没有遗漏首请求、终止请求或压力请求。

真正改善的链条是：保护多个连续 host 前缀 → 压力后仍可恢复 → 少重算 → 缩短这组序列及各会话。Full 在此资源比例下进行了大量写入，而后续大多数请求的实际 host 命中为 0；这是匹配 compiled Full 的真实性能结果，不是人为关闭其读写。但没有采集 Full 的逐块驻留树，不能把失效原因进一步宣称为某一种已严格隔离的 LRU 因果。强 StaticFair 已保存大部分收益，因此不能仅对 Full 得到正数就宣称动态机制价值很大。

M1 对 Fair 少重算 6432 tokens，却多写 280 MiB、多读 175.875 MiB。所有方法实际 host 池都是 224 MiB，GPU KV 池都是 512 MiB、18720 tokens；**没有证明减少显存/CPU RSS或 KV memory-time**。此小模型配置中更便宜的 StaticFair 是重要的 Pareto 对照。

## 实现及安全性

- `slackkv/m1/native.py`：增加可选 terminal-store gate；默认关闭，实验显式开启。独立有效 M1 请求的 `end_after_request=true` 只拒绝尚未发出的可选 store。观察/注册、已有 KV 的 lookup/load、正常计算、在途 DMA 结算与真正完成时的 end/tombstone 均保留。
- `slackkv/m1/spec.py`：增加 `TerminalCPUOffloadingSpec` / `TerminalCPUManager`，供 native Full 和 StaticFair 使用同等、显式独立 terminal 信号。只修改 store 准入，其余使用原生行为。历史无 gate 的 Full 数据保留。通用终止优化不归因于 max-min 创新。
- `experiments/system_m1/contention.py` 与配置：新增单个冻结场景；既有 smoke runner 复用原授权、锁、连续账本、超时和自有进程组清理。附加 runner 源码哈希纳入准备核验。
- 扩展 native audit，覆盖 terminal 读取保持到引用结清、新代 terminal 不写且结束、已有在途写入仍结清、Full 普通 store 与 terminal read 不退化。原 120 次交错与 TP 元数据检查保留。

本地六个相关测试文件 **62 项 PASS**；最终 M1 聚焦回归 **25 项 PASS**，compileall PASS。Ubuntu **19 组 CPU/native 审计 PASS**，禁止 CUDA 初始化；这不是 19 个 GPU correctness case。真实 GPU content2 两个恢复请求分别具有非零 host 命中，完整 16 IDs 与各自独立 cold 参考相同；两个参考零 local/host 命中且输出分别以 CEDAR17/MAPLE42 开头，IDs 彼此不同。最终 512 个 host 槽全部空闲，无 pending read/store 或物理成员。被测新 terminal 请求 D2H=0，三个性能方法的四个 terminal 请求 D2H 均为0。

上述仍不覆盖全部 KV 字节、GPU 强制取消/在途 end 碰撞、TP2 模型 DMA、共享前缀、分支请求或无界公平性。所有已测请求正常完成，只证明被测序列活性。压力任务是独立无 host 写入的模型请求，耗时 M1 833.21/835.62/837.73ms、Fair 838.14/838.75/840.42ms、Full 840.59/836.90/840.71ms；这不是同时运行的后台服务公平性证明。

## 场景和测量规则

Qwen2.5-1.5B-Instruct，FP16、TP1，vLLM 0.23.0 / Torch 2.11.0+cu126，单 RTX A6000。沿用已有 overlay 两个 observe hook、compiled/CUDA Graph 路径；不更改 GPU 调度器、attention 或 DMA。性能 `VLLM_BATCH_INVARIANT=0`，内容检查为1，二者不混合比较。

host 为8192 tokens ×28672 bytes=224 MiB。A/B/S 初始需求8192/6144/1024 tokens，共15360，约420 MiB；S 退出后加入4096-token D，三会话基本需求18432（原四者合计19456），持续超出 host。每次续接追加64 tokens，原历史前缀严格保留。四个独立根输入无公共首块，没有用盐消除共享；仅独立 cold oracle 使用盐。StaticFair 对四个预先配置的会话各固定2048-token上限，不知道未来长度、不借出空闲份额。M1按已知活跃需求 max-min。Full仍按每个完整当前输入发起原生可选 offload，公共 terminal gate 除外。

固定17请求：A0/B0/S0/pressure0/A1/B1/S_end/D0/pressure1/A2/B2/D1/D_end/pressure2/A3/B_end/A_end。会话内串行，整个驱动也串行，属于 **synthetic resource contention，非计算并发或生产 trace**。工具间隔每次0.15s，输出固定16 tokens、ignore_eos；使用正确 Qwen 聊天格式。增长部分为冻结的简短确认/后续用户检查，不是已测输出反馈的真实 Agent 任务，不能评价任务成功率。

各方法先做五次相同长度1024/4096/6144/8192/16384、不同根的预热，全部禁止 host 写入。等待传输稳定并检查 M1 无成员/pending；未调用清空 GPU API。warmup 输入未用于正式测量，GPU残余由自然 allocator 回收。启动分别约26.02s，预热约2.64–2.65s，均计入 GPU 预算并单列；正式首请求 M1/Fair/Full 为402.30/401.03/405.26ms，全部纳入会话时间。不存在事后扣除旧 cold-start 的0.67s。

### 物理链条与代价

M1实际 ready 深度：A独占512块 → B加入各256 → S加入 A/B各224、S64；S末次增长使 A/B先小幅减为222，S结束后目标256但尚未立即物化；D加入后三者170–171块。D结束后 A/B目标256，实际仍约170；A3普通请求将 A重建到256。B结束后 A目标512，但最后 terminal请求不为将来补写，随后正常释放。不能把目标增加直接算成完成恢复。

压力后 A1/B1各真实恢复3584 tokens，A2/B2/D1/A3/B_end各恢复2720。Full A1恢复1024，其余这些请求恢复0；Full与M1末次A_end均有8384 GPU local命中，表明实验并未强制清空 GPU。M1适用于需要保护多个可复用前缀的容量区间；不会因当前GPU命中就自动取消普通host保留。

缩额、余数轮转与下一次扩额会造成回收/重写。包括 terminal 输入增长仍参与观察和分配，因此可能短暂缩小其他成员。这是当前多写成本的重要待查来源，**本次没有再调参或修改准入规则**。该配置的动态分配增量收益小，不支持立刻开展更大矩阵。

## 失败、原始证据与预算

第一版 `contention.py/` run为 **FAILED**：未使用聊天格式，cold A输出了包含密码的整段前文，cold B输出无关代码；oracle严格断言拒绝。恢复与cold IDs相同不能弥补参考本身不可区分的问题。修正为Qwen聊天格式后另冻结 `preparation_chat/`，未拿失败版本和新版本进行性能比较，也没有降低断言。

同时修复附加源码校验循环变量覆盖run名称的工程缺陷，因此首次run落在名为`contention.py`的目录。所有资源仍计入原family账本，未覆盖历史。旧runner及输入生成源码通过记录中的SHA恢复并验证一致，见 `source_before_chat/` 与 `failed_source_verification.json`。

| 本轮run | 状态 | GPU·小时（含启动、编译、清理） |
|---|---|---:|
| contention.py（预期content1） | FAILED | 0.017632695216666938 |
| m1_followup_content2 | PASS | 0.01767409546472158 |
| m1_followup_m11 | COMPLETE | 0.013270473302497218 |
| m1_followup_fair1 | COMPLETE | 0.013308428501388537 |
| m1_followup_full1 | COMPLETE | 0.013463693147500129 |

本轮新增 **0.07534938563277440 GPUh**；同一0.5授权累计 **0.21011529948083787 GPUh**，连续历史 **4.402103319208365 GPUh**。五个run全部确认自有进程组退出，两GPU最后各19 MiB/0%。没有借用旧32B的4.5例外或重新计数。现有 `nvidia_cutlass_dsl` .pth环境警告保留，未修改共享环境。

原始run保存在 `results/system_m1_smoke/{contention.py,m1_followup_content2,m1_followup_m11,m1_followup_fair1,m1_followup_full1}/`；配置、两版准备、CPU审计、分析、复算和SHA清单在 `results/system_m1_followup/`。服务器原始文件保持不变，Git版本仅无损压缩log/prom/大型inputs，297个原文件SHA核验通过。Linux原始分析与本地压缩版复算完全一致；另从每请求来源及原始时间戳独立复核输入、16输出工作量、130304正式输入/272输出、来源守恒、H2D=external×28672。来源指标初始零标签可能不存在，其presence单独记录，并用五次零cache命中的warmup和新进程来源证明初始为零；缺失迁移counter仍保留null。

准备元数据中旧`after_join_nominal_demand=19456`实际指四个配置会话的合计，不能作为S退出后的同时活跃量；修正未来准备字段为配置合计19456、退出后加入D的基础需求18432。再次CPU准备核验正式输入SHA未变（`9057fb7c…`），未重跑或修改旧GPU数据。执行字节保存在`executed_sources/`，元数据说明见`provenance_notes.json`。

## 决策与限制

本轮接纳 Astra 的小修补、等价预热和高于host的多会话筛选；不将终止优化包装为算法创新，不强行要求动态方案全面胜出。动态 M1 的实际可用性有所推进；增量价值相对 StaticFair 仍小，因此先保留条件启用和清晰回退，不默认扩大架构。

下一步优先处理/量化真实共享 system-prefix 覆盖及最小共享计费；当前任何共有块导致整代停止可选写入，独立路径的正结果不能外推。**本轮未新增共享覆盖比例实验**，共享参考与多所有者释放仍待实现。若之后验证更大模型的重算/迁移比有价值，先提出有成本依据的单点迁移验证，不直接开启32B矩阵。此次未跑新 NoHost，不可复用旧workload的NoHost数字来声称此场景收益。正式统计、并发SLO、GPU memory-time和生产质量仍UNKNOWN。

不改变最终Agent Serving目标；值得研究的问题是“保护可用连续前缀所节省的重算，是否值得动态重分配造成的迁移”，而非单纯提升Promotion成功率。max-min、公平前缀与terminal gate均使用成熟机制，尚未证明独立论文创新。按本交接完成提交后停止本轮，等待新证据审查。
