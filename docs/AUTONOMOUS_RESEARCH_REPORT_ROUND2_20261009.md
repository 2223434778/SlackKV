# 自主研究第二轮：容量转折与两个失败的需求调度架构

本轮完成三个真实研究闭环：容量干预 → 有界续接优先级 → 软会话工作集准入。**没有得到可默认保留的新加速算法。**两个候选均未通过提前冻结的整体筛选门槛，停止扩展；没有调参挑选好结果。

本轮按停止条件 **B（限定范围的负结论与战略调整）**结束，非预算耗尽。可靠结论是：在这个负载上，用局部优先级改善 P95，或用逻辑驻留减少搬运/分配失败，都不足以满足整体目标；继续围绕这两项实现或原始 JIT 做矩阵不合理。**不能由两个单次失败推出所有准入、保留或 backfilling 算法无效，更不能声称不存在新系统算法。**项目最终目标保持；技术路线和评价重点需要改变。

## 1. 主要瓶颈：容量影响关键路径，但不是全部等待的解释

Compiled OnDemand、相同 40 个实际输入、相同 Qwen2.5-1.5B FP16 / K2 / CPU 8 GiB / BGv2，仅改变 GPU KV：

| GPU KV | 实际 blocks | Resume P95 ms | Resume median ms | Session median ms | 容量失败续接* | Agent source 恢复 MiB | BG wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 GiB | 4681 | 5199.92 | 2140.74 | 23525.76 | 31/32 | 19280.19 | 67189.95 |
| 4 GiB | 9362 | 1163.51 | 201.69 | 18780.22 | 9/32 | 9925.13 | 67717.66 |
| 8 GiB | 18724 | 445.32 | 111.15 | 18837.08 | 0/32 | 0 | 67851.80 |

\* 至少观测到一次真实分配失败的请求数，不是失败持续时间或所有排队请求的完整容量状态。每个请求最终有 allocator 观测，不表示整个排队区间都被检查。

2→4 GiB 的 P95 降低约 77.62%、session median 降低约 20.17%；后台增加约 0.79%。4→8 GiB 继续改善尾部，却没有继续降低 session median。这是资源干预，不是 SlackKV 创新或同资源加速。三点各一次，初始请求发出顺序略有差异，不能当成精确因果份额或统计显著性结果。

各配置**自己的** P95 请求，server arrival→首次模型准入分别为 5058.18、955.82、3.66 ms；对应客户端 TTFT 为 5199.92、1163.51、445.32 ms。8 GiB 的剩余延迟主要位于准入之后，包含增量 Prefill、执行、输出传递等，不能继续归因为容量排队。这是不同尾部请求的路径分解，不是同一个请求在三个配置中的独立成本相减。

CPU source 的 pre-first 远程等待 median 在 2/4 GiB 分别约 49.01/41.38 ms；固定路径删除远程等待后 P95 改善约 0.96%/4.79%；8 GiB 为 0。这不构成重调度后的全局因果上界。

静态全工作集模型：同一轮八个请求的 prompt+声明输出上限，加后台 9500+2048 tokens 和 null block，需要约 **4.81→5.39 GiB**。模型不是实际同时驻留轨迹，也没有作为候选算法的未来信息。它解释为什么 2 GiB 是强压力条件、8 GiB 可以覆盖本次需求；见 `results/autonomous_capacity/footprint_model.json`。A6000 物理显存不是 2 GiB。

## 2. 探索与淘汰：one-turn continuation credit

假设：真实到达的短续接若能在新长上下文冷 Prefill 前获得有界机会，可减少完成后又重新排队的成本。

实现只改变 WAITING 顺序：每个较早冷队头至多被一次成功续接 allocation 越过，跨 tick 保留计数；async restore 与随后的正常准入不重复花 credit；分配失败仍停止，不按容量回填；取消清理；有物理 Lease 不重排。未改变 KV 内容、allocator、K2、模型工作或后台规则，默认关闭。

| 2 GiB 匹配配对 | Compiled OnDemand | credit | 变化 |
|---|---:|---:|---:|
| Resume P95 ms | 6061.25 | 3642.52 | -39.90% |
| Resume median ms | 1829.48 | 2207.02 | +20.64% |
| Session median ms | 22855.42 | 24244.98 | +6.08% |
| BG wall ms | 67864.35 | 67756.38 | -0.16% |
| 实际 KV H2D MiB | 19485.81 | 19204.94 | -1.44% |

全部 40 输入匹配、执行配置/源码匹配；四次 credit，四个不同冷队头。三个 session E2E 改善、五个恶化。P95 排名改变不等于整体改善，session 门槛失败，停止该实现，不调整 credit 或反序扩大。这里只能支持这次筛选观察；初始 issue 顺序不同及 baseline 波动限制稳定性结论。

## 3. 探索与淘汰：two-call soft working-set epochs

假设：避免新会话冷 Prefill 不断驱逐短工具等待中的会话，可能真正减少周转，而不仅是搬运提前。

最小实现：按当前真实请求的 prompt+声明 max output 估计 block charge，结合真实 pool 大小与未分组后台需求，控制新 program 准入。一个逻辑 epoch 最多完成两次调用，idle 最长一秒；terminal、cancel 或过期释放逻辑状态。已入组需求与空 cohort 始终回到原始 allocator 的进度路径。没有 physical pin、Lease、新 allocator reservation 或提前 Promotion。已发送的新会话在引擎内等待，**全部等待计入 E2E**。

这个 charge 是工作集估计，**不是实际 free blocks 或物理容量证书**。未来上下文增长可能使估计超预算；真实 allocator 仍决定能否分配。只验证 sequential programs，未证明并行 DAG 的性能或活性。

| 2 GiB 匹配配对 | Compiled OnDemand | soft epochs | 变化 |
|---|---:|---:|---:|
| Resume P95 ms | 4782.34 | 9889.70 | +106.80% |
| Resume median ms | 1943.93 | 1281.59 | -34.07% |
| Session median ms | 22170.16 | 25356.69 | +14.37% |
| BG wall ms | 67402.00 | 67347.36 | -0.08% |
| 实际 KV H2D MiB | 18888.63 | 11597.69 | -38.60% |
| 观测容量失败续接 | 28/32 | 2/32 | 不作持续时间归因 |

全部 40 输入匹配，初始 issue 顺序均为 E00…E07。27 个逻辑 epoch 被准入，最终 active programs 为 0，无实际 preemption。六个 session E2E 恶化、两个改善。停止此 lifetime/call-count 方案，不调参。

P95 请求 E02:turn:2 到第一次 allocator 检查已约 **9720.85 ms**，arrival→admission 约 9765.21 ms。epoch blocked 样本跨度约 7350.43 ms；E07 初始请求跨度约 9310.41 ms。稀疏样本不能证明连续阻塞时长，但与源码和迟到的 allocator 检查共同表明逻辑准入引入了长等待。减少 allocator 失败也因为更多请求先被 gate 挡住，**不能据 2/32 推断容量压力已消除**。提升命中、减少 H2D 或 allocator failure 都不是独立验收目标。

## 4. 强基线与验收

两个候选各自使用同源码、同模型、同 2 GiB、同 K2、同输入与采样的 Compiled OnDemand。独立审计确认七个有效 run 均有 VLLM_COMPILE / FULL_AND_PIECEWISE 日志。未与 eager、batch-invariant 或不同 KV budget 横向归属算法收益。

门槛在候选运行前记录：P95 改善至少 10%、session median 不恶化、BG wall/TPOT 和实际 H2D 不恶化超过 5%、生命周期有效，才反序和第二配置扩展。两项均失败，所以未扩展；背景 workload 没有削减。所有主要结果只是单次筛选，**无统计显著性、稳定收益或生产普适性声明**。

七个 run 各完成 40 Agent 请求、32 续接和 8192 BG 输出：共 280/224/57344。混合生成吞吐使用 BG 首 token→最后完成窗口，本次后台尾部主导这个窗口，约 174–176 tokens/s；它不是独立稳态负载容量指标。两个候选的混合吞吐变化都低于 0.2%。

## 5. 正确性、活性、公平性与计量限制

- 40 项本地状态/分析/原始证据回归测试通过。两个 runtime 阶段各 15 项 native prefetch、connector-first、gate-capacity 单元测试通过，CPU-only；没有重跑完整历史 Lease/HOL GPU adversarial 矩阵。
- 有效 run 的生命周期、pause、K2、后台固定输出、容量检查、OOM/服务器存活、既有 physical/logical lease 检查均通过；soft epoch 最后状态为 0。默认开关关闭可回退 OnDemand；原历史 runtime 未修改。这是本有限负载的活性证据，不是无限过载下公平性证明。
- 实际 preemption counter：capacity-2 为 2，credit control/candidate 各 1，其他有效 run 为 0。source 事件数量不能替代这个 counter；credit candidate 的 32 source events 不表示没有 preemption。
- 每个 run 的 D2H counter 都为 5172.5625 MiB。credit H2D counter 比 source-token 换算多 **458752 bytes＝一个 16-token block**，原因 UNKNOWN；最终迁移门槛使用真实 counter，不强迫两者相等。8 GiB source external restore 为 0，但 H2D series 缺失，actual counter 记 UNKNOWN，不能缺测填零。
- 两个候选与各自 control 的输出 hash 均非全一致；本轮没执行新的 batch-invariant 内容 oracle。先前 cold/batch 的数值差异解释不能自动证明新候选的 KV 内容正确。生命周期通过、数值差异、KV 身份/内容与生成质量严格区分；**内容/语义质量等价仍 UNKNOWN**。没有候选被接受部署。
- owned/active/idle block union 的完整 GPU KV memory-time 仍 UNKNOWN。仅能报告已分配 pool 在 BG 窗口内的成本：2/4/8 GiB 约 134.38/270.87/542.81 GiB·s；这是 pool reservation 的资源成本，不能冒充有效 KV 驻留积分。
- 采样对 baseline/candidate 相同，instrumentation CPU 成本没有独立量化；不能宣称零开销。

## 6. 创新性与下一方向

本轮两项都是已有思想的可检验变体，**尚无成立的新颖性贡献**：

- [InferCept](https://arxiv.org/abs/2402.01869)研究 augmented inference 截停期间 KV 保留/交换；工具等待 KV 管理本身已有。
- [FastServe](https://arxiv.org/abs/2305.05920)已有 token 级抢占与分层 state 管理；调度与迁移协作本身已有。
- [KVFlow](https://arxiv.org/abs/2507.07400)使用 workflow 信息管理前缀和预取；本 PoC 只使用已到达 demand，未证明这种区别产生整体收益。
- [Continuum v7](https://arxiv.org/html/2511.02230v7)已有 TTL-aware priority 和 program FCFS。credit 的有界实际需求重排、epoch 的逻辑准入与 physical pin 分离是实验差异；本次结果没有证明优于其方法，未运行其完整实现。
- [MORI](https://arxiv.org/html/2606.00866v1)已有相对 idleness、分层放置和 admission。本 epoch 是有界逻辑工作集，没有 phase estimator 或 replica affinity，不能把它包装为新的“分层准入”。
- [Autellix](https://arxiv.org/abs/2502.13965)已有 program-level history/prioritization；一般程序连续性与防饥饿不是新问题。
- [Sarathi-Serve](https://arxiv.org/html/2403.02310v3)已有 chunked prefill 与 decode 协同；调整普通 chunk 参数也不能成为创新。

本次检索主要覆盖摘要及 Continuum/MORI/Sarathi 全文相关段落，没有完成全部作者代码与同硬件 SOTA 复现；不借用其 speedup 作为 SlackKV 预期。

**最终目标不改变，建议改变研究问题的组织方式：**从“提前恢复 KV”转为“在明确 SLO 与公平约束下，提高有限显存的有效并发/降低成本”。后续应先使用稳态多轮到达、明确的 cold-start 与 continuation 分层指标、较长真实会话及一个有依据的更大模型，保留本次负载与匹配基线；用少量点确认容量转折是否仍存在。当前 workload 的短 5-turn、32-tail 样本及后台长尾不足以建立通用算法优势。

下一轮优先做带真实执行时间/owned block union 的关键路径测量与成本模型；只有能够预测同资源 P95/E2E 或同 SLO 显存收益的候选才进入实机。允许进一步考察改变 **active KV footprint** 的执行方案，但 exact attention 的带宽/重读成本和现有相关实现必须先验证，不能直接声称 partial KV 有价值。历史 Phase0 peer/CPU 带宽相近，无证据把 GPU1 当成自动更快的 KV tier。不继续调本轮 credit、epoch 或原 JIT/Frontier 矩阵。

最重要不确定性：初始顺序与 scheduler 时序的方差、完整 owned KV 轨迹、数值/内容正确性、新模型/稳态真实会话的泛化，以及与现有系统的实质区别。新的目标下还有哪些可避免计算/排队尚未确定；本报告不是“整个项目失败”的结论。

## 7. 实验、失败、预算与复现

起点 `e239088b3891396cde654669d74020e8fce82ecd`，本机/server 一致、origin/main 无新增；历史未跟踪数据保留。Ubuntu 双 A6000，仅 GPU0；沿用 Python 3.11.16 / Torch 2.11.0+cu126 / CUDA12.6 / driver550.54.14、固定 Qwen snapshot。模型加载、编译和退出全部计入 budget；没有改依赖或其他服务。

| 执行 | 状态 | 含启停秒 |
|---|---|---:|
| capacity-2 | VALID | 107.19 |
| capacity-4 | FAILED（workload 前旧固定池验证器拒绝） | 60.07 |
| capacity-4-retry | VALID | 107.75 |
| capacity-8 | VALID | 130.45 |
| credit-control-2 | VALID | 109.85 |
| credit-candidate-2 | VALID | 107.74 |
| epoch-control-2 | VALID | 107.36 |
| epoch-candidate-2 | VALID | 109.37 |

累计 **0.2332685893 GPU·小时**，最长 130.45 秒。CPU-only 单元/离线检查不占 GPU；GPU1 未使用。最后两 GPU 均 19 MiB / 0%、无 compute process。没有用 run 重命名重置预算。

4 GiB 失败日志与新预算匹配验证器都保留；不是放宽检查，第二轮用 pinned 28672 bytes/token×16 对应实际池数，历史固定实验验证器不变。离线 audit 的 derived-summary hash 重算、missing counter 与 one-block mismatch，以及一次 PowerShell quoting 错误，记录在 `results/autonomous_capacity/analysis_failures.txt`。

原始 run/输入/输出与完整日志无损 gzip，87 个导出逻辑文件 hash 已验证；服务器未压缩原件保留。重算后的 summary 与冻结 `exported_round_summary.json` 分开，原 SHA 校验不被更新值替代。证据入口：`results/autonomous_capacity/README.md`、`round_summary.json`、`independent_audit.json`、`SHA256SUMS.json`；预注册 `protocol.md`；精确参数与 argv 在各 `execution.json`。

仓库根目录离线复算：

```sh
python -m experiments.autonomous_capacity.summary results/autonomous_capacity
python -m experiments.autonomous_capacity.audit results/autonomous_capacity
python -m pytest -q tests/test_autonomous_capacity.py tests/test_autonomous_admission.py tests/test_research_reassessment.py --basetemp=.pytest_cache/round2-reproduce
```

服务器 venv 下实际执行命令见 `results/autonomous_capacity/commands.sh`。再次实测必须使用新 run 名并继续同一 ledger；示例：

```sh
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
SRC=/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round2-source
$PY -m experiments.autonomous_capacity.run --name reproduce-control --kv-gib 2 --trace --runtime-source "$SRC" --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl.gz
$PY -m experiments.autonomous_capacity.run --name reproduce-epoch --kv-gib 2 --trace --working-set-epochs --runtime-source "$SRC" --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl.gz
```

新实测不是本轮进一步授权消耗的承诺；本轮已依据科学停止条件结束。固定输入是明确的资源回放，替换 synthetic client 的输出反馈，仍保留 finish-relative 工具 pause 与完整模型 token 工作；不等价于语义 Agent 成功率测试。

修改文件：`experiments/autonomous_admission/run.py`（扩展 opt-in runner）、`experiments/autonomous_capacity/`（两候选、预算布局、patch、复算/审计/导出）、`tests/test_autonomous_capacity.py`、本报告及研究/决策/实验日志、`results/autonomous_capacity/`。所有机制默认关闭，保留可审查 delta 与完整负结果；最终 Git commit 见本报告的提交及 main 历史。
