# 部分前缀的三个问题：NoHost、公平分配与自然 GPU 回收

2026-10-10。起点`67eff77`；实验定义提交`158778f`，冻结协议`docs/HOST_PREFIX_FOLLOWUP_PROTOCOL.md`。16个真实GPU run全部完成；原始输入无损归档后在本机重新计算哈希、工作量、计时与来源守恒。没有修改vLLM运行时、KV内容格式、调度器或GPU内核。

**决定：保留简单部分前缀准入作为容量受限的工程候选，但不升级为已验证的广泛加速算法。NoHost补强了单会话机制正结果；静态均分保住了多会话前缀；无reset下有小幅整体收益，但未达到预注册10%主门槛。GPU-hot时没有收益，host充足时普通完整Offload足够。暂不开发复杂adaptive frontier，不扩大正式矩阵。**

## 1. 部分前缀能否胜过 NoHost？

**能，在本次强制GPU冷却、host256MiB的单会话诊断条件下。** 两个预定种子，第二组反序。NoHost仅设置max_offload_tokens=0，仍开启GPU APC；不是关闭所有缓存。模型、compiled执行、GPU/CPU预算、实际输入和输出严格匹配。

| 种子 | NoHost Resume P95 ms | 部分前缀 P95 ms | 变化 | BG中位变化 |
|---|---:|---:|---:|---:|
|773311|1736.43|1425.02|−17.93%|−7.91%|
|884422|1790.65|1430.84|−20.09%|−9.35%|

每run只有5个续接，nearest-rank P95实际为最大值，两个种子不是统计显著性证明。第一轮冷请求也保留在工作量与流量统计中。两个匹配对照都通过该阶段“P95至少改善10%、后台不恶化超过5%”的门槛。

每个部分前缀run五次恢复各9360tokens，累计external46800，计算输入来源由222144降至175344。H2D1,341,849,600bytes、D2H268,369,920bytes；NoHost两方向均为0。**相对NoHost，收益以额外迁移换取避免重复Prefill，不能声称所有资源成本同时下降。** 这也不能把DMA时间与重算时间的差简单相加作因果分解。

这些诊断包含每轮GPU reset和后台join，计时跨度不作为自然Agent Session E2E。来源：`a1nohost/a1fair/a2fair/a2nohost`原始requests、inputs、before/after.prom及analysis.json。

## 2. 多会话下，简单公平分配能否保住收益？

**保住了前缀与较小收益，但没有达到原先10%的速度门槛。** 两个互不共享前缀的会话交替6轮，总host仍256MiB。实际585个16-token物理CPU块分为292/293块，即4672/4688tokens，不让各会话分别占满整个tier。

| 会话 | NoHost P95 ms | 静态均分 P95 ms | 相对NoHost | 相对Full |
|---|---:|---:|---:|---:|
|A|1813.44|1661.16|−8.40%|−8.43%|
|B|1815.22|1653.75|−8.90%|−9.05%|

各自平均请求服务时间相对NoHost改善7.63%/7.98%，后台中位改善4.76%；没有牺牲另一会话制造单方加速。Full累计D2H9,920,053,248bytes但external=0；均分只写一次两个前缀，总D2H268,369,920bytes，再恢复累计46800tokens。串行Agent的逐请求指标证明：A五次各4672、B五次各4688，没有被对方覆盖。

该阶段为单次匹配筛选，仍有reset和join；诊断跨度不改称自然会话E2E。固定总容量下每会话留存深度减半，收益收缩是需要报告的tradeoff；不为跨过10%而事后调额度。

静态额度只对本次固定成员、非共享、稳定前缀的准入有效。`max_offload_tokens`不是通用物理配额/即时回收API，没有实现动态加入退出或共享前缀计费控制器。CPU native metadata审查验证了真实块保留、在途load引用、引用释放后替换已结束会话以及取消store回收。v1人工联合保护键场景保留；v2仅提交新请求自身键，使用真实A/B在途读引用保护，结果PASS。它不是GPU取消/抢占或内容一致性的完整证明。

## 3. 不强制清空 GPU 时，收益是否仍然存在？

**在受控自然驱逐压力下观察到约8.9%的完整会话收益；GPU-hot下没有额外收益。尚不能宣称稳定普适。** 本阶段完全不调用reset。四个独立闭环会话，每个6轮、轮间完成后等待0.2s；后台独立完成12个8192输入/256输出任务，无逐轮join。Session E2E取本会话第一次issue到最后一次done，包含工具间隔和资源竞争；这是synthetic closed-loop，不是真实生产trace。

全部同栈Compiled OnDemand。各组总36个实际输入、3456个规定输出严格匹配；GPU容量取服务器日志真实池大小。GPU-hot实际1,355,712tokens（约36.2GiB），受限两组74,896tokens（配置2GiB）。

| GPU / host | 方法 | Session E2E中位 ms | BG全部12任务完成 ms | GPU hit tokens | Host restore tokens |
|---|---|---:|---:|---:|---:|
|GPU-hot /256MiB|NoHost|10654.67|35171.43|576000|0|
|同上|静态均分|10824.81|34779.63|576000|0|
|同上|Full|11154.94|35011.84|576000|0|
|2GiB /512MiB|NoHost|44508.69|70270.17|0|0|
|同上|静态均分|40557.84|66169.61|41792|51808|
|同上|Full|44506.51|70101.54|0|0|
|2GiB /8GiB|NoHost|44422.65|70257.98|0|0|
|同上|静态额度（此时全存）|11095.74|35822.54|0|576000|
|同上|Full|10992.22|35089.49|8768|567232|

受控压力组，相对NoHost：会话E2E中位−8.8766%、后台makespan−5.8354%；四会话E2E分别−8.19/−13.71/−8.60/−3.43%。一个会话的续接P95反而+1.41%，故不是全指标改善。相对Full：会话中位−8.8721%、后台−5.6089%。两对都未达到主指标至少10%的预注册门槛，即使所有会话E2E及后台退化约束通过，也不改变阈值。

该组静态额度为4672/4672/4672/4704tokens。合计GPU+host复用93600tokens，来源统计计算输入由790272降到696672；来源统计的Agent历史重复Prefill由576000降到482400（−16.25%），不等于16.25%的端到端加速。H2D1,485,438,976、D2H536,739,840，总2,022,178,816bytes；Full写19,840,106,496bytes且没有host命中。NoHost不迁移，因此该方案相对NoHost增加CPU/GPU迁移；相对Full减少无效写入。并发指标快照有重叠，只以最终守恒做聚合来源结论，不虚构逐请求归因。

GPU-hot组GPU命中相同且host恢复为0，不能兑现host前缀；静态均分相对NoHost中位+1.60%，一个会话E2E+8.52%，触发5%负对照退化警示。只有一次并发运行，不能区分随机入队/数值执行波动与策略影响。不能用其余会话改善掩盖这项退化。

host8GiB时每会话额度74896tokens，大于本次上下文，因此所有实际store上限与Full完全相同，根本没有部分截断。静态策略相对NoHost会话中位−75.02%、后台−49.01%，**全部归因于普通完整Offload**；相对Full反而+0.94%/+2.09%，一个会话+5.46%触发负对照警示。两个相同有效策略的单次结果也提示并发筛选有波动，不能把小差异解释为算法成本或收益。

512MiB等host预算和2GiB GPU KV是人为受控资源点，不是这台A6000的物理内存限制。服务器约118GiB可用RAM，未证明实际部署必须承受这种host压力。自然GPU驱逐证据比强制reset更强，但仍不是自然生产host不足的证据。三种资源点不能混合计算算法加速；所有百分比都来自本配置匹配基线。

## 实现、安全性与失败

新增followup_screen/queue/analyze/export及fair_native_audit，复用原生store准入；静态物理块额度向下对齐、余数给最后一个成员。保留NoHost和Full回退，不引入pins、leases或逻辑reservations，没有强制释放在用块。旧calibrate和新runner均累计四个结果族预算，拒绝未解决RUNNING。

- 本轮16个GPU run全部COMPLETE、server_exit_code=0，全部规定工作完成，最终running/waiting=0；各run在30min以内，最长117.63s。
- 30项CPU测试通过；native v2 metadata生命周期检查通过。归档后本机独立检查全部payload SHA、长度、输出量、源计数、实际GPU池及H2D=external×28672。source守恒不是全面GPU tensor内容证明。
- 沿用之前oracle2单个batch-invariant输入的内容证据，没有新增多会话tensor内容oracle或Agent任务质量评价。并发数值输出可能变化；性能模式不与invariant模式混合。GPU取消/抢占、动态配额、无界公平性与共享前缀仍未全面验证。
- 本机pytest第一次26pass/1setup error，因沙箱临时目录不可写；指定workspace --basetemp后断言不变、最终30pass。环境既有cutlass_dsl .pth警告原样保留，未改环境。
- 运行尚未结束时分析正确拒绝，未将未完成预算作为最终数值。另一次本机tar data-filter路径检查失败；随后命令错误写出空目录0GPU摘要，现存tool_failures/invalid_empty_analysis_attempt1.json并标INVALID。修复空目录必须UNKNOWN/拒绝，添加测试；安全验证归档仅普通文件、无绝对/上级/链接路径后逐字节解压，再重算全部16run。有效测量未修改。

## 后续判断与研究价值

本轮没有产生新的复杂算法。证明了现成部分store接口可以胜过NoHost，且简单静态额度可避免交替相互清空；无reset下的小幅整体改善给出机制延续证据。然而整体主门槛未通过，适用范围只在GPU和host同时受限的受控点，真实host压力UNKNOWN。

复用已有组件仍然合理，不以每个部分原创作门槛。[此前文献比较](AUTONOMOUS_RESEARCH_HOST_PREFIX_20261010.md)仍适用：写入准入、连续前缀与公平尾部已有相关工作。本轮没有完整EfficientAgent/RR-Evict同栈对照，也不能声称超过它们；当前价值是简单组合的可复现实证。

下一技术决策：不在当前合成配置上调动态评分。若继续，应先用目标模型、真实上下文/会话并发估计GPU及pinned-host联合容量，证明合理host预算下存在写入浪费；与普通Full、NoHost和此静态基线对比，再决定需要何种深度选择。若可用host足以全存，直接保留完整Offload；若GPU足以留存，host恢复策略没有额外收益空间。当前工作没有要求改变项目最终目标，也没有证据足以进入大规模正式矩阵。

## 数据、资源与复现

结果目录：[host_prefix_followup_20261010](../results/host_prefix_followup_20261010/)；[analysis.json](../results/host_prefix_followup_20261010/analysis.json)、metrics.csv、budget.json、environment.json、diagnostics.json及manifest.json。每run保存execution、workload、完整日志gzip、requests.json.gz、inputs.json.xz、before/after.prom。xz无损原始payload；完整未压缩服务器原件保留。所有16run部署runner SHA为`7ef4ab4aea2d575485a8cc3adbe29e8f5ffe14bd9418eab6aec2d579e24c2b68`，冻结源码在provenance。code_git记录67eff77是审查起点，不是声称服务器干净HEAD；服务器仍dirty d403714，实际runtime是隔离round4-source，scheduler SHA在execution中。

本轮GPU0新增0.3645630389GPUh（约1312.43s），GPU1未用；含启动/编译/预热/执行/退出。保守累计0.2004867526+0.5462201082+0.1388070682+0.3645630389=**1.2500769679/4GPUh**，剩余2.7499230321。最终两卡19MiB/0%，8000端口空闲。CPU归档/复算不分配GPU。

```sh
# Ubuntu仓库；换一个未用过的name-prefix，不得覆盖历史run。
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source:/media/a6000/D/ZhaoBinHe/SlackKV
python -m experiments.host_prefix.followup_queue --stage a --name-prefix replay1- --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
python -m experiments.host_prefix.followup_queue --stage b --name-prefix replay1- --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
python -m experiments.host_prefix.followup_queue --stage hot --name-prefix replay1- --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
python -m experiments.host_prefix.followup_queue --stage pressure --name-prefix replay1- --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
python -m experiments.host_prefix.followup_queue --stage ample --name-prefix replay1- --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
# CPU原生检查：CUDA_VISIBLE_DEVICES='' python -m experiments.host_prefix.fair_native_audit --output <new-output.json>
python -m experiments.host_prefix.followup_analyze --directory results/host_prefix_followup_20261010
# 新复现组增加 --name-prefix replay1-；所有组的预算仍累计，不因过滤分析重置。
python -m pytest -q --basetemp .tmp/<new-test-directory> tests/test_host_prefix_followup.py tests/test_host_prefix.py tests/test_opportunity_decision.py
```

实际server argv、输入种子、调用时间及逐run预算以execution.json/原始requests为准。provenance中的冻结queue保持原运行顺序；预算没有通过新目录重置。
