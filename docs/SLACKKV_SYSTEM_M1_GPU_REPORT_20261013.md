# M1：授权 GPU 验证与动态机制筛选

基线 `efb74ae`；用户在交付后明确批准新增 **0.5 GPU·小时**。
本轮完成代码修复、CPU/native 验证、真实 GPU 内容检查和匹配性能筛选，M2/M3 保持关闭。
结果目录：`results/system_m1_smoke/`；最终源码以本报告所在提交为准，实际执行版本有独立快照及 SHA。

## 结论

**M1 的正常恢复/结束路径在本次真实模型 smoke 中通过；动态借用及退出扩额确实生效，但尚未证明稳定优于强 Full 基线。**
两组 M1/Fair 的 median 会话 E2E 分别 **+2.83%、−6.01%**，首请求成本是重要混杂因素，原负结果完整保留。
两组 M1/true NoHost 分别 **−9.20%、−10.00%**；不是统计显著或生产泛化结论。
重复 M1 与单次 Full 的 median 接近（−0.58%），不能据此宣称优于 Full。
整个协议中 M1 多出一次未被重用的终止代写入；缓存管理收益与迁移代价须一起评价。

## 环境、输入和执行范围

- 已有 Qwen2.5-1.5B-Instruct revision `989aa798...`，FP16、TP1、RTX A6000 GPU0；vLLM0.23.0 / Torch2.11.0+cu126，native Python overlay。
- 编译及 CUDA Graph 原路径，未使用 enforce-eager。GPU KV512MiB，实际池 **18,720 tokens**；model limit18,432。M1/Fair/Full host224MiB、512blocks，28,672bytes/token；NoHost 不创建 connector。
- 两个独立首块前缀会话：长8192、短1024；三个16384-token 一次性压力请求。串行 closed-loop、0.15s间隔、每请求16输出，不强制清空 GPU。
- 性能序列固定12请求、108,544输入/192输出 tokens，含末尾同代迟到和新代诊断。会话 E2E 定义为首请求 issue 到各自显式结束请求完成；末尾诊断另计，整体迁移流量不丢弃。
- 正确性 run 单独使用 batch-invariant，13请求/208输出；只对一个恢复长输入的16个 token IDs 建立独立冷 prefill oracle。性能模式不使用 invariant，M1/Fair 详细状态 trace 均关闭，source trace规则相同。
- 这是受限合成机制 smoke：不是真实生产 trace，也不是并发多Agent/后台公平性或SLO并发容量测试。StaticFair 为两个预配置Agent各4096-token固定cap，不借短需求剩余、不根据未来长度调整。

## 失败、最小修复和真实性能 gate

| run | 结果 | 消耗 GPUh | 原因与处理 |
|---|---|---:|---|
| m1_content1 | FAILED，0请求 |0.02012536|子进程PATH缺conda bin，FlashInfer找不到已有ninja；修复PATH，新增预检和环境回归|
| m1_content2 | FAILED，0请求 |0.00975751|隔离代码目录缺旧scheduler的3个实验模块；补齐同版本依赖，CPU导入Scheduler/AsyncScheduler通过且CUDA未初始化，相关算法开关为0|
| m1_content3 | FAILED，13请求 |0.01314655|host restore及cold IDs相同，但无物理状态trace，不能满足完整gate；不事后改为PASS|
| m1_content4 | PASS，13请求 |0.01313582|logger从外部模块层级改为vllm配置层级，新增真实INFO可见性回归，KV策略未改变|

环境 .pth `nvidia_cutlass_dsl` 警告未阻断最终运行，未安装/修改共享环境。退出时 semaphore cleanup 警告原样保留。
全部11个GPU run均退出且显存释放已确认；未终止无关服务。一个 Full 启动 SSH超时后确认没有execution/launch记录才重试，没有重复GPU占用。
最终本地75项相关回归通过，native16组（含120次交错）通过；历史失败/通过证据未覆盖。

真实 content4：

1. 长请求先保存512块；短请求加入后长保留448、短64，确认只回收合法尾部。
2. 普通压力后长续接：local1264 + host5904，确有 H2D；独立cold local/external均0，16输出IDs一致。
3. 短会话结束后长目标512、实际暂为448；下一正常请求实际补到512，未重置旧cursor或额外重算填池。
4. 每个状态快照 `ready + pending_store + free = 512`；结束同代迟到请求记录明确拒绝写入，仍可正常使用GPU命中。
5. 更高generation通过正常GPU已存在KV保存512块，不需清空GPU或人为Transformer forward；最后结束后members/待完成批次空，512块全部free。admitted/reclaimed各1152块。

这验证了被测正常模型路径，不等于KV tensors逐字节核验、所有输入/布局/TP2验证、任意GPU取消或在途end重叠证明。所有语义marker都是CEDAR17，未来须更强的不同身份/内容oracle；本次不夸大其检测范围。

## 性能结果：保留所有首请求

| 方法 / 次序 | 长E2E s | 短E2E s | Median s | 首请求 ms |
|---|---:|---:|---:|---:|
| M1 1 |5.990|1.751|3.871|1124.36|
| Fair 1 |5.671|1.857|3.764|454.22|
| Full 1 |5.333|1.773|3.553|458.14|
| NoHost 1 |6.566|1.960|4.263|1113.95|
| Fair 2 |5.666|1.851|3.758|451.85|
| M1 2 |5.314|1.751|3.532|462.91|
| NoHost 2 |5.895|1.955|3.925|450.03|

第一组 M1/Fair 长 +5.63%、短 −5.71%；反序第二组长 −6.21%、短 −5.43%。每次均完整计首请求，没用扣除成本的counterfactual替代实测。
首轮M1准入约26ms、Fair约23ms，额外约0.67s主要在准入之后；重复后消失，NoHost也有相似现象。支持首次真实请求成本需要控制，**具体JIT/缓存归因仍UNKNOWN**。
没有事后修改 workload、参数、阈值或删首请求。反序及NoHost重复用于检查这一已观察到的混杂，不开展大规模矩阵、不挑最好run。

机制证据与整体结论分开：long3两方法local同为2304，M1 host5888、Fair1792；有效cache覆盖8192 vs4096。第一组TTFT42.47 vs186.12ms，动态扩额减少后续输入重算。
这仍未证明整体普遍胜出。Full 是相同224MiB host预算的 compiled native OnDemand 强基线；相对它没有明确性能优势。
三个压力请求E2E约825–836ms；各请求计时及首token到响应结束窗口保存。窗口只是proxy而非TPOT，且没有并发后台服务，不能由此宣布后台公平性已证明。

## 迁移、资源与不利结果

| 方法 | prompt source compute tokens | 真实host恢复tokens | 全协议D2H MiB | 全协议H2D MiB |
|---|---:|---:|---:|---:|
| M1（两次相同）|60448|18736|504|512.3125|
| Fair（两次相同）|70704|8480|140|231.875|
| Full|61440|17744|336|485.1875|
| NoHost（两次相同）|79184|0|不适用/计数缺失|不适用/计数缺失|

所有源计数 compute/local-hit/external 总和与108,544输入严格吻合；有connector的真实H2D bytes严格等于恢复tokens×28,672。
source compute是输入来源计费，不是Transformer forward调用数。NoHost external0有直接计数；迁移labels不存在仍保留null，不凭空填0。
M1更充分使用host容量，较Fair减少10,256输入重算tokens，但迁移量显著增大；不把局部TTFT优势包装成免费收益。
M1/Fair/Full始终同一224MiB物理host池，未证明RSS或active GPU footprint下降；GPU KV memory-time未测。

重要额外成本：实际会话结束时 M1 D2H **280MiB**、Full336MiB；但M1在最后的新代、立即结束的诊断请求又写 **224MiB**，全协议变504MiB。此写入本次没有后续重用。
这是“保留目标”未利用已知终止信号的机会；后续可研究停止不必要terminal optional-store，但须保留当前请求的合法read和原生引用结算。**本轮未实现这个新策略，也未删诊断流量改善数字。**

## 资源和下一步判断

新增 **0.13476591384806347 / 0.5 GPUh**（11个run含失败、启动、编译、运行、退出和每run计费裕量）；历史4.191988019727527，连续 **4.326753933575591 GPUh**。
剩余额度未消耗，本轮GPU已停；不重命名/重置账本。最终两卡均19MiB/0%，owned groups清理确认。
324原始文件长度/SHA核验，日志、prom和inputs压缩存档；`analyze_smoke`已在压缩-only数据独立复算，`analysis_verified.json`可定位全部数值。
NoHost旧records.params包含未发送的构造诊断dict，实际wire没有kv_transfer_params；保留原记录并在provenance_notes说明，未来runner修为记录实际payload/None。这是记录修复，未改已执行请求。

下一步优先：先预注册等价首请求warmup/执行状态，再评估M1 vsFull；结合当前GPU命中和明确terminal信号研究无用写入准入，随后处理常见共享prefix覆盖。
不直接启动M2/M3、不把max-min或CUDA Graph当创新；当前价值是可运行的安全组合机制及可测代价，论文级新颖性/稳健收益仍待更强工作负载证据。
应保留项目目标，停止盲目增设机制。当前最重要不确定性是：在强Full且正常warm serving、多Agent及共享prefix条件下，动态前缀是否有足够端到端净收益。
