# Deployment opportunity CPU audit — frozen before execution

基线 `834ed5a`；不调整前轮10%性能门槛。本阶段检验自然部署机会，不进行性能筛选，也不把 proxy trace 映射当作 GPU 实验。

- 输入：既有15条完整 Weka 文件前缀，SHA256 `d078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994`。保留 producer-filtered/nonrandom、proxy token、非真实 Agent DAG 限制。
- 单位：每个 trace/model 单独审计。共享身份为完整前缀 trie 节点，不能仅凭最后一个 hash 共享，也不能把所有 trace 的零点叠加成生产并发。
- active：endpoint `[t,t+api_time)` 内的输入前缀 union；另记录全输出预留数值诊断，不将其当成 GPU 实际占用。
- future-needed：已经完成且之后还有输入到达使用的块。保留全部这类块与 active 的 union，代表离线知道未来、无驱逐、input-only 的保留目标，既不是可执行策略，也不是最小容量下界。
- intervening footprint：同 container/model 完成后到下次到达前，期间发起的输入 prefix union。不宣称真实 reuse-distance cache miss，也不把 container 等同真实会话。
- 容量：从实际模型 config 计算 FP16 KV bytes/token；记录当前服务器 RAM、其他服务及 cgroup，核对 native host budget 是否已经跨 TP rank 聚合。较大模型只作资源可行性条件分析，不下载权重、不声称延迟。
- 决策：若合理 GPU/host 配置可承载目标工作集，则优先普通 GPU/Full；若 proxy/目标模型和到达过程无法可靠映射，则部署价值 UNKNOWN，停止在任意小 host cap 上搜索算法。只有可信双重压力才扩大 GPU 对照。
- 资源：本阶段新增 GPU 0；连续已用 `1.2500769679056248 / 4 GPUh`，不重置预算。

执行 CPU 审计后新增一项预先定义的资源诊断：32 GiB `torch.zeros(pin_memory=True)`，4 CPU threads、单次60秒watchdog、无模型/无GPU tensor。安全门要求两卡空闲且起始 MemAvailable ≥80 GiB，48 GiB剩余为本次安全guard，**不是生产预算**。检查 `is_pinned` 与首尾零值，记录RSS/系统内存/删除后缓存行为；VmLck/VmPin为零不能推断未pin。保守按1 GPU计入整个进程时间，并纳入后续两类runner连续预算。
