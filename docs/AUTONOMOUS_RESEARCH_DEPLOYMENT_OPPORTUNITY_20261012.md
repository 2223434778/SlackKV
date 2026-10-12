# 部分前缀的部署机会审计

2026-10-11启动，2026-10-12恢复执行；基线 `834ed5a`。回应 `RESEARCH_REVIEW_FOLLOWUP_20261011.md`：本阶段判断合理资源下是否存在机会，不重新解释前轮10%性能门槛。

## 结论与决策

**目前没有足够证据把部分前缀升级为自然部署中的端到端优化。** 已完成真实来源的 CPU 工作集审计、模型配置容量计算和服务器资源检查。1.5B 小模型当前的合理 GPU/host 容量尚无双重不足证据；32B TP2 长上下文是更有压力的可行性候选，但实际可用 KV 池、host 成本、需求和延迟均未实测。停止在任意512MiB host上扩展复杂算法，保留既有部分前缀机制证据和完整 offload 强基线。

这是证据边界，不是“部分前缀无价值”的负结论。当前真实部署价值仍为 **UNKNOWN**。原始JIT问题和项目最终目标未改变；没有新增算法加速声明。

## 1. 新的测量是什么

复用固定revision的Weka文件前缀，先核对完整源SHA256：`d078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994`。15条完整trace、2661请求、27个独立trace/model。没有将各trace从零开始的时间轴叠加；输入身份采用完整前缀trie节点，不仅比较末块hash。

- **累计唯一输入**：全部历史输入prefix节点，不能直接用作同时驻留量。
- **活跃输入峰值**：endpoint `[t,t+api_time)` 内输入prefix union。不是A6000真实计算并发。生成输出没有哈希身份，不能假装可共享；另保存全输出预留数值诊断。
- **active + future-needed union**：已经完成、之后仍有输入到达使用的块与活跃输入union。用离线未来信息保留全部这类输入，属于无驱逐保留目标，不是在线策略、最优容量下界或实际缓存命中。
- **intervening input footprint**：同container/model完成后到下一次到达前发起的输入prefix union。container不等于真实session；该量不包含输出增长，也不等于实际cache miss。

| 独立namespace指标 | 中位 | P95 | 最大 |
|---|---:|---:|---:|
| 累计唯一输入 proxy tokens | 222720 | 1454080 | 2760192 |
| 活跃输入峰值 proxy tokens | 165248 | 704768 | 769344 |
| active + future-needed 峰值 proxy tokens | 165248 | 800000 | 1332032 |
| endpoint inflight请求数峰值 | 2 | 5 | 5 |

只有3/27个namespace的最后一项输入union容量峰值大于活跃输入峰值，比分别约1.172/1.578/1.731；这不表示另外24个没有idle KV，只表示离线保留目标没有增加它们的整个trace峰值。累计最大276万与活跃最大77万差距大，否定“累计输入直接证明并发容量不足”的分析方式。

**不能将此表乘Qwen bytes/token当作真实Qwen部署：** source使用64-token proxy blocks，原模型为Claude；93.05%的请求代理长度超过旧Qwen32K设置。文件前缀不是随机生产样本，endpoint耗时不是目标硬件计算成本，也没有真实工具DAG或完整跨trace到达过程。

## 2. 模型与资源核算

FP16 KV/token = `2 × layers × kv_heads × head_dim × 2 bytes`。配置与索引元数据已归档，未下载32B权重。

| 项目 | Qwen2.5-1.5B | Qwen2.5-32B |
|---|---:|---:|
| KV bytes/token（全模型） | 28672 | 262144 |
| 32K不共享输入KV | 0.875 GiB | 8 GiB |
| 8个32K输入 | 7 GiB | 64 GiB |
| 16个32K输入 | 14 GiB | 128 GiB |
| 32个32K输入 | 28 GiB | 256 GiB |

1.5B既有compiled/natural85% GPU池实测为1,355,712 tokens/replica，输入容量约36.2GiB，可容纳41个完整32K输入的算术等价量；这不承诺41个并发的输出增长、SLO或背景吞吐。32个输入28GiB在单卡输入池内，64个输入56GiB host也低于当前空闲RAM。实际后台、输出增长和服务目标必须另外计入，不能因此宣称所有1.5B工作负载无需offload。

32B官方revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`，权重索引payload65,527,752,704bytes≈61.0275GiB。双卡各49,140MiB、85%资源减权重，剩余最多约20.5526GiB **总KV上界**；尚未扣activation、CUDA Graph、通信、临时buffer，不能当成已测allocator capacity。TP2 KV每rank128KiB/token；仅从上界看，3个完整32K输入24GiB已经超出。该目标值得资源可行性验证，但模型未缓存、真实吞吐未验证。

16个32K输入128GiB完整host副本超过当前idle MemAvailable，并不能证明整个GPU+host层级一定放不下：exclusive placement可避免hot重复副本，full offload则可能保留GPU/host重复块；CPU模型加载、runtime headroom也未测。没有足够的真实16会话长上下文需求，因此不把这个人为枚举点当生产瓶颈。

官方来源：[32B配置](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/config.json)、[权重索引](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/model.safetensors.index.json)。仅是容量算术，不是新算法性能。

## 3. 合理host预算与实现语义

2026-10-11快照：MemTotal134,919,331,840bytes≈125.65GiB；MemAvailable127,637,286,912bytes≈118.87GiB。其他主要RSS为桌面、mysqld、远程桌面等；cgroup memory.limit_in_bytes为很大哨兵值，没有发现512MiB或其他严格研究进程上限。可用内存不代表可无成本pin满，也不支持凭空设小host预算。

核对当前native源码：`cpu/spec.py`以本rank bytes/block乘world_size计算全局block大小，`num_blocks=floor(cpu_bytes_to_use/global bytes/block)`；各worker只分配本地page。因此这个参数已经跨TP worker聚合，不能再乘TP2。默认worker走`torch.zeros(...,pin_memory=is_pin_memory_available())`，预先分配并清零；需要真实RSS/系统内存测量。源码路径和SHA已保存。

为此定义32GiB单次pinned-host probe：4 CPU threads、60秒watchdog、无模型/无GPU tensor、起始RAM至少80GiB，48GiB剩余仅是这次安全guard，不是生产预算。检查实际pin属性与首尾零值，记录分配、del+GC后RSS；VmLck/VmPin为零不能证明未pin。保守按1GPU计全进程，后续runner已纳入连续预算。

**probe没有启动。** 额度恢复后2026-10-12服务器仍可SSH、GPU两卡16MiB/0%，但原`/media/a6000/D`不存在；lsblk显示两块3.7T ext4磁盘未挂载，原repo/runtime/source均不可访问。第一次upload失败；只尝试创建不存在的项目临时目录，因父挂载点不可写失败，没有改系统配置。未挂盘、未sudo、未安装替代环境、未在缺失历史账本下运行GPU。已请求恢复原研究盘或提供新路径。该环境问题与算法失败分开记录。

当前服务器可用RAM比前一快照更大，是系统状态变化，不能混成一次测量。32GiB pin是否成功、代价和释放均为UNKNOWN。

## 4. 实现、验证与失败

- 新CPU audit、容量算术、官方metadata-only获取器及未执行pin probe；未改serving算法/生命周期。
- 新测试涵盖共享前缀身份、half-open事件、完成后可用性、输出预留、零时长、model隔离；固定随机事件与独立区间集合参考逐点一致。
- 共38个相关测试通过，含前轮host-prefix/机会判断回归。更新预算测试，确保新probe family计费且RUNNING阻止后续启动。
- 官方HTTPS第一次因本机sandbox `[WinError 10013]`失败；按授权使用正常network escalation成功，固定revision和文件SHA。没有模型下载。
- 初次定位vLLM源码路径不存在，后在已知isolated source找到并审计；初次lsblk使用当前系统不支持的MOUNTPOINTS，改用MOUNTPOINT成功。均不是GPU实验失败。
- CPU raw proxy input未重提交；固定source manifest在既有results下。新逐时间事件和27 namespace汇总JSON可复算。

## 5. 相对强基线有什么新增性能证据

**本阶段没有新GPU性能对照，新增加速数值为UNKNOWN。** 沿用前轮结论：单会话部分前缀胜NoHost约18–20%；两会话保留前缀但约8.4–8.9%P95；无reset受控四会话E2E约−8.88%，未过10%门槛。GPU-hot无机制收益；充足host的ordinary Full显著更好。这里没有重测8.9%可重复性，也没有把CPU分析视为支持稳定加速。

正确性方面只有CPU数据审计/既有回归，未补GPU内容oracle、取消或性能公平性实验。没有新请求调度改变，所以也没有新的公平性收益声明。潜在创新仍是容量受限下可持续部分检查点与写入准入/公平保留的组合，已有RR-Evict/EfficientAgent/MORI等相关方法不能省略；本阶段没有扩大新颖性主张。

## 6. 下一技术决定

保持静态机制和Full/NoHost强基线，优先恢复已有研究环境、完成真实host成本测量。然后根据合理host空间与实际target工作集决定是否值得32B TP2或其他长上下文配置验证；不预设一定需要partial，不为10%调小预算。若Full在可接受内存成本下已足够好，接受有限工程结果；若要转成内存效率目标，另行预注册SLO及公平性条件，不能事后追认旧结果通过。

目前最重要不确定性是：**有没有可验证的目标部署需求，使GPU工作集超出自然池、合理pinned-host容量也不足，而且Full/NoHost的系统关键路径确实值得partial优化？** 服务器路径恢复是测量的外部依赖，不是研究目标变更。

## 复现与归档

结果：`results/deployment_opportunity_20261011/{footprints.json,capacity.json,server_inventory.json,environment_block_20261012.json,qwen32b_metadata/}`；协议：`docs/DEPLOYMENT_OPPORTUNITY_PROTOCOL_20261011.md`。

```powershell
python -m pytest -q --basetemp .tmp/pytest_deployment_repro tests/test_deployment_opportunity.py tests/test_host_prefix_followup.py tests/test_host_prefix.py tests/test_opportunity_decision.py
python -m experiments.deployment_opportunity.analyze --input .tmp/opportunity_trace_prefix.jsonl --expected-sha256 d078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994 --output results/deployment_opportunity_repro
python -m experiments.deployment_opportunity.capacity --directory results/deployment_opportunity_20261011
```

原始文件获取见`experiments/opportunity_decision/fetch.py`与既有source_manifest。metadata重新获取会解析当时main，精确复核本次应使用已冻结revision URLs和manifest SHA。本阶段新增GPU **0**，连续预算仍为 **1.2500769679056248 / 4 GPUh**。没有重置预算；pin脚本未启动不计作GPU成功实验。
