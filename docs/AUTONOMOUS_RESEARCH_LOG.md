# 自主研究日志

## 2026-10-09：需求准入诊断

- 问题：历史 2 GiB KV 配置下 P95 Resume TTFT 约 6.4 秒，直接异步 KV 恢复仅约 60–84 ms。剩余时间的可优化部分尚未识别。
- 决策：先观察真实 allocator 检查，避免根据非原子的控制器快照推断容量因果。保持既有 BGv2 workload、K2 和 OnDemand，使用独立运行时副本，不改历史环境。
- 预期收益：本轮诊断本身不声称加速；确定容量失败、门控和请求竞争是否值得改变架构。
- 实现：可选 allocator 采样，结果变化即记录，否则每 gate 每请求至多每 100 ms 一次；保留完整日志、异常和保守 GPU 时间。默认关闭，不改变分配结果。
- 协议：40 个 Agent 请求、32 个续接边界、固定 4×2048 后台输出，绝对时钟等待，GPU KV 2 GiB；单项上限 30 分钟，本轮上限 4 GPU·小时。诊断与性能验证区分，未被调用的 allocator 检查为 UNKNOWN，不能由头部失败外推整条队列。
- 运行命令：`python -m experiments.autonomous_admission.run --name diagnostic-k2 --trace`，在服务器项目根目录，指定现有 venv Python。
- 启动失败：首次因绝对时钟函数名引用错误，在加载模型前退出；已修正并保存失败日志，未消耗 GPU 推理时间。
- 诊断有效：40 请求/32 边界完成，P95 Resume TTFT 6459.75 ms；28/32 续接在首次准入前观测到容量失败，无缺失 allocator 观测。253 个失败样本全部 FULL_SEQUENCE，失败时 reservation 均为 0；无 preemption、OOM、lease 泄漏。后台 8192 tokens 完成，wall 133372.65 ms。观测支持真实容量阻塞，不能据此量化 HOL 的因果贡献。
- 成本：含启动/退出的单 GPU 保守计时 169.52 秒（0.04709 GPU·小时）。

## 2026-10-09：有限容量回填，预先冻结筛选协议

- 假设：某些队头因完整 KV 容量不足而无法准入时，后面较小请求可能仍然能够实际分配。有限回填或可减少闲置容量；与已有 backfilling 调度相近，暂不主张新颖性。
- 实现：仅在真实 `allocate_slots` 返回 None 后跳过；同一容量失败请求最多消耗 2 次额外成功准入，每个 scheduler tick 最多一个回填；计数跨 tick 保留，头部准入/取消即清理。保留正常 full-sequence/reservation、K2、remote 状态和实际请求所有权检查；有 lease 时关闭额外绕过，不用于 FullSlackKV。有限回填不等于普遍的 liveness 证明。
- 预注册：先执行未采样的 OnDemand 和回填各一次，固定 K2/2 GiB/BGv2。只有 P95 至少改善 5%、session median 不恶化超过 2%、后台 wall 不恶化超过 3%，且所有现有正确性检查通过，才扩展反序复验；否则停止该方向，不调阈值或负载挑结果。两次为筛选，不构成稳定显著性证明。
- 文件：`backfill.py`、`backfill.patch`、`run.py`；新增跨 tick 上限、无效扫描和取消清理测试。19 项独立分析/状态测试通过。
- 运行命令：`python -m experiments.autonomous_admission.run --name screen-ondemand --backfill 0`，随后同条件 `--name screen-backfill --backfill 2`。诊断补丁在两者均关闭采样；调度补丁默认 limit 0，只改变候选的回填开关。
- 状态：准备进行真实系统筛选。
