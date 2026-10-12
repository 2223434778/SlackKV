# SlackKV M1：动态 Host 前缀与原生生命周期验证

日期：2026-10-12。实现基线：`f2d6af0`，以本报告所在提交为最终代码版本。
主架构：`SLACKKV_SYSTEM_ARCHITECTURE_20261012.md`，特别是第 9 节。

## 交付结论

M1 已有可加载的最小实现和真实 vLLM CPU/native 元数据验证，尚未完成模型/GPU验证。
本地 64 项测试通过，其中 12 项是新增控制器测试；Ubuntu native 扩展审计通过 12 组场景，包含 120 次生命周期交错操作。
这些不是 120 个 GPU 请求或性能样本。两版审计均通过，没有通过削弱断言掩盖失败。

本轮新增 GPU 消耗为 **0 GPU·小时**。连续历史账本仍为 **4.191988019727527 GPU·小时**；此前 4.5 的例外仅用于上一组三方法对照，未挪用其余额。
M1 相对 Compiled OnDemand / StaticFair 的 TTFT、E2E、吞吐、后台公平性和 KV 内容正确性均为 **UNKNOWN**。
历史 32B StaticFair 的整体收益是启动这一机制的依据，不能作为动态 M1 的实测结果。

## 实现与设计选择

- `slackkv/m1/controller.py`：按实际已知输入前缀需求进行整数 max-min 分配，轮转余数。短需求先饱和，剩余容量分给长需求；总需求装得下时可完整保存。不是任意时间保证填满物理池。
- `slackkv/m1/native.py`：扩展真实 CPUOffloadingManager，将逐批 store 准入和尾部回收放在同一锁下。逻辑控制器只管理身份与目标，不直接修改物理缓存。
- `slackkv/m1/spec.py`：通过原生 OffloadingSpecFactory 的 `spec_name` / `spec_module_path` 加载，继续使用原始 worker、传输 handler 和按需恢复。
- `experiments/system_m1/install.py`：创建独立 native Python overlay，仅在两处更新 offload keys 后加入可选 observe hook。未改 GPU allocator、attention、scheduler 优先级或 DMA。原基线源码和历史实验保持原样。
- `experiments/system_m1/native_audit.py` 与 `tests/test_system_m1.py`：验证控制器及实际 native manager、任务构建器、Factory、TP 完成汇总。

不创建未物化的长期 reservation。请求首次有效观察时冻结 session/generation/native hash 路径和准入 ceiling；每批仍使用当前配额、真实净新增量和已确认空闲槽位。
例如 A 原 ceiling=10、仅写入 2 块，B 加入后各目标 5：A 后续最多补到 5，不能靠旧 ceiling 把 B 的份额驱逐。

只有 native `ref_cnt == 0` 的尾块可回收；`-1` 的 pending store 和正引用的 load 块都受保护。最深块受保护时停止该尾部回收，不跨过它删祖先。
写入失败产生孤立后缀时，先等引用结清，再按最深到最浅安全回收。移除同时更新原生 policy、free list 和移除事件。
每次快照验证 `ready + pending_store + free = capacity`；load 保护是 ready 的子集，不重复计费。

旧 generation 的有效完成仍结清 native 引用，不能因逻辑过期忽略。使用 request-context 身份和批次 keys 跟踪待完成任务；重复旧通知不会污染新物理槽。
TP 完成继续经过原生 pending_count 汇总：第一 worker 返回不将全局 pending 块改为 ready。
尾部缩额不重置 `next_stored_block_idx`；当前请求错过的深度允许在下一正常请求重建，不追加人为重算。

## 实际验证

| 验证 | 结果 | 证据 |
|---|---|---|
| 控制器与历史相关回归 | 64 passed，0.39 s | `results/system_m1_20261012/local_validation.json` |
| 初版 native 审计 | PASS，8 组，27.55575 s | `native_audit_v1.json` 与对应日志、冻结源码 |
| 扩展 native 审计 | PASS，12 组，6.68947 s | `native_audit_v2.json` 与对应日志、冻结源码 |
| 隔离安装 | COMPLETE，disabled-hook AST 等价 | `installation.json` |

扩展审计覆盖：

1. 旧 10-block plan / 新 5-block 目标的实际逐批准入。
2. deepest load 保护、延期回收和严格尾部移除事件顺序。
3. pending-store `-1`、旧 generation、迟到与重复完成的物理结算。
4. 同 native 身份的新 generation 等旧引用结束，在下个正常请求重建；文本 request ID 复用仍隔离。
5. 祖先写入失败、孤立在途后缀的延期债务与最终释放。
6. 共享/未知请求停止可选写入，仍可通过原生身份读取已有合法 KV。
7. 短需求 1 / 长需求 10 在 10 块池得到 1 / 9；结束短会话后长会话下个请求可增到 10。
8. 120 次真实 manager 交错操作，取消/失败/加载保护/generation 变化后回收全部 13 个槽位，contexts 和 plans 归零。
9. 跨线程 membership 事件被拒绝；native reset 后迟到通知不破坏槽位。
10. 真实 Factory 与两个 scheduler hook；TP2 元数据按全局 40 MiB / 4 MiB 每块得到 10 块，所有 worker 完成才 ready；关闭 M1 返回原始 CPU manager。
11. 真实 store-job builder 的部分 src/dst 选择、原生 cursor 保留、后续批次不写洞、下一正常请求补齐，以及未调用首次 lookup 时的第二 hook。
12. 非 LRU / sliding attention 在创建 handler 前被拒绝。

Ubuntu 使用 vLLM 0.23.0 原生源码、Torch 2.11.0+cu126；`CUDA_VISIBLE_DEVICES=''`，审计额外禁止 `torch.cuda._lazy_init`。
请求输入是合成的 Python 元数据对象；native RequestOffloadState、SchedulerOutput、Factory、CPU manager 和任务构建器是真实代码。
TP2 测试验证汇总和容量元数据，不是启动两个 GPU worker。没有模型 forward、GPU tensor 或 DMA，也没有证明 KV 内容正确。
运行时间只是审计 CPU wall time，不用于算法速度比较；第二版 import/cache 状态不同。

## 配置和调用契约

先运行 `results/system_m1_20261012/commands.sh` 的 CPU 安装/审计部分。安装目标必须是新目录，脚本不覆盖历史 native tree。
测试 overlay：`/media/a6000/D/ZhaoBinHe/slackkv-system-m1-native`；代码模块通过 `PYTHONPATH` 另行加入。
开启 M1 的 KV transfer 配置示例（**没有执行 GPU serving**）：

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "SlackKVM1Spec",
    "spec_module_path": "slackkv.m1.spec",
    "block_size": 16,
    "cpu_bytes_to_use": 67108864,
    "slackkv_m1_enabled": true,
    "slackkv_m1_trace": true
  }
}
```

`cpu_bytes_to_use` 沿用原生全局 TP 预算语义，不能每 rank 再扩大一次。实际池块数由模型布局决定；上述 64 MiB 是 smoke 示例，不是性能资源设置。
客户端在普通请求 `kv_transfer_params` 中附加：

```json
{"slackkv_m1": {"session_id": "agent-01", "generation": 0, "end_after_request": false}}
```

相同会话串行续接沿用 generation；最后一个普通请求设 `end_after_request: true`。工具等待中的单请求完成不会删除会话。
复用/重写会话需增加 generation。无后续请求的退出只能由 scheduler 线程调用 `manager.end_session(...)`；本版没有新增 HTTP 事件服务。
首次普通请求观察是 join，不依赖未来长度。不能从外部线程直接更新成员；RLock 不意味着原生 lookup→load 路径支持任意多线程并发。
`slackkv_m1_trace` 默认关闭；开启后输出 `SLACKKV_M1_STATE` 的目标、实际连续深度、pending/read 引用和延期债务，不增加网络上报。

启动回退：`slackkv_m1_enabled: false` 使用原始 CPUOffloadingManager；原 Full / NoHost / StaticFair 配置不改默认值。
这证明 Factory 的关闭路径与 hook 移除后的 AST 等价，不能保证新增 getattr 检查完全零性能开销。没有实现有在途任务时任意热切换到 Full/StaticFair。

## 限制、风险与异常

- 首版支持一个 full-attention group、LRU、原生 TP；DCP/PCP/PP、Eagle、sliding/chunked attention 和附加 store-threshold filter 不支持。
- 同 generation 并行分支、输入重写/截断和跨会话共享尚无完整公平计费：停止该请求可选写入、记录原因，保留正常 native read/compute。共享块不复制计费，不通过 session 盐消除共享。这是保守覆盖限制，不是完整共享优化。
- shared/rewritten/parallel fallback 在该 generation 内保持；新 generation 可重新加入。旧物理引用仍可导致临时不能写入，需要后续正常请求重建。
- 未显式结束的会话会继续参与公平分配；没有 TTL、预测工具完成或自动不活跃淘汰。最新 generation tombstone 按历史 session ID 增长，生产级 metadata GC 尚未实现。
- 已知输入短需求可借出容量，但 frozen ceiling、原生 cursor、保护引用和回收欠账可能让物理池暂时未填满。手动 legacy `max_offload_tokens` cap 仍受尊重，但控制器需求取实际输入，不针对外部 cap 做进一步 work-conserving 规划。
- 管理器适配使用已核验 native 私有 policy/free 接口，升级 vLLM 必须重新审计；安装器检测两个 anchor，不满足则失败，不猜测新 API。
- max-min 只代表缓存资源公平，不保证延迟公平、无后台代价或整体加速。optional store 被跳过不阻塞正常推理；真正模型级活性仍需 GPU验证。
- 期间一次工具自动审批服务因 429/retry 上限未执行 native 命令；未绕过审批，稍后同范围重试成功。不是 native 实验失败，也没有 GPU 消耗。额度恢复不视为新 GPU 授权。
- 两次 native 审计均 PASS；未发生需要隐藏或丢弃的失败 run。源码 SHA、日志和版本快照全保留。

## 下一步的具体 smoke（尚未执行）

建议新授权 **0.5 GPU·小时** 上限，单卡、每 run 最长 10 min，包含模型加载/编译/退出和失败成本；不足则停止，不改名重置账本。
先用已有 Qwen2.5-1.5B compiled 路径、单张 A6000、固定 GPU KV budget，host 小池按真实 native block size 换算。
固定同输入及生成 token 数，分开跑 M1 / StaticFair / Full / NoHost，不把这个正确性 smoke 当正式性能矩阵。

优先场景：短/长串行会话加入和显式结束、缩额遇到实际 load/store、取消后继续；不强制清空 GPU。用状态日志核对实际回收、迁移及无 orphan/reference 泄漏。
对一组必要输入另用 batch-invariant 模式和独立冷 prefill 比较逐 token；该 correctness 配置不混入普通性能对照。
预注册最低验收为请求完成、原生物理计数守恒、无内容/身份错误、无保护块提前释放。失败先修生命周期，不扩性能评价。
通过后再用非均匀长度/动态成员匹配 StaticFair，验证 M1 独有的借用与退出能力；不能只与 thrashing Full 比较就声称创新或 SOTA。
M2 仅补测量显示确实需要的压力反馈；M3 预取继续关闭。当前完成的是 M1 CPU/native 里程碑，不提前开展下一阶段。
