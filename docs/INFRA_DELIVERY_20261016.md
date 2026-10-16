# 工程交付：E1 → E2 → E3

**已交付可运行的单机 infra 原型。** StaticFair 为主线，NoHost/Full 为配置对照；真实多轮、本地工具、生命周期、受管进程、可信报告与固定用途 benchmark 已打通。目标是能运行和解释核心资源管理，不以论文创新或所有负载都更快为验收标准。

## 从哪里开始

1. 阅读 [README](../README.md)，先跑 CPU doctor、单测和保存记录的报告重建。
2. 按 [操作指南](INFRA_OPERATIONS.md)接入受测 Linux/runtime/model，再运行真实 demo。
3. 沿 [架构](INFRA_ARCHITECTURE.md)阅读代码，用 [动手清单](INFRA_HANDS_ON.md)形成自己的操作记录。
4. [讲解材料](INFRA_INTERVIEW.md)提供表达模板；只描述自己实际理解和完成的贡献。

## 实现与提交

| 阶段 | 交付 | 验证/状态 |
|---|---|---|
| E1，`b6773c7` | 打包/CLI、集中配置、固定 runtime、doctor、真实双会话、本地工具、StaticFair、terminal、owned 生命周期 | 两次真实 demo；最终独立 host 内容 oracle；[详情](INFRA_E1_STATUS_20261016.md) |
| E2，`c45cd25` | 三用途冻结 benchmark、compare、首段断流/新generation、独立停止、CPU CI、原始证据复核 | 九个 benchmark PASS，可靠性运行 STOPPED/清理确认；[详情](INFRA_E2_STATUS_20261016.md) |
| E3，本交付提交 | 安装/架构/排障/动手/讲解材料；压缩归档报告及独立输出；terminal 结算槽位保护；构建生成版本文件边界 | Windows 68 PASS /3 SKIP，Linux 71 PASS；离线 wheel/隔离安装与 CLI 7 项检查 PASS |

代码集中在 `slackkv/infra/`；配置在 `configs/local.json` 和 `configs/bench-*.json`；测试为 `tests/test_infra*.py`，保留 `test_system_m1*.py` 生命周期回归。脚本 `scripts/archive_infra.py` 无损归档，`scripts/verify_infra.py` 独立复算。`.github/workflows/infra-cpu.yml` 提供 CPU CI；本交付不以尚未核查的远端 Actions 状态宣称 CI 已绿。

E3 的修正没有改变 scheduler、allocator、connector、模型工作量或 GPU 默认配置；最终 Python/native CPU 校验与 E2 冻结的 GPU 源码分别留证，不能把此前 GPU run 伪称为最终 E3 每行代码都重新跑过。

## 实际跑过什么

- **真实流程：** 每次 2 会话 ×3 轮，跨会话请求有交叠；真实 assistant 输出进入下一轮。查本地 JSON 是客户端驱动，非模型自主选工具；遵守 EOS。
- **host 内容：** 固定受压场景、不 reset GPU；GPU 前缀2304、host额外1792tokens，实际 H2D49MiB / D2H112MiB。密码位置3018:3023确在恢复区间；完整输出IDs与独立 cold 一致，cold 无缓存命中。oracle 使用 batch-invariant，性能测试没有混用该模式。
- **用途对比：** hot、host-enough、host-limited 各一次 NoHost/Full/Fair。冻结输入、规定16实际输出、相同资源与编译模式，每个case内实际GPU池均18720tokens；无GPU清空。host受限全序列分别3778.93/3590.18/3369.38ms；host足够Full/Fair来源和迁移计数完全相同；hot零host恢复，时间差不归因 offload。单次工程筛选不证明稳定加速。
- **可靠性：** 实际首段SSE后断流，不提交半轮；逻辑close后换generation/salt可用，最终scheduler idle。独立stop确认owner退出0与STOPPED/cleanup；server abort与自然结束的区别仍 UNKNOWN。
- **CPU/打包：** Linux最终71 PASS（真实native metadata，禁止CUDA初始化）、Windows68 PASS/3 SKIP。wheel从本地离线构建，隔离安装后从项目目录外运行doctor、报告及help；工具JSON包含在包内。固定runtime的1806 Python文件与编译库manifest、doctor校验通过。

## 数据、复现和预算

原始入口：[工程数据](../results/infra_engineering_20261016/)。E1原始manifest保留，E2 [manifest_e2.json](../results/infra_engineering_20261016/manifest_e2.json)覆盖261服务器原文件。`provenance/e2_source`记录真实GPU源码；每个run的execution含命令、版本、源码hash、成本和清理结果。三种对照数值见comparison-*.json与run内CSV/JSON/Markdown。

```bash
python -m pytest tests/test_infra.py tests/test_infra_reliability.py tests/test_infra_benchmark.py tests/test_infra_native.py tests/test_system_m1.py tests/test_system_m1_shared.py -q
slackkv report --directory results/infra_engineering_20261016/runs/e1_demo2 --output .tmp/reader-report
python scripts/verify_infra.py --directory results/infra_engineering_20261016 --output .tmp/reader-verification.json
```

上述命令不启动模型；native测试在无依赖环境明确skip。GPU demo/bench/verify-host/reliability/stop的实际入口与配置见操作指南，需有适用资源授权并使用连续账本和唯一run名。

工程共15次GPU run：13 COMPLETE、1 FAILED、1 STOPPED，全部确认清理；含启动/编译/退出合计 **0.1454743116594586 /0.5 GPUh**，GPU1未用。历史连续总计 **4.7163514802356055 GPUh**，未挪用旧专用余额；E3新增GPU **0**。最后检查两卡各19MiB、利用率0%。剩余额度不是自动执行更多实验的要求。

E3新增证据：[validation.json](../results/infra_engineering_20261016/e3/validation.json)、[package_validation.json](../results/infra_engineering_20261016/e3/package_validation.json)、Linux最终测试日志和doctor/runtime校验。初次Linux最小上传漏了旧实验测试夹具，61 PASS/10 FAIL；保留失败日志，补齐夹具后71 PASS，没有削弱断言。受测环境原有cutlass `.pth` 告警保留，未修改服务器包。

## 当前边界

实测环境为Python3.11.16、torch2.11.0+cu126、vLLM distribution0.23.0+cu126且源码0fc695fc6、transformers5.17.0，RTX A6000/CUDA12.6/driver550.54.14，Qwen2.5-1.5B-Instruct。已验证的是固定干净上游Python与匹配既有编译库的overlay；**干净机器从零编译整个vLLM栈尚未重做**，源码构建路径不能当成已实测承诺。

SessionPool是应用客户端协议，不是全局多租户网关。只绑定本机，没有认证/持久化/多进程全局槽位。配额为写入准入，空槽/短会话不借出；独立close只逻辑关闭，非物理即时free。未验证多机、多模型、任意取消时序或生产SLO。实际host占用/memory-time未观测，指标缺失保持UNKNOWN，首段内容间隔代理不叫TPOT分布。压力请求不是并发后台服务，不能宣称全面后台公平性。

vLLM提供执行、GPU allocator、KV索引、LRU、DMA与CUDA优化；本项目新增会话/配额/terminal适配、运行管理、观测与可复现验证。旧M1、32B结果和全部历史失败保留，不删除代码、不继续M2/M3或研究调参。
