# SlackKV 实验与验证数据

[项目主页](../README.md) · [运行与复现命令](../docs/INFRA_OPERATIONS.md)

## 工程验证

| 内容 | 证据 |
| --- | --- |
| 已验证的功能与环境范围 | [交付报告](../docs/INFRA_DELIVERY_20261016.md) · [工程数据摘要](infra_engineering_20261016/summary.md) |
| 真实两会话、三轮 Demo | [运行摘要](infra_engineering_20261016/runs/e1_demo2/summary.md) · [会话记录](infra_engineering_20261016/runs/e1_demo2/sessions.json) · [请求时序 CSV](infra_engineering_20261016/runs/e1_demo2/requests.csv) |
| CPU KV 恢复与独立 cold 对照 | [内容校验](infra_engineering_20261016/runs/e1_host3/oracle.json) |
| 三种策略各一次匹配对照 | [GPU 热缓存](infra_engineering_20261016/comparison-hot.json) · [CPU 容量充足](infra_engineering_20261016/comparison-host-enough.json) · [CPU 容量受限](infra_engineering_20261016/comparison-host-limited.json) |
| 断流、恢复与进程清理 | [验证口径与结果](../docs/INFRA_E2_STATUS_20261016.md) · [执行记录](infra_engineering_20261016/runs/e2_reliability1/execution.json) |
| 原始资产和来源 | [E1 清单](infra_engineering_20261016/manifest.json) · [E2 清单](infra_engineering_20261016/manifest_e2.json) · [E2 源码快照](infra_engineering_20261016/provenance/e2_source/) |

## 历史研究

| 主题 | 数据与说明 |
| --- | --- |
| 准入和显存容量 | [准入诊断](autonomous_admission/) · [容量与调度](autonomous_capacity/) |
| 部分前缀 | [首次对照](host_prefix_screen_20261010/) · [多会话复验](host_prefix_followup_20261010/) · [32B 原始数据](large_model_20261012/) · [32B 实验报告](../docs/AUTONOMOUS_RESEARCH_LARGE_MODEL_20261012.md) |
| 动态 M1 | [实测验证](system_m1_smoke/) · [成本分析](system_m1_cost/) · [策略复盘](../docs/RESEARCH_DIRECTION_AFTER_M1.md) |
| 早期阶段、失败与诊断 | [按时间查看实验记录](../docs/experiment_log.md) |

从已归档的运行记录生成报告时，请使用新的输出路径。原始日志和指标中的 `.gz` 文件为无损归档。工程验证与历史实验采用各自的模型、资源和工作负载，跨场景比较前请核对配置和指标口径。
