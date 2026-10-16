# E1：真实多会话纵向流程已运行

基线d38e1b8；日期2026-10-16（用户时区）。代码版本为本报告所在提交。先交付E1，再继续E2/E3；不是宣布生产级完成。

- 已实现：无重型依赖的打包与CLI、集中配置、固定版本/源码树/编译库校验、doctor、owned serve/stop、真实历史SessionPool、本地JSON工具、原生StaticFair/Full/NoHost入口、JSON/CSV/Markdown报告。
- 主线直接使用vLLM0fc695fc6原生max_offload_tokens，不加载旧研究scheduler。仅在原生factory加载薄terminal spec；原生allocator/worker/DMA不改。1805上游Python源码树hash固定，overlay加生成version文件后1806文件，编译库逐一hash。
- 实测栈：Python3.11.16 / torch2.11.0+cu126 / vLLM distribution0.23.0+cu126 / transformers5.17.0，CUDA12.6 / driver550.54.14 / 单RTX A6000；已有本地Qwen2.5-1.5B-Instruct。没有下载模型或升级环境。

## 验证

CPU：19项新测试PASS；真实native CPU审计6条契约PASS（禁止CUDA初始化）。wheel离线构建成功；首次Windows默认pip临时目录权限失败，改用workspace临时目录后成功。Windows离线报告读取曾因默认GBK失败，新增显式UTF8及非ASCII报告回归；Linux执行字节另存provenance，UTF8修补不改变已测Linux行为。已存在的cutlass .pth告警与退出semaphore告警保留，未修环境。独立doctor最初stdout/stderr混在doctor.json，保留该无效文件；有效机器可读输出为doctor_verified.json及每run的doctor.json。

GPU（原新工程账本；每个run均确认owned进程退出、GPU回到19MiB）：

| run | 结果 | GPUh |
|---|---|---:|
| e1_demo1 | 两会话三轮成功；后来发现上下文预检token计数问题，本次小输入不触发超限 | 0.012892430726933525 |
| e1_host1 | FAILED：transformers5默认BatchEncoding，len误数2个字段，导致尺寸检查拒绝；未发推理请求 | 0.010612019835567076 |
| e1_host2 | 平坦ID修正后恢复/cold完整IDs一致、有真实host流量；marker仍在GPU前缀，不作为最终host敏感内容验证 | 0.010008468955283106 |
| e1_demo2 | 修正后两会话各3轮、真实assistant历史、真实本地工具、跨会话请求交叠PASS | 0.008955471526391597 |
| e1_host3 | 最终受压恢复/独立cold/marker位于host区间PASS | 0.010009540269448836 |

新工程E1累计 **约0.052478 /0.5 GPUh**；连续历史 **4.623355099889771 GPUh**，不使用旧M1余额。启动/模型加载/编译/客户端/退出均计费，CPU检查0GPUh。每次小于600秒，GPU1未使用。

e1_demo2实际messages/requests/tools/report留在原目录，两个会话历史各7条（system+3user+3assistant）；下一轮明确包含先前实际assistant内容。最后回答使用Lin/Chen、发布日期及testing/review资料；工具由客户端选择，不冒充模型自主工具调用。它是自由生成demo，不用于政策吞吐排名。

e1_host3：8192token文本、16384token不相干压力，无GPU清空；恢复GPU-local **2304**、host **1792**，H2D **51,380,224 bytes（49MiB）**、累计D2H **117,440,512 bytes（112MiB）**。区分性密码位于token **[3018,3023)**，完整落在真实host恢复区间[2304,4096)，不在GPU/新计算部分。独立salted cold实际compute8192、GPU/host0，完整输出IDs与恢复一致，答案CEDAR17。batch-invariant仅用于该正确性检查，不与demo性能混比。H2D初始动态label缺失时，依据新服务、零external源及无prefetch的明确零起点推导；不是通用缺失填零。

## 边界和剩余项

StaticFair与Full/NoHost入口及同预算命令生成已测CPU；本E1真实GPU仅StaticFair，其他模式/小benchmark留E2。会话槽位/单在途限制在单SessionPool内，直接vLLM请求/多个Pool不提供全局租户隔离。cache_salt隔离会话，不提供跨会话共享。独立close仅逻辑关闭，物理缓存随native LRU/引用结算，没有有序物理close API。

取消为SSE边界合作检查/读超时，不声称已确认立即server abort。上下文预检修正有回归，失败不提交半轮，失败身份需close/换generation。更完整启动失败/停止、HTTP中断、流式输出观测和小型严格对照在E2补。干净环境源码重编译安装尚未执行；匹配已有编译库+完整固定上游Python overlay路径已实测。

## 启动和证据

可复制的通用命令见README；实际服务器argv/模型/config/源码hash/进程出生时间/时限/退出与清理在每run execution.json。新原始数据在results/infra_engineering_20261016，历史原始数据不覆盖。102服务器原文件解压长度/SHA验证；早期代码在provenance下保留，最终运行代码hash对照本提交。

```bash
python -m slackkv.infra doctor --config configs/local.json --config-only
python -m slackkv.infra demo --config /absolute/demo_config.json --ledger /absolute/infra_engineering_20261016 --name unique-demo
python -m slackkv.infra verify-host --config /absolute/oracle_config.json --ledger /absolute/infra_engineering_20261016 --name unique-host
python -m slackkv.infra report --directory /absolute/infra_engineering_20261016/runs/unique-demo
python -m pytest tests/test_infra.py -q
```

GPU命令仅在适用预算与同一连续账本下运行；不要重用已有run名。用户可先运行CPU自检和阅读已保存真实报告，无需GPU。E1完成并提交后，按交接继续E2/E3，不恢复研究矩阵。
