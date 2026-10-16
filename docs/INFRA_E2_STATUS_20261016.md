# E2：可靠性、指标与三种固定用途场景

基线E1提交b6773c7。E2新增固定输入benchmark/compare、真实断流检查、STOPPED确认、纯CPU CI和独立归档复核；不恢复研究调参。实际源码在results/infra_engineering_20261016/provenance/e2_source，运行版本由execution.json与plan hash固定。

## 实际验证

- Windows CPU：65 PASS /3 SKIP（POSIX进程两项、缺vLLM的native一项）；Linux pinned环境27 PASS，包含真实小子进程失败/回收、正常停止和真实CPU native metadata；CUDA初始化被禁止，没有GPU成本。
- 九个GPU用途smoke全部COMPLETE，三个case各一次NoHost/Full/StaticFair。每个case内资源、原始token输入、16tokens输出工作量、预热/编译模式匹配，实际GPU KV均18,720tokens；两agent请求均有真实交叠。NoHost是真无connector，Full/Fair同connector/terminal gate。
- `e2_reliability1`真实收到首段SSE后关闭客户端流，失败历史仅system一条，close后generation更新、新请求成功，scheduler最终running/waiting0。独立`slackkv stop`给验证过出生时间的owner发信号；owner exit0，状态STOPPED/cleanup_verified。
- 每个run含argv、配置、源码hash、日志、时戳、warmup、实际usage/IDs、prom、JSON/CSV/Markdown。benchmark compare拒绝不同资源/输入/实测GPU池、未结算或重复policy。不以自由聊天输出排名。

## 数值与口径

GPU KV固定512MiB，hot/host-limited host224MiB，host-enough host1GiB；都是明示工程配置，不是A6000物理内存限制。hot两会话各3轮，1024→1088→1152输入；另两case为8192→8256→8320，轮间各一次16K独立压力；没有GPU reset。warmup三/四形状、不同salt、zero host store。正式hot为6请求/6528输入/96输出；其余8请求/82304输入/128输出。

| 场景/方法 | 全序列 ms | 重算tokens | GPU命中 | host恢复 | D2H MiB | H2D MiB |
|---|---:|---:|---:|---:|---:|---:|
| hot NoHost | 517.36 | 2304 | 4224 | 0 | 0 | 0 |
| hot Full | 411.00 | 2304 | 4224 | 0 | 59.5 | UNKNOWN |
| hot Fair | 414.00 | 2304 | 4224 | 0 | 59.5 | UNKNOWN |
| enough NoHost | 3820.39 | 82304 | 0 | 0 | 0 | 0 |
| enough Full | 2750.54 | 49408 | 0 | 32896 | 451.5 | 899.5 |
| enough Fair | 2755.49 | 49408 | 0 | 32896 | 451.5 | 899.5 |
| limited NoHost | 3778.93 | 80000 | 2304 | 0 | 0 | 0 |
| limited Full | 3590.18 | 74176 | 0 | 8128 | 673.75 | 222.25 |
| limited Fair | 3369.38 | 65920 | 4608 | 11776 | 224 | 322 |

**不将hot的时间差称为offload收益**：三者compute/GPU命中完全相同，host恢复都是0；单次差可能包含执行布局、驱动与测量噪声，没有因果分解。hot H2D label未出现，保留UNKNOWN；NoHost的0由确实无connector证明，不是把缺失counter补零。

host足够时Full/Fair来源与迁移完全一致，Fair并无额外机制优势，时间约+0.18%。host受限时Fair相对Full全序列约−6.15%、相对NoHost约−10.84%，少重算/少D2H，但H2D高于Full。每个case一次、短工作量、没有预注册提速验收；数字只用于工程取舍说明，不宣称稳定加速、生产SLO或论文结果。

session E2E、逐请求首段内容TTFT/响应/输出区间代理及完整字节保留在benchmark.json/CSV与comparison-*.json；没有逐token时戳，不称TPOT分布。全局counter不能在并发时归因单请求；host实际占用/RSS/memory-time未观测，仍UNKNOWN。压力请求不是并发后台服务，未以它宣称后台公平性已全面验证。

## 资源与失败

E2十个run新增 **0.09299638034583445 GPUh**；工程E1+E2 **0.1454743116594586/0.5 GPUh**，历史累计 **4.7163514802356055**，剩余约0.354526工程GPUh。所有15个工程run清理已确认（13 COMPLETE、1早期FAILED、1 STOPPED），GPU均回到19MiB、GPU1未用。没有因为换阶段重置额度；E3不再启动GPU。

E2 GPU无失败，无重跑选最好结果。E1失败与较弱oracle原样保留。261服务器原文件解压长度/hash独立通过；独立脚本从原始requests/plan/prom/时戳复算三case工作量、来源、迁移、重叠、实际GPU池、源码与owner清理。结果：[e2_verified.json](../results/infra_engineering_20261016/e2_verified.json)。CI配置已加入，远端CI结果未作为本报告的验收证据；上述是本地/服务器实际执行结果。

## 复现

```bash
python -m slackkv.infra bench-prepare --config configs/bench-host-limited.json --scenario host-limited --directory results/new-plan
python -m slackkv.infra bench --config configs/bench-host-limited.json --plan results/new-plan --policy fair --ledger /absolute/original-engineering-ledger --name unique-fair
python scripts/verify_infra.py --directory results/infra_engineering_20261016 --output .tmp/verified_unique.json
```

其他模式保持同一plan/config，CLI实际argv见e2_launch.json；完整操作者命令会在E3指南补齐。`reliability`只连接匹配的live owner，`stop`要求PID出生时间一致，未知owner不发信号。新读者可只做CPU验证、查看已存报告；GPU复跑须适用授权、唯一名称和连续账本。
