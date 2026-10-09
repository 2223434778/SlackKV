# Round3 summary

目的：判断dynamic shared CPU-prefix fork是否产生可优化的重复GPU物化；以成本模型淘汰完整active-prefix streaming，并重新审查创新空间。

设置：单A6000 GPU0、Qwen2.5-1.5B FP16、Compiled OnDemand/APC、CPU8GiB；synthetic fork3episodes×8branches×2turns，每call128outputs，BG8192prompt/1024outputs×3。对照4/8GiB、K2/K8、cold/decode BG分别标记；invariant只用于内容oracle。历史40/32/BG8192 workload另作回归。

| 默认模式run | Workflow median ms | Child TTFT P95 ms | BG median ms | H2D MiB |
|---|---:|---:|---:|---:|
| cold 4GiB K8 | 4360.56 | 709.27 | 8574.71 | 2016 |
| decode 4GiB K8 | 4134.64 | 480.42 | 8568.08 | 2016 |
| decode 8GiB K8 | 4068.17 | 495.85 | 8541.77 | 2016 |
| decode 4GiB K2 | 4154.34 | 474.15 | 8615.56 | 2016 |

所有完整fork run：三次ancestor external恢复、21次首轮local ancestor hit、24次第二轮local hit，D2H2310MiB、preemptions0、final running/waiting0。原生`_blocks_being_loaded`已保护在途共享，因此未实现coalescing候选；不是算法速度配对，也不宣称统计显著。invariant独立16/16cold oracle精确匹配。旧负载回归valid，P95Resume4853.24ms、session median23878.57ms、BGwall67764.82ms，28/32capacity-failed resumes。

成本屏：shared-root-only每step streaming180.39GB，历史26.28GB/s条件下6.86秒DMA估计；unique fork<1GiB、已有驻留更快，不开发该regime的PoC。peer输入方向历史带宽仅比CPU高约0.33%（传输时间改善）；不能声称整体收益。

异常：部分run后helper参数失败；retry错误派生source_delta零值INVALID；live-ledger分析、CPU测试PYTHONPATH与上传父目录失败均保留。raw source与actual transfer独立审计；missing/failed不补0。52localtests/15runtimeCPUtests，101raw hashes verified；7GPU executions合计0.20048675264813934GPUh，最长260.33sec，两卡最后空闲。

结论：停止条件B，范围限于当前方法/条件。没有新算法、没有相对强基线的算法加速。下一研究重点先验证真实长上下文Agent的host-miss/recompute/readiness critical path，不能继续普通TTL/去重/JIT参数迭代。完整报告、文献/不确定性：[AUTONOMOUS_RESEARCH_ROUND3.md](../../docs/AUTONOMOUS_RESEARCH_ROUND3.md)；复现：[README.md](README.md)。
