# 代码阅读与动手清单

按顺序完成，每步保留自己的命令与观察；CPU步骤无需占GPU。GPU步骤必须使用适用授权与同一连续账本，不能重用旧run名称。

1. **CPU启动。** 安装`.[test]`，运行`slackkv doctor --config configs/local.json --config-only`和`pytest tests/test_infra.py -q`。解释为什么PASS不证明模型/KV正确。
2. **读主路径。** 按`config.py→sessions.py→http.py→native.py→service.py`各找一个关键不变量：对齐额度、同会话单在途、完整SSE后提交、terminal不破坏load、只清自有进程。
3. **亲自跑真实demo。** 按[操作指南](INFRA_OPERATIONS.md)准备固定runtime，核对doctor，运行唯一名称demo。打开sessions.json，指出第二轮真实assistant历史；在tools.json说明是谁选择的工具。
4. **解释时延。** 在requests.json中找issue/first/done，在requests.csv中找它们对应的时延指标，说明first-content平均间隔为什么不是逐token TPOT。对照Prometheus累计值，指出哪些是全局数据、哪些缺失为UNKNOWN。没有GPU可用时先按操作指南将归档demo重建到新目录。
5. **改一个预算。** 先只在CPU复制配置并改host_kv_bytes，读取`layout()`计算配额，不启动GPU矩阵。预测短会话/空槽是否借出，然后用测试核对；不要预言一定更快。
6. **复现一个安全失败。** 将配置port设为0或ctx设为小于输出上限，确认doctor/schema报错且没有服务进程。运行`test_context_rejection...`，确认history没变。不要为故障演示杀别人的GPU服务。
7. **读一个真实恢复。** 打开e1_host3/oracle.json：GPU2304、host1792，密码位置3018:3023位于哪里？为什么51,380,224=1792×28672？为什么cold0hit和完整IDs比“答案差不多”更有力？
8. **亲手停止。** 在适用剩余额度内serve，另一个终端stop；等待execution.json的STOPPED/cleanup_verified。解释PID出生时间为什么重要，为什么RUNNING不能随手改成COMPLETE。
9. **理解负结果。** 读[历史M1成本报告](SLACKKV_SYSTEM_M1_COST_20261013.md)，用自己的话解释“CPU局部24倍改善但整体未改善”。再读benchmark三个用途场景，不把单次数字当生产结论。

完成后尝试在不看文档的情况下画出一次store/load与终止的路径，并说明哪些代码来自vLLM、哪些是本项目应用层。面试中只描述自己实际掌握与完成的部分。
