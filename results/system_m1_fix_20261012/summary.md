# M1 correction — 2026-10-12/13

目标：修复已结束 generation 经 Member GC 后复活；验证原生引用仍正确结算；准备有界 GPU smoke。
交接基线4d1da69，旧实现92e65cb，最终代码以本目录所在提交为准。

结果：新增回归在旧 controller 上1 failed/15 passed；修复后完整本地74 passed。
扩展 native 审计在旧 controller 上 FAILED，真实 late 请求重新获得 store 资格；修复后15组PASS、含120次交错。
失败断言/堆栈和两种源码SHA均保留。旧版64项/v1-v2结果未覆盖或否定该缺陷，不能追溯重标。

配置：Windows Python3.13.5；Ubuntu Python3.11、vLLM0.23.0/Torch2.11.0+cu126真实元数据路径，CUDA_VISIBLE_DEVICES=''并禁止lazy CUDA初始化。没有GPU tensors/DMA/model forward。
测试原生 store/load 跨终止/新代、reset后不复活、普通finish允许续接、共享system prefix不同后缀。
共享诊断一次fallback_shared_prefix；合法native read仍通过，没有session盐。

终止契约：只接受已见且等于latest的end；重复/旧/未知/未来end无副作用。终止bit跟随latest tombstone，不随Member GC或cache reset清除；更高generation明确重新加入。没有TTL。

GPU：NOT_STARTED，内容NOT_RUN，动态收益UNKNOWN。新增0，连续4.191988019727527GPUh不变。请求新增0.5GPUh尚未获批。
smoke runner/config和实际CPU生成输入已准备；v3为最终配置/runner匹配版本。
v1保留但不适合512MiB KV的32K max_model_len；v2修正容量，v3增加runner SHA冻结校验。都没有启动GPU。
v3：compiled TP1，512MiB GPU KV，18,432 model limit，28,672bytes/token，host224MiB/512blocks；长8192/短1024/普通压力16384输入，输出16。
不清空GPU，独立salt仅用于cold correctness oracle；M1/StaticFair性能模式均无详细M1 state trace，普通source trace规则相同。

异常：环境启动时nvidia_cutlass_dsl .pth导入警告；CPU验证/准备均完成，未修改共享环境。GPU启动可行性仍需实际验证。
原始输入以gzip保存，plan中的SHA是解压原字节SHA；服务器原始JSON保留。三版准备日志、失败、最终源码都可定位。
复现见commands.sh；完整解释见docs/SLACKKV_SYSTEM_M1_REPORT.md“修正与后续验证”。
