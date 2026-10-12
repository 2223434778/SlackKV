# system_m1_20261012

目标：实现动态 host 连续前缀分配，验证真实 native 生命周期契约。
基线源码版本 f2d6af0；最终实现以此目录所在 Git 提交为准。
不是模型性能实验，没有模拟数据冒充 GPU 测量。

配置：Windows Python 3.13.5 本地纯控制器与历史回归；Ubuntu native vLLM 0.23.0 / Torch 2.11.0+cu126；禁用 CUDA 设备与 lazy init。
实际比较：开启 M1 的 native 元数据路径，关闭 M1 的原始 manager Factory 路径，以及关闭 hook 后 AST 一致性。
未执行 Full / NoHost / StaticFair GPU 性能比较。

结果：本地 64 passed；native v1 8 组 PASS（27.555750608 s），v2 12 组 PASS（6.689466011 s），每版包含 120 次交错操作。安装成功，两个 opt-in hook，原源码未修改。
10 块池旧计划与新成员竞争、TP pending 汇总、引用保护尾部回收、取消/失败/迟到完成和 native store cursor 场景通过。

异常：一次工具审批服务 429 导致命令未执行，稍后重试成功；无 native FAILED run。源版本/日志分别保存，不覆写初版。
限制：合成请求元数据，未验证 GPU tensors/DMA/KV 内容、生成质量、实际模型活性或性能；共享公平降级、显式 session end、无 tombstone GC。

新增 GPU 0，历史连续账本 4.191988019727527 GPUh 不变。后续模型 smoke 尚未执行，需要新的数值授权。
详细接口、限制和具体 smoke 计划：`docs/SLACKKV_SYSTEM_M1_REPORT.md`。
复现：`commands.sh`；原始 JSON、日志、安装 SHA、source_v1/source_v2 完整保留。
