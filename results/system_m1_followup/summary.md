# M1 有界净收益验证

完整解释与限制见 `docs/SLACKKV_SYSTEM_M1_NET_BENEFIT_20261013.md`。

目的：验证 terminal 可选写入 gate、双答案内容 oracle、等价预热，以及与带相同终止优化的 compiled StaticFair/Full 进行一次独立前缀资源竞争筛选。

配置：Qwen2.5-1.5B FP16 TP1，单 A6000，GPU KV 512 MiB/18720 tokens，host 224 MiB/8192 tokens。四个会话 8K/6K/1K/4K 前缀随续接增长，三个 16K 压力请求。五种独立根预热，没有 GPU clear API。驱动串行，属于合成资源竞争，非生产或同时计算竞争。

结果：content2 PASS，两个真实 host 恢复与独立零命中 cold 各16 IDs相同，CEDAR17/MAPLE42可区分，新 terminal D2H=0。62项本地相关测试、19组 CPU/native 审计 PASS。性能各一次，17请求/130304输入/272输出；M1 median E2E 5234.78ms，Fair 5330.23、Full 5652.81，变化−1.79%/−7.40%；10个续接的 nearest-rank P95 252.17ms vs 262.41/345.56，变化−3.90%/−27.03%。四会话本次均改善，不宣称统计稳定。M1 D2H/H2D为476/595.875 MiB，Fair196/420，Full1641.5/28。StaticFair以更低迁移成本保留了大部分收益。

异常：首次内容运行 FAILED，plain completion不能作区分性 oracle；修正聊天格式后另冻结输入。循环变量覆盖导致首次run落到`contention.py/`，原记录和预算保留，旧源码SHA与执行记录一致。保留现有 cutlass .pth 警告，未修改共享环境。执行准备中19456字段为四配置会话合计；S退出后加入D的实际基础需求18432，未来元数据已澄清，旧记录未改。

结论：完成本轮有界审查里程碑，保留 M1 竞争场景可选路径，不默认替换 StaticFair，不开启 M2/M3。共享覆盖、并发后台公平性、memory-time、稳定收益和生产质量仍 UNKNOWN；未执行新 NoHost，不能挪用旧负载结果。

预算：五个新run花费0.07534938563277440 GPUh，0.5授权累计0.21011529948083787，历史连续总额4.402103319208365。全部自有进程组退出，两卡最终19MiB/0%；提交推送后停止本轮。

原始run目录在`results/system_m1_smoke/`。无损`.gz`对应`raw_manifest.json`中的服务器原字节；`analysis_verified.json`从压缩数据复算Linux结果，`spotcheck.json`独立核验每请求来源及时间戳。命令在`commands.sh`，配置/源码SHA在准备与执行记录。
