# M1 有界共享前缀里程碑

目的：修复terminal无意义挤占，支持公共system-root只收费一次和连续私有tail，限定验证结束、读取和原生引用结算。

配置：Qwen2.5-1.5B FP16 TP1 compiled，单A6000，GPU KV512MiB/18720tokens，host224MiB/8192tokens；512-token共享root，四个私有会话8K/6K/1K/4K增长，19正式串行请求及四个16K压力请求。公共根计费StaticFair每会话2432-token上限，不借用；所有方法公共terminal gate、五形状独立warmup、无GPU清空、两个轻量checkpoint。性能无详细状态日志。

结果：87本地相关回归、最终29组CPU/native审计PASS。content4三个真实恢复（含A结束后B）与零命中cold各完整16IDs一致，CEDAR17/MAPLE42可区分；70状态守恒、公共root32物理块、最终512free。原三个content失败保留：格式要求不合、前缀priming产生错误码、选择题cold不敏感。最后identity oracle接受码或固定肯定答案，拒绝错误/双码/否定/EOS后码；没有降低完整IDs比较。旧失败未改为PASS。

单次19请求、153088输入/304输出性能：M1/Fair/Full median E2E5380.69/5428.11/5512.98ms，M1变化−0.87%/−2.40%；11续接P95239.90/255.81/306.01ms，−6.22%/−21.60%。M1/Fair/Full D2H537.25/185.5/957.25MiB，H2D560/381.5/316.75MiB。M1对Fair少6528重算tokens，但总迁移多93.52%，不证明稳定优势。所有四个会话本次均改善；不外推生产/并发后台/SLO/质量。

新增7run0.10946156016611189GPUh，原0.5授权累计0.3195768596469497，历史连续4.511564879374477。全部清理，两卡19MiB/0%。保留M1可选与StaticFair，不扩旧矩阵/M2M3；提交后停止本轮。报告`docs/SLACKKV_SYSTEM_M1_SHARED_20261013.md`，数据、失败、SOURCE SHA、复现命令和独立spotcheck均归档。433原文件哈希核验、压缩版复算通过；预算浮点末位差单独披露。

最终复核补充：共享独占尾部deferred诊断此前会因减去含公共root的target而低报，修正后新增native回归，audit4最终29组PASS，无分配/回收/DMA变化。435原文件（含audit4两文件）哈希核验通过。content机制记录定位112MiB首复用前回收和89.25MiB末段GPU命中后结束的未复用写入，见unused_write_evidence.json；不作完整性能额外DMA归因。
