# M1 增量：terminal 保留语义与有界共享前缀

日期：2026-10-13。交接基线 `b242f2d`，实现版本为本报告所在提交。接纳 Astra 的两个增量，主架构保留，M2/M3 关闭。

## 结论

**terminal 无意义挤占已修复；共享 system-root + 独立私有尾部已完成实现及限定验证。动态分配仍仅保留为可选路径，不默认替换 StaticFair。**

真实 GPU 检查验证：共享根512 tokens只占32个物理host块；A/B各自私有marker恢复与零命中cold参考各16 IDs相同；A结束后B仍能恢复2560个host tokens（另有512 GPU-local tokens），输出与B自己的cold一致；最后512个host槽全部空闲。87项相关本地测试、最终29组CPU/native审计通过。不是完整KV逐字节或所有并发路径证明。

归档复核补充修正了共享模式`deferred_reclaim_blocks`诊断：独占块数不能直接减去含公共根的逻辑target，否则受保护的独占尾部会低报；现按独占key集合减去保留前缀集合计数。新增针对性native场景通过，最终第四版审计 **29组PASS**。不改变分配、引用、回收或DMA行为；GPU按旧诊断版本实际字节归档，不改旧快照。

关闭详细状态日志、各方法采用公共terminal gate和两个公共轻量checkpoint后，19请求共享场景各一次匹配筛选结果为：

| 指标 | 动态M1 | 公共根计费后的StaticFair | Full |
|---|---:|---:|---:|
| A会话E2E / ms | 8023.18 | 8099.11 | 8260.90 |
| B会话E2E / ms | 9144.38 | 9263.76 | 9360.18 |
| S会话E2E / ms | 2247.23 | 2319.02 | 2400.93 |
| D会话E2E / ms | 2738.19 | 2757.10 | 2765.07 |
| 四会话E2E median / ms | 5380.69 | 5428.11 | 5512.98 |
| 全序列完成 / ms | 9734.08 | 9832.79 | 9935.20 |
| 续接nearest-rank P95 / ms | 239.90 | 255.81 | 306.01 |
| 输入重算tokens | 108096 | 114624 | 116992 |
| GPU本地命中tokens | 24512 | 24512 | 24512 |
| host恢复tokens | 20480 | 13952 | 11584 |
| D2H / MiB | 537.25 | 185.5 | 957.25 |
| H2D / MiB | 560 | 381.5 | 316.75 |

M1相对StaticFair：median −0.87%、全序列 −1.00%、续接P95 −6.22%，总DMA **+93.52%**。相对Full分别−2.40%、−2.02%、−21.60%，总DMA−13.87%。四会话本次均改善，但11个resume的nearest-rank P95就是最大值，不能声称稳定长尾。多省6528个重算tokens，却比Fair多迁移530.25 MiB；此配置不支持默认采用动态策略，更不值得围绕这一序列调参。

共享机制和terminal修补首先是可用性/安全性工程进展，未证明新颖SOTA。最重要的后续问题是减少没有足够复用收益的重分配与回收重写，需先记录成本来源；本次不宣称已解释此前280 MiB或全部本轮额外迁移。

对content4固定路径还能定位两处未被host复用的写入：A0实际保存512块，B/S加入后在首次A1恢复前只剩256块，尾部256块（112MiB）已经回收；B_after_a_end则从192→396块，补写204块（89.25MiB，prom窗口与admitted计数一致），随后唯一B_end使用6336个GPU-local tokens、host0，再结束。证据见`unused_write_evidence.json`。两处合计201.25MiB，仅是content机制run的观测路径证据，不是生产收益上界、免费的删除反事实或性能run额外351.75MiB的完整因果归因。它比继续扩大同序列更支持研究“什么时候值得借用并写入”。

## 1. 实现与不变量

### terminal需求与当前读取分离

`Member.retention_blocks`独立于完整native请求路径。只有非terminal请求更新未来保留需求；新terminal会话登记身份、最终结束，但需求为0。已有会话terminal输入正常增长，保留原需求和目标，不为了不可选写入的数据扩额，也不因这次观察轮转余数。结束零需求会话也不触发无意义重分配。

native observe仍在lookup前执行。当前已有前缀不提前清空、不提前end，原生lookup/load正常；`max_offload_tokens=0`停止未来store，已发出的DMA仍按原ticket结清。真正完成时执行end和tombstone。新generation的正常退休语义、未知/旧/未来end的规则保留。

两个真实CPU/native反例已验证：512块A驻留时，新的512块terminal T不减A目标或释放A尾块；普通N加入仍将A缩至256。9块奇数预算下A terminal输入8→12，未来需求仍8、目标及cursor不变，当前读取受保护、B前缀保留。

### 有界共享root与私有tail

`slackkv_shared_prefix_blocks`为显式、默认0的支持边界；本场景设32。独立默认模式保留原单owner/fallback路径。共享模式使用`key_refs: native key → logical member set`，**不是删除shared检查后继续单owner**。native身份由真实offload hashes提供，不由session ID替代；正常请求没有cache salt。

对已知未来需求做max-min逻辑深度分配，成本为各目标前缀key集合的并集。相同公共块计一次；公共根本身也必须装得下，不为完整root预留无限容量。CPU例子：root3、A/B各6私有块，预算9，可以各保留深度6，distinct总量9而非逻辑和12。root6/预算4时只能共享保留前4块。

逐批store继续受冻结ceiling、当前target、真实空闲槽位和净新增块限制；复用原生CPU manager/worker/DMA，禁止M1调用native LRU隐式驱逐。已存在或pending公共key不会重复创建store ticket。

回收在同一manager锁内执行：先计算各存活成员实际连续物化前缀所需并集，再对pending store、正load引用及当前保护key求祖先闭包；只按深度从尾到根移除零引用、无人需要的物理块，同步policy/free/events。逻辑保留引用与native DMA `ref_cnt`分开。已结束A的逻辑引用可移除，B所需公共祖先仍保留；最后一个保留需要结束且引用结清后才释放。失败头块造成的孤立后缀，在在途引用结清后回收。

支持多个串行会话在声明root范围内共享同根，之后独立增长；超界公共私有块、合并路径、同会话并行/改写/截断仍明确fallback停止可选写入，合法native读保留。未实现任意共享DAG、fork、预测、GPU预取。两个上下文完全相同且共享深度超过声明root也会fallback；不能把这个版本说成一般共享覆盖完成。

物理`ready+pending+free=capacity`保持512。`target_distinct_blocks`是逻辑并集目标，不是未物化reservation；每会话逻辑深度之和在共享模式可超过512，不得拿它当物理占用。`physical_shared_blocks`按注册路径的多逻辑引用计数，不等同DMA引用或未来保留额度。物理pool字节仍固定224 MiB，没有证明减少CPU RSS或GPU memory-time。

## 2. 验证范围

87项本地回归覆盖原独立路径、terminal反例、公共计费/最后引用、root大于容量、unsupported形状、退休private身份等待、公共checkpoint配置与严格marker判断。三版CPU/native审计分别25/26/28组PASS，最后版本保留原120次元数据交错和TP完成汇总检查，并扩展：

- 公共块不重复store，两会话均保存私有尾部；A结束后B可读并扩额，最后load完成才全部释放。
- 深层private load保护零引用公共/私有祖先，load完成后才回收，无祖先洞。
- pending公共root去重，root写失败、孤立private后缀回收、重试及重复旧completion不会删除新公共块。
- 旧generation pending写结清后公共root仍可被B/新代使用，旧end/重复失败通知不污染新代。
- native reset的flush/stale-job边界下，旧ticket不能修改复用的公共槽。

这是CPU/native元数据测试，不是28个GPU故障注入。真实content4采用batch-invariant模式，先检查三个独立salt的cold参考，再执行共享恢复；A1、B1、B_after_a_end的全部16 IDs分别与cold一致，三个cold均零local/host命中。前两个输出彼此不同，marker放在私有历史起始处，而非公共root或最后问题中。

70个content状态快照守恒，最大实际公共共享块32，无`shared_prefix`或`unsupported_shared_shape`fallback；最后无物理成员、pending read/store，512free。新terminal请求D2H=0。性能run没有详细M1状态事件（三个均0），每方法恰好两个公共pool checkpoint，预热后512free，最终无在途传输；M1最终无成员且512free。不能从checkpoint推算memory-time。

未覆盖GPU强制取消/在途end碰撞、TP2模型DMA、PP/DCP、非full-attention布局、实际fork/DAG、无界饥饿、长期服务器增长或完整内容字节校验。所有被测请求推进完成，不宣称一般liveness/fairness证明。

## 3. 场景、强基线及测量

模型/栈沿用Qwen2.5-1.5B-Instruct FP16 TP1，vLLM0.23.0、Torch2.11.0+cu126，单RTX A6000、compiled/CUDA Graph，GPU KV512MiB（真实18720 tokens）/host224MiB（8192 tokens，28672bytes/token）。512MiB是受控压力预算，不是48GiB A6000的物理上限。

A/B/S基础长度8192/6144/1024，共享root512，distinct初始需求14336 tokens，超过host8192。S退出后D以4096加入，A+B+D基础logical需求18432、distinct17408；每次续接追加64 tokens。公共root按16-token native块对齐；四个首private块不同。marker是CEDAR17或MAPLE42，公共内容不含身份答案。压力和warmup使用不同root，没有给正式输入提前命中。

19请求序列固定：A0/B0/S0/pressure0/A1/B1/S_end/D0/pressure1/A2/B2/D1/D_end/pressure2/A3/A_end/pressure3/B_after_a_end/B_end。增加末尾压力与B恢复是为验证A结束后的共享生命周期，不根据延迟调参。驱动及同会话均串行，合成资源竞争，不是生产trace或同时计算竞争。固定0.15s工具间隔，19正式请求153088输入/304输出，ignore_eos使每请求16输出；生成后的填充内容不代表任务质量。

StaticFair加强为公共root计费一次，固定四个已配置会话各私有份额1920，logical深度上限512+1920=2432；四者distinct总预算`4×2432−3×512=8192`，短会话未用份额不借出，也不知道未来长度。此规则在任何性能run之前冻结。Full与Fair使用同等独立terminal-store gate，原始native读写/LRU保持；M1也不预取。三个性能方法模型、资源、全部输入、输出工作量、terminal信号与预热相同。

五次独立预热形状1024/4096/6144/8192/16384，禁止host store，warmup会话正常终止。启动各约26.02s、warmup2.62–2.63s单列并计入预算；正式首请求419.96/399.31/405.62ms全部计入。未调用GPU clear API，三个方法实际GPU-local命中都是24512 tokens。M1多出的首请求约15–21ms保留，不据此反序重跑或扣除。

性能batch-invariant=0，content=1，不混合性能。详细状态仅用于content，性能采用同等source指标、HTTP计时和公共两个轻量checkpoint，修复了旧筛选只对M1输出详细状态的测量限制，但不能回推旧57条日志开销或将本次与旧序列直接比较。

四个pressure请求M1约828.81–830.58ms，Fair约830.82–838.94ms，Full约838.39–845.84ms。观察到这些请求没有退步，但它们是串行独立推理，未测同时后台服务公平性。新NoHost未运行，不能借用旧NoHost数据声称本场景加速。此次不同的共享输入、顺序和Fair规则不与旧独立场景做直接百分比归因。

## 4. 三次失败与oracle修订

所有失败都保留，并计入同一预算。没有将前几次失败 retroactively 改记PASS。

1. content1：恢复执行、70状态守恒和清理完成；A1/B1及A结束后的B恢复16 IDs均与cold一致，但B3 cold回答为`The secret password at the beginning is **MAPLE42**.`，旧`startswith`格式断言失败。这不等于被测KV错误，也未据此接受旧run。
2. content2：尝试assistant回答前缀，三个cold输出错误数字密码，不能作identity oracle，FAILED。不能因为恢复等于一个错误参考就宣称内容正确。
3. content3：尝试相同候选码选择任务，一个cold B输出CEDAR17，FAILED；因先做cold预检，没有继续正式恢复序列。

content4恢复content1原输入（SHA相同），将“模型只输出码的格式约束”和“身份内容敏感性”分开。identity oracle要求EOS前唯一正确码，恢复与cold完整16 IDs相同；错误码、双码、重复、EOS后才出现码均拒绝。交付代码进一步收窄为**码本身，或固定肯定句**两种完整答案文法，拒绝仅出现正确词或否定句；额外回归覆盖这些拒绝条件，离线重验三个实际cold答案均通过。没有放宽完整16-ID比较，也没有改写旧记录；不宣称模型总能遵守“只输出”格式或评估生成质量。

五版准备及四版实际content源码SHA均保存并验证。性能使用最后相同原输入，公共根StaticFair计费规则另在性能之前冻结；content4只运行M1，Fair规则修正不改变其执行。GPU之后controller修改collector契约docstring、native修正上述诊断计数，oracle文法收紧离线检查；实际执行字节在`executed_sources/`，GPU内容/性能数据未重写，最终诊断有新增CPU/native回归。

## 5. 数据、复现与资源

原始run在`results/system_m1_smoke/m1_shared_{content1,content2,content3,content4,m11,fair1,full1}/`；审计、五版准备、分析、source验证、复现命令及清单在`results/system_m1_shared/`。433个初始原文件及后补audit4两文件，共435个服务器原文件SHA核验通过，Git中log/prom/大inputs无损gzip，原服务器文件保留。`analysis_verified.json`从压缩数据复算；全部性能指标与Linux分析完全一致，预算求和仅末位浮点顺序有约5.6e−17GPUh差，单列说明，不掩盖实质差异。`spotcheck.json`独立从每请求admission来源和raw timestamps核对153088/304、各会话E2E、来源守恒、H2D=external×28672、公共checkpoint及三个真实内容对照。

| 新run | 状态 | GPU·小时 |
|---|---|---:|
| content1 | FAILED | 0.01845790642833082 |
| content2 | FAILED | 0.01848323913388918 |
| content3 | FAILED | 0.013113411633336606 |
| content4 | COMPLETE / content PASS | 0.018495397286391786 |
| m11 | COMPLETE | 0.013614495415555818 |
| fair1 | COMPLETE | 0.013641070907777046 |
| full1 | COMPLETE | 0.013656039360830619 |

本增量 **0.10946156016611189 GPUh**；原0.5授权连续累计 **0.3195768596469497/0.5**，历史累计 **4.511564879374477 GPUh**。早期4.5仅限制旧32B三方法对照，未挪用其额度；此处按独立已批准0.5记账，未扩授权、未重置family。含所有启动、编译、失败、推理、退出与清理。CPU/native/tokenizer准备未初始化CUDA，GPU成本0。七run全部自有进程组退出，两GPU最后各19MiB/0%。现有cutlass.pth警告保留，未修改共享环境或其他服务。

复现见`results/system_m1_shared/commands.sh`，复算加`--variant shared_v1`选择固定cohort；全family仍参与预算统计，不因variant隔离而重置。准备和运行需唯一目录名与同一ledger；命令文件不是新GPU授权。

## 下一决定

保持动态M1和公共根计费的StaticFair都可用；共享compatibility不等于动态策略优势。此场景M1增量不到1%且DMA近翻倍，不扩大同一序列或32B矩阵。下一步先对回收/扩额的实际复用与重写建立事件成本记录，再判断是否值得引入更稳健的retention决策；共享深度变化与生产输入覆盖也仍待测。当前不能直接上M3或把公平quota包装成论文创新。按本交接提交/push后停止，供下一审查基于实际代码和负结果决策。
