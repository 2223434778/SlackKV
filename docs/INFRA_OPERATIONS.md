# 安装、运行、复现与排障

## 环境边界

实际GPU验证使用Python3.11.16、torch2.11.0+cu126、vLLM distribution0.23.0+cu126、transformers5.17.0、CUDA12.6/driver550.54.14/RTX A6000。源码固定0fc695fc6，完整Python树校验；不是“任意最新vLLM均可”。CPU schema/协议测试支持Python3.11+，Windows不能启动POSIX受管GPU服务。

已实测的是**独立固定上游Python+匹配既有编译库**。干净机器从源码完整编译未在本交付重做；下列源码安装路径应按该commit的build requirements核对，CUDA/torch不匹配应停止，不覆盖正在服务的旧环境。

## 1. 独立环境和固定源码

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
slackkv doctor --config configs/local.json --config-only

# 将上游源码放在本项目外，避免误提交嵌套仓库。
export UPSTREAM="$HOME/src/vllm-0fc695fc6"
git clone https://github.com/vllm-project/vllm.git "$UPSTREAM"
git -C "$UPSTREAM" checkout 0fc695fc6
```

已有匹配vLLM环境可以只在其中安装本项目（不安装/升级runtime依赖），并找出编译库根目录：

```bash
python -m pip install --no-deps -e .
export COMPILED="$(python -c 'import importlib.util,pathlib;print(pathlib.Path(importlib.util.find_spec("vllm").origin).parents[1])')"
slackkv prepare-runtime --source "$UPSTREAM" --compiled "$COMPILED" --target "$PWD/.runtime/native"
```

新建环境需先准备与验证表一致的torch/CUDA及上游编译依赖，然后构建固定源码（此分支本次未重新编译验证）：

```bash
# 在独立环境按上游该commit要求安装build依赖和CUDA12.6 torch。
# 不让另一个CUDA版本的torch替换已选定的2.11.0+cu126。
VLLM_TARGET_DEVICE=cuda VLLM_VERSION_OVERRIDE=0.23.0+cu126 python -m pip install "$UPSTREAM"
python -m pip install transformers==5.17.0
# 再按上面的COMPILED/prepare-runtime步骤创建overlay，doctor核对版本。
```

`prepare-runtime`不修改donor、既有服务或上游源码，不下载模型；目标必须全新且独立。原生max_offload_tokens已经存在，不需要应用旧research patch。生成version文件和编译库来自同一已匹配commit。重复同参数会校验目标hash；未知/部分目录拒绝覆盖。

上游树hash仅排除构建生成的顶层`vllm/_version.py`，其他Python源码仍逐字节（统一换行）匹配。donor的生成版本文件必须包含固定commit，复制后连同编译库一并hash；若构建override产生的文件缺少commit，不手改以绕过检查，先核对源码构建产物，当前干净构建路径仍标未验证。

## 2. 模型和doctor

```bash
export SLACKKV_MODEL=/absolute/path/to/local/Qwen2.5-1.5B-Instruct
export SLACKKV_RUNTIME="$PWD/.runtime/native"
slackkv doctor --config configs/local.json
```

模型自行提前准备；受测snapshot为989aa7980e4cf806f80c7fef2b1adb7bc71aa306。doctor不自动下载，不初始化模型CUDA；仅查model/config、包版本、hash、选中GPU和可用RAM/端口。无GPU或只有文档阅读需求时使用`--config-only`，它不证明KV内容正确。

## 3. 一次真实demo与报告

```bash
# 仅一次初始化，已有适用授权才运行GPU；已有账本直接复用。
slackkv init-budget --ledger "$PWD/results/infra_engineering_20261016"
slackkv demo --config configs/local.json --ledger "$PWD/results/infra_engineering_20261016" --name reader-demo-unique
slackkv report --directory results/infra_engineering_20261016/runs/reader-demo-unique
```

查看requests.json、sessions.json、tools.json、requests.csv、summary.md、server.log和execution.json。demo真实输出参与后续输入，不与其他policy的自由输出强行配对。使用新的唯一run名，不能覆盖历史。服务自行退出；独立serve需Ctrl+C或`slackkv stop --directory <run-dir>`，并等待execution.json确认STOPPED/cleanup_verified。

无GPU也能使用仓库内已归档的真实demo。日志和指标使用无损gzip，报告命令支持压缩指标：

```bash
slackkv report --directory results/infra_engineering_20261016/runs/e1_demo2 --output .tmp/reader-report
python scripts/verify_infra.py --directory results/infra_engineering_20261016 --output .tmp/reader-verification.json
```

不要对历史目录省略`--output`；默认写回输入目录，适合刚结束的自己的run。`report`用于真实demo；benchmark已有`benchmark_summary.md`、CSV和JSON，使用`bench-compare`复算对照。

## 4. 固定用途benchmark

准备是CPU/tokenization，不启动服务；执行时每条命令自己启停服务，使用同一账本：

```bash
slackkv bench-prepare --config configs/bench-host-limited.json --scenario host-limited --directory results/my-fixed-plan
slackkv bench --config configs/bench-host-limited.json --plan results/my-fixed-plan --policy nohost --ledger "$PWD/results/infra_engineering_20261016" --name reader-nohost-unique
slackkv bench --config configs/bench-host-limited.json --plan results/my-fixed-plan --policy full --ledger "$PWD/results/infra_engineering_20261016" --name reader-full-unique
slackkv bench --config configs/bench-host-limited.json --plan results/my-fixed-plan --policy fair --ledger "$PWD/results/infra_engineering_20261016" --name reader-fair-unique
slackkv bench-compare --directories results/infra_engineering_20261016/runs/reader-nohost-unique results/infra_engineering_20261016/runs/reader-full-unique results/infra_engineering_20261016/runs/reader-fair-unique --output results/my-comparison.json
```

hot与host-enough使用对应config和scenario，不能把不同资源的结果混比。host-enough配置1GiB host，其余两场景224MiB；GPU均为明确512MiB实验池，非A6000物理限制。case内三方法配置相同，只有policy不同。hot没有压力请求，另两场景各有两个16K压力请求。shape预热使用新salt和zero host store。

每个输入token序列冻结，输出规定16tokens、ignore_eos=true并核对实际IDs/usage；这是有意的合成benchmark，**不是真实聊天**。真实demo不设置ignore_eos。compare拒绝未完成、不同输入/资源/实测GPU池或重复policy。各case一次，不声称稳定速度或SLO；允许Fair更慢。修改源码或输入后应新建plan，不能悄悄改冻结计划。

## 5. 验证host和失败路径

复制local.json为独立oracle配置，明确设置batch_invariant=true，其他资源不变：

```bash
slackkv verify-host --config /absolute/oracle.json --ledger "$PWD/results/infra_engineering_20261016" --name reader-host-unique
```

该检查先CPU确认固定尺寸/marker位置，才启动GPU。保存实际source/bytes、完整输出IDs、cold零命中以及marker确在host区间的断言；不能拿其时间和非invariant benchmark比较。

对自己前台启动且ready的服务，可以从另一终端运行真实断流检查：

```bash
slackkv reliability --config configs/local.json --directory /absolute/owned-run
slackkv stop --directory /absolute/owned-run
```

它验证收到首段内容后的HTTP关闭、失败历史原子性、新generation请求可用和scheduler最终idle。服务端abort/自然完成区别仍UNKNOWN，不用逻辑close冒充物理回收。

## 常见故障

| 现象 | 定位/处理 |
|---|---|
| 缺模型/占用端口/选中GPU忙 | doctor会拒绝；换明确配置或等资源，不杀其他服务 |
| 包版本或upstream树不匹配 | 使用固定环境/固定commit；不要关闭hash检查或升级现有运行环境碰运气 |
| 目标overlay已存在但无manifest | 保留失败目录，使用新的独立目标；不自动递归删除 |
| 请求超上下文/槽位/同会话并发 | 明确拒绝；先查看history/state，失败身份close后新generation，不能原salt盲目重试 |
| 不完整SSE、请求超时 | history未提交；若有失败快照查看client_failure.json，再查server.log/执行状态 |
| RUNNING或cleanup未确认 | 停止新增GPU工作，核查精确owner PID/出生时间/进程组和GPU状态；不要改成COMPLETE或清空账本 |
| 字节为UNKNOWN | 检查metric起点及label；不能用0替代缺失，NoHost由实际无connector配置另证 |
| Windows中文报告失败 | 工程主线显式UTF8；CPU测试含非ASCII。不要修改旧raw数据编码 |
| cutlass .pth/semaphore告警 | 受测环境原有告警保留；是否影响本run看真实退出/cleanup，不能仅凭告警判成功或失败 |

CI只运行CPU配置/协议/控制器与POSIX小进程测试，缺native时明确skip；GPU验证依靠保存的真实记录。没有Docker/Kubernetes/Redis/多机依赖。历史32B数据无需重跑即可阅读，旧研究授权不能用于新矩阵。
