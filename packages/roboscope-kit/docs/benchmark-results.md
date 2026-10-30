# RoboScope 推理 SDK 评测与吞吐量优化报告

更新时间：2026-07-22

## 实验定位

本报告验证的是：**将任意模型推理代码接入 RoboScope Kit 后，平台能否保持官方评测语义，并通过平台侧调度和运行时优化提高吞吐量。**

实验中选择 StarVLA 作为一份具体的推理代码样例，但报告对象是 SDK 接入、调参和 benchmark runtime，不把 StarVLA 作为标题或结论主体。SDK 用户只实现 `Policy.load()` / `Policy.act()`；平台负责进程生命周期、持久化任务队列、资源采样、K/g 自动选择、失败恢复和结果完整性校验。

LIBERO 与 RoboTwin 的 simulator、协议、模型 checkpoint 和实验 manifest 不同，以下指标完全分开汇报，不做跨 benchmark 吞吐量比较。

# LIBERO

## LIBERO 结论

- 官方原生协议 naive `K=1、g=1` 已完成 LIBERO-10 全量评测：**500/500 episodes、448/500 成功（89.6%）**，共执行 **138,223 policy steps**；评测墙钟 **4,053.13 s**，吞吐 **34.10 steps/s、0.1234 episodes/s**。
- 单 policy server 冷启动为 **25.12 s**，包含冷启动的端到端墙钟为 **4,078.25 s**。GPU 平均利用率 **13.8%**、P95 **31%**、峰值显存 **9,393 MiB**；进程累计平均 CPU 约 **9.14 核**，server 与 evaluator 峰值 RSS 上界合计约 **6.84 GiB**。
- 当前最早归档的未调优基线为 `K=6、g=2`：LIBERO-10、500 episodes，墙钟 **981.79 s**，成功 **448/500（89.6%）**。归档材料未记录该次运行的总 steps，因此 steps/s 留空。
- 后续官方协议运行在 `K=10、g=4` 下将墙钟降至 **699.14 s**，成功结果保持 448/500。
- 完整吞吐优化后，`K=36、g=16`、持久化队列、chunk size 1 的历史最好单次运行为 **180.40 s、765.39 steps/s、2.772 episodes/s**；相对官方 naive K=1/g=1，墙钟缩短 **95.5%（22.47×）**。相对 K=6/g=2 最早归档未调优基线，墙钟缩短 **81.6%（5.44×）**。
- SDK 重构的同机相邻 A/B 中，旧路径为 **385.21 steps/s**，SDK 路径为 **390.05 steps/s**，提升 **1.26%**；500/500 个终态记录完整，无缺失或重复。
- 冻结代码调参选择 `K=34、g=12`：60 秒稳态中位数 **823.67 steps/s**，比 K36/g16 高 1.37%。500-episode 整场测试则是 K36/g16 的中位墙钟更短，说明稳态吞吐和尾部完成时间的最优点不同。

## LIBERO 官方 naive 基线与优化结果

| 阶段 | 配置 | 实验范围 | 墙钟时间 | steps/s | episodes/s | 成功结果 | 说明 |
|---|---|---|---:|---:|---:|---:|---|
| 官方协议 naive | K=1，g=1 | LIBERO-10，500 ep | **4,053.13 s** | **34.10** | **0.1234** | **448/500（89.6%）** | 官方原生 `eval_libero.py`；138,223 steps；冷启动 25.12 s |
| 最早归档未调优基线 | K=6，g=2 | LIBERO-10，500 ep | **981.79 s** |  | 0.509 | 448/500 | 官方部署协议；总 steps 未归档 |
| 官方协议并发调优 | K=10，g=4 | LIBERO-10，500 ep | 699.14 s |  | 0.715 | 448/500 | 与 naive 每任务结果一致 |
| 静态 episode shards | K=36，g=16，chunk 13–14 | LIBERO-10，500 ep | 238.87 s | 575.65 | 2.093 | 451/500 | 吞吐实验，episode sharding 改变 RNG progression |
| 持久化队列 | K=36，g=16，chunk 1 | LIBERO-10，500 ep | **180.40 s** | **765.39** | **2.772** | 449/500 | 历史最好单次运行 |
| 同机旧路径 | K=36，g=16 | LIBERO-10，500 ep | 357.04 s | 385.21 | 1.400 | 451/500 | SDK 重构相邻 A/B 对照 |
| 同机 SDK 路径 | K=36，g=16 | LIBERO-10，500 ep | **351.29 s** | **390.05** | **1.423** | 452/500 | 500/500 记录完整 |

不同日期的绝对吞吐受到主机状态影响。SDK 是否回退只采用“同机相邻 A/B”判断；981.79→180.40 s 用于描述同一类 500-episode 工作负载的历史优化过程。

## LIBERO naive 资源占用

| 指标 | 结果 |
|---|---:|
| GPU 利用率（平均 / P95 / 峰值） | 13.8% / 31% / 37% |
| GPU 显存（平均 / 峰值） | 8,983 / 9,393 MiB |
| GPU 功耗（平均 / 峰值） | 95.6 / 114.4 W |
| GPU 峰值温度 | 43°C |
| 进程累计平均 CPU（server + evaluator） | 约 9.14 核 |
| server / evaluator 峰值 RSS | 3.05 / 3.79 GiB |

该次运行固定 `CUDA_VISIBLE_DEVICES=0`，使用 StarVLA commit `7e4a47360093d79dc42b883885c04fbe790c5b60` 和 checkpoint SHA-256 `67d26b27713d44c14af4ddb6eab05cd57e388f10caad153b657c8fdc64ded92f`。总 steps 由官方 evaluator 输出的 500 个视频逐帧计数得到；官方路径每个 policy action 写入一帧。

## LIBERO 调参与消融

| 消融项 | 配置 | 实验范围 | steps/s | 墙钟时间 | 结果 | 判断 |
|---|---|---|---:|---:|---|---|
| SDK 重构 | 旧路径 | 500 ep，同机 A/B | 385.21 | 357.04 s | 451/500 | 对照 |
| SDK 重构 | SDK 路径 | 500 ep，同机 A/B | **390.05** | **351.29 s** | 452/500；记录完整 | 保留 |
| 稳态 K/g | K34/g12 | 60 s × 3 | **823.67** |  | inference P95 214.16 ms | 稳态指标胜出 |
| 稳态 K/g | K36/g16 | 60 s × 3 | 812.53 |  | inference P95 263.43 ms | 稳态略低 |
| 整场 K/g | K34/g12 | 500 ep × 3 | 747.43 | 184.36 s | 成功数中位数 451 | 尾部稍慢 |
| 整场 K/g | K36/g16 | 500 ep × 3 | **755.59** | **181.78 s** | 成功数中位数 451 | 整场墙钟胜出 |
| Executor | static shards，chunk 13–14 | 500 ep | 575.65 | 238.87 s | 451/500 | 基线 |
| Executor | persistent queue，chunk 5 | 500 ep | 676.66 | 204.48 s | 445/500 | 改善 |
| Executor | persistent queue，chunk 2 | 500 ep | 734.30 | 187.66 s | 453/500 | 改善 |
| Executor | persistent queue，chunk 1 | 500 ep | **746.10** | **184.60 s** | 451/500 | 保留 |

# RoboTwin

## RoboTwin 结论

- StarVLA 官方多 GPU launcher 基线已在 4×RTX 4090 上完成 **50 tasks × 10 episodes = 500 episodes**：墙钟 **6,925.87 s**，共执行 **212,326 policy steps**，吞吐 **30.66 steps/s、0.07219 episodes/s**，成功 **227/500（45.4%）**。
- 该基线直接使用官方 `examples/Robotwin/eval_files/start_eval.sh`，配置为 4 GPU、`jobs_per_gpu=1`、共 4 slots；未使用 RoboScope SDK、Autotune、worker pool、persistent queue 或 RoboTwin runtime 修改。官方 RoboTwin evaluator 仅将每任务测试数从 100 改为 10，因此本行不使用 K/g 标记。
- 完整性校验通过：50/50 tasks、500/500 结果、500/500 可解析视频，`complete=true`，无 frame error 或汇总 issue。GPU 平均利用率 **39.5%**、P95 **68%**，单卡峰值显存 **19,447 MiB**，四卡平均总功耗 **588.7 W**。
- 相同 500-episode 范围下，SDK/runtime 的 4 GPU、1 lane × 8 processes/GPU 配置为 **491 s、423.28 steps/s、1.018 episodes/s、234/500**；相对官方基线，墙钟缩短 **92.9%（14.11×）**，steps/s 提高 **13.81×**。两次运行的总 steps 和成功数不同，因此吞吐对比用于衡量端到端执行效率，不作为严格 paired correctness 结论。
- pristine official evaluator 另有 50 tasks × 50 episodes 官方评分参考：**2,500/2,500** 条记录，成功 **1,159/2,500（46.36%）**，总计 **1,042,569 policy steps**；该历史记录没有完整墙钟或吞吐量。
- pristine official evaluator 路径的主要开销是每 episode 重建 simulator：`setup_demo` 约 22.7 s，其中 cuRobo `MotionGen` warmup 约 20.6 s；cold URDF load 约 8.7 s，首次 RT render 约 5 s。
- 当前完整优化配置为 1 lane × 8 processes/GPU、固定 batch 1、0 ms wait、每 16 条记录回收 worker。在 4×RTX 4090 上完成 2,500 episodes，达到 **475.04 steps/s、1.139 episodes/s**，墙钟 **2,194.7 s**，零 OOM、零 worker failure。
- 优化路径成功 1,158/2,500（46.32%），相对 pristine official 评分参考为 -0.04 个百分点，McNemar `p=1.0`；全部 2,500 个终止步数一致。
- 完整运行早于 `turn_switch` 最后一动作修复。修复后只完成了该任务 50 episodes 的严格一致性复验，尚未重新运行完整 2,500 episodes。

## RoboTwin 官方基线与优化结果

| 阶段 | 配置 | 实验范围 | 墙钟时间 | steps/s | episodes/s | 成功结果 | 正确性/说明 |
|---|---|---|---:|---:|---:|---:|---|
| StarVLA 官方 4-GPU 调度基线 | 官方 launcher；4 GPU × 1 job/GPU，共 4 slots | 50 tasks × 10 | **6,925.87 s** | **30.66** | **0.07219** | **227/500（45.4%）** | 无 SDK/Autotune/runtime 修改；212,326 steps；500/500 视频完整 |
| SDK/runtime 500-episode 优化 | 4 GPU；1 lane × 8 processes/GPU；fixed batch 1；0 ms | 50 tasks × 10 | **491 s** | **423.28** | **1.018** | **234/500（46.8%）** | 相同 episode 数；墙钟为官方基线的 1/14.11 |
| 官方评分参考 | pristine official `eval_policy.py` | 50 tasks × 50 |  |  |  | **1,159/2,500（46.36%）** | 2,500/2,500；1,042,569 steps；历史运行未记录墙钟 |
| 早期 SDK/runtime 基线 | 原生评测调度，K=24/g=4 | 24 tasks × 1 |  | 25.063 |  |  | 不同 manifest，仅作早期吞吐基线 |
| 完整优化基线 | 1 lane × 4 processes/GPU，adaptive batch 2 | 50 tasks × 50 |  | 315.23 |  | 1,154/2,500 | 完整 manifest，零 OOM |
| 当前完整配置 | 1 lane × 8 processes/GPU，fixed batch 1，recycle=16 | 50 tasks × 50 | **2,194.7 s** | **475.04** | **1.139** | **1,158/2,500（46.32%）** | 终止步数全部与官方一致 |

## RoboTwin 官方 4-GPU 基线资源占用

| 指标 | 结果 |
|---|---:|
| GPU 利用率（四卡样本平均 / P95 / 峰值） | 39.5% / 68% / 100% |
| 单卡显存（样本平均 / P95 / 峰值） | 14,588 / 15,878 / 19,447 MiB |
| 四卡总功耗（平均 / P95 / 峰值） | 588.7 / 720.9 / 794.5 W |
| GPU 温度（平均 / P95 / 峰值） | 48.0 / 55 / 61°C |
| 进程总 RSS（平均 / P95 / 峰值） | 29.1 / 32.2 / 65.3 GiB |

运行时间为 2026-07-22 18:13:18–20:08:44（Asia/Shanghai），StarVLA commit 为 `7e4a47360093d79dc42b883885c04fbe790c5b60`，RoboTwin commit 为 `c3ddfa8b97d5519efa828b075999bd0006778e5e`，checkpoint SHA-256 为 `8ccee8ddb33fa5df83a9a5e2b7d4f737db93f19a051b23f738af8a484fe2d7e5`。官方 evaluator 会在运行时用专家轨迹在线筛选 valid seed，这部分开销包含在墙钟内。

## RoboTwin 消融实验

| 消融项 | 配置 | 实验范围 | steps/s | episodes/s | 峰值显存 MiB/GPU | 正确性/结果 | 判断 |
|---|---|---|---:|---:|---:|---|---|
| Serving 模式 | adaptive batch 2 / 12 ms | 50×10，1×5 | 323.42 |  |  | batch 2 改变数值路径 | 拒绝 |
| Serving 模式 | fixed batch 1 / 0 ms | 50×10，1×5 | **342.07** | 0.823 | 32,647 | singleton 与官方数值一致 | 保留 |
| Worker 数 | 1×5 | 50×10 | 342.07 | 0.823 | 32,647 |  | 继续扩容 |
| Worker 数 | 1×6 | 50×10 | 368.37 | 0.886 | 36,277 |  | 继续扩容 |
| Worker 数 | 1×7 | 50×10 | 401.44 | 0.966 | 39,838 |  | 继续扩容 |
| Worker 数 | 1×8 | 50×10 | **423.28** | **1.018** | 44,029 | 491 s；234/500 | 保留 |
| Worker 数 | 1×9 | 50×10 | 385.37 | 0.927 | 46,809 | queue wait p50 184.9 ms | 拒绝 |
| Worker 回收 | recycle=16 | 50×10 | 396.02 |  |  | 500/500；零 OOM | 完整运行采用 |
| Worker 回收 | recycle=20 | 50×10 | 378.46 |  |  | 500/500；零 OOM | 单次较低，不能证明 16 必然更优 |
| 全局 cache 清理 | 每次任务切换 | 50×10 | 315.5 |  |  |  | 拒绝 |
| 全局 cache 清理 | 4,096 MiB 阈值 | 50×10 | 299.6 |  |  |  | 拒绝 |
| 调度策略 | strict task affinity | 50×10 | 263.22 |  |  | 尾部拖长 | 拒绝 |
| Lane 调度 | async，2 lanes × 4 processes/GPU | 28×3 | 213.96 |  |  |  | 移除 |
| Batch size | batch 4 | 28×3 | 141.73–200.01 |  |  | 延迟超过 batch 2 两倍 | 拒绝 |
| 无模型仿真 | 2 lanes × 8 processes/GPU | 28×3 | 395.25 |  | 27,163 | boundary rendering | 仅表示仿真吞吐，不与真实模型直接比较 |
| GPU 扩展 | 1 GPU，冠军同 manifest | 50×50 |  |  |  |  | 未完成 |
| GPU 扩展 | 4 GPU，冠军同 manifest | 50×50 | 475.04 | 1.139 | 44,501 | 2,500/2,500 完整 | 已完成 |
| 稳定性复跑 | 1×8 完整重复运行 | 50×50 |  |  |  |  | 未完成 |
| 修复后全量一致性 | `turn_switch` 修复后全套复跑 | 50×50 |  |  |  |  | 未完成 |

## RoboTwin 当前瓶颈

冠军运行累计 worker busy time 中，physics/control 占 49.5%，model client wait 占 27.3%，observation capture 占 23.2%。后续应优先优化 singleton 推理延迟与排队、原生 physics/control 循环、SAPIEN task-scoped cache 生命周期和观测序列化，而不是继续增加 lane、worker 或 batch size。

## RoboTwin 基准机规格

本报告所有 RoboTwin 运行使用 2026-07-18 在 `robotwin-vectorized-4090` 上采集的以下机器规格（容器名、进程 ID、利用率、温度与空闲内存是运行期状态，不算机器规格，已排除）：

| 组件 | 规格 |
|---|---|
| OS | Ubuntu 22.04.3 LTS，Linux 5.15.0-78-generic，x86-64 |
| CPU | 2 路 Intel Xeon Gold 6459C，每路 32 物理核、每核 2 线程 |
| CPU 容量 | 64 物理核、128 逻辑 CPU（`0-127`），最高 3.9 GHz |
| CPU 缓存 | L2 合计 128 MiB，L3 合计 120 MiB |
| NUMA | 2 节点：node 0 CPU `0-31,64-95`；node 1 CPU `32-63,96-127` |
| RAM | 容器可见 754 GiB，无 swap |
| GPU | 4 × NVIDIA GeForce RTX 4090，各 49,140 MiB，450 W 功耗上限 |
| GPU 驱动 | NVIDIA 595.58.03 |
| GPU 拓扑 | 4 张卡全部位于 NUMA node 1，CPU 亲和 `32-63,96-127` |
| GPU 互联 | 仅 PCIe，无 NVLink；GPU 1 与 GPU 2 共享同一 PCIe host bridge |
| cgroup | cgroup v2；评测 shell 观测到 CPU 亲和 `0-127` |
| 评测 Python | `/root/miniconda3/envs/robotwin` 中的 Python 3.10.20 |

GPU 上报显存高于零售 24 GiB RTX 4090，因此容量决策必须使用实测的 49,140 MiB，不能从产品型号推断显存。四张卡都在 NUMA node 1 意味着 CPU 绑定是一个实验变量：把 worker 绑到 node 0 可能增加 PCIe 流量开销，把所有 worker 集中到 node 1 又可能造成 CPU 争抢；两种策略都不能未经受控 A/B 就假定更快。

刷新该硬件快照的命令：`lscpu`、`free -h`、`nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit --format=csv`、`nvidia-smi topo -m`、`taskset -pc $$`、`cat /proc/self/cgroup`。

## RoboTwin 保真度验证（bit-exact trace）

优化运行时与 pristine official `eval_policy.py` 在相同 `(task, seed)` 对上逐 action vector、逐张边界 RGB 图做哈希对比，结果全部 bit-exact（`compare.json` 中 `all_exact: true`）：

| task:seed | steps (official = optimized) | action max abs diff | 匹配的边界图 |
|---|---:|---:|---:|
| adjust_bottle:100001 | 400 = 400 | 0.0 | 25/25 |
| beat_block_hammer:100000 | 400 = 400 | 0.0 | 25/25 |
| click_bell:100000 | 400 = 400 | 0.0 | 25/25 |
| open_laptop:100000 | 700 = 700 | 0.0 | 44/44 |

open_laptop 覆盖 700-step 长 horizon 与 expert-state restore 路径。该摘要是 4 task × 1 episode 的 focused scope；套件级对齐仍以 50×50 paired 运行（46.36% vs 46.32%，-0.04 pp，McNemar `p=1.0`，终止步数全部一致）为准。

## RoboTwin 冠军运行资源剖面

50×50 冠军运行（1 lane × 8 processes/GPU、fixed batch 1、recycle=16，2,194.7 s、475.04 steps/s）实测：

| 指标 | 结果 |
|---|---:|
| 请求延迟 p50 / p95 | 202.8 / 383.2 ms |
| 请求 queue wait p50 / p95 | 100.3 / 303.4 ms |
| 推理 prediction p50 / p95 | 69.1 / 204.3 ms |
| GPU 平均利用率 | 67.9% |
| CPU 平均占用 | 26.0 逻辑核 |
| 峰值显存 | 44,501 MiB（最繁忙的 49,140 MiB 卡上还剩 4,639 MiB） |
| 配置级开销 | 墙钟超过最慢 shard 墙钟 204.9 s（9.3%） |
| shard 墙钟离散 | 4.6% |

worker memory 增长不是 Python/CUDA 泄漏：30-episode 同任务压力运行把 worker 显存稳定在 3,286 MiB。增长发生在同一个持久 worker 打开新任务时——RoboTwin 以 `clear_cache=False` 关闭每个 scene，SAPIEN 把任务渲染资产保留在进程级全局缓存里，缓存生命周期长于任务，显存随 worker 见过的不同任务数增长。SAPIEN 在此环境只暴露整缓存清除，调用后下一 episode 要重建 ray tracing pipeline（约 10.7 s）；每次任务切换都清除会把 50×10 降到 315.5 steps/s，4,096 MiB 阈值降为 299.6 steps/s，task-contiguous 调度引入不可接受的终端 shard 长尾（严格亲和 263.22 steps/s），三者均被拒绝。最终采用 SDK 通用 worker 计划回收（每 16 条提交记录回收一次，146 次回收、零 OOM、零 worker 失败、峰值 44,501 MiB）；这是对第三方全局缓存的显式隔离措施，不代表已实现选择性缓存淘汰。

不要在没有新证据时把 worker 加到每 GPU 8 个以上：9 个 worker 把 50×10 从 423.28 降到 385.37 steps/s，queue wait p50 从 126.9 升到 184.9 ms，峰值显存从 44,029 升到 46,809 MiB。需要官方 parity 时也不要降低 ray-tracing 保真度或恢复 batch-2 serving。

# 待补实验

- LIBERO K=6/g=2 最早归档基线的总 steps 与 steps/s。
- RoboTwin 50 tasks × 50 episodes 官方 launcher 全量运行的完整墙钟与吞吐量。
- RoboTwin 冠军配置同 manifest 的 1 GPU/4 GPU 扩展效率。
- RoboTwin 1×8 完整运行重复性。
- `turn_switch` 修复后的 2,500-episode 全量一致性复跑。

# 数据来源

- `roboscope-kit/packages/roboscope-kit/README.md`
- Git 历史中的 LIBERO 500-episode 与 RoboTwin paired-report 结构化实验记录
- 2026-07-22 AutoDL 4×RTX 4090 官方原生 LIBERO K=1/g=1 全量运行日志与资源采样
- 2026-07-22 AutoDL 4×RTX 4090 StarVLA 官方 RoboTwin 多 GPU launcher 500-episode 运行日志、视频逐帧校验与资源采样
- 本报告的 RoboTwin 基准机规格、保真度验证与冠军运行资源剖面来自已归档的 RoboTwin vectorized runtime 实验记录，现已并入本报告正文
