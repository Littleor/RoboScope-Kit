# StarVLA 官方镜像技术方案

## 1. 目标

RoboScope 模型评测保留两条独立通路：

```text
通路 A：用户自定义镜像评测
通路 B：StarVLA 官方镜像评测
```

两条通路都支持快速测试和正式评测。快速测试与正式评测互为独立操作。

| 评测通路 | 快速测试 | 正式评测 |
| --- | --- | --- |
| 用户自定义镜像 | 支持 | 支持 |
| StarVLA 官方镜像 | 支持 | 支持 |

StarVLA 官方镜像只提供 StarVLA runtime 和 SDK Policy。它仅根据 Benchmark 名称选择 pinned StarVLA 参考客户端的图像预处理，不包含也不识别任何 Benchmark 的 simulator、数据集、suite、evaluator、worker、相机布局、action 维度或计分逻辑。

Benchmark simulator、数据集和薄 launcher 位于 `/root/autodl-fs/benchmarks`，
由平台在评测任务启动时提供。完整控制 SDK 和独立 Control Python 随镜像
烘焙在 `/opt/roboscope/control-runtime`；`evaluation-kit-root.txt` 和
`control-python.txt` 始终引用这一固定镜像内路径。

“用户自定义镜像”不是一份平台预制镜像，而是平台现有的镜像制作机制：用户从可用镜像创建开发实例，安装或修改模型、依赖和 SDK Policy，停止实例后保存为个人镜像，再使用该个人镜像执行快速测试或正式评测。

当前平台使用已经完成实机验证的旧专用镜像作为官方基础镜像。它包含较多 LIBERO 和 StarVLA 专用内容，不是理想的最小通用基础镜像，但现阶段继续作为用户创建开发实例和保存个人镜像的起点。基础镜像和 StarVLA 官方镜像必须烘焙同一提交的 Control Runtime；个人镜像从新版基础镜像继承它。

## 2. 用户自定义镜像机制与评测通路

现有用户流程为：

1. 从当前官方基础镜像或已有个人镜像创建开发实例。
2. 安装或修改任意具身模型、Python 环境和依赖。
3. 实现并配置 RoboScope SDK `Policy`。
4. 停止实例并保存为新的个人镜像。
5. 使用个人镜像执行快速测试或正式评测。

用户镜像负责：

- 模型代码和 Python 环境。
- 模型依赖。
- RoboScope SDK `Policy` 实现。
- `/workspace/roboscope-policy.json`。
- observation 到模型输入的转换。
- 模型输出到目标动作的转换。

平台将用户选择的模型权重目录下载到评测实例，并通过 `SessionContext.checkpoint_path` 交给用户 Policy。用户 Policy 自行解释目录内容。

评测页面现有“个人镜像”列表即为该通路的镜像选择入口。本任务不改变个人镜像的创建、保存、查询或评测协议。

## 3. StarVLA 官方镜像通路

用户只需要：

1. 上传一个可离线加载的 StarVLA run snapshot。
2. 选择 StarVLA 官方镜像。
3. 直接执行快速测试或正式评测。

用户无需创建评测开发实例、编写 SDK Policy、安装 StarVLA、配置 Python 环境或保存个人镜像。

官方镜像负责：

- 固定版本的 StarVLA 源码和 Python 环境。
- RoboScope Kit（`roboscope-kit`）。
- `/workspace/roboscope-policy.json`。
- StarVLA run snapshot 结构验证。
- StarVLA config 驱动的 framework、processor 和推理初始化。
- StarVLA full checkpoint strict load。
- 按当前 Benchmark 的 pinned StarVLA 参考客户端预处理配置指定的图像 feature，再传给 StarVLA。
- 返回配置指定形状和变换后的 action chunk。

`/root/autodl-fs/benchmarks` 下的 Benchmark runtime 负责：

- Benchmark simulator、数据集、suite 和 evaluator。
- Benchmark worker。
- Benchmark 原生 observation feature。
- episode 调度、K/g 探测、进度、结果和清理。
- 将 StarVLA 返回的通用 action chunk 应用到当前 Benchmark。

## 4. 目录边界

新镜像源码位于：

```text
official_images/starvla/
```

该目录只允许包含：

```text
official_images/starvla/
├── AGENT.md
├── starvla-official-image-design.md
├── build_official_image_bundle.sh
├── install_official_image_bundle.sh
├── verify_official_image.sh
├── prepare_official_image_for_bake.sh
├── startbenchmark.sh
├── resetenv.sh
├── roboscope-policy.json
├── starvla_official_policy.py
├── starvla_full_checkpoint.py
└── export_starvla_eval_run.py
```

该目录只允许使用 Benchmark 名称选择 StarVLA 上游参考预处理；禁止包含 suite、simulator、evaluator、worker、相机布局、action 维度、gripper 语义或计分逻辑。

已经完成实机验证的旧专用镜像当前仍提供基础镜像内容。其构建链路在
`benchmarks/libero_official/` 中增加同一套镜像内 Control Runtime，但不改变
Benchmark simulator 或用户 Policy 协议。

## 5. 启动协议

两类镜像都从固定入口启动：

```bash
/root/startbenchmark.sh <suite> <checkpoint_source> [evaluation_num]
```

StarVLA 官方镜像中的 `/root/startbenchmark.sh` 是薄代理：

```text
/root/startbenchmark.sh
    -> $ROBOSCOPE_PLATFORM_ENTRYPOINT
        -> 当前 Benchmark runtime
        -> /workspace/roboscope-policy.json
        -> /opt/roboscope/starvla/starvla_official_policy.py
```

Cloud 根据本次评测的 Benchmark 设置绝对路径
`ROBOSCOPE_PLATFORM_ENTRYPOINT`，其目标位于 AutoDL 文件存储中的对应
Benchmark runtime。镜像不得猜测 Benchmark，也不得写死一个不存在的统一目录。

`/root/resetenv.sh` 同样通过可选的 `ROBOSCOPE_PLATFORM_RESET --reset` 代理到平台提供的
reset 入口；平台未提供 reset 脚本时不执行额外操作。

官方镜像不实现 Benchmark runtime，也不把 Benchmark runtime 打进镜像 bundle。

## 6. StarVLA run snapshot

官方通路接收标准 StarVLA 训练输出快照：

```text
<run>/
├── config.yaml
├── dataset_statistics.json
├── checkpoints/
│   └── <selected full checkpoint>
└── <config.yaml 引用的本地模型构造资源>
```

要求：

- `config.yaml` 描述 framework、processor、模型构造资源和推理参数。
- `dataset_statistics.json` 与训练配置配套。
- `checkpoints/` 中固定一个本次评测选择的完整 checkpoint。
- 配置引用的本地资源保持可解析的相对路径。
- 整个目录可以在关闭外部模型下载的环境中加载。

## 7. 配置驱动加载

加载链路为：

```text
checkpoint_path 或 run_dir
    -> 定位唯一 run snapshot
    -> 定位唯一 selected checkpoint
    -> read_mode_config(checkpoint)
    -> config.yaml
    -> dataset_statistics.json
    -> build_framework(config)
    -> strict state-dict load
```

`checkpoint_path` 只用于定位 checkpoint 和 run 目录。模型构造由相邻 `config.yaml` 驱动。

`starvla_full_checkpoint.py` 提供显式 API：

```python
load_full_checkpoint(checkpoint_path)
```

它负责读取配置、创建 empty-weight framework、读取 state dict，并使用 `strict=True, assign=True` 加载。该 loader 不依赖训练数据，也不根据 Benchmark 推断模型资产。

Policy 在进入离线加载前读取 `framework.qwenvl.base_vlm`。如果 run snapshot
中尚无该目录，则使用平台注入的 `BENCHMARK_BASE_*` 只读凭证，从
`oss://<bucket>/benchmark/basemodel/<base-model>/` 自动下载模型构造所需的
config、tokenizer 和 processor metadata，并排除所有权重文件。并发 Policy
replica 通过同一份本地 cache 和文件锁只下载一次，再把 cache 链接到配置声明的
run snapshot 相对路径。完整模型参数仍只来自用户上传的 full checkpoint。

## 8. 通用输入配置

StarVLA run snapshot 的 `inference.inputs` 声明 Policy 需要从平台 Observation 读取的 feature 名称：

```yaml
inference:
  dtype: bfloat16
  unnorm_key: franka
  inputs:
    images:
      - images.primary
    language: task
    state: robot_state
  kwargs:
    do_sample: false
    use_ddim: true
    num_ddim_steps: 10
```

Policy 按 Benchmark 名称复现镜像所固定 StarVLA 上游参考客户端的图像预处理，再与 language feature 组装为 StarVLA example：RoboTwin、RoboCasa 和 LIBERO+ 使用 OpenCV `INTER_AREA` 缩放到 `224×224`；LIBERO 使用 PIL 双线性缩放到 `224×224`；CALVIN 使用上游 `resize_with_pad` 等比例缩放并补黑边到 `224×224`，再执行 `convert_to_uint8`。`224×224` 是 pinned StarVLA 参考客户端的模型输入尺寸，不是 Benchmark Observation 规范。该预处理属于官方 StarVLA Policy，不属于 Benchmark Runtime；自定义镜像仍收到 Benchmark 原始 Observation。它不要求也不改写用户上传的 StarVLA `config.yaml`。配置声明 state 时，Policy 先按 checkpoint 对应的训练时 DataConfig 拆分原始 state，复用同一套 transform pipeline 处理，再组装为 StarVLA example：

```python
{
    "image": [features[name] for name in config.inputs.images],
    "lang": features[config.inputs.language],
    "state": training_transform(features[config.inputs.state]),
}
```

`state` 可省略。Policy 不内置任何 Benchmark feature 名称，只按 Benchmark 名称选择图像预处理；上例只是说明配置格式。

Policy 在 `load()` 时通过 `PolicyCapabilities.required_features` 返回配置声明的 feature。平台 Benchmark runtime 负责提供这些 feature。

## 9. 通用动作配置

StarVLA config 中的 `framework.action_model.action_dim` 声明动作维度，`action_horizon` 声明 chunk 长度。

可选 `inference.action_transforms` 只支持通用数值变换：

```yaml
inference:
  action_transforms:
    - operation: threshold
      index: 6
      threshold: 0
      below: -1
      above: 1
    - operation: affine
      index: 6
      scale: -1
      offset: 0
```

支持：

- `threshold`：按阈值输出离散值。
- `affine`：执行 `value * scale + offset`。

Policy 不理解任何维度的业务语义。维度含义由训练配置、数据配置和当前 Benchmark runtime 共同约定。

## 10. 平台边界

平台读取用户选择的上传目录，签发只读下载凭证，选择官方镜像或用户镜像，然后启动固定 Benchmark 命令并轮询进度和结果。

## 11. 快速测试与正式评测

快速测试验证：

- run snapshot 结构完整。
- StarVLA config 可以读取。
- full checkpoint 可以 strict-load。
- 配置声明的 Observation feature 存在。
- 配置的 action 维度和 chunk 长度匹配模型输出。
- 一个真实 episode 可以完成。

正式评测使用同一镜像、同一 Policy 配置和同一模型目录协议，执行完整 episode manifest。

快速测试不是保存镜像或提交正式评测的前置条件。

## 12. Run snapshot 导出

`export_starvla_eval_run.py` 负责从训练输出导出评测快照：

- 复制 `config.yaml`。
- 复制 `dataset_statistics.json`。
- 复制一个 selected checkpoint。
- 复制配置引用的本地资源。
- 固化 `inference` 配置。
- 执行离线 strict-load preflight。

示例：

```bash
python /opt/roboscope/starvla/export_starvla_eval_run.py \
  --run-dir <training-run> \
  --checkpoint <training-run>/checkpoints/<checkpoint> \
  --output <eval-run> \
  --resource-root <training-workspace>
```

## 13. 镜像构建与验证

构建 bundle：

```bash
official_images/starvla/build_official_image_bundle.sh \
  --ref <engine-commit> \
  --output-dir dist/official-image
```

安装 bundle：

```bash
./install.sh
```

bundle 同时包含两份职责不同的 SDK：StarVLA Python 只安装通用 contracts、
Policy、protocol 和 server 组成的 Policy-core wheel；独立 Control Python
使用完整 SDK source，负责 Benchmark 注册表、client、配置加载器和 runtime。

StarVLA 镜像安装和烘焙前会删除基础镜像遗留的
`/root/roboscope-kit`，完整控制 SDK 只从固定的
`/opt/roboscope/control-runtime` 加载。

验证内容包括：

- bundle payload checksum。
- 安装后 runtime checksum。
- Policy-core SDK 版本。
- 镜像内 Control Python、完整控制 SDK、RoboScope Kit commit 和 SDK 版本。
- 安装后的 SDK 不包含具体 Benchmark 注册表或 `roboscope_kit.runtime`。
- 通用 StarVLA Policy import。
- `/workspace/roboscope-policy.json` 与 manifest 一致。
- pinned StarVLA commit。
- 清理后的 workspace 只保留默认 Policy 配置。

Benchmark 快速测试与正式评测由镜像内 Control Runtime 驱动，并使用
`/root/autodl-fs/benchmarks` 下的 simulator、数据集和薄 launcher。

## 14. 发布顺序

1. 用同一 RoboScope Kit 提交构建基础镜像和 StarVLA bundle。
2. 分别在全新 AutoDL 实例安装并验证镜像内 Control Runtime。
3. 保存新的基础镜像和 StarVLA 官方镜像。
4. 部署只引用固定镜像内路径的 `/root/autodl-fs/benchmarks` 薄 launcher。
5. 用个人镜像和 StarVLA 官方镜像完成快速测试与正式路径 smoke。
6. 更新 Cloud UUID；旧个人镜像从新版基础镜像重建后再评测。
