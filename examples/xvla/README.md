# X-VLA Policy 示例

一个 `roboscope-kit` `Policy`，运行 X-VLA-0.9B（[2toINF/X-VLA](https://github.com/2toINF/X-VLA)，
arXiv:2510.10274）的公开 checkpoint，覆盖 calvin、libero、libero_plus、robotwin 四个评测集，
展示有状态 Policy 和多 Benchmark action contract 的写法。公开榜单中的 X-VLA 条目属于上游论文与社区成绩记录。

## 验证范围与研究对照（2026-08-15）

| 评测集 | demo 使用的 X-VLA checkpoint | 上游论文/社区指标（研究对照） | 说明 |
|---|---|---|---|
| calvin（task_ABC_D） | `X-VLA-Calvin-ABC_D` | avg_len **4.43** | 社区指标可对照 |
| libero（4 suites） | `X-VLA-Libero` | 98.2/98.6/97.8/97.6，平均 **98.1** | 社区指标可对照 |
| libero_plus（4 suites） | `X-VLA-Libero` | 无条目 | 如实报告，无社区参照 |
| robotwin（demo_clean/demo_randomized） | `X-VLA-RoboTwin2` | easy **70.0** / hard **39.0** | benchmark protocol 一致时具备直接可比性 |
| robocasa（tabletop） | — | 无条目 | **N/A**：X-VLA 公开资产范围仅覆盖表中其余 Benchmark |

RoboTwin 研究对照数字来自 X-VLA 上游固定提交
[`bcc5152`](https://github.com/2toINF/X-VLA/blob/bcc515295e9286394db607e9fe673b41f25df627/evaluation/robotwin-2.0/README.md)
的 leaderboard settings 结果表。平台结果在 benchmark protocol 与 cache provenance
一致时具备直接可比性。

## 运行环境

- conda env `xvla`（python 3.10 + torch 2.1.2 cu121 + transformers 4.51.3 + X-VLA
  requirements + `sapien==3.0.0b1` + `mplib==0.2.1`）
- `/opt/roboscope/xvla/X-VLA`：上游开源仓库代码
- `/opt/roboscope/xvla/checkpoints/{calvin,libero,robotwin}`：权重（HF 下载）
- `/opt/roboscope/xvla/xvla_demo_policy.py`：demo SDK Policy 适配器
- `/workspace/roboscope-policy.json`：策略入口（schema_version 2）

权重放在 `/opt/roboscope/xvla/checkpoints/` 下，策略忽略 `SessionContext.checkpoint_path`，
平台侧 model artifact 用占位文件即可。

## 适配要点（2026-08-15 按社区 Recipe 严格移植，不调参）

- **libero / libero_plus**：与 libero_client.py 逐字一致——绝对 EE 动作
  `[pos3, axis-angle3, grip±1]`、30 步 chunk、proprio 跟踪（首帧真实状态→chunk 末目标）、
  domain_id=3、steps=10。proprio 旋转用 **site 系**（= 控制器 ee_ori_mat，观测
  `robot0_eef_quat` 是 body 系 xyzw，差恒定 90° z 旋转，`_libero_site_rotmat`）。
  worker 补丁（SDK 内）：`use_delta=False`（Recipe 原句）+ MAX_STEPS 对齐
  Recipe 的 800/900 步 horizon。
- **calvin**：与 calvin_client.py 一致——绝对 `(pos, quat_xyzw, grip)` 元组、
  取模型 chunk 前 20 行、domain_id=2、gripper 阈值 0.8。平台 calvin 规格 7 维
  相对动作，本 Policy 通过 `PolicyCapabilities` 声明
  `calvin.absolute-compact-quat-v1`；benchmark adapter 根据协商后的 contract
  选择绝对语义。适配器先把单位 quat 规范到 `w>=0`，再编码 xyz 3 维；worker 保留 xyz
  原值并由单位约束重建非负 w（仅容忍浮点误差导致的轻微越界），env 收到与
  Recipe 等价的绝对动作。
- **robotwin**：X-VLA 是 EE 策略（16 维 pos+quat+grip），平台 RoboTwin 规格 14 维
  qpos。本 Policy 通过 `PolicyCapabilities` 声明 `robotwin.ee-compact-quat-v1`，
  benchmark adapter 根据协商后的 contract 选择 EE 执行路径；适配器把每臂动作编码为
  `[pos3, canonical quat_xyz3, grip]`，launcher 重建完整 16 维动作并调用原生
  `take_action(..., action_type="ee")`。首个 chunk 的 proprio 用 SAPIEN FK 还原真实
  endpose；后续 chunk 按 Recipe 使用上个 chunk 的最后目标 pose，并从本次真实 joints
  更新 gripper。seed cache 仍只负责跳过 expert `play_once`，EE 执行保留原生
  `CuroboPlanner`。StarVLA 显式声明 qpos contract。
- robocasa 未声明 capability（X-VLA 无 RoboCasa 权重，见范围表）。

## 实机验证

- `--quick-test` 与其它 preflight 用于有界 smoke；完整 suite 提供研究指标对照证据。
- full-run gate 运行平台定义的完整 suite。`--episodes 1` 只缩减每个 task 的 episode
  数，并不把 suite 变成小样本；例如 LIBERO+ 仍会展开 10030 个 variants。
- `outer rc=0`、testlog 为 `finish`、errors.jsonl 无 terminal failure
- 指标与上游论文/社区 X-VLA 条目按相同 protocol 比对
