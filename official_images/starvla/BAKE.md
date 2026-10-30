# StarVLA 镜像烘焙

## 构建 bundle（需公网）

合入 RoboScope Kit `main` 后，以目标提交构建 bundle。构建脚本会下载并校验
Control Runtime 的全部固定安装产物（uv 二进制、CPython 3.11.13 standalone
archive、`uv.lock` 解析出的 numpy wheel），随 bundle 一起交付：

```bash
official_images/starvla/build_official_image_bundle.sh --ref <target-commit>
```

产物版本由 `benchmarks/starvla_platform_runtime/control-runtime-dependencies.json`
锁定（numpy 直接从 SDK `uv.lock` 解析，不单独维护）；URL 与 SHA-256 在
`control-runtime-artifacts/control-runtime-artifacts.json` 中汇总，并被 bundle
`payload.sha256` 覆盖。更换 uv / CPython 版本时同步更新该 JSON 即可。

## AutoDL 离线安装与验证

用当前官方镜像创建全新 AutoDL 实例，上传并解压 bundle，直接运行安装、验证脚本。
安装全程只使用 bundle 内的本地产物，不访问 PyPI、GitHub 或其他包源，无需手动安装
`uv`：

```bash
tar -xzf roboscope-official-starvla-image-<commit>.tar.gz
cd roboscope-official-starvla-image-<commit>
./install.sh
```

核对镜像内 `/opt/roboscope/control-runtime`、SDK、StarVLA commit 和 checksum。
以同一提交部署 `/root/autodl-fs/benchmarks` 下的薄 launcher，跑真实快速测试与
正式路径 smoke。

部署 RoboTwin 时，使用同一提交的解压 bundle 安装准备阶段的 Control Runtime。
该流程复用离线安装器及 bundle 中校验过的 CPython、numpy 和 uv，避免重新联网下载：

```bash
ROBOSCOPE_INSTALL_RUNTIMES='calvin robocasa liberoplus robotwin' \
ROBOSCOPE_CONTROL_RUNTIME_BUNDLE_DIR=/root/roboscope-official-starvla-image-<commit> \
  bash benchmarks/starvla_platform_runtime/install-platform-runtimes.sh \
    /root/autodl-fs/benchmarks <target-commit>
```

SDK 0.7.0b51 起，RoboTwin 的两个完整正式 seed cache 随镜像固定为
`robotwin2-platform-native-step-lim-v2`。准备阶段从镜像归档恢复至实例本地并校验
generation 摘要；Quick Test 的成功不能代替该检查。来源与新旧成绩比较限制见
[固定输入协议](../../benchmarks/starvla_platform_runtime/robotwin/SEED_PROTOCOL.md)。

## 清理与保存

清理下载包、bundle 解压目录、临时脚本、`/root/.cache` 缓存和临时 SSH key 后关机
保存，等待镜像可用。将新 UUID、SDK 版本、创建时间写入 Cloud 配置，部署后冷启动复验。

复用验收实例烘焙时，提交的清理脚本会将 `/root/autodl-fs` 软链接恢复为
AutoDL 的规范挂载入口 `/autodl-fs/data`，避免把实例本地的临时 runtime 入口保存进镜像。
