# RIR 重构版使用与变更说明

版本：0.2.12。日期：2026-09-05。
基于 `alexandergwm/Tools` 的 `codex/rir-asio-audit-20260905` 分支，提交
`a29a30be2ad83388619267643711d0c7be8a44c0`。

这是可以安装运行的 Python 源码包，需要 Python 和项目依赖；不包含预编译 EXE 或嵌入式 Python。
原下载源码单独保留，本次修改在独立副本完成，没有向 GitHub 提交或推送。

## 快速开始

Windows 解压后进入 `python_acoustic_capture` 文件夹，双击 `start_gui_windows.bat`。
启动器会让你选择 Python 环境，并安装缺少的依赖。声卡驱动和 ASIO/TotalMix 仍需使用现场配置。

也可以在该文件夹中打开 PowerShell，手动安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m acoustic_capture rir configs/simulated.yaml
```

最后一条命令使用模拟后端，不访问扬声器或麦克风；结果写入解压目录下的 `runs`。
正常模拟示例会采集 5 次，并产生 `processed/average_rir.wav` 和 `metrics/summary.json`。
已有示例配置中的作者 E 盘路径均已改为相对于 YAML 的 `../audio` 和 `../runs`，可在其他盘解压。

需要测试 GUI 的音频/语音流程时，可先生成合成演示音频：

```powershell
.\.venv\Scripts\python.exe -m acoustic_capture demo-audio --output-dir audio
.\.venv\Scripts\python.exe -m acoustic_capture gui configs/simulated.yaml
```

真实测量使用 `configs/rme_ucx.yaml`，先按现场设备调整声卡、输入输出通道与元数据；更多信息见
[README.md](README.md) 和 [WINDOWS_RME_GUIDE.md](WINDOWS_RME_GUIDE.md)。

## 结构调整

| 模块 | 职责 |
| --- | --- |
| `signals.py` | ESS、静音和播放通道路由 |
| `audio.py` | 声卡生命周期、音频回调、取消与完整性检查 |
| `deconvolution.py`（新增） | 正则化逆滤波、IR 提取、共同裁剪窗口验证 |
| `timing.py`（新增） | 直达声检测、GCC-PHAT、低频群延迟、扫频片段时序诊断 |
| `rir_quality.py`（新增） | 重构验收、延迟有效性与最终推荐范围 |
| `rir.py` | 串联采集、基础质检、重复一致性、平均与结果保存 |
| `storage.py` | 运行快照、文件与清单；区分原始音频格式和 FLOAT 处理数据 |

原来从 `acoustic_capture.rir` 导入的主要数学函数继续重导出，减少外部调用的迁移成本。
ESS 与 Kirkeby 正则化公式保持原有算法和仓库 golden 行为，没有悄悄替换成另一套逆滤波。

## 审计问题的处理

| 编号 | 原问题 | 本版行为 |
| --- | --- | --- |
| F01 | GUI 读取未初始化变量 | 构建控件前初始化；真实 GUI 工作流测试通过 |
| F02 | 保存设置改写运行中的配置 | GUI、音频后端、RIR 入口和运行存储各持有独立快照；编辑用于下一次运行 |
| F03 | 两次结果绕过一致性检查 | 两次也检查；相互矛盾时不猜哪次正确，两次都保留原始数据但不进入均值；两次延迟波动也参与最终质量 |
| F04 | 延迟配置未应用、提前通道被截 | 频带、最大时差和容差实际传入；分析窗口覆盖正负范围；超范围或歧义峰标为无效 |
| F05 | 跨频率空洞错误展开相位 | 在连续可靠频段内展开，各段使用独立截距，不强行连接陷波两侧的相位分支 |
| F06 | 固定频响被判为硬件时钟漂移 | 保留时序趋势数值，明确 `clock_drift_confirmed=false`；仅凭该趋势不再触发硬件时钟失败判定 |
| F07 | 裁剪遗漏直达声或补零冒充尾声 | 逐通道检测并验证共同窗口；不足时拒绝该 take，保存原始录音和诊断 IR |
| F08 | PCM 保存削波，而报告使用浮点值 | RIR、逆滤波器和处理后响应固定为 FLOAT；原始录音仍遵循配置的 PCM/FLOAT 偏好 |
| F09 | 整数流缺少满量程转换 | 当前明确只支持 `float32`，配置和真实后端都阻止不支持的类型 |
| F10 | 未录满却报告完成 | 核对实际帧数，回调异常返回工作线程；非主动短录报错，取消只返回实录片段 |

基于声学 ESS 的 ppm 趋势还不能分离设备时钟缩放与固定系统相位。这里修复的是错误诊断和
错误验收，没有声称新增了有线 loopback 校准或经验证的硬件时钟估计器。

## 重构验收有两种明确用途

默认 `repeats.reconstruction_policy: diagnostic` 保持有意提取早期 RIR 的用途：
相关性、NMSE 和回卷文件保留，最终质量的 `training_recommendation_scope` 明确为
`recording_and_repeat_consistency_only`。这不表示短 RIR 已经解释完整房间尾声。

如果需要完整响应重构验收，在 YAML 的 `repeats` 中设置：

```yaml
repeats:
  strategy: reconstruct_average
  fixed_count: 5
  delay_low_hz: 100.0
  delay_high_hz: 1000.0
  delay_max_ms: 3.0
  delay_agreement_samples: 2.0
  reconstruction_policy: full_response
  minimum_reconstruction_correlation: 0.90
  maximum_reconstruction_nmse_db: -10.0
```

该模式对每个 take 和最终平均 RIR 都检查相关性与绝对 NMSE；非有限或缺失指标也不能通过。
上述阈值可按实验目标配置，绝对 NMSE 越负越好；尺度无关 NMSE 仍只作诊断，不隐式改变 RIR 增益。

消费者应检查每路 `gcc_phat.valid` 和 `low_frequency_group_delay.valid`。为兼容旧代码，
无效 GCC 的样点返回值仍可能是 0，但此时 `valid=false`，微秒和等效路径差为空，不能解释成零时差。
新增 `estimators_agree` 和 `estimators_agree_within_configured_tolerance` 按配置阈值判断；
旧 `estimators_agree_within_one_sample` 保持字面含义，仍只表示不超过 1 样点。

独立调用 `extract_rir` 时，窗口不足默认抛出 `RIRWindowError`，其中保留部分结果和诊断。
采集工作流会接住它并保存原始数据，再排除该 take；不会把补零尾部当成测量结果接受。

## 验证结果

最终完整测试：**133 passed，0 failed，0 skipped**，用时 **13.93 s**。
结果记录在 [verification/pytest-results.xml](verification/pytest-results.xml)。
默认模拟配置通过真实 CLI 完成 5 次 RIR 采集；5 次均被接受，平均 RIR 为 24000 样点，
回卷最低相关性 **0.9960**，最差通道的中位 NMSE **−20.742 dB**。
命令输出见 [verification/cli-smoke.log](verification/cli-smoke.log)，结构化结果见
[verification/validation.json](verification/validation.json)。

覆盖的反例包括：±120/±144 样点时差、超范围延迟、无信号、陷波后的 144 样点群延迟、
固定滤波器时序趋势、两次矛盾结果、PCM 录音与大于 1 的 RIR 增益、裁剪不足、采集中修改配置、
回调异常、提前停止、主动取消，以及“每次自身回卷通过但平均后不通过”的完整响应验收。

旧测试里的失效导入、覆盖同名测试的重复定义、已移除 GUI 按钮和过期笛卡尔积假设已更新。
场景测试继续检查固定任务数、随机种子复现、扫描顺序不影响任务和续采；底层 ACQUA/campaign
模块测试保留。实验汇总测试增加了录音尾部余量，没有撤掉新的裁剪边界保护。

本次使用现有 Python 3.11、NumPy/SciPy 和模拟/假声卡完成离线验证，包含实际 Tk 界面工作流。
没有进行真实 RME/ASIO 长时间播录、物理接线校准或独立 MATLAB 执行，不能据此认证实机性能。

重新运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

ZIP 保留源码、配置、启动器、构建脚本、测试、说明及本次验证记录；不包含虚拟环境、缓存、
临时录音、原分支的旧 portable 发布产物和历史 `verification_10_pairs` 数据。
准确变更清单与基线文件校验值见 [verification/changes.json](verification/changes.json)。
