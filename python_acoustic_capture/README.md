# 声学采集工具

面向 Windows + RME Fireface UCX 的多通道 RIR 与语音增强数据采集工具（默认双麦）。核心流程与
GUI 分离；同一份 YAML 配置既能在命令行使用，也能在图形界面修改和运行。

## 1. 安装

建议在 Windows PowerShell 中使用 Python 3.10–3.12：

```powershell
cd python_acoustic_capture
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

列出 PortAudio 能看到的设备：

```powershell
acoustic-capture devices
```

把输出中的 RME 设备名称或数字编号填入 `configs/rme_ucx.yaml`。不同驱动安装中显示的名称
可能不同，因此示例里的 `ASIO Fireface USB` 只是占位值。

> `sounddevice` 只能使用当前 PortAudio 构建所提供的 host API。如果设备列表中没有 RME
> ASIO，请先确认 RME 驱动、采样率和 TotalMix；必要时安装带 ASIO 支持的 PortAudio。
> RME 的 WDM/WASAPI 设备也能用于功能验证，但正式多通道实验应先做同步和丢帧测试。

## 2. RME/TotalMix 准备

1. 将两个麦克风固定接到模拟输入 1、2，关闭自动增益和系统音效。
2. 将目标声源和干扰扬声器分别路由到两个独立输出。
3. 固定 48 kHz、内部时钟（或明确记录外部时钟），并保存 TotalMix snapshot。
4. 先用低播放电平测试；确认扬声器、人工头和耳机安全后再逐步提高。
5. 两个输入增益必须记录到 `metadata.rme`，同一 block 内不得改变。

通道编号在 YAML 中均为 **1-based**，与声卡面板的直观编号一致。

## 3. 使用 GUI

### 双击启动

- Windows：双击 `start_gui_windows.bat`
- macOS（推荐）：双击 `声学采集工具.app`
- macOS（终端备用）：双击 `start_gui_macos.command`

启动器第一次运行时会在项目目录创建 `.venv`、安装程序，然后打开 GUI；后续双击仍会同步
本地代码更新。macOS 可用于查看界面、编辑配置以及使用 `simulated` 后端演练流程，但 RME
ASIO 实际采集需要在 Windows/RME 环境验证。

Windows 首次启动还会在缺少示例文件时生成一组无版权的合成流程验收音频。正式采集前必须在
GUI 中把它们换成自己的干净语音和干扰素材。RME 接线、TotalMix 路由和上机检查步骤见
[`WINDOWS_RME_GUIDE.md`](WINDOWS_RME_GUIDE.md)；双击 `check_audio_windows.bat` 可以检查设备、
48 kHz、输入/输出通道，并在确认后进行 5 秒双麦克风录制。

`.app` 启动器不读取交互式 `.zshrc`，因此不会被 oh-my-zsh 更新提示打断。首次打开会显示
初始化通知，安装日志保存在 `mac_gui_launcher.log`。如果 macOS 提示文件来自未知开发者，
可在 Finder 中右键该 `.app` 并选择“打开”。

### 命令启动

```powershell
acoustic-capture-gui
# 或
acoustic-capture gui
```

打开 `configs/rme_ucx.yaml` 后，可以修改声卡、通道、扫频、重复次数、播放电平、源文件、
保存目录和实验元数据。场景 block 的四项可以独立勾选。点击运行前 GUI 会保存并校验配置；
采集在后台线程执行，日志显示在窗口底部。

GUI 顶部提供三个模式，底层关系是“一个通用 I/O 引擎 + 两个专业工作流”：

- `基础播录`：选择“仅播放”“仅录制”或“同步播放并录制”，完成一次最基础操作；
- `房间脉冲响应采集`：复用同时播录，并增加指数扫频、重复、反卷积、质检与平均；
- `语音增强数据采集`：复用基础播录，并编排环境、目标、干扰和混合场景。

录制设备和播放设备可以分别选择；设备下拉框显示各设备支持的最大输入/输出通道数。
录制通道使用逗号分隔，例如 `1,2` 或 `1,2,3,4`；各模式的输出通道独立设置。

采集完成后，右侧结果面板会自动加载本次运行：

- 上图显示实际保存的播放/原始参考；
- 中图显示麦克风录音，按照实际输入通道数绘制；
- RIR 测试时，下图显示计算得到的平均或逐次 RIR；
- 三个下拉框可切换本次运行中的其他 take、场景项目或参考文件；
- “试听播放信号”和“试听录制信号”可试听原始及录制信号；
- 试听支持自动选择有效通道、全部通道混音或指定单个通道，并默认衰减 12 dB；
- 鼠标停在任意结果图上时，按住 `Ctrl`（macOS 也支持 `Command`）并滚动滚轮可围绕鼠标位置缩放时间轴；
- 放大后按住鼠标左键左右拖动，可以平移当前时间范围；
- 勾选“显示语谱图”可在波形和语谱图之间切换，语谱图使用“监听/语谱图通道”中的选择；
- “打开历史结果”可以查看过去保存的测试结果。

`audio.input_channels` 支持一个或多个输入，例如 `[1, 2, 3, 4]`。采集、RIR 计算、质量指标、
WAV 保存和结果图会自动跟随通道数扩展。

实验元数据使用 JSON 格式。建议每次改变人工头、重新佩戴或移动声源时更新
`session_name`、`wearing_id` 和几何参数。

## 4. 命令行

```powershell
acoustic-capture validate configs/rme_ucx.yaml
acoustic-capture devices
acoustic-capture hardware-check configs/rme_ucx.yaml
acoustic-capture check-input configs/rme_ucx.yaml --duration 10
acoustic-capture io configs/rme_ucx.yaml --action play_record
acoustic-capture rir configs/rme_ucx.yaml
acoustic-capture rir configs/rme_ucx.yaml --output-channel 2
acoustic-capture scene configs/rme_ucx.yaml
acoustic-capture demo-audio --output-dir audio
```

无声卡演练可用：

```powershell
acoustic-capture rir configs/simulated.yaml
```

场景模拟还需要把两个 WAV 放到 `audio/target.wav` 和 `audio/interferer.wav`。

## 5. 场景 block 怎么选

`scene.items` 支持任意组合：

| 项目 | 保存内容 | 用途 |
|---|---|---|
| `ambient` | 双麦环境底噪 | 底噪/SNR/设备状态 |
| `target_only` | 目标单独发声的双麦录音 | 声学目标参考、监督标签 |
| `interferer_only` | 干扰单独发声的双麦录音 | 干扰声学参考 |
| `mixture` | 两者同时发声的双麦录音 | 真实混合输入 |

推荐正式采集使用完整四项。若场地时间紧，只做真实微调数据，最少选择
`target_only + mixture`；如果需要分析或重构真实混合，保留全部四项。

`repetitions` 表示整个所选序列重复几次。各项之间物理位置和输入增益必须保持不变。
`countdown_s` 会在每一项开始前留出准备时间，GUI 日志会显示当前项目。
分别录制的 target/interferer 相加不保证严格等于真实 mixture，因此三者都应保留。

### 4 秒文件夹批量模式

在“语音增强数据采集”模式中，把“语音来源模式”设为“文件夹批量”，然后分别选择目标语音目录
和干扰声音目录。程序递归扫描配置的扩展名，并支持：

- `cycle`：按排序顺序循环配对，数量较少的一侧循环使用；
- `cartesian`：目标与干扰做全部组合。

`scene.duration_s` 建议设为 `4.0`。长文件截取前 4 秒，短文件在末尾补零，不会重复语音内容。
每个源文件组合会依次执行所勾选的仅目标、仅干扰和真实混合播录。环境底噪在每次 repetition
开始时只录一次，供该 repetition 下的全部样本引用。

运行目录会额外保存：

- `labels.jsonl`：适合 Python/PyTorch 数据加载；
- `labels.csv`：UTF-8 BOM，便于通用表格工具读取；
- `labels.xlsx`：包含“标签”“采集参数”“文件索引”“汇总”“字段说明”五个工作表，可填写人工标签、
  数据集划分、是否有效和备注，并记录所有源文件、录音、播放参考和指标文件的相对路径。

## 6. RIR 重复与平均

默认最少 4 次、最多 10 次。达到最少次数后，最近若干次满足相关性和峰值漂移条件会自动
停止。每次原始录音和反卷积结果都会保存；削波或相关性不合格的 take 会被拒绝但不会删除。

输出包括：

- `processed/average_rir.wav`：对齐后的有效 take 均值；
- `processed/median_rir.wav`：对突发异常更稳健的逐样本中位数；
- `metrics/take_NNN.json`：削波、峰值、漂移、相关性和后端状态；
- `metrics/summary.json`：接受和拒绝的 take 编号。

“重新佩戴”必须建立新的 session，不能作为同一个 RIR 的重复 take 参与平均。

### ESS 生成公式

设扫频起始频率为 $f_0$、终止频率为 $f_1$、扫频时间为 $T$、数字峰值电平为 $L$ dBFS：

```text
f(t) = f0 · (f1/f0)^(t/T)
φ(t) = 2π f0 T / ln(f1/f0) · [(f1/f0)^(t/T) - 1]
s(t) = 10^(L/20) · w(t) · sin(φ(t)),  0 ≤ t < T
x(t) = 前静默 ⊕ s(t) ⊕ 后静默
```

`w(t)` 是由 `fade_s` 控制的正弦平方淡入淡出窗。GUI 可直接修改 `start_hz`、`end_hz`、
`duration_s`、`pre_silence_s`、`post_silence_s`、`fade_s`、`level_dbfs` 和最终 RIR 截取长度。
实现位于 `signals.py`，直接按上述解析相位公式生成，不依赖黑盒扫频调用。

## 7. 每次运行的归档

```text
runs/<time>_<session>_<kind>/
  config_resolved.yaml       # 实际使用的完整配置
  manifest.json              # 状态、主机、软件版本、Git commit、全部文件哈希
  report.md                  # 人可直接阅读的本次运行报告
  raw/                       # 原始双麦录音，永不覆盖
  references/                # sweep、逆滤波器、实际播放矩阵、源文件哈希
  processed/                 # RIR 均值/中位数及逐次 RIR
  metrics/                   # 每条录音质检和汇总
  logs/
```

WAV 默认是 48 kHz、双通道、32-bit float。数据集发布时可以再批量转换为 24-bit PCM，
采集母版建议保留 float。

## 8. 实验组织原则

- 人工头改变、耳机重新佩戴、麦克风位置改变：新 session。
- 声源高度、距离、方位角或俯仰角改变：新 session。
- 固定几何下的随机噪声重复：同一 session 内的 take/repetition。
- 每个 session 拍摄一张位置照片，并把照片名写入 metadata（照片暂由人工放进运行目录）。
- 正式批量前使用声级计确定播放电平，并记录声卡数字电平、模拟增益和测点 SPL。

## 9. 测试

```powershell
pip install -e .[dev]
pytest
```

测试使用模拟声学后端，不需要声卡，会验证扫频、通道路由、RIR 完整流程、可选 block 和
归档结构。它不能替代 UCX 上的人工验收：首次上机仍需检查通道映射、延迟、削波和 TotalMix
路由。

## 模块索引

- `config.py`：所有可修改参数和校验规则
- `audio.py`：真实/模拟音频后端
- `signals.py`：ESS、重采样、电平和输出路由
- `rir.py`：反卷积、重复质检和平均
- `scene.py`：可选语音 block
- `quality.py`：通用指标
- `storage.py`：目录、manifest 和哈希
- `cli.py`：命令行入口
- `gui.py`：只负责参数编辑和启动任务
