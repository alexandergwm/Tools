# 声学采集工具

面向 Windows + RME Fireface UCX 的多通道 RIR 与语音增强数据采集工具（默认双麦）。核心流程与
GUI 分离；同一份 YAML 配置既能在命令行使用，也能在图形界面修改和运行。

## 1. 安装

建议在 Windows PowerShell 中使用 64 位 Python 3.10–3.12：

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

启动器第一次运行时会弹出文件选择框，要求选择一个 64 位 Python 3.10 或更高版本的
`python.exe`。随后可选择“用它创建/复用项目 `.venv`”（推荐）或“直接使用所选 venv/Conda
环境”。选择结果仅保存在本机的 `.python-selection.json`；需要更换时双击
`choose_python_windows.bat`。启动器会安装缺少的依赖并打开 GUI。

Windows 首次启动还会在缺少示例文件时生成一组无版权的合成流程验收音频。正式采集前必须在
GUI 中把它们换成自己的干净语音和干扰素材。RME 接线、TotalMix 路由和上机检查步骤见
[`WINDOWS_RME_GUIDE.md`](WINDOWS_RME_GUIDE.md)；双击 `check_audio_windows.bat` 可以检查设备、
48 kHz、输入/输出通道，并在确认后进行 5 秒双麦克风录制。

### 命令启动

```powershell
acoustic-capture-gui
# 或
acoustic-capture gui
```

打开 `configs/rme_ucx.yaml` 后，可以修改声卡、通道、扫频、重复次数、播放电平、源文件、
保存目录和实验元数据。场景 block 的四项可以独立勾选。点击运行前 GUI 会保存并校验配置；
采集在后台线程执行，日志显示在窗口底部。
采集期间“停止当前测试”按钮会立即请求终止音频流。已完整完成的 RIR take 会保留并生成标记为
`partial_average` 的部分均值；正在播放但未完成的一次不会进入均值。语音增强采集同样会保留
已经完整写入的配对序列，并把运行状态标记为 `cancelled`。

GUI 顶部只保留两个入口：

- `RIR 采集`：默认只显示声卡、输入/输出通道、扫频播放电平、有效 RIR 次数和保存位置。
  常规实验直接使用默认 ESS；扫频频率、时长、淡入淡出、RIR 截取和质量阈值统一放在
  “显示高级设置”中。
- `音频 / 语音采集`：把普通播录和语音增强合并。默认“标准监督：目标 + 干扰 + MIXED”在一次连续
  播录中采 `target_only + interferer_only + mixture`；纯干扰可带回服务器做离线混合。也可选择含底噪的完整监督采集、只采
  干净目标、只采干扰、普通单文件同步播录、仅录制或仅播放。界面只显示当前方案实际需要的
  文件、声源和通道。

如果只想把麦克风原始数据录成一个文件，不需要进入上述实验流程：先设置录制协议、录制设备、
采样率和录制通道，然后点击工具栏的 `● 快速录音`。在弹出的窗口中选择一个 `.wav` 保存位置后
会立即开始录制；点击 `停止录制 / 测试` 即完成写盘，右侧会自动显示刚录好的多通道波形。
快速录音不要求实验名称、播放设备、时长、ESS 参数、语音素材或标签。录音采用流式写盘，时长
不受内存容量限制；应保证保存磁盘有足够空间。

先在“音频协议 / 主机接口”中选择 `MME`、`Windows WASAPI`、`Windows WDM-KS` 或 `ASIO`，
再选择设备。录制设备下拉框只显示该协议下具有输入能力的设备，播放设备下拉框只显示该协议下
具有输出能力的设备；切换协议时两项会清空，避免沿用另一协议的设备编号。录制通道使用逗号
分隔，例如 `1,2` 或 `1,2,3,4`；“人工嘴 / 目标源输出通道”和“干扰源输出通道”独立设置。
`mixed` 时两路必须不同，分别接人工嘴和干扰扬声器。正式同步采集的输入和输出必须
属于同一个协议，配置检查也会阻止协议不匹配的设备。

采集完成后，右侧结果面板会自动加载本次运行：

- 上图显示实际保存的播放/原始参考；
- 中图显示麦克风录音，按照实际输入通道数绘制；
- RIR 测试时，下图显示计算得到的平均或逐次 RIR；
- 三个下拉框可切换本次运行中的其他 take、场景项目或参考文件；
- “试听播放信号”和“试听录制信号”可试听原始及录制信号；
- 试听支持自动选择有效通道、全部通道混音或指定单个通道，并默认衰减 12 dB；
- 鼠标停在任意结果图上时，按住 `Ctrl` 并滚动滚轮可围绕鼠标位置缩放时间轴；
- 放大后按住鼠标左键左右拖动，可以平移当前时间范围；
- 勾选“显示语谱图”可在波形和语谱图之间切换，语谱图使用“监听/语谱图通道”中的选择；
- “打开历史结果”可以查看过去保存的测试结果。

`audio.input_channels` 支持一个或多个输入，例如 `[1, 2, 3, 4]`。采集、RIR 计算、质量指标、
WAV 保存和结果图会自动跟随通道数扩展。
RIR 数据集索引会同时保存 WAV 列号、实际声卡输入通道和麦克风编号；例如选择 `1,3,5` 时，
不会被误标成连续的 `1,2,3`。

点击“开始新的 RIR 实验”后，GUI 会先强制输入本次实验名称，再开始扫频。
一次点击会严格按照 `repeats.fixed_count` 采集多次 IR、做录音与回卷重构质检，并只对本实验中的
有效 IR 求对齐算术均值。改变角度、高度、重新佩戴或移动麦杆后，再点击同一个开始按钮并输入新的自由名称；
不会与上一次实验求平均。语音增强实验同样在点击开始时命名。名称会同时写入运行目录、
`experiment_id`、`scene_id` 和清单元数据。
自动展开的 RIR 实验名包含人工头、耳机型号/个体、佩戴、麦杆、声源、角度、高度和距离。
生成训练数据集时还会重新核对物理条件指纹；即使误用了同一个名称，不同物理条件也不会被
静默合并。重复完成的计划实验、条件不匹配、跨数据集划分泄漏都会令 `training_ready=false`。

### 现场工作流：Excel 清单或手工命名

GUI 不要求测试工程师逐项评价。现场只有两种主流程：

1. 点击顶部“测试清单 (.xlsx)”，选择清单中的一行。工具自动带入工作流、实验名、佩戴、麦杆、
   声源角度/高度等条件；采集完成后自动把 `status`、`completed_run` 和完成时间回写到同一个
   `.xlsx`。Excel 必须在采集结束前关闭，否则 Windows 文件锁会阻止回写，但不会影响 WAV 保存。
2. 不选择清单，点击开始后手工输入本次实验名。每次重新佩戴、移动角度/高度或改变麦杆姿态，
   使用一个新实验名。

空白模板随 Portable 一起提供为 `acoustic_capture_checklist_template.xlsx`，也可在 GUI 的“文件”
菜单新建。RIR YAML 计划可以直接转成预填清单：

```powershell
acoustic-capture checklist acoustic_capture_checklist.xlsx `
  --rir-plan configs\lab_rir_experiment_plan.yaml
```

阵列坐标、校准阈值等专业字段属于项目模板，不需要每次现场填写。默认 `capture_profile: standard`
只自动提醒；只有项目负责人明确切换为 `production` 后，缺少会破坏训练正确性的条件才会阻止开始。

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

推荐正式采集至少使用 `target_only + interferer_only + mixture`；若还要记录环境底噪，使用完整四项。

`repetitions` 表示整个所选序列重复几次。各项之间物理位置和输入增益必须保持不变。
`capture_strategy: paired_sequence` 会把所选片段拼成一次连续全双工声卡流；默认监督方案的
顺序是 `target_only → interferer_only → mixture`。中间插入
`gap_s` 秒数字静音，然后按同一个采样时间轴切回各双麦片段。
`countdown_s` 只在整组配对序列开始前执行一次。

其中 target-only 和 mixture 会重复播放完全相同的目标语音样本、起点和长度；两者共享一次
ASIO 流的硬件延迟，因此 target-only 可以作为 mixture 的样本级对齐监督标签。若配置中包含
`mixture`，默认必须同时包含 `target_only`，否则配置检查失败。默认同时录制 `interferer_only`，
用于加性一致性检查和离线混合；只采纯目标或纯干扰仍然允许，
但标签中的 `supervision_ready` 会是“否”。分别录制的 target/interferer 相加不保证严格等于
真实 mixture，因此三种实采录音仍应保留。

服务器训练汇总要求人工头、耳机型号/个体、佩戴、麦杆、麦克风通道身份，以及目标和干扰源的
位置、方位、俯仰、高度、距离完整。缺项时原始 WAV 仍保留，但该条不会进入
`supervised_pairs.csv`，并在 `speech_samples.csv/.xlsx` 的 `dataset_validation_error` 中说明原因。
断点续采会核对播放电平、通道、素材大小/修改时间和整份物理标签，条件变化时拒绝把新数据接到
旧实验后面。

严格监督还要求录制输入和两路扬声器输出使用同一块可验证的双工声卡。正式实验应让人工嘴和
干扰扬声器都接 RME 输出，并在 GUI 中让输入、输出都选择同一个 `ASIO Fireface USB`。
如果 RME 负责录音、Beosound 作为另一台 Windows 播放设备，流程可以试跑，但标签中的
`shared_hardware_clock` 和 `supervision_ready` 会是“否”，不会进入监督训练索引。

### 4 秒文件夹批量模式

在“音频 / 语音采集”中选择语音采集方案，把“素材选择方式”设为“文件夹批量”，然后分别选择目标语音目录
和干扰声音目录。程序递归扫描配置的扩展名，并支持：

- `cycle`：按排序顺序循环配对，数量较少的一侧循环使用；
- `cartesian`：目标与干扰做全部组合。

大素材库会在后台扫描，开始采集时也只读取、重采样并计算当前一对文件的 SHA256，不会先把
整个目录的音频读进内存。2000 条 target + 2000 条 interferer 使用 `cycle` 时形成 2000 组任务；
不要误选 `cartesian`，因为它会形成 4,000,000 组。程序对超过 100,000 组的笛卡尔组合会直接
给出提示并阻止启动，避免实验电脑因任务表过大而长时间无响应。

正式开始前会逐个读取音频文件头，损坏或空音频会在扬声器发声之前报出。每完成一组配对，
程序都会立即落盘标签和断点；停止、驱动报错或断电后，用同一实验名重新开始，GUI 会询问是否从
下一条续采。预检还会按片段长度、通道数、重复数和播放参考设置估算整批磁盘占用，空间不足时
直接阻止开始。

素材逐文件元数据可由两个可选 CSV 提供：目标表使用
`relative_path,speaker_id,utterance_id`，干扰表使用
`relative_path,noise_id,noise_class`。`relative_path` 相对于对应素材文件夹，这些值会逐对写入标签，
用于监督训练索引和数据划分泄漏检查。

`scene.duration_s` 建议设为 `4.0`。长文件截取前 4 秒，短文件在末尾补零，不会重复语音内容。
每个源文件组合会在一次连续播录中完成所勾选的仅目标、仅干扰和真实混合片段。环境底噪在
每次 repetition 开始时单独录一次，供该 repetition 下的全部样本引用。

运行目录会额外保存：

- `labels.jsonl`：适合 Python/PyTorch 数据加载；
- `labels.csv`：UTF-8 BOM，便于通用表格工具读取；
- `supervised_pairs.csv/jsonl`：只包含已经通过对齐与质量检查的 mixture/target-only 监督对；
- `labels.xlsx`：包含“标签”“采集参数”“实验条件”“监督配对”“文件索引”“汇总”“字段说明”，
  人工头、耳机、佩戴、麦杆和声源几何均为独立列，不必从 JSON 中解析；
- `raw/*_paired_sequence_mics.wav`：未切分的一次连续双麦录音；
- `references/*_paired_sequence_playback.wav`：未切分的连续播放矩阵；
- `metrics/*_paired_sequence_layout.json`：三个片段的精确起止采样点和对齐方式。

在 `labels.xlsx` 的“标签”页修改人工标签、train/valid/test、是否有效或备注后，从 GUI“文件 →
导入人工质检 labels.xlsx”，或运行 `acoustic-capture labels-import <运行目录>`。这会生成
`labels_reviewed.jsonl`；之后 `speech-dataset` 会优先读取质检版，同时保留原始自动标签不被覆盖。

多耳机、多次佩戴、麦杆姿态和干扰源几何位置的 `scene_id` 命名、三类现场 YAML 以及
服务器聚合规则见 [SPEECH_DATASET_DESIGN.md](SPEECH_DATASET_DESIGN.md)。

全部语音实验完成后，可生成可搬到服务器的统一多通道数据集：

```powershell
acoustic-capture speech-dataset runs datasets\headset_speech_generalization_v1 `
  --project-id headset_speech_generalization_v1
```

程序会把每个被引用的多通道 WAV 只复制一次，生成 `speech_samples.csv/jsonl`、
`supervised_pairs.csv/jsonl`、`speech_dataset.xlsx` 和数据集 manifest，并再次检查监督对的采样率、
帧数和通道数完全一致。只想在原位置建立索引时可加 `--index-only`。
默认音频目录按训练用途分为 `audio/target-mixed/`、`audio/target-only/` 和
`audio/interferer-only/`（底噪在 `audio/ambient/`）；每个目录下再按 run ID 隔离，Excel/CSV 中保存
三条精确相对路径，离线混合时无需从文件名反推配对关系。

汇总器还会生成 `split_leakage_report.json`，自动检查相同目标音频内容或同一物理佩戴条件是否同时
出现在 train/valid/test；Excel 中的“划分泄漏检查”工作表给测试工程师直接查看。每组监督采集还会
自动计算 `mixture` 与 `target_only + interferer_only` 的分通道相关性、相对残差和估计 SIR。该指标
用于发现接线、播放或场景变化异常，不要求现场人工评分，也不会默认用一个武断阈值拒绝真实录音。

## 6. RIR 重复与平均

GUI 的“RIR 重复采集方式”有两种选择：

- `固定次数 + 重构质检 + 全部有效平均（推荐）`：严格完成设定次数。每个 take 反卷积得到 RIR 后，
  再用原始 ESS 与该 RIR 卷积，和对应的真实麦克风扫频录音比较 MSE、RMSE、NMSE、相关系数及
  仅供诊断的尺度无关 NMSE。通过录音基础质检的 take 全部参与对齐算术平均；
- `固定次数 + 保留原始 take，之后再选`：采集和指标相同，但把生成的均值标记为参考结果，
  `metrics/summary.json` 中为 `selection_deferred: true`，方便之后从原始 take 离线重选。

对应 YAML：

```yaml
repeats:
  strategy: reconstruct_average  # 或 fixed_count（延后选择）
  fixed_count: 5
```

默认严格录制 5 次，不使用收敛提前停止。每次原始录音、完整反卷积 IR、对齐裁剪 IR、ESS 回卷重构信号
和指标都会保存；削波或相关性不合格的 take 会被拒绝但不会删除。任一麦克风的扫频相对前置底噪低于
`minimum_sweep_snr_db`、发生丢帧/削波、相关性不足或峰值漂移超限时，该 take 不进入均值。

输出包括：

- `average_rir.wav` / `selected_rir.wav` / `all_accepted_mean_rir.wav`：全部有效 take 对齐后的算术均值；
- `median_rir.wav`：逐采样中位数，供诊断；
- `recon_NNN.wav`：原始 ESS 与该次 RIR 卷积得到的重构扫频响应；
- `mean_recon.wav`：最终平均 RIR 与原始 ESS 卷积得到的重构响应。

所有麦克风始终共用同一组 accepted take。平均前不对每条 RIR 做 peak/RMS 归一化，避免破坏
通道间增益和相位。绝对 MSE 不拟合增益；同时提供拟合单个标量后的尺度无关 NMSE，便于区分
整体增益偏差和时频形状错误。最终平均 RIR 还会分别回卷并和每个 accepted take 的真实录音比较。

- `processed/average_rir.wav`：全部有效 take 的对齐算术均值；
- `processed/all_accepted_mean_rir.wav`：所有有效 take 对齐后的直接均值；
- `processed/median_rir.wav`：对突发异常更稳健的逐样本中位数；
- `metrics/take_NNN.json`：削波、峰值、漂移、相关性和后端状态；
- `metrics/summary.json`：接受/拒绝的 take 编号、选择方法，以及每个原始 take 的离线重选路径。

去卷积使用与 MATLAB `impzest` 同类的频率相关 Kirkeby 正则化逆滤波，降低扫频带外噪声被
反向放大的风险。第 1 个麦克风的直达峰提供每次 take 的公共裁剪和对齐基准；所有麦克风只做
同一个时间平移，因此麦克风之间真实的到达时差（ITD）会保留。每次还会保存未裁剪的
`take_NNN_full_ir.wav` 以及各麦克风相对麦克风 1 的峰值偏移。

“重新佩戴”必须建立新的 session，不能作为同一个 RIR 的重复 take 参与平均。

RIR 模式右侧的 ESS 预览分为三张图：第一张只是“前静音 → ESS → 后静音”的播放时间轴，
不表示音频振幅，也不能据此判断掉音；第二张显示 ESS 起始段的真实连续波形；第三张显示扫频瞬时频率的
对数轨迹。实际测试完成后，右侧才会自动切换为播放参考、麦克风录音和计算所得 RIR。

### ESS 生成公式

设扫频起始频率为 $f_0$、终止频率为 $f_1$、扫频时间为 $T$、数字峰值电平为 $L$ dBFS：

```text
f(t) = f0 · (f1/f0)^(t/T)
φ(t) = 2π f0 T / ln(f1/f0) · [(f1/f0)^(t/T) - 1]
s(t) = 10^(L/20) · w(t) · sin(φ(t)),  0 ≤ t < T
x(t) = 前静默 ⊕ s(t) ⊕ 后静默
```

`w(t)` 使用正弦窗。默认向上扫频淡入 80 ms、淡出 5 ms，与 MATLAB R2024b `sweeptone`
一致；GUI 中分别对应 `fade_in_s` 和 `fade_out_s`。GUI 还可直接修改 `start_hz`、`end_hz`、
`duration_s`、`pre_silence_s`、`post_silence_s`、`level_dbfs` 和最终 RIR 截取长度。实现位于
`signals.py`，直接按上述解析相位公式生成。

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

多耳机、多次重新佩戴、麦杆位置和干扰源角度/高度的批量实验，使用
`configs/lab_rir_experiment_plan.yaml`。矩阵中的 `artificial_heads`、耳机、佩戴、麦杆和声源姿态
共同决定独立实验；`plan-expand` 会生成逐条件 YAML。全部采集后，`rir-dataset` 会复制每次
独立实验的多通道均值 RIR，并另外导出每个麦克风各自的 mean IR，
再生成 CSV/JSONL 服务器索引。它不会跨佩戴、角度或高度做均值。完整命名和
train/valid/test 防泄漏规则见 `EXPERIMENT_DATASET_DESIGN.md`。

如果按 GUI 中“每次开始时手动命名实验”的方式采集，不需要先维护计划矩阵，使用：

```powershell
acoustic-capture rir-collect runs datasets\rir_manual `
  --project-id headset_rir_generalization_v1
```

它把每一个完成的运行视为独立实验，即使两个手动名称相同也绝不会跨运行求均值；输出包括
多通道 mean IR、每麦单通道 mean IR、CSV/JSONL 索引和 `rir_dataset.xlsx` 汇总表。

## 9. 测试

```powershell
pip install -e .[dev]
pytest
```

测试使用模拟声学后端，不需要声卡，会验证扫频、通道路由、RIR 完整流程、可选 block 和
归档结构。它不能替代 UCX 上的人工验收：首次上机仍需检查通道映射、延迟、削波和 TotalMix
路由。

## 10. Windows Portable EXE

语音采集开始后，右侧会即时显示当前素材对的 target 与 interferer 波形，以及红色播放/录制游标；
RIR 预览同样显示扫频播放位置。素材预检和文件夹扫描在后台执行，因此网络盘、同步盘或异常声卡驱动
不会阻塞 GUI 主界面；可随时点击“停止当前测试”。

无需 Python 的 Portable 版本可通过以下脚本重新生成：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable_windows.ps1
```

构建输出：

- `portable_release/dist/AcousticCapturePortable/AcousticCapture.exe`：推荐的文件夹版，启动快；
- `portable_release/AcousticCapturePortable-folder.zip`：可直接复制到实验室电脑并解压；
- `portable_release/dist/AcousticCapturePortable.exe`：单 EXE 版，首次启动会稍慢。

首次启动会在 EXE 同目录释放相对路径配置、示例音频、`runs`、`datasets` 和 `logs`，不会要求
选择 Python 解释器。RME 驱动仍需由实验室电脑安装；打包文件已经包含支持 ASIO 的 PortAudio。

Portable 首次启动还会释放 `acoustic_capture_checklist_template.xlsx`，可直接编辑后由 GUI 选行采集。

## 模块索引

- `config.py`：所有可修改参数和校验规则
- `audio.py`：真实/模拟音频后端
- `signals.py`：ESS、重采样、电平和输出路由
- `rir.py`：反卷积、重复质检和平均
- `scene.py`：可选语音 block
- `quality.py`：通用指标
- `professional.py`：可选的项目级专业预检、阵列定义和配置指纹
- `checklist.py`：Excel 测试清单读取、选行应用和完成状态回写
- `storage.py`：目录、manifest 和哈希
- `cli.py`：命令行入口
- `gui.py`：只负责参数编辑和启动任务
