# 真实语音增强数据采集设计

训练汇总只有在人工头、耳机型号/个体、佩戴、麦杆、麦克风通道身份，以及目标/干扰声源的位置、角度、高度和距离完整时才标记 `training_ready`。断点续采还会核对播放电平、通道、素材文件状态和整份物理元数据，避免把不同现场条件拼进同一个运行。

## 一次实验的边界

一个 `scene_id` 只表示一个固定物理状态：耳机型号、耳机个体、一次佩戴、麦杆姿态、
人工头位置、干扰音箱位置和房间状态都不变。重新佩戴、移动麦杆、改变干扰源角度、
高度或距离，都新建一个 `scene_id`，不跨场景求平均。

建议命名：

```text
hs01_w01_b00_int090_h170_d100_supervised
```

- `hs01`：耳机个体
- `w01`：第 1 次独立佩戴
- `b00`：麦杆标准姿态
- `int090`：干扰源方位角 +90°
- `h170`：干扰源高度 1.70 m
- `d100`：距离 1.00 m
- `supervised_pair`：只产生 mixture 和与之配对的 target（默认，训练所需的最小现场方案）
- `supervised`：产生 mixture、配对 target 和额外 interferer-only（用于物理一致性诊断）

`headset_model_id`、`headset_unit_id`、`wearing_id`、`boom_pose_id` 和完整声源几何信息
仍应分别写在 YAML 的 `metadata` 中；文件名只是方便人快速识别。

## 监督配对为什么能对齐

监督场景使用：

```yaml
scene:
  items: [target_only, interferer_only, mixture]
  capture_strategy: paired_sequence
  require_supervised_pair: true
  gap_s: 1.0
```

程序先加载并定长同一份 target 和 interferer 数字波形，然后在**一次连续声卡流**中播放：

```text
静音 | target_only | 静音 | interferer_only | 静音 | mixture | 静音
```

`target_only` 与 `mixture` 的 target 输出通道复用完全相同的 float32 样本；`interferer_only` 与
`mixture` 的 interferer 输出通道也复用相同样本。输入和输出流只打开一次，因此三个片段
共享同一个声卡时钟和固定 I/O 延迟。麦克风连续录音再按已知的样本边界切片，得到等长的
两通道 `target_recording`、`interferer_recording` 和 `mixture_recording`。

训练时每一行标签中：

- 输入：`mixture_recording`
- 监督目标：同一行的 `target_recording`
- 可用条件：`supervision_ready == 是` 且 `valid == 是`
- 两个耳机麦克风都保存在同一个双通道 WAV 中，不拆成两个独立实验

这里的严格对齐还依赖同一块硬件时钟：输入和 target/interferer 输出必须都来自同一个可验证的
双工设备（正式实验使用同一个 RME ASIO 设备）。若输入为 RME、输出为另一台 Windows 播放
设备，即使不是蓝牙，也不能证明两个设备时钟和延迟一致；程序会把
`shared_hardware_clock` 与 `supervision_ready` 标成“否”。

不要把另一次独立运行产生的纯净录音当成某个 mixture 的逐样本标签。独立运行的
target-only 数据可以扩充纯净语料或用于验证，但不具备与旧 mixture 相同的声卡时基和现场噪声。

## 三类 YAML

- `configs/lab_speech_supervised_pair.yaml`：首选；同一连续流采 target-only、mixture。
- `configs/lab_speech_supervised.yaml`：完整诊断版；额外采 interferer-only。
- `configs/lab_speech_target_only.yaml`：仅采人工嘴发声的额外纯净场景。
- `configs/lab_speech_interferer_only.yaml`：仅采干扰源发声的额外纯干扰场景。

每移动一次干扰源或重新佩戴一次，复制相应 YAML，至少修改：

```yaml
storage:
  session_name: 新的唯一场景名
metadata:
  scene_id: 与 session_name 相同
  headset_unit_id: hs01
  wearing_id: w02
  boom_pose_id: b01_up
  interferer:
    position_id: intm090_h120_d150
    azimuth_deg: -90
    height_m: 1.20
    distance_m: 1.50
```

## 输出与服务器聚合

每个源文件对和每次重复产生 `labels.csv`、`labels.xlsx`、`labels.jsonl` 中的一行。
`sample_id` 以 `scene_id__` 开头，因此不同佩戴和不同干扰位置不会重名。
人工头、耳机型号/个体、佩戴、麦杆、房间以及 target/interferer 的位置、角度、高度和距离
同时被展开成普通列；完整原始 metadata 仍保留在 `metadata_json`。

批量素材建议附带逐文件索引 CSV。目标索引列为
`relative_path,speaker_id,utterance_id`，干扰索引列为
`relative_path,noise_id,noise_class`；路径相对于各自素材文件夹。这样 2000 条素材不会错误地共用
一个全局 speaker/noise 标签。每完成一条配对都会同步写 `labels.partial.jsonl` 和断点，失败或停止
后重用同一实验名即可从下一条续采。

关键文件：

```text
raw/*_paired_sequence_mics.wav              原始连续双麦录音
references/*_paired_sequence_playback.wav  原始连续多通道播放参考
metrics/*_paired_sequence_layout.json      每段精确起止样本
raw/*_target_only_mics.wav                  切出的双通道监督 target
raw/*_interferer_only_mics.wav              切出的双通道纯干扰
raw/*_mixture_mics.wav                      切出的双通道混合输入
labels.csv / labels.xlsx / labels.jsonl     路径、场景元数据和质检状态
supervised_pairs.csv / supervised_pairs.jsonl  只含本次运行中合格的监督对
```

人工质检不要直接替换自动标签。编辑 `labels.xlsx` 的“标签”工作表后，通过 GUI 导入，生成
`labels_reviewed.jsonl`。服务器汇总器优先使用 reviewed 文件，同时保留原始标签供审计。

推荐不要手工拼路径，而是运行：

```powershell
acoustic-capture speech-dataset runs datasets\headset_speech_generalization_v1 `
  --project-id headset_speech_generalization_v1
```

汇总器为每行生成跨运行唯一的 `dataset_sample_id`，复制所引用的多通道录音，并再次验证
mixture 与 target-only 的采样率、帧数和通道数。数据集中的音频默认分别放入
`audio/target-mixed/`、`audio/target-only/`、`audio/interferer-only/`，并按 run ID 建子目录。
监督关系仍以 `supervision_pair_id` 和索引中的三条相对路径为准，不依赖三个目录中的排序。
训练只读取统一索引中
`dataset_supervision_ready == 是` 的行，输入为 `mixture_recording`，标签为同一行的
`target_recording`。数据集划分应按 `wearing_id` 或
`scene_id` 分组后再分 train/validation/test，避免同一次佩戴的近重复样本泄漏到不同集合。

工具会自动生成 `split_leakage_report.json`，检查相同目标素材 SHA256 和同一物理采集分组是否跨
train/valid/test。监督采集还会生成分通道混合可加性残差、相关系数和估计 SIR，用于自动发现路由、
漏播或场景变化。真实顺序录制会包含环境噪声与微小变化，因此 `target_only` 是严格复用数字源和
同一连续硬件流的监督参考，不应被误解为与 mixture 内目标声像逐样本完全相等的数学分量。

## 明天现场检查

1. GUI 中输入和输出都选同一块 RME 的 ASIO 设备；输入通道为 `1,2`，target/干扰输出为不同通道。
2. 先用较低电平试听，确认人工嘴只响应 target 通道，干扰音箱只响应 interferer 通道。
3. 监督场景保留 `gap_s: 1.0`；若房间混响尾音超过 1 秒再增大。
4. 先跑一个短场景，检查结果页三段波形、无削波/丢帧，且表中 `supervision_ready` 为“是”。
5. 固定物理状态后批量采音频；移动声源或重新佩戴前先结束当前 scene。
