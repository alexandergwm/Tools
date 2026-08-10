# 耳机、佩戴、麦杆和声源位置 RIR 实验设计

## 1. 实验层级

数据必须按以下层级组织。重新佩戴、改变麦杆、改变声源角度或高度都会开始一次新实验：

```text
project
└─ experiment_id             # 一个确定的佩戴、麦杆、声源角度和高度
   ├─ take_001 ... take_005  # 实验内部重复 ESS
   ├─ mean_ir_mic_01.wav     # 麦克风 1 的实验内均值 IR
   └─ mean_ir_mic_02.wav     # 麦克风 2 的实验内均值 IR
```

- `experiment`：一次确定的耳机单体、佩戴、麦杆和声源几何，是数据集基本单元。
- `take`：只是在该实验内部固定全部物理条件后重复播放 ESS。
- 仅在同一 `experiment_id` 内对通过质检的 take 对齐并求均值。
- 跨 `wearing_id`、角度、高度、麦杆或耳机单体都不做波形均值，也不生成跨实验 stack。
- 双麦录音分别形成 `mean_ir_mic_01` 和 `mean_ir_mic_02`；双通道 WAV 只是这两个 IR 的容器。

## 2. 推荐采集设计

先做可解释的分块实验，不要一次同时改变麦杆和干扰源位置：

1. 每个耳机单体至少 3 次、建议 5 次重新佩戴。
2. 每次佩戴先固定干扰源，测人工嘴在 `nominal / up / down` 三个麦杆位置的 RIR。
3. 麦杆回到 `nominal`，再测干扰源的角度×高度网格。
4. 距离、人工头高度、输入增益、数字播放电平和测点 SPL 必须记录且在一个比较组内固定。
5. 每个物理条件至少保存 5 个通过质检的 ESS take。

`configs/lab_rir_experiment_plan.yaml` 默认生成 54 个条件：2 个耳机、3 次佩戴、人工嘴 3 个麦杆位置、干扰源 6 个位置。若增加到 5 次佩戴，只需在 `matrix.wearings` 添加 `w04` 和 `w05`。

## 3. 坐标和编号

- `headset_model_id`：型号，不写自然语言长名称，例如 `model_a`。
- `headset_unit_id`：实物单体编号，例如 `hs01`、`hs02`；同型号的两个实物也必须不同。
- `wearing_id`：`w01`、`w02`；取下再戴才递增。
- `boom_pose_id`：`b00_nominal`、`b01_up`、`b02_down`；同时在备注记录毫米偏移。
- `source_role`：只使用 `mouth` 或 `interferer`。
- `azimuth_deg`：人工头正前方为 0°；必须在计划中写明正方向。
- `elevation_deg`、`source_height_cm`、`distance_cm`：保存数值，不把单位混进单元格。
- `dataset_split`：按整个 `wearing_id` 划分，不能把同一佩戴的 take 随机拆到 train/test。

实验编号由工具自动生成，例如：

```text
hm-model-a__hu-hs01__w-w03__b-b00-nominal__src-interferer-interferer-speaker-01__azp090__elp017__d100
```

时间戳只属于运行目录，不进入稳定的实验编号。若重测同一实验，数据集汇总器选取最新的已完成运行，并在索引中记录重复运行数。

## 4. 明天的操作顺序

1. 修改 `configs/lab_rir_base.yaml` 中 RME 增益、房间和 SPL 校准信息。
2. 修改计划 YAML 中两个耳机型号/单体编号以及真实距离、高度和角度。
3. 展开计划：

   ```powershell
   .\.venv\Scripts\python.exe -m acoustic_capture plan-expand configs\lab_rir_experiment_plan.yaml
   ```

4. 生成的 54 个 YAML 位于 `configs/generated/lab_rir_v1/`，一份 YAML 就是一次实验，按文件名前三位顺序采集。
5. 每次移动声源、改变高度或重新佩戴后，先核对 GUI 左侧元数据和 `experiment_order`，再开始 RIR。
6. 当天结束后生成服务器数据包：

   ```powershell
   .\.venv\Scripts\python.exe -m acoustic_capture rir-dataset configs\lab_rir_experiment_plan.yaml
   ```

## 5. 服务器数据包

```text
datasets/headset_rir_generalization_v1/
├─ dataset_manifest.json
├─ indexes/
│  ├─ rir_experiments.csv
│  └─ rir_experiments.jsonl
└─ rir/
   └─ experiments/{train,valid,test}/
      ├─ <experiment_id>__mean-ir-2ch.wav
      ├─ <experiment_id>__mean-ir-mic-01.wav
      └─ <experiment_id>__mean-ir-mic-02.wav
```

训练时读取 `rir_experiments.jsonl`，一行对应一次独立实验。佩戴、角度和高度差异通过多行实验样本体现，训练程序自行采样这些实验，采集工具不再跨实验平均。

## 6. Excel 总结结构

- **实验总览**：计划数、完成数、缺失数、训练/验证/测试数量和关键规则。
- **采集计划**：一行一个 experiment，供现场按顺序执行。
- **RIR 实验索引**：一行一个独立实验，分别记录两个麦克风的 mean IR 路径。
- **实验统计**：按耳机、佩戴、声源角色统计完成数量，只统计条数，不计算任何波形均值。
- **字段说明**：字段类型、单位和是否进入文件名。

Excel 用来查看和现场管理，训练程序应以 JSONL 为权威索引；任何人工修订都应同步回计划 YAML，而不是只在 Excel 中改颜色。
