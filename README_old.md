# 医保 AI 识图项目运行指南

**目标一、二扩展（2026-09-14）：**已新增 ConvNeXt-Tiny 三分类、DINOv2 判重、数据索引、训练恢复、校准、推理及服务代码。请按 [目标一与目标二训练和集成指南](DETECTION_README.md) 使用 `configs/config_detection.yaml` 和 `python -m scripts.detection`。新流程只加载两个目标模型；不需要训练以下旧流程的其他四份权重。真实容器运行及赛事效果尚待验收。

下文描述原 `configs/config.yaml` 对应的完整六模型流程，保留供其他目标参考。

本文根据当前仓库源码整理，目的是说明：先运行什么、每条命令读取什么、会生成什么文件，以及下一步如何使用这些文件。

**旧流程文档核对状态（2026-09-13）：**下文输出是源码定义的预期输出，未执行训练或推理，不代表已经运行成功。项目未附带真实影像、标签或训练后权重；新增检测依赖见 `requirements-detection.txt`，新流程的验证边界见上述指南。

## 1. 先理解整体执行顺序

从零训练到预测，按下面的顺序操作：

1. 准备 Python 环境、影像和 Excel 标签。
2. 修改 `configs/config.yaml` 中的路径和运行参数。
3. 运行数据检查，查看缺失或损坏文件报告。
4. 分别训练异常识别、序列识别、核心区域分割、异常区域分割、影像特征预测、重复检测，得到 **6 份 `best.pth`**。
5. 可选：运行验证集模拟，检查完整预测流程能否产生结果。
6. 根据用途选择：用命令行对指定目录批量预测，或启动 HTTP 服务接收平台任务。

**训练顺序并非模型依赖顺序。** 各训练入口都直接使用影像和标签，不读取其他任务训练出的权重。例如，特征模型训练时用 Excel 中的真实序列类别组织输入，不需要先用序列模型预测；它也不读取分割模型的输出。

**完整推理则有固定顺序：**每例异常识别 → 序列识别及模态选择 → 分割 → 影像特征预测；全部病例处理结束后，再执行批次内重复检测。特征模型使用选中的影像，当前并不把分割 mask 作为输入。

如果已有与当前模型结构匹配的全部权重，可以跳过训练，直接执行第 7 节或第 8 节；指定目录推理和 HTTP 服务不依赖训练 Excel。

## 2. 各目录负责什么

- [`configs/config.yaml`](configs/config.yaml)：数据路径、训练参数、输出位置和服务参数。
- [`train/`](train/)：5 个训练入口；其中 `train_segmentation.py` 需要用不同参数运行两次。
- [`scripts/`](scripts/)：数据检查、指定目录预测、验证集模拟。
- [`service/server.py`](service/server.py)：加载模型，提供 HTTP 接口，接收任务并回调结果路径。
- [`src/common/`](src/common/)：读取配置、随机种子、设备选择、日志和 checkpoint 保存。
- [`src/data/`](src/data/)：影像路径规则、NIfTI 读取与预处理、病例划分。
- [`src/models/basic3d.py`](src/models/basic3d.py)：通用 3D 编码器、分类模型、多模态特征模型。
- [`src/tasks/`](src/tasks/)：各任务的数据集、分割与重复检测模型、特征标签定义。
- [`src/pipeline/`](src/pipeline/)：加载 6 个模型、单病例预测、批量预测和重复检测。
- `__pycache__/`：Python 字节码缓存，不需要手动执行。

`src/` 中的文件主要被入口脚本导入，不需要逐个运行。

## 3. 准备运行环境

### 3.1 命令执行位置

**下文所有命令都在项目根目录执行**，也就是同时包含 `configs/`、`train/`、`scripts/`、`service/` 和本 README 的目录。

统一使用 `python -m 包名.模块名`。例如 `train_duplicate.py` 和 `simulate_validation.py` 没有自行添加项目根目录到导入路径，使用这种方式可以从根目录导入 `src`。

先在终端选择或激活你实际使用的 Python 环境；PyCharm 中配置了项目解释器，并不意味着终端已经能执行 `python`。

```powershell
python --version
```

预期输出：解释器版本，不生成项目结果文件。服务使用了 `str | None` 类型注解，按源码需要 Python 3.10 或更高版本；具体 Python 与依赖版本组合尚未在本项目验证。

### 3.2 依赖

源码直接用到的第三方依赖包括：`torch`、`numpy`、`pandas`、`scikit-learn`、`nibabel`、`monai`、`PyYAML`、`requests`、`uvicorn`、`fastapi`、`pydantic`。读取 `.xlsx` 还需可用的 Excel 引擎，例如 `openpyxl`。服务调用 `model_dump()`，需要支持该方法的 Pydantic 版本（v2 API）。

仓库没有提供 `requirements.txt` 或 `pyproject.toml`。下面是根据源码依赖整理的安装起点，**不是经过测试的版本锁定方案**；GPU 环境中的 PyTorch 安装还需与你的系统和 CUDA 环境匹配。

```powershell
python -m pip install torch numpy pandas scikit-learn nibabel monai PyYAML openpyxl requests uvicorn fastapi "pydantic>=2"
```

预期输出：安装日志；依赖安装到当前 Python 环境，不生成模型或预测结果。不要覆盖已有的赛事指定环境；已有环境可先做下面的导入检查。

```powershell
python -c "import torch, numpy, pandas, sklearn, nibabel, monai, yaml, openpyxl, requests, uvicorn, fastapi; from pydantic import BaseModel; assert hasattr(BaseModel, 'model_dump'); print('Imports OK'); print('CUDA available:', torch.cuda.is_available())"
```

预期输出：`Imports OK` 和 CUDA 可用性，不生成文件。这个检查只验证导入及一个服务所需 API，不验证训练、显存容量或全部版本兼容性。

## 4. 准备数据和配置

### 4.1 训练标签

将下面 5 个 Excel 文件放到 `paths.labels_dir` 指向的目录。默认是项目根目录下的 `labels/`。

- `1_abnormal.xlsx`：异常识别标签。代码读取 `AccessionNumber`、`SeriesUid`、`Label`；类别为 `true`、`fake`、`compositing`。其他任务还用它提取 `true` 病例进行划分或负样本构建。这里的 `true` 表示影像真实性类别，不能理解为“没有肿瘤”。
- `2_duplicate.xlsx`：重复病例正样本对。代码读取 `src_img`、`desc_img`，**将这两列当作病例目录编号**，不是直接当作影像文件路径。表内每一对都按正样本处理，负样本由程序生成。
- `3_serieslabel.xlsx`：序列类别。代码读取 `AccessionNumber`、`SeriesUid`、`SeriesLabel`，支持 `T1CE`、`T2`、`FLAIR`。
- `4_masklabel.xlsx`：分割标注索引。代码读取 `AccessionNumber`、`SeriesUid`、`SeriesLabel`、`Task`、`Maskname`。`Task` 为 `core` 或 `abnormal`；`Maskname` 是对应序列目录中的 mask 文件名。
- `5_characteristics.xlsx`：病例级影像特征。必须有 `AccessionNumber`；特征列包括 `Glioma`、`Enhancement`、`Necrosis`、`CysticChange`、`Hemorrhage`、`Calcification`、`Margin`、`Lobulation`、`Morphology`、`WHO_grade`、`EnhancementPattern`、`Signal_T2WI`、`Signal_FLAIR`、`Location`。缺失特征会被损失掩码忽略；非空值必须符合代码定义。

特征取值的准确映射见 [`src/tasks/characteristics/schema.py`](src/tasks/characteristics/schema.py) 和 [`dataset.py`](src/tasks/characteristics/dataset.py)。例如 `Glioma` 为 `No/Yes`，`Margin` 为 `Unclear/Clear`，`WHO_grade` 为 1–4；`Location` 多标签用 `|` 分隔。这里没有真实 Excel，尚未核实实际数据是否符合这些约定。病例和序列编号应保留完整文本，避免 Excel 数字转换造成前导零丢失或编号变化。

### 4.2 训练影像目录

`paths.annotation_root` 应指向下面的 `annotation/`：

```text
annotation/
├── <真实病例编号>/
│   └── <序列编号>/
│       ├── <序列编号>.nii.gz
│       └── <Maskname 指定的文件>       # 有分割标注时
├── fake/
│   └── <病例编号>/<序列编号>/<序列编号>.nii.gz
├── compositing/
│   └── <病例编号>/<序列编号>/<序列编号>.nii.gz
└── duplicate/
    └── <病例编号>/<序列编号>/<序列编号>.nii.gz
```

路径规则见 [`src/data/paths.py`](src/data/paths.py)。重复检测训练时，正样本两侧都从 `annotation/duplicate/` 读取；生成的负样本从 `annotation/<病例编号>/` 读取。

代码读取的是 NIfTI `.nii.gz`，当前项目没有 DICOM 转换入口。预处理在数据读取时执行：强度处理、归一化、缩放到 `data.target_shape`；不会额外生成一套预处理影像文件。

### 4.3 修改配置路径

编辑 [`configs/config.yaml`](configs/config.yaml)，至少核对：

- `paths.annotation_root`：实际训练影像根目录。当前是 `/2026aicompetition/datasets/training/annotation`。
- `paths.labels_dir`：标签目录，默认 `./labels`。
- `paths.checkpoints_dir`：模型权重目录，默认 `./checkpoints`。
- `paths.output_dir`：数据报告、共享划分、模拟验证输出目录，默认 `./outputs`。
- `paths.competition_log_dir`：JSONL 训练日志目录，当前是 `/2026aicompetition/workspace/logs`。
- `paths.answer_root`：HTTP 服务任务结果根目录，当前是 `/2026aicompetition/workspace/answer`。
- `service.callback_url`：预测成功后的通知地址，默认空字符串，表示跳过回调。

Windows 本地运行时，将赛事路径换成实际存在或可写的位置。下面只展示可替换的 `paths` 段，其他配置段应保留；示例中的 `D:/实际数据目录/annotation` 必须替换：

```yaml
paths:
  annotation_root: D:/实际数据目录/annotation
  labels_dir: ./labels
  checkpoints_dir: ./checkpoints
  output_dir: ./outputs
  competition_log_dir: ./outputs/logs
  answer_root: ./outputs/answer
```

路径解析细节：`load_config()` 只把 `labels_dir`、`checkpoints_dir`、`output_dir` 的相对路径转换为“配置文件所在目录的上一级”下的绝对路径。因此建议把自定义配置也放在 `configs/` 中。其他配置路径和命令行 `--input`、`--output` 的相对路径按进程当前工作目录解析。

后文输出路径按默认的 `checkpoints_dir=./checkpoints`、`output_dir=./outputs` 描述；若改配置，应替换相应根路径。训练日志和服务结果的位置分别由自己的路径配置决定。

### 4.4 运行参数及当前生效范围

- `project.device: cuda`：只有此值为 `cuda` 且 CUDA 可用时选择 GPU，否则选择 CPU；当前实现不支持通过这里指定 `cuda:1` 等设备字符串。
- `project.seed: 42`：训练随机种子。
- `project.num_workers: 4`：多数训练 DataLoader 的工作进程数；重复检测训练固定为 0。本地排查数据加载问题时可把此配置暂设为 0。
- `data.target_shape: [96, 128, 128]`：预处理后张量的 D、H、W。显存和运行时间还受病例序列数等因素影响，没有实测硬件要求。
- `training.epochs: 20`：每个训练命令默认训练 20 轮。
- `training.batch_size: 2`：只被异常识别和序列识别训练用于训练批量；分割、特征、重复检测训练固定 batch size 为 1。
- `training.lr: 0.0003`、`weight_decay: 0.00001`：各任务 AdamW 参数。
- `split.val_ratio: 0.1`、`random_state: 42`：训练病例划分参数。
- `project.amp`：当前只有异常识别训练使用该配置启用 CUDA 混合精度。
- `abnormal.num_classes`、`sequence.num_classes`、`segmentation.base_channels`：当前模型构造没有读取这些配置；分类数和分割网络通道由源码固定。
- `validation_simulation.enabled`：当前模拟脚本不检查这个开关；执行命令就会运行。

## 5. 第一步：检查训练数据

```powershell
python -m scripts.validate_data --config configs/config.yaml
```

入口：[`scripts/validate_data.py`](scripts/validate_data.py)。

**输入：**5 个标签表的路径，以及 `annotation_root` 下的影像和 mask；不需要模型权重。

**做什么：**检查 5 个 Excel 是否存在，逐行检查 `1_abnormal`、`3_serieslabel` 中的影像，以及 `4_masklabel` 中的影像和 mask。检查会实际读取 NIfTI 数据，不只是检查文件名。

**输出：**

```text
outputs/data_validation/invalid_files.csv
```

发现问题时，每条记录包含 `table`、`row`（Excel 行号，计入表头）、`AccessionNumber`、`SeriesUid`、`type`、`path`。控制台还会显示标签统计和无效文件数量。

**运行后做什么：**查看报告，修正数据位置、缺失文件或损坏文件，再开始训练。脚本不会修复文件、删除标签或把报告自动传给训练脚本；各训练数据集会自行筛选可用数据。

限制：

- `2_duplicate.xlsx` 和 `5_characteristics.xlsx` 只检查是否存在，没有逐行内容检查。
- 没有问题时，当前实现仍写 CSV，但由于空记录没有预设列，文件可能没有可用表头。
- 缺少 Excel 会记录错误并跳过，不保证以失败退出码结束；因此“命令执行完”不等于数据齐全。
- Excel 本身无法读取、缺少字段或出现未知标签仍可能抛异常；这不是完整的数据规范校验。

## 6. 第二步：训练并得到 6 份权重

下面给出便于操作的顺序，逐条执行并检查结果。各模型独立训练；没有自动串联所有任务的总训练脚本。

### 6.1 异常识别：真实 / 非人体 / 拼接

```powershell
python -m train.train_abnormal --config configs/config.yaml
```

- 入口：[`train/train_abnormal.py`](train/train_abnormal.py)。
- 输入：`1_abnormal.xlsx` 和对应的 `true/fake/compositing` 影像。
- 训练：三分类模型，按病例划分训练/验证数据；验证准确率按序列样本统计。
- 输出权重：`checkpoints/abnormal/epoch_<轮数>.pth`，以及验证准确率最高的 `checkpoints/abnormal/best.pth`。
- 输出 JSONL 日志：`<competition_log_dir>/abnormal_train.jsonl`。
- 不保存独立的病例划分 JSON；不会生成病例预测文件。

### 6.2 序列识别：T1CE / T2 / FLAIR

```powershell
python -m train.train_sequence --config configs/config.yaml
```

- 入口：[`train/train_sequence.py`](train/train_sequence.py)。
- 输入：`3_serieslabel.xlsx`、`1_abnormal.xlsx`（用于共享划分）及真实病例影像。
- 训练：三种序列类别分类；验证准确率按序列样本统计，控制台打印混淆矩阵。
- 输出权重：`checkpoints/sequence/epoch_<轮数>.pth`、验证准确率最高的 `checkpoints/sequence/best.pth`。
- 输出 JSONL 日志：`<competition_log_dir>/sequence_train.jsonl`。
- 首次运行且划分不存在时，生成 `outputs/splits/true_case_split_0.1.json`。

### 6.3 核心区域分割

```powershell
python -m train.train_segmentation --config configs/config.yaml --task core
```

- 入口：[`train/train_segmentation.py`](train/train_segmentation.py)。
- 输入：`4_masklabel.xlsx` 中 `Task=core` 的影像及 mask，另读 `1_abnormal.xlsx` 创建或复用共享划分。
- 训练：3D UNet 二值分割；同一病例、序列、序列类别和任务组内的多个 mask 合并为并集。
- 输出权重：`checkpoints/segmentation/core/epoch_<轮数>.pth`、验证 Dice 最高的 `checkpoints/segmentation/core/best.pth`。
- 输出 JSONL 日志：`<competition_log_dir>/segmentation_core_train.jsonl`。
- 训练过程中不会导出预测 mask 文件。

### 6.4 异常区域分割

```powershell
python -m train.train_segmentation --config configs/config.yaml --task abnormal
```

- 输入：与上一步相同的表和路径规则，但只选取 `Task=abnormal` 的标注。
- 输出权重：`checkpoints/segmentation/abnormal/epoch_<轮数>.pth`、验证 Dice 最高的 `checkpoints/segmentation/abnormal/best.pth`。
- 输出 JSONL 日志：`<competition_log_dir>/segmentation_abnormal_train.jsonl`。
- `--task core` 和 `--task abnormal` 是两次独立训练，必须分别运行才能得到两个分割模型。

### 6.5 病例级影像特征预测

```powershell
python -m train.train_characteristics --config configs/config.yaml
```

- 入口：[`train/train_characteristics.py`](train/train_characteristics.py)。
- 输入：`5_characteristics.xlsx`、`3_serieslabel.xlsx`、`1_abnormal.xlsx` 和真实病例影像。
- 训练：按真实序列类别组织 T1CE、T2、FLAIR 多模态输入，预测胶质瘤、分级、位置、形态等多个标签。缺失模态使用零张量和模态掩码；缺失特征标签不计入对应任务损失。
- 输出权重：`checkpoints/characteristics/epoch_<轮数>.pth`、验证损失最低的 `checkpoints/characteristics/best.pth`。
- 输出 JSONL 日志：`<competition_log_dir>/characteristics_train.jsonl`。
- 使用共享真实病例划分，不读取前面训练得到的序列模型或分割权重。

### 6.6 重复病例检测

```powershell
python -m train.train_duplicate --config configs/config.yaml
```

- 入口：[`train/train_duplicate.py`](train/train_duplicate.py)。
- 输入：`2_duplicate.xlsx` 中的正样本对，`1_abnormal.xlsx` 中的真实病例，以及 `annotation/duplicate/` 和普通真实病例目录。
- 训练：编码病例内多个序列，学习重复病例相似度。负样本从真实病例抽取，排除标签表中已知的正样本对；`negative_ratio=3` 是划分及过滤前的目标负样本数量比例，不保证最终数据集仍为 3:1。
- 划分：对正样本涉及病例与真实病例的并集单独划分；一对病例只有两侧都位于同一子集才被保留，不使用共享划分 JSON。
- 输出权重：`checkpoints/duplicate/epoch_<轮数>.pth`、验证损失最低的 `checkpoints/duplicate/best.pth`。
- 输出 JSONL 日志：`<competition_log_dir>/duplicate_train.jsonl`，当前只写验证阶段记录。
- 控制台打印验证 loss、accuracy、AUC；验证标签不同时包含正负两类时，代码将 AUC 设为 0，不能据此解读模型性能。
- 本命令不生成 `duplicate_pairs.jsonl`，该文件在推理阶段生成。

### 6.7 训练结束后应检查什么

正常完成有效训练后，完整推理要求下面 6 个文件都存在且与当前模型结构匹配：

```text
checkpoints/
├── abnormal/best.pth
├── sequence/best.pth
├── segmentation/core/best.pth
├── segmentation/abnormal/best.pth
├── characteristics/best.pth
└── duplicate/best.pth
```

每个任务还保存 `epoch_1.pth`、`epoch_2.pth` 等逐轮文件。checkpoint 内含 `epoch`、`model`、`optimizer` 以及任务额外字段；推理加载器读取 `best.pth` 中的 `model`。它不会自动查找最新的 `epoch_*.pth`。

所有训练入口都从头创建模型，没有 `--resume` 或自动恢复逻辑。重复运行到相同目录会覆盖同名权重；JSONL 日志则追加。若需要保留不同实验，可复制配置到 `configs/` 中并设置不同的输出、权重和日志目录。

JSONL 日志优先写到 `paths.competition_log_dir`；仅在创建该目录失败时，代码回退到 `outputs/logs/`。若目录创建成功但后续写文件失败，没有再次回退。普通运行日志通过控制台输出，不会自动保存成 `.log` 文件。验证 accuracy、Dice、AUC 等主要出现在控制台或 checkpoint 中，JSONL 并不完整记录这些指标。

还要确认训练/验证数据数量和指标合理：过滤缺失文件后可能没有样本，小数据集也可能无法完成划分；分割 Dice 为 NaN 时可能不产生新的最佳权重。仅发现一个历史 `best.pth`，不能证明本次训练有效。

### 6.8 共享划分如何工作

序列、两个分割、特征训练，以及验证模拟共用：

```text
outputs/splits/true_case_split_0.1.json
```

该文件首次创建时只从 `1_abnormal.xlsx` 的 `Label=true` 提取唯一病例编号，保存 `train`、`val`、`meta`。训练使用 `split` 配置，模拟脚本使用 `validation_simulation.ratio/random_state`；**哪个入口首次创建文件，就决定实际划分**。之后只要文件存在，后续入口直接复用，不核对新配置或标签表是否变化。

文件名里的 `0.1` 是源码写死的，不能仅凭名字判断实际比例。若要开启新的划分实验，使用新的 `output_dir`，并配套新的权重目录重新训练相关模型，避免新划分与旧权重混用。

异常识别和重复检测使用各自的病例划分。因此模拟验证病例不能保证没有出现在这两个模型的训练集中，当前模拟流程不能直接当作全部任务严格隔离的联合评估。

## 7. 第三步：运行完整预测

训练完成后，可以选择模拟验证或指定目录推理；两者都调用 `load_all_models()` 和 `run_batch()`，**需要完整的 6 份权重**。即使只关心其中一个结果，当前入口也没有只加载单任务模型的选项。

### 7.1 可选：模拟验证集预测

```powershell
python -m scripts.simulate_validation --config configs/config.yaml
```

入口：[`scripts/simulate_validation.py`](scripts/simulate_validation.py)。

**输入：**已有共享划分，或用 `1_abnormal.xlsx` 首次生成划分；`annotation_root` 下的验证病例；6 份模型权重。

**做什么：**准备验证病例目录，优先为整个病例目录创建符号链接，失败时复制目录；保存名单；执行完整批量推理。

默认输出：

```text
outputs/
├── splits/true_case_split_0.1.json       # 没有时创建，有则复用
└── validation_simulation/
    ├── split.json                     # 字段为 train 和 validation
    ├── input/<病例编号>/              # 原病例目录的链接或副本
    └── results/                       # 第 7.3 节的结果格式
```

`validation_simulation.output_subdir` 决定上面的模拟目录名。`input/` 中是原病例目录的链接或完整副本，可能同时包含原有 mask；推理只按 `<序列编号>.nii.gz` 寻找影像。

**它不计算准确率、Dice 或比赛总分，也不输出评分报告。** 它的用途是检查留出病例上的完整推理能否运行、结果文件是否生成。它只选取真实类别病例，不是对全部异常类别或重复正样本的完整评测。

重复运行时，已经存在的病例目标目录会被跳过，脚本不清理旧输入或旧结果。改变数据或划分后，应使用新的模拟目录或新的实验输出目录，以免混入旧病例；还应核对划分是否与训练权重一致。

### 7.2 对指定目录批量预测

将待预测数据准备为下面的结构；命令示例假设放在项目根目录的 `inference_input/`：

```text
inference_input/
├── <病例 A>/
│   ├── <序列 1>/<序列 1>.nii.gz
│   └── <序列 2>/<序列 2>.nii.gz
└── <病例 B>/
    └── <序列 3>/<序列 3>.nii.gz
```

```powershell
python -m scripts.test_pipeline --config configs/config.yaml --input "./inference_input" --output "./outputs/inference_results"
```

入口：[`scripts/test_pipeline.py`](scripts/test_pipeline.py)。

**输入：**`--input` 指向的病例集合和配置中的 6 份权重。数据目录需自行准备，命令不会下载影像。

**输出：**写入 `--output` 指定的 `outputs/inference_results/`，结构见下一节。此路径由命令行决定，不会自动替换成配置的 `answer_root` 或 `output_dir`。

输入根目录的每个直接子目录都会被当成病例。因此不要直接传含有 `fake/`、`compositing/`、`duplicate/` 等分类子目录的训练 `annotation/` 根目录，也不要把输出目录放在输入目录之内。

脚本名字中的 `test` 表示运行预测，并非自动化单元测试；它没有测试断言和评分逻辑。

### 7.3 预测到底生成什么

三个推理入口（模拟验证、指定目录、HTTP 服务）都采用以下结果结构：

```text
<本次结果根目录>/
├── <病例 A>/
│   ├── prediction.json
│   ├── <选中的 T1CE 序列>/<序列编号>_core.nii.gz
│   └── <选中的 FLAIR 或 T2 序列>/<序列编号>_abnormal.nii.gz
├── <病例 B>/
│   └── ...
└── duplicate_pairs.jsonl
```

**`prediction.json`：每病例一份。** 包含：

- `AccessionNumber`：病例目录名。
- `IsNotHumanBodyProb`、`IsStitchedProb`：对病例内有效序列的异常分类概率取平均后，分别取 fake 和 compositing 类概率。即使被判异常，当前仍继续下游预测。
- `ProcessingTime_ms`：单病例处理计时，不包含模型启动加载和后续整批重复检测，也不是 HTTP 请求总耗时。
- `SegmentationMaskURI`：已生成 mask 的本地路径；`core` 对应核心分割，`flair` 对应异常分割，即使异常分割输入回退到 T2，键名仍是 `flair`。
- `Prediction`：肿瘤概率、位置、形态、WHO 分级、强化及强化模式、坏死、囊变、出血、钙化、边缘、分叶、T2/FLAIR 信号等结果。肿瘤概率小于 0.5 时，`WHO_Grade.predicted` 为 `null`。
- `Interpretation.Conclusion` 和 `Interpretation.AttentionMapURI`：当前都是空字符串，没有生成自然语言结论或注意力图。

**分割文件：按模态条件生成。** 序列模型对每个有效序列分类，每种模态保留置信度最高的一条。存在 T1CE 时才生成核心 mask；存在 FLAIR 时用 FLAIR 生成异常 mask，否则尝试 T2，两者都没有则不生成异常 mask。mask 被缩放回对应影像原始尺寸，使用参考影像的 affine/header，保存为二值 NIfTI。缺少某种 mask 不一定表示推理报错，也可能是未选出所需模态。

**`duplicate_pairs.jsonl`：每批次一份，每行一对候选病例。** 字段为 `StudyUID`、`StudyUID_dup`、`PairProb`。代码计算当前输入批次内病例的余弦相似度，各病例取 Top-K（默认 50，实际不超过其他病例数），合并无序重复对后按分值降序写出。`PairProb` 是 `(cosine + 1) / 2` 的映射值，没有经过概率校准，也没有阈值过滤，因此不是“确认重复的病例清单”。有效编码病例不足 2 个时文件为空。

当前重复检测是批次内部比较，不搜索外部历史数据库。序列类别虽用于模态选择，但没有单独导出序列分类表。

批处理会忽略不可读的序列；但某病例一个有效序列都没有时，`run_case()` 会抛异常，`run_batch()` 没有逐病例异常隔离，后续病例与重复检测可能不再执行。已写出的部分结果仍会留在磁盘。

## 8. 可选：作为 HTTP 服务运行

当需要赛事平台或其他程序通过接口提交任务时使用此方式；手动批量预测不需要先启动服务。

### 8.1 启动服务

```powershell
python -m service.server --config configs/config.yaml
```

入口：[`service/server.py`](service/server.py)。

**输入：**配置和全部 6 份权重。启动时先加载模型，成功后监听 `service.host/service.port`，默认 `0.0.0.0:8000`。

**输出：**控制台启动日志和一个持续运行的 HTTP 服务。启动本身不创建病例预测结果；需收到任务后才生成文件。模型在启动时加载并复用，更新磁盘权重后需要重启服务才能重新加载。

不要用 `uvicorn service.server:app` 代替此命令：当前初始化写在 `main()`，直接导入 `app` 会跳过配置、设备及模型初始化。

### 8.2 检查服务在线

保持服务终端运行，在另一个 PowerShell 终端执行：

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/health"
```

预期返回的 JSON 内容为 `{"status":"active"}`（PowerShell 可能显示为对象）。不生成文件，也不检查任务完成度或实际预测能力。

### 8.3 提交一个预测任务

下面是 PowerShell 示例。**先将 `dataset_path` 改为服务所在机器可访问的实际病例集合绝对路径**，目录结构与第 7.2 节相同。

```powershell
$inferenceRequest = @{
    request_id = "req_001"
    team_id = "team_001"
    track_code = "track_001"
    input = @{
        evaluation_id = "eval_001"
        dataset_path = "D:/实际数据目录/inference_input"
    }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/call" -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($inferenceRequest))
```

设置 `$inferenceRequest` 只是在当前 PowerShell 会话构造请求字符串，不生成文件。POST 请求的即时响应为 `{"status":"received"}`，表示已接收并安排后台处理，**不表示推理成功**。

字段说明：`request_id` 是必填请求标识；`team_id` 和 `track_code` 可省略，当前不参与推理；`input` 必填，内部要提供 `dataset_path` 和评测编号，评测编号兼容 `evaluation_id` 或 `evaluationId`。`input` 当前只是普通字典，没有严格校验这些内部字段。

后台任务读取该目录并输出到：

```text
<paths.answer_root>/eval_001/
```

原始赛事配置下是 `/2026aicompetition/workspace/answer/eval_001/`；使用第 4.3 节的本地示例配置则是项目的 `outputs/answer/eval_001/`。具体结果结构见第 7.3 节。服务不接收影像文件上传，调用方不能传自己机器上只有自己可见的路径。

### 8.4 完成通知与失败行为

若 `service.callback_url` 非空，后台推理成功后会向该地址 POST：

```json
{
  "request_id": "req_001",
  "evaluationId": "eval_001",
  "predPath": "<paths.answer_root>/eval_001"
}
```

这里只通知结果目录路径，不上传结果文件；接收方需能访问或按约定获取该位置的数据。未配置回调时跳过通知，结果仍写盘。

当前后台任务在服务进程内运行，没有持久化任务队列、查询任务状态接口或自动重试。推理调用阶段异常只记日志，没有失败回调；回调异常也只记日志，HTTP 非成功状态仅记录状态码，没有强制报错或重试。因此需要结合服务日志、输出文件和回调响应判断完成情况。

代码没有显式任务串行队列或模型访问锁，多请求可能重叠执行；相同 `evaluation_id` 会写同一个目录。当前本地验证可逐个提交任务，并为每次运行使用不同评测编号。结束服务可在服务终端按 `Ctrl+C`，尚未完成的进程内任务不会自动恢复。

## 9. 常见问题与运行边界

- **找不到 `python`：**先选择并激活实际解释器环境，确认第 3 节版本命令可用。这是本次终端实际遇到的问题，不能据此判断 PyCharm 的其他解释器环境也不存在。
- **`No module named src`：**确认当前目录是项目根目录，并使用本文的 `python -m ...` 形式。
- **找不到 `best.pth`：**先完成对应训练或准备匹配权重，核对 `checkpoints_dir`。三个完整推理入口及服务启动都会加载全部 6 份。
- **标签或影像找不到：**先核对第 4 节路径规则；默认赛事 Linux 路径需要按本地实际环境修改。Excel 行里的编号必须与文件夹及影像名一致。
- **训练数据数量为 0，或划分报错：**检查有效病例数、标签类别数、验证比例及数据过滤日志。脚本不会自动保证每个子集都有足够样本。
- **改了划分比例却没变化：**查看共享划分文件是否被复用，以及当前命令读取的是 `split` 还是 `validation_simulation` 配置。
- **已生成部分结果却没有重复检测文件：**先查看前面是否有病例报错。重复检测在所有病例主流程完成之后才执行。
- **模拟脚本完成但没有分数：**当前没有端到端评分脚本，模拟只输出预测。单任务训练中的验证指标也不等价于完整赛事评分。
- **显存或内存不足：**当前没有实测容量保证。异常和序列训练可调批量；其他训练批量已固定为 1，重复检测会一次编码病例内多个序列。推理还会同时加载全部模型；不能假定只改 `training.batch_size` 就能降低推理占用。
- **重跑后出现旧结果：**权重和预测中的同名文件会被覆盖，但旧输出目录不会完整清理；新实验或新预测批次应使用独立输出位置。

## 10. 复核源码时从哪里看

- 命令参数及逐轮输出：[`train/`](train/)、[`scripts/`](scripts/)、[`service/server.py`](service/server.py)。
- 配置路径解析：[`src/common/config.py`](src/common/config.py)。
- 权重保存内容与设备选择：[`src/common/train_utils.py`](src/common/train_utils.py)。
- 训练日志写法：[`src/common/logging_utils.py`](src/common/logging_utils.py)。
- 划分复用逻辑：[`src/data/split.py`](src/data/split.py)。
- NIfTI 检查、预处理和分割保存：[`src/data/nifti.py`](src/data/nifti.py)。
- 完整推理要求的权重路径：[`src/pipeline/load_models.py`](src/pipeline/load_models.py)。
- 病例及批次执行顺序：[`src/pipeline/batch_pipeline.py`](src/pipeline/batch_pipeline.py)。
- `prediction.json` 与 mask 的准确字段和生成条件：[`src/pipeline/case_pipeline.py`](src/pipeline/case_pipeline.py)。
- 重复候选对字段及分值算法：[`src/pipeline/duplicate_pipeline.py`](src/pipeline/duplicate_pipeline.py)。

本文描述的是当前实现；后续若调整参数、数据格式、模型结构或输出协议，应同步更新对应章节。
