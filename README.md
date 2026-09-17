# Goal 1 / Goal 2 训练与预测

本项目实现赛道四自建模型组中 Goal 1 和 Goal 2 的独立训练、checkpoint 保存/加载及预测闭环。代码不会自动下载权重，也不会执行比赛说明或 `model_all` 文档中的命令。

## 三个任务及基线

- Goal 1 真假人体识别：MedicalNet 3D ResNet-10 三分类，类别顺序固定为 `true / fake / composition`，提交字段 `IsNotHumanBodyProb = P(fake)`。
- Goal 2 拼接影像识别：与 Goal 1 共用同一个三分类模型及 checkpoint，提交字段 `IsStitchedProb = P(composition)`。
- Goal 2 重复影像识别：Siamese MedicalNet 3D ResNet-10。两侧共享编码器；一个检查内的多序列 embedding 先平均再 L2 归一化，由可训练的余弦评分器输出 `PairProb`。

MedicalNet 在赛方允许模型清单中。尽管云容器提供 NVIDIA RTX PRO 5000 Blackwell（73415 MiB），本方案仍优先选用任务匹配、实现成熟且成本适中的 ResNet-10，而非单纯增大参数量。

## 数据目录

推荐把项目放在 `/2026aicompetition/workspace/goal1and2`，标注放在项目内：

```text
goal1and2/
├── configs/config.yaml
├── labels/
│   ├── 1_abnormal.xlsx
│   └── 2_duplicate.xlsx
├── checkpoints/
└── ...
```

`paths.labels_dir: ./labels` 相对项目根目录解析。Excel 也可以放在其他持久化目录，只需修改该配置；不要把自建 Excel 写入只读官方数据目录。

`1_abnormal.xlsx` 必须包含 `AccessionNumber`、`SeriesUid`、`Label`。标签支持 `true`、`fake`、`composition` 和别名 `compositing`。`nonhuman` 是否并入 `fake` 由 `abnormal.label_aliases` 控制，默认并入，但这是需要结合最终赛方数据确认的解释。

`2_duplicate.xlsx` 必须包含 `src_img`、`desc_img`，两列均为 `duplicate/` 下检查的 AccessionNumber。读取 Excel 时统一使用 `dtype=str`；建议在 Excel 中也把检查号和序列号设为文本，避免前导零丢失。

默认官方影像结构为：

```text
annotation/
├── <true accession>/<series>/<series>.nii.gz
├── fake/<fake accession>/<series>/<series>.nii.gz
├── Composition/<composition accession>/<series>/<series>.nii.gz
└── duplicate/<accession>/<series>/<series>.nii.gz
```

支持 `.nii` 和 `.nii.gz`。如果序列目录内文件名不等于 SeriesUid，但只有一个有效 NIfTI，也可解析。Goal 1/2 不要求 `3_serieslabel.xlsx`、`4_masklabel.xlsx` 或 `5_characteristics.xlsx`。

## 配置和权重

主要配置位于 `configs/config.yaml`：

- `paths.annotation_root`：训练影像根目录。
- `paths.labels_dir`：两个 Excel 所在目录。
- `abnormal.checkpoint_path`、`duplicate.checkpoint_path`：训练保存及推理加载的 best checkpoint。
- `abnormal.pretrained_path`、`duplicate.pretrained_path`：可选的官方 MedicalNet ResNet-10 预训练权重；留空即随机初始化，代码不会自动下载。
- `abnormal.resume_path`、`duplicate.resume_path`：恢复训练 checkpoint；留空则不恢复。
- `data.target_shape`、`training.batch_size`、`duplicate.series_batch_size`：可按 72GB GPU 实测调整。默认开启 CUDA AMP，分类 batch size 为 4。
- `data.case_aggregation`：检查内序列 softmax 概率聚合，支持 `max`（默认）或 `mean`。

checkpoint 会记录模型名、类别顺序/embedding 维度、输入尺寸、检查聚合方式、轮次及预训练来源。推理使用严格参数加载。

## 安装、校验与训练

```bash
pip install -r requirements.txt
python scripts/validate_data.py --config configs/config.yaml --tasks goal1 goal2
python train/train_abnormal.py --config configs/config.yaml
python train/train_duplicate.py --config configs/config.yaml
```

三分类使用 `CrossEntropyLoss`，验证报告检查级 accuracy、macro-F1、fake-vs-rest 和 composition-vs-rest 的 AUROC/AUPRC。重复任务使用 `BCEWithLogitsLoss`，报告 AUROC、AUC-PR、Recall@10%FPR 和 Precision@15%Recall。重复关系按连通分量划分训练/验证，避免重复组泄漏；负样本只从 `duplicate/` 检查集合抽取。

## 独立预测

Goal 1：

```bash
python scripts/test_pipeline.py --task goal1 --config configs/config.yaml --input <测试集目录> --output <结果目录>
```

Goal 2：

```bash
python scripts/test_pipeline.py --task goal2 --config configs/config.yaml --input <测试集目录> --output <结果目录>
```

每个检查的 `<结果目录>/<AccessionNumber>/prediction.json` 始终同时包含 `IsNotHumanBodyProb` 和 `IsStitchedProb`。Goal 2 还会在结果根目录生成 `duplicate_pairs.jsonl`，去除自配对和重复对，并用全局贪心选择确保每个 ID 出现次数不超过 `duplicate.topk_candidates`（默认 200）。重复预测至少需要两个可读检查。

## 最小闭环测试

```bash
python scripts/smoke_test_goal1and2.py
```

该脚本只创建临时 synthetic NIfTI 和 CSV 等价 manifest，验证两个 Dataset/DataLoader、forward、loss、backward、checkpoint 严格重载、Goal 1/2 独立推理，以及 JSON/JSONL 约束。它不代表真实数据训练或指标已经验证。

## 已知问题与平台验证清单

- 当前按 NIfTI 实现；比赛资料中的 DICOM/NIfTI 表述冲突尚需在平台确认。如果最终输入为 DICOM，应新增显式转换/读取层，不应把格式猜测写死。
- 官方拼接目录按 `Composition/` 解释；Excel 同时接受 `composition/compositing`。目录名可通过 `data.source_dirs` 修改。
- `nonhuman` 默认映射到 `fake`，请用最终标注定义复核；可在 `abnormal.label_aliases` 中调整。
- 需在真实 Excel、真实 NIfTI 目录上执行 `validate_data.py`，并验证真实类别分布、损坏文件和重复组数量。
- 为同时得到无泄漏的训练集和验证集，重复标注至少需要两个互不连通的阳性重复组；不足时训练入口会明确报错，需与赛方确认可用的验证策略。
- 需用赛方提供的 MedicalNet 权重实测键名兼容性；未配置权重时模型从随机初始化训练。
- 需在大赛 Blackwell 容器中确认 PyTorch/CUDA 版本、AMP、合适 batch size、I/O 吞吐和最终指标。
