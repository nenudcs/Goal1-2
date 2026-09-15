# 目标一与目标二训练和集成指南

本扩展在原项目中实现 ConvNeXt-Tiny 序列三分类与 DINOv2 检查重复检测，保留原有其他任务。所有命令、模型运行和测试都在赛事容器内执行；代码不自动下载源码、权重或依赖。没有采用旁边 `goal2_project` 的实现。

当前交付是代码与容器验收用例，不包含已训练权重，也未在真实赛事容器中验证效果。本地完成了 54 个 Python 文件的静态语法检查及新增模块的本地导入引用检查，未执行项目代码。12 项合成数据工程测试须在容器中运行。目标三至五由队友合并；目标一二单独输出不能代表完成全部赛道。

## 1. 上传及环境检查

上传完整的“医保AI识图”文件夹到队伍工作区，以下示例假设代码位于 `/2026aicompetition/workspace/医保AI识图`：

```bash
cd /2026aicompetition/workspace/医保AI识图
python -m scripts.detection doctor --metadata-only
python -m scripts.detection doctor
```

默认配置是 `configs/config_detection.yaml`。使用其他配置时，`--config` 放在子命令之前：

```bash
python -m scripts.detection --config configs/config_detection.yaml doctor
```

先检查配置中的原图根目录、标签目录、两个基线源码目录和权重路径。普通 Linux 容器及 CUDA 是运行前提；代码不会退回本地 Windows 或 CPU。`doctor --metadata-only` 仅检查资源信息，不能代表 GPU 验收。

完整 `doctor` 实际加载两份官方权重，检查 ConvNeXt 全主干反向传播、DINOv2 前向、配对头反向及保存后重载一致性，结果写入 `outputs_detection/doctor.json`。每次先标记为运行中，失败后写入错误及 `gpu_verified=false`；只有全部检查通过才记录成功。

使用赛事提供的 Blackwell 兼容镜像，不要直接套用旧基线的 PyTorch 2.0 环境。`requirements-detection.txt` 是依赖清单；缺失依赖需从赛事允许的资源补齐。压缩 DICOM 可能另需 `pylibjpeg` 等解码插件。程序不会安装依赖、升级驱动或联网取模型。

预训练权重与训练产物分开：前者只读取公共模型目录，后者写入 `checkpoints_detection`。原始目录清单可能省略子目录，找不到源码时应核对容器实际内容，不能据清单缺项认定资源不存在。

## 2. 数据索引及负例口径

兼容准备好的结构：`<原图根目录>/<检查号>/<序列号>/<序列号>.nii.gz`，以及特殊目录 `fake`、`Composition`、`composition`、`compositing`、`duplicate`。影像通过统一索引查找，配对标签不会改变读取路径。同编号在不同来源文件夹出现时会报错，需先消除歧义。

标签目录包含 `1_abnormal.xlsx` 与 `2_duplicate.xlsx`，也允许同名 CSV；同一表不能同时存在 XLSX 和 CSV。

- 三分类表：`AccessionNumber, SeriesUid, Label`，Label 为 `true/fake/compositing`；也接受 `composition` 拼写。标签是逐序列监督。
- 重复表：`src_img, desc_img` 表示正对；若存在 `label` 列，则值为 `0/1`，空标签为未知，不用于监督。
- 检查号及序列号必须是文本。XLSX 数字编号和公式会被拒绝，避免长编号精度与前导零损失。已经损失的数字不能靠转字符串恢复。
- 可在三分类表提供 `PatientId`，或通过 `patient_metadata_file` 指定 `AccessionNumber,PatientId` 表。缺失患者信息会在摘要中明确提示。

重复负例默认 `negative_policy: unknown`。这时允许索引与检索诊断，禁止判重训练、正式判重验证及完整推理。确定以下一种事实后修改配置，并重新索引：

1. `explicit`：重复表含明确 `label=0` 的记录，或通过 `negative_pair_file` 指定单独负对表。单独负对表无 label 列时全部为明确负对。
2. `closed_world`：队友或组织方已经确认某个检查集合内的重复清单完整；必须通过 `closed_world_ids_file` 指定带 `AccessionNumber` 列的 XLSX/CSV，只有这个集合内未列出的对可生成负例。此模式会枚举集合内配对，较大集合建议提供明确负对文件。

未知配对、不同行号、不同检查号或不同患者号，都不自动成为负例。确认口径是数据前提，不是通过改配置就能获得的事实。

```bash
python -m scripts.detection index
```

输出 `outputs_detection/manifest.json` 和 `data_summary.json`，包含有效影像、分组划分、类别及配对数量、数据内容校验值与警告。索引会实际解压影像并计算校验值，首次运行可能较久。

优先采用 `split_file`；未指定时探测标签目录和原图根目录中的唯一 `official_split.csv`、`splits.csv`、`split.csv` 或 `split.json`。固定表需完整覆盖检查号，并提供 `train/val/calibration` 三种集合；两集合划分需先由队友补充校准集合。没有固定划分时，按患者和已知正对连通组生成约 80/10/10 划分，不拆散同源分组。正对关系只用于分组，不推导新的正标签。

每个集合都需要足够的三类序列、检查级真假/拼接正负样本，以及正式判重所需的明确正负对。无法满足时相应训练会停止。跨集合负对不用于单集合训练或评价。

NIfTI 保留物理空间信息，转换为规范 RAS 顺序，不压缩成固定三维立方体。推理也支持具有完整几何信息的单帧灰度 DICOM 序列；按物理位置排序并执行同一后续预处理。多帧、多时相混合、非均匀层间距、缺失几何及不支持的像素类型会明确报错，需在容器中先转换或拆分。不会静默漏读。NIfTI 只接受精确命名的原图文件，忽略其他掩码文件；若有人把掩码伪装成原图文件名，仍需人工核对数据来源。

## 3. 训练、恢复及验证

```bash
python -m scripts.detection train-abnormal
python -m scripts.detection evaluate --branch abnormal
# 确认负例口径、重新 index 且 duplicate_ready=true 后：
python -m scripts.detection train-duplicate
python -m scripts.detection evaluate
```

旧训练入口传入新配置也会进入新流程，例如 `python -m train.train_abnormal --config configs/config_detection.yaml`。不传新配置时仍是旧模型。

三分类使用完整视图和覆盖边缘的局部视图，邻近三片作为三个输入通道。每序列仅计算一次分类损失；训练默认采样 64 个视图，验证与推理遍历全部视图并分块聚合。前 5 个 epoch 冻结主干，之后微调整个主干。检查级真假、拼接分数分别取对应类别的最大序列概率，再单独校准；这两个检查级字段允许同时较高。

DINOv2 始终冻结，缓存每序列和检查的特征；病例内容、源码或权重变化会产生不同缓存键。检索同时考虑检查及序列相似度。配对头只在训练集合内从明确负例池挖掘难负例，校准使用完整的已标注校准对。

训练没有墙钟截止时间，默认 `max_epochs: null`。连续 10 次验证指标未改善时早停，可配置 `patience`；如设置 `max_epochs`，它仅控制轮次。三分类选模指标为检查级真假与拼接的平均 AP，判重使用候选遗漏按零分、最终 200 对裁剪后的 AP。

每个分支保存 `last.pth`、`best.pth`、`calibration.json`、`validation.json`。`last.pth` 按完整 epoch 保存模型、优化器及随机状态；中途终止会从最近完整 epoch 恢复，未完成轮次可能重跑。默认再次执行训练命令自动恢复；需要在早停后继续时：

```bash
python -m scripts.detection train-abnormal --continue-training
python -m scripts.detection train-duplicate --continue-training
```

`--continue-training` 清零早停计数，保留模型及优化器状态。数据、结构、预处理或实现发生变化会阻止错误恢复，需建立新实验目录。修改轮次上限与 patience 不会使恢复失效。负例口径明确后，若原图、三分类标签及三集合划分完全不变，可继续使用之前的三分类权重；分组变化则需要重新训练。

校准始终使用独立 calibration 集，并与 best 权重校验值绑定。校准分数与真值无正相关时会报错，保留已验证权重供诊断，不生成可提交的校准文件。不要用其他实验的校准参数绕过检查。

日志固定写入 `/2026aicompetition/workspace/logs`，包含实际训练和验证阶段；写入失败不会回退目录。GPU、内存不足及损坏影像均明确失败，不会通过跳过切片继续输出。可在新实验配置中降低训练视图数或微批次，但修改预处理或训练配置后应核对恢复兼容性。

内部评价包含三分类逐类 precision/recall/F1、检查级 AP/ROC-AUC、判重候选召回与最终保留正对比例，以及 Recall@10%FPR、Precision@15%Recall。AP 按分数组合后的阶梯 PR 定义计算；工作点使用离散阈值，不插值。未知对不纳入标签全集，未提交的已标注对按 0 分计入。官方门槛、负对全集与插值口径仍需官方评分细则核实，不能把内部验证当成独立测试成绩。

## 4. 检索诊断、正式推理及合并

尚未确认负例时可以生成候选检查号诊断：

```bash
python -m scripts.detection retrieve --output outputs_detection/retrieval_diagnostic.json
```

这个文件不含重复概率，不是提交文件。正式推理需要两份验证并校准完成的模型：

```bash
python -m scripts.detection infer --input /data/testset \
  --output /2026aicompetition/workspace/answer/manual_001
python -m scripts.detection validate-output --input /data/testset \
  --output /2026aicompetition/workspace/answer/manual_001
```

输出每检查的 `prediction.json` 与根目录 `duplicate_pairs.jsonl`。`StudyUID/StudyUID_dup` 使用 AccessionNumber，不使用 DICOM StudyInstanceUID。每检查最多关联 200 对；少于两个检查时不能满足官方非空对文件要求，会明确报错。

结果先在临时目录完成校验，再整体发布。相同输入、模型和合并来源的已完成结果可复用；其他已有输出目录拒绝覆盖。`detection_manifest.json` 用于恢复及来源核对，不包含新增诊断结论。结果完整性校验针对目标一二，不验证目标三至五的医学正确性。

队友合并示例：

```bash
python -m scripts.detection merge \
  --detection /2026aicompetition/workspace/answer/manual_001 \
  --teammate /2026aicompetition/workspace/teammate_result \
  --output /2026aicompetition/workspace/answer/merged_001
```

合并要求检查号集合完全相同。复制队友完整结果树，仅更新真假、拼接字段及重复对文件，保留其他预测和相对掩码路径。源目录不变；目标目录必须不存在。队友引用的文件需实际存在且使用源结果树内的相对路径，绝对路径、外部 URI、符号链接及路径逃逸会被拒绝。

Python 公共入口位于 `src.detection.pipeline`：`load_detection_models(cfg, device=None)`、`run_detection_batch(dataset_path, output_dir, models, cfg, merge_from=None)`。模型加载一次并复用；批量函数返回目标一二完整性报告。结果合并入口为 `src.detection.submission.merge_results(...)`。

## 5. 服务和容器验收

```bash
python -m scripts.detection serve
# 等价入口：
python -m service.server --config configs/config_detection.yaml
```

服务以前台进程运行，固定监听 8000，提供 `/health`、`POST /call`、`GET /jobs/{request_id}`、`POST /retry`。独立 ASGI 入口也可用 `service.detection_server:app`，通过 `DETECTION_CONFIG` 指定配置。

`/call` 使用规范的嵌套 `input.evaluation_id` 与 `input.dataset_path`。任务持久化后快速返回 200，单工作线程串行推理；重复 request_id 的相同请求幂等，不同输入或重复占用 evaluation_id 返回 409。只运行一个服务进程，文件锁阻止重复工作线程。

重启后恢复排队/运行任务；已完成推理且等待回调的任务只恢复回调，并重新校验磁盘结果。健康检查依据模型加载与工作线程状态，不被 GPU 推理阻塞。失败任务可用 `POST /retry` 请求体 `{"request_id":"原请求号"}` 重试；失败原因保存在任务 JSON 中。回调重试采用至少一次交付，网络中断时可能重复发送相同 request_id，需在平台联调中确认上游幂等处理。不要手动删除或修改正在运行的任务文件。

默认 `callback_enabled: false`，任务完成为 `awaiting_merge`，由队友统一合并并回调。若把本服务用于合并后的提交，需同时配置 `merge_from_template`（例如 `/2026aicompetition/workspace/team_results/{evaluation_id}`）、真实 `callback_url` 和 `callback_enabled: true`；只有队友结果已准备齐全且合并校验通过才回调。回调默认使用字符串 evaluationId 和绝对 predPath；平台若要求其他类型或相对路径，分别调整 `callback_evaluation_id_type`、`callback_pred_path_template` 并实际联调。回调非 2xx 会重试，最终失败可查询并手动重试。

容器内执行工程用例：

```bash
python -m unittest discover -s tests -p 'test_detection_*.py' -v
```

测试涵盖文本编号、未知标签、患者/正对分组、NIfTI 方向、DICOM 几何排序、全部视图覆盖、分块聚合一致性、梯度与重载、配对对称性、200 对限制、漏检计分、训练恢复、校准绑定、HTTP 幂等、故障重试、回调恢复和结果合并。测试只用合成数据及小模型，不能代替完整 `doctor`、真实数据训练或平台评分。缺少依赖会明确失败，不以跳过测试冒充验收通过。
