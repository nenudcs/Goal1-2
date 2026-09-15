from pathlib import Path
import json
import time
import torch
import torch.nn.functional as F

from src.data.nifti import load_preprocessed_volume, save_mask_like_reference, check_nifti
from src.tasks.sequence.dataset import INV_LABEL_MAP
from src.tasks.characteristics.schema import (
    MODALITIES, BINARY_FIELDS, CATEGORICAL_FIELDS, LOCATION_CLASSES
)

def list_series(case_dir: Path):
    items = []
    for sd in sorted(case_dir.iterdir()):
        if not sd.is_dir():
            continue
        p = sd / f"{sd.name}.nii.gz"
        if check_nifti(p):
            items.append((sd.name, p))
    return items

@torch.no_grad()
def run_case(case_dir, output_case_dir, models, cfg, device):
    """
    正式病例主链:
      abnormal -> sequence -> segmentation -> characteristics
    """
    t0 = time.time()
    accession = case_dir.name
    series = list_series(case_dir)
    if not series:
        raise RuntimeError(f"{accession}: 未找到 SeriesUid.nii.gz")

    shape = cfg["data"]["target_shape"]
    clip = cfg["data"]["intensity_clip_percentiles"]

    # -------- 1) abnormal --------
    abnormal_probs = []
    cached = {}
    for suid, p in series:
        x, _ = load_preprocessed_volume(p, shape, clip)
        cached[suid] = (x, p)
        logits = models["abnormal"](x.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0].cpu()
        abnormal_probs.append(probs)

    case_abn = torch.stack(abnormal_probs).mean(0)
    fake_prob = float(case_abn[1])
    stitched_prob = float(case_abn[2])

    # V1 不做硬截断，仍继续下游，避免 abnormal 一次误判导致所有后续任务丢失。
    # 后续可根据验证集增加高置信阈值。

    # -------- 2) sequence recognition --------
    modality_candidates = {m: [] for m in MODALITIES}
    for suid, (x, p) in cached.items():
        logits = models["sequence"](x.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(probs.argmax())
        pred_label = INV_LABEL_MAP[pred_idx]
        modality_candidates[pred_label].append((float(probs[pred_idx]), suid, x, p))

    # 每个模态保留最高置信的 Series
    selected = {}
    for mod, candidates in modality_candidates.items():
        if candidates:
            selected[mod] = sorted(candidates, reverse=True, key=lambda z: z[0])[0]

    # -------- 3) segmentation --------
    output_case_dir = Path(output_case_dir)
    output_case_dir.mkdir(parents=True, exist_ok=True)

    mask_uri = {}

    if "T1CE" in selected:
        _, suid, x, p = selected["T1CE"]
        logits = models["seg_core"](x.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0,0] > 0.5).float().cpu().numpy()
        out = output_case_dir / suid / f"{suid}_core.nii.gz"
        save_mask_like_reference(pred, p, out)
        mask_uri["core"] = str(out)

    abnormal_source = "FLAIR" if "FLAIR" in selected else ("T2" if "T2" in selected else None)
    if abnormal_source:
        _, suid, x, p = selected[abnormal_source]
        logits = models["seg_abnormal"](x.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0,0] > 0.5).float().cpu().numpy()
        out = output_case_dir / suid / f"{suid}_abnormal.nii.gz"
        save_mask_like_reference(pred, p, out)
        mask_uri["flair"] = str(out)

    # -------- 4) characteristics --------
    volumes = []
    mmask = []
    for mod in MODALITIES:
        if mod in selected:
            volumes.append(selected[mod][2])
            mmask.append(1.0)
        else:
            volumes.append(torch.zeros((1, *shape), dtype=torch.float32))
            mmask.append(0.0)

    xmulti = torch.stack(volumes, 0).unsqueeze(0).to(device)
    mmask_t = torch.tensor([mmask], dtype=torch.float32, device=device)
    outputs = models["characteristics"](xmulti, mmask_t)

    pred = {}

    for field, mapping in BINARY_FIELDS.items():
        p = F.softmax(outputs[field], dim=1)[0]
        pos_prob = float(p[1].cpu())
        pred[field] = {"probability": pos_prob, "predicted": int(p.argmax().cpu())}

    for field, classes in CATEGORICAL_FIELDS.items():
        p = F.softmax(outputs[field], dim=1)[0].cpu()
        pred[field] = {
            "predicted": classes[int(p.argmax())],
            "probabilities": {c: float(p[i]) for i, c in enumerate(classes)}
        }

    locp = torch.sigmoid(outputs["Location"])[0].cpu()
    loc_probs = {c: float(locp[i]) for i, c in enumerate(LOCATION_CLASSES)}
    pred["Location"] = {
        "predicted": max(loc_probs, key=loc_probs.get),
        "probabilities": loc_probs,
    }

    # -------- 比赛输出 JSON --------
    glioma_prob = pred["Glioma"]["probability"]
    enhancement_prob = pred["Enhancement"]["probability"]

    result = {
        "AccessionNumber": accession,
        "IsNotHumanBodyProb": fake_prob,
        "IsStitchedProb": stitched_prob,
        "ProcessingTime_ms": int((time.time() - t0) * 1000),
        "SegmentationMaskURI": mask_uri,
        "Prediction": {
            "TumorProbability": glioma_prob,
            "Location": pred["Location"]["predicted"],
            "Morphology": {
                "predicted": "Irregular" if pred["Morphology"]["predicted"] == 1 else "Regular",
                "probabilities": {
                    "Regular": 1.0 - pred["Morphology"]["probability"],
                    "Irregular": pred["Morphology"]["probability"],
                },
            },
            "WHO_Grade": {
                "predicted": int(pred["WHO_grade"]["predicted"]) if glioma_prob >= 0.5 else None,
                "probabilities": pred["WHO_grade"]["probabilities"],
            },
            "Enhancement": {
                "present": enhancement_prob >= 0.5,
                "EnhancementProbability": enhancement_prob,
            },
            "EnhancementPattern": pred["EnhancementPattern"],
            "Necrosis": {
                "present": pred["Necrosis"]["probability"] >= 0.5,
                "NecrosisProbability": pred["Necrosis"]["probability"],
            },
            "CysticChange": {
                "present": pred["CysticChange"]["probability"] >= 0.5,
                "CysticChangeProbability": pred["CysticChange"]["probability"],
            },
            "Hemorrhage": {
                "present": pred["Hemorrhage"]["probability"] >= 0.5,
                "HemorrhageProbability": pred["Hemorrhage"]["probability"],
            },
            "Calcification": {
                "present": pred["Calcification"]["probability"] >= 0.5,
                "CalcificationProbability": pred["Calcification"]["probability"],
            },
            "Margin": {
                "clear": pred["Margin"]["probability"] >= 0.5,
                "MarginClearProbability": pred["Margin"]["probability"],
            },
            "Lobulation": {
                "present": pred["Lobulation"]["probability"] >= 0.5,
                "LobulationProbability": pred["Lobulation"]["probability"],
            },
            "Signal_T2WI": pred["Signal_T2WI"],
            "Signal_FLAIR": pred["Signal_FLAIR"],
        },
        "Interpretation": {
            "Conclusion": "",
            "AttentionMapURI": "",
        },
    }

    with (output_case_dir / "prediction.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
