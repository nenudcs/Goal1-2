from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.seed import seed_everything
from src.common.logging_utils import build_logger, CompetitionJSONLLogger
from src.common.train_utils import get_device, save_checkpoint
from src.data.split import get_or_create_true_case_split
from src.models.basic3d import MultiModalClassifier
from src.tasks.characteristics.dataset import CharacteristicsDataset
from src.tasks.characteristics.schema import BINARY_FIELDS, CATEGORICAL_FIELDS, LOCATION_CLASSES

HEADS = {
    **{k: 2 for k in BINARY_FIELDS},
    **{k: len(v) for k, v in CATEGORICAL_FIELDS.items()},
    "Location": len(LOCATION_CLASSES),
}

def masked_multitask_loss(outputs, targets, masks):
    total = 0.0
    active = 0
    details = {}

    for name, logits in outputs.items():
        mask = masks[name].to(logits.device).float()

        if name == "Location":
            y = targets[name].to(logits.device).float()
            per_sample = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1)
        else:
            y = targets[name].to(logits.device).long()
            per_sample = F.cross_entropy(logits, y, reduction="none")

        denom = mask.sum()
        if denom.item() > 0:
            loss = (per_sample * mask).sum() / denom
            total = total + loss
            active += 1
            details[name] = float(loss.detach().cpu())

    if active == 0:
        # 理论上不应发生，但保留保护逻辑
        total = next(iter(outputs.values())).sum() * 0.0

    return total / max(active, 1), details

def main(cfg):
    logger = build_logger("train_characteristics")
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)

    cpath = Path(cfg["paths"]["labels_dir"]) / "5_characteristics.xlsx"
    spath = Path(cfg["paths"]["labels_dir"]) / "3_serieslabel.xlsx"

    cdf = pd.read_excel(cpath)
    cdf["AccessionNumber"] = cdf["AccessionNumber"].astype(str)
    train_ids, val_ids = get_or_create_true_case_split(
        Path(cfg["paths"]["labels_dir"])/"1_abnormal.xlsx",
        Path(cfg["paths"]["output_dir"])/"splits"/"true_case_split_0.1.json",
        cfg["split"]["val_ratio"], cfg["split"]["random_state"]
    )

    logger.info("========== Characteristics training ==========")
    logger.info("Train cases=%d | Val cases=%d | Device=%s",
                len(train_ids), len(val_ids), device)
    logger.info("Heads=%s", HEADS)

    for col in cdf.columns:
        if col == "AccessionNumber":
            continue
        logger.info("Label non-empty | %-20s = %d/%d",
                    col, cdf[col].notna().sum(), len(cdf))

    shape = cfg["data"]["target_shape"]
    clip = cfg["data"]["intensity_clip_percentiles"]

    train_ds = CharacteristicsDataset(
        cpath, spath, cfg["paths"]["annotation_root"], train_ids, shape, clip,
        modality_dropout=cfg["characteristics"]["modality_dropout"], training=True
    )
    val_ds = CharacteristicsDataset(
        cpath, spath, cfg["paths"]["annotation_root"], val_ids, shape, clip,
        modality_dropout=0.0, training=False
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg["project"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg["project"]["num_workers"])

    model = MultiModalClassifier(
        feature_dim=cfg["characteristics"]["feature_dim"], heads=HEADS
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])

    jlog = CompetitionJSONLLogger(
        cfg["paths"]["competition_log_dir"], "characteristics_train.jsonl",
        fallback_dir=str(Path(cfg["paths"]["output_dir"]) / "logs")
    )

    best = float("inf")
    global_step = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_losses = []

        for step, (x, mmask, targets, masks, acc) in enumerate(train_loader, 1):
            global_step += 1
            x, mmask = x.to(device), mmask.to(device)
            opt.zero_grad(set_to_none=True)

            outputs = model(x, mmask)
            loss, details = masked_multitask_loss(outputs, targets, masks)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

            if step % cfg["training"]["log_every"] == 0 or step == 1:
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d global=%d loss=%.6f "
                    "modalities=%s case=%s task_losses=%s",
                    epoch, step, len(train_loader), global_step, loss.item(),
                    mmask[0].detach().cpu().tolist(), acc[0],
                    {k: round(v, 4) for k, v in details.items()}
                )
                jlog.write(epoch, global_step, "train", "training",
                           "official/train_v2", loss.item(), opt.param_groups[0]["lr"])

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, mmask, targets, masks, acc in val_loader:
                x, mmask = x.to(device), mmask.to(device)
                outputs = model(x, mmask)
                loss, _ = masked_multitask_loss(outputs, targets, masks)
                val_losses.append(loss.item())

        val_loss = sum(val_losses)/max(len(val_losses), 1)
        logger.info("[VAL] epoch=%d train_loss=%.6f val_loss=%.6f",
                    epoch, sum(train_losses)/max(len(train_losses),1), val_loss)

        ckpt_dir = Path(cfg["paths"]["checkpoints_dir"]) / "characteristics"
        ckpt = ckpt_dir / f"epoch_{epoch}.pth"
        save_checkpoint(model, opt, epoch, ckpt, {"heads": HEADS, "val_loss": val_loss})
        jlog.write(epoch, global_step, "val", "training", "internal/validation_v2",
                   val_loss, opt.param_groups[0]["lr"], ckpt)

        if val_loss < best:
            best = val_loss
            save_checkpoint(model, opt, epoch, ckpt_dir / "best.pth",
                            {"heads": HEADS, "val_loss": val_loss})
            logger.info("New best characteristics model: val_loss=%.6f", val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
