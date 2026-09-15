from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from src.common.config import load_config
from src.common.seed import seed_everything
from src.common.logging_utils import build_logger, CompetitionJSONLLogger
from src.common.train_utils import get_device, save_checkpoint
from src.data.split import get_or_create_true_case_split
from src.tasks.segmentation.dataset import SegmentationDataset
from src.tasks.segmentation.model import build_segmentation_model

def main(cfg, task):
    logger = build_logger(f"train_segmentation_{task}")
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)

    xlsx = Path(cfg["paths"]["labels_dir"]) / "4_masklabel.xlsx"
    df = pd.read_excel(xlsx)
    df["AccessionNumber"] = df["AccessionNumber"].astype(str)
    df["Task"] = df["Task"].astype(str).str.lower()
    task_df = df[df["Task"] == task]

    train_ids, val_ids = get_or_create_true_case_split(
        Path(cfg["paths"]["labels_dir"])/"1_abnormal.xlsx",
        Path(cfg["paths"]["output_dir"])/"splits"/"true_case_split_0.1.json",
        cfg["split"]["val_ratio"], cfg["split"]["random_state"]
    )

    logger.info("========== Segmentation training ==========")
    logger.info("Task=%s | Train cases=%d | Val cases=%d | Device=%s",
                task, len(train_ids), len(val_ids), device)
    logger.info("Mask rows=%d | Unique series=%d", len(task_df),
                task_df[["AccessionNumber","SeriesUid"]].drop_duplicates().shape[0])

    shape = cfg["data"]["target_shape"]
    clip = cfg["data"]["intensity_clip_percentiles"]
    train_ds = SegmentationDataset(xlsx, cfg["paths"]["annotation_root"],
                                   train_ids, task, shape, clip)
    val_ds = SegmentationDataset(xlsx, cfg["paths"]["annotation_root"],
                                 val_ids, task, shape, clip)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg["project"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg["project"]["num_workers"])

    model = build_segmentation_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    criterion = DiceCELoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    jlog = CompetitionJSONLLogger(
        cfg["paths"]["competition_log_dir"], f"segmentation_{task}_train.jsonl",
        fallback_dir=str(Path(cfg["paths"]["output_dir"]) / "logs")
    )

    best_dice = -1.0
    global_step = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_losses = []

        for step, (x, y, acc, suid, img_path) in enumerate(train_loader, 1):
            global_step += 1
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

            if step % cfg["training"]["log_every"] == 0 or step == 1:
                pos_voxels = int(y.sum().item())
                logger.info(
                    "[TRAIN] task=%s epoch=%d step=%d/%d global=%d "
                    "loss=%.6f positive_voxels=%d case=%s series=%s",
                    task, epoch, step, len(train_loader), global_step,
                    loss.item(), pos_voxels, acc[0], suid[0]
                )
                jlog.write(epoch, global_step, "train", "training",
                           "official/train_v2", loss.item(), opt.param_groups[0]["lr"])

        model.eval()
        dice_metric.reset()
        val_losses = []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_losses.append(criterion(logits, y).item())
                pred = (torch.sigmoid(logits) > 0.5).float()
                dice_metric(y_pred=pred, y=y)

        val_dice = float(dice_metric.aggregate().item()) if len(val_loader) else 0.0
        val_loss = sum(val_losses)/max(len(val_losses),1)

        logger.info(
            "[VAL] task=%s epoch=%d train_loss=%.6f val_loss=%.6f dice=%.4f",
            task, epoch, sum(train_losses)/max(len(train_losses),1), val_loss, val_dice
        )

        ckpt_dir = Path(cfg["paths"]["checkpoints_dir"]) / "segmentation" / task
        ckpt = ckpt_dir / f"epoch_{epoch}.pth"
        save_checkpoint(model, opt, epoch, ckpt, {"val_dice": val_dice, "task": task})
        jlog.write(epoch, global_step, "val", "training", "internal/validation_v2",
                   val_loss, opt.param_groups[0]["lr"], ckpt)

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, opt, epoch, ckpt_dir / "best.pth",
                            {"val_dice": val_dice, "task": task})
            logger.info("New best segmentation model | task=%s dice=%.4f", task, val_dice)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--task", required=True, choices=["core", "abnormal"])
    args = parser.parse_args()
    main(load_config(args.config), args.task)
