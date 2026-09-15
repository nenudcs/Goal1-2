import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.paths import PathResolver
from src.data.nifti import load_preprocessed_volume, check_nifti
from src.tasks.characteristics.schema import (
    MODALITIES, BINARY_FIELDS, CATEGORICAL_FIELDS, LOCATION_CLASSES
)

def _is_missing(v):
    return pd.isna(v) or str(v).strip() == ""

class CharacteristicsDataset(Dataset):
    """
    使用 3_serieslabel.xlsx 的金标准模态建立训练输入。
    正式端到端推理时由 sequence model 决定模态。
    """
    def __init__(
        self, characteristics_xlsx, series_xlsx, annotation_root,
        accessions, target_shape, clip_percentiles,
        modality_dropout=0.0, training=True
    ):
        cdf = pd.read_excel(characteristics_xlsx)
        sdf = pd.read_excel(series_xlsx)

        cdf["AccessionNumber"] = cdf["AccessionNumber"].astype(str)
        sdf["AccessionNumber"] = sdf["AccessionNumber"].astype(str)
        sdf["SeriesUid"] = sdf["SeriesUid"].astype(str)
        sdf["SeriesLabel"] = sdf["SeriesLabel"].astype(str).str.strip().str.upper()

        self.cdf = cdf[cdf["AccessionNumber"].isin(set(accessions))].reset_index(drop=True)
        self.resolver = PathResolver(annotation_root)
        self.target_shape = target_shape
        self.clip_percentiles = clip_percentiles
        self.modality_dropout = float(modality_dropout)
        self.training = training
        self.series_map = {}
        missing = []
        for acc, g in sdf.groupby("AccessionNumber"):
            valid = {}
            for _, row in g.iterrows():
                mod=row["SeriesLabel"]; suid=row["SeriesUid"]
                if mod not in MODALITIES: continue
                path=self.resolver.series_path(str(acc),str(suid),'true')
                if check_nifti(path): valid[mod]=suid
                else: missing.append((str(acc),str(suid),mod,str(path)))
            self.series_map[str(acc)] = valid
        valid_idx=[]; skipped_cases=[]
        for idx,row in self.cdf.iterrows():
            acc=str(row["AccessionNumber"])
            if self.series_map.get(acc): valid_idx.append(idx)
            else: skipped_cases.append(acc)
        self.cdf=self.cdf.loc[valid_idx].reset_index(drop=True)
        print(f'[CharacteristicsDataset] valid_cases={len(self.cdf)} missing_or_broken_modalities={len(missing)} cases_without_any_valid_series={len(skipped_cases)}')
        for x in missing: print(f'[CharacteristicsDataset][MISSING MODALITY] {x}')
        for x in skipped_cases: print(f'[CharacteristicsDataset][SKIP CASE] {x}')

    def __len__(self):
        return len(self.cdf)

    def __getitem__(self, idx):
        r = self.cdf.iloc[idx]
        acc = str(r["AccessionNumber"])
        smap = self.series_map.get(acc, {})

        vols = []
        present = []

        for mod in MODALITIES:
            suid = smap.get(mod)
            if suid is None:
                vols.append(torch.zeros((1, *self.target_shape), dtype=torch.float32))
                present.append(0.0)
                continue

            path = self.resolver.series_path(acc, suid, "true")
            x, _ = load_preprocessed_volume(path, self.target_shape, self.clip_percentiles)

            # 模态缺失增强：仅在训练时启用
            if self.training and random.random() < self.modality_dropout:
                vols.append(torch.zeros_like(x))
                present.append(0.0)
            else:
                vols.append(x)
                present.append(1.0)

        # 避免所有模态都被 dropout
        if sum(present) == 0 and smap:
            mod = next(iter(smap.keys()))
            mi = MODALITIES.index(mod)
            path = self.resolver.series_path(acc, smap[mod], "true")
            vols[mi], _ = load_preprocessed_volume(path, self.target_shape, self.clip_percentiles)
            present[mi] = 1.0

        x = torch.stack(vols, dim=0)   # [3,1,D,H,W]
        modality_mask = torch.tensor(present, dtype=torch.float32)

        targets = {}
        masks = {}

        for field, mapping in BINARY_FIELDS.items():
            v = r.get(field, np.nan)
            if _is_missing(v):
                targets[field] = torch.tensor(0, dtype=torch.long)
                masks[field] = torch.tensor(0.0)
            else:
                key = str(v).strip()
                # true/false 可能被 Excel 读成布尔
                if key in ("True", "FALSE", "False", "TRUE"):
                    key = key.lower()
                targets[field] = torch.tensor(mapping[key], dtype=torch.long)
                masks[field] = torch.tensor(1.0)

        for field, classes in CATEGORICAL_FIELDS.items():
            v = r.get(field, np.nan)
            if _is_missing(v):
                targets[field] = torch.tensor(0, dtype=torch.long)
                masks[field] = torch.tensor(0.0)
            else:
                key = str(v).strip()
                if field == "WHO_grade":
                    key = key.replace(".0", "")
                targets[field] = torch.tensor(classes.index(key), dtype=torch.long)
                masks[field] = torch.tensor(1.0)

        # Location 多标签
        v = r.get("Location", np.nan)
        loc_target = torch.zeros(len(LOCATION_CLASSES), dtype=torch.float32)
        if _is_missing(v):
            masks["Location"] = torch.tensor(0.0)
        else:
            for token in str(v).split("|"):
                token = token.strip()
                if token in LOCATION_CLASSES:
                    loc_target[LOCATION_CLASSES.index(token)] = 1.0
            masks["Location"] = torch.tensor(1.0)
        targets["Location"] = loc_target

        return x, modality_mask, targets, masks, acc
