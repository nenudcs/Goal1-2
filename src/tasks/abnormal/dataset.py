import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.paths import PathResolver
from src.data.nifti import load_preprocessed_volume, check_nifti

LABEL_MAP = {"true": 0, "fake": 1, "compositing": 2}

class AbnormalDataset(Dataset):
    def __init__(self, xlsx_path, annotation_root, accessions, target_shape, clip_percentiles):
        self.df = pd.read_excel(xlsx_path)
        self.df["AccessionNumber"] = self.df["AccessionNumber"].astype(str)
        self.df["SeriesUid"] = self.df["SeriesUid"].astype(str)
        self.df["Label"] = self.df["Label"].astype(str).str.strip().str.lower()
        self.df = self.df[self.df["AccessionNumber"].isin(set(accessions))].reset_index(drop=True)

        self.resolver = PathResolver(annotation_root)
        self.target_shape = target_shape
        self.clip_percentiles = clip_percentiles

        original_count = len(self.df)
        valid_rows, skipped = [], []
        for idx, r in self.df.iterrows():
            source = r["Label"] if "Label" in r else "true"
            path = self.resolver.series_path(r["AccessionNumber"], r["SeriesUid"], source)
            if check_nifti(path):
                valid_rows.append(idx)
            else:
                skipped.append((r["AccessionNumber"], r["SeriesUid"], str(path)))
        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        print(f"[{self.__class__.__name__}] original={original_count} valid={len(self.df)} missing_or_broken_skipped={len(skipped)}")
        for acc,suid,path in skipped:
            print(f"[{self.__class__.__name__}][SKIP] AccessionNumber={acc} SeriesUid={suid} path={path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        source = r["Label"]
        path = self.resolver.series_path(r["AccessionNumber"], r["SeriesUid"], source)
        x, _ = load_preprocessed_volume(path, self.target_shape, self.clip_percentiles)
        y = LABEL_MAP[source]
        return x, torch.tensor(y, dtype=torch.long), r["AccessionNumber"], r["SeriesUid"]
