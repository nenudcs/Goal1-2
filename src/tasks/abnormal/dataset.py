import torch
from torch.utils.data import Dataset

from src.data.nifti import check_nifti, load_preprocessed_volume
from src.data.paths import PathResolver, read_manifest


CLASS_NAMES = ("true", "fake", "composition")
LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


def normalize_abnormal_labels(frame, aliases=None):
    required = {"AccessionNumber", "SeriesUid", "Label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Abnormal manifest is missing columns: {sorted(missing)}")
    aliases = aliases or {
        "true": "true", "fake": "fake", "nonhuman": "fake",
        "composition": "composition", "compositing": "composition",
    }
    frame = frame.copy()
    raw_labels = frame["Label"].astype(str).str.strip().str.lower()
    for column in ("AccessionNumber", "SeriesUid"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["Label"] = raw_labels.map(aliases)
    if frame["Label"].isna().any():
        bad = sorted(set(raw_labels[frame["Label"].isna()]))
        raise ValueError(f"Unsupported abnormal labels: {bad}")
    return frame


class AbnormalDataset(Dataset):
    def __init__(
        self, manifest_path, annotation_root, accessions, target_shape,
        clip_percentiles, label_aliases=None, source_dirs=None,
    ):
        aliases = {str(k).lower(): str(v).lower() for k, v in (label_aliases or {}).items()}
        self.df = normalize_abnormal_labels(read_manifest(manifest_path), aliases or None)
        if accessions is not None:
            allowed = set(map(str, accessions))
            self.df = self.df[self.df["AccessionNumber"].isin(allowed)].reset_index(drop=True)

        self.resolver = PathResolver(annotation_root, source_dirs=source_dirs)
        self.target_shape = target_shape
        self.clip_percentiles = clip_percentiles
        valid_rows, skipped = [], []
        for idx, row in self.df.iterrows():
            path = self.resolver.series_path(
                row["AccessionNumber"], row["SeriesUid"], row["Label"]
            )
            if check_nifti(path):
                valid_rows.append(idx)
            else:
                skipped.append((row["AccessionNumber"], row["SeriesUid"], str(path)))
        original_count = len(self.df)
        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        print(
            f"[AbnormalDataset] original={original_count} valid={len(self.df)} "
            f"missing_or_broken_skipped={len(skipped)}"
        )
        for accession, series_uid, path in skipped:
            print(
                f"[AbnormalDataset][SKIP] AccessionNumber={accession} "
                f"SeriesUid={series_uid} path={path}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.resolver.series_path(
            row["AccessionNumber"], row["SeriesUid"], row["Label"]
        )
        volume, _ = load_preprocessed_volume(
            path, self.target_shape, self.clip_percentiles
        )
        return (
            volume,
            torch.tensor(LABEL_MAP[row["Label"]], dtype=torch.long),
            row["AccessionNumber"],
            row["SeriesUid"],
        )
