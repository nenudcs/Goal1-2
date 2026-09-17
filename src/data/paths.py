from pathlib import Path

import pandas as pd

from src.data.nifti import check_nifti


DEFAULT_SOURCE_DIRS = {
    "true": "",
    "normal": "",
    "fake": "fake",
    "nonhuman": "fake",
    "composition": "Composition",
    "compositing": "Composition",
    "duplicate": "duplicate",
}


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def find_series_nifti(series_dir):
    series_dir = Path(series_dir)
    if not series_dir.is_dir():
        return None
    preferred = (
        series_dir / f"{series_dir.name}.nii.gz",
        series_dir / f"{series_dir.name}.nii",
    )
    for path in preferred:
        if check_nifti(path):
            return path
    candidates = sorted(series_dir.glob("*.nii.gz")) + sorted(series_dir.glob("*.nii"))
    valid = [path for path in candidates if check_nifti(path)]
    return valid[0] if len(valid) == 1 else None


def list_case_series(case_dir):
    case_dir = Path(case_dir)
    items = []
    if not case_dir.is_dir():
        return items
    for series_dir in sorted(case_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        path = find_series_nifti(series_dir)
        if path is not None:
            items.append((series_dir.name, path))
    return items


class PathResolver:
    """Resolve configurable competition source folders and NIfTI filenames."""

    def __init__(self, annotation_root: str, source_dirs=None):
        self.root = Path(annotation_root)
        self.source_dirs = dict(DEFAULT_SOURCE_DIRS)
        self.source_dirs.update(source_dirs or {})

    def source_root(self, source="true"):
        source = str(source).strip().lower()
        if source not in self.source_dirs:
            raise ValueError(f"Unknown source={source}")
        relative = str(self.source_dirs[source]).strip()
        return self.root if not relative else self.root / relative

    def series_dir(self, accession: str, series_uid: str, source: str = "true") -> Path:
        return self.source_root(source) / str(accession) / str(series_uid)

    def series_path(self, accession: str, series_uid: str, source: str = "true") -> Path:
        series_dir = self.series_dir(accession, series_uid, source)
        found = find_series_nifti(series_dir)
        return found or series_dir / f"{series_uid}.nii.gz"

    def mask_path(self, accession: str, series_uid: str, mask_name: str) -> Path:
        return self.root / str(accession) / str(series_uid) / str(mask_name)

    def duplicate_accession_dir(self, accession: str) -> Path:
        return self.source_root("duplicate") / str(accession)

    def duplicate_accessions(self):
        root = self.source_root("duplicate")
        if not root.is_dir():
            return []
        return sorted(path.name for path in root.iterdir() if path.is_dir())

    @staticmethod
    def list_case_series(case_dir):
        return [path for _, path in list_case_series(case_dir)]
