from pathlib import Path

class PathResolver:
    """
    统一解决赛事目录中的路径差异。

    normal/true:
      root/{AccessionNumber}/{SeriesUid}/{SeriesUid}.nii.gz

    fake:
      root/fake/{AccessionNumber}/{SeriesUid}/{SeriesUid}.nii.gz

    compositing:
      root/compositing/{AccessionNumber}/{SeriesUid}/{SeriesUid}.nii.gz

    duplicate:
      root/duplicate/{AccessionNumber}/{SeriesUid}/{SeriesUid}.nii.gz
    """
    def __init__(self, annotation_root: str):
        self.root = Path(annotation_root)

    def series_path(self, accession: str, series_uid: str, source: str = "true") -> Path:
        source = str(source).strip().lower()

        if source in ("true", "normal", ""):
            base = self.root
        elif source == "fake":
            base = self.root / "fake"
        elif source in ("compositing", "composition"):
            choices = [self.root / name for name in ("Composition", "composition", "compositing")
                       if (self.root / name / str(accession) / str(series_uid)).is_dir()]
            if len(choices) > 1:
                raise ValueError(f"拼接影像路径存在歧义: {choices}")
            base = choices[0] if choices else self.root / "compositing"
        elif source == "duplicate":
            base = self.root / "duplicate"
        else:
            raise ValueError(f"未知 source={source}")

        return base / str(accession) / str(series_uid) / f"{series_uid}.nii.gz"

    def series_dir(self, accession: str, series_uid: str, source: str = "true") -> Path:
        return self.series_path(accession, series_uid, source).parent

    def mask_path(self, accession: str, series_uid: str, mask_name: str) -> Path:
        # 当前 mask 仅用于正常训练数据
        return self.root / str(accession) / str(series_uid) / str(mask_name)

    def duplicate_accession_dir(self, accession: str) -> Path:
        return self.root / "duplicate" / str(accession)
