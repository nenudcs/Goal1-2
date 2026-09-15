from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

def load_nifti(path):
    path = Path(path)
    nii = nib.load(str(path))
    arr = nii.get_fdata(dtype=np.float32)
    return arr, nii.affine, nii.header

def robust_normalize(arr: np.ndarray, p_low=0.5, p_high=99.5):
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    x[~finite] = 0

    nonzero = x[np.abs(x) > 1e-8]
    if nonzero.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(nonzero, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6

    x = np.clip(x, lo, hi)

    vals = x[np.abs(x) > 1e-8]
    mean = vals.mean() if vals.size else 0.0
    std = vals.std() if vals.size else 1.0
    std = max(float(std), 1e-6)

    x = (x - mean) / std
    return x.astype(np.float32)

def resize_volume(arr: np.ndarray, target_shape):
    """
    将 3D volume resize 到固定 shape。
    输入约定为 nibabel 得到的 (X,Y,Z)，内部统一成 tensor [1,1,D,H,W]。
    """
    x = torch.from_numpy(arr).float()
    x = x.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # Z,Y,X -> 1,1,D,H,W
    x = F.interpolate(x, size=tuple(target_shape), mode="trilinear", align_corners=False)
    return x.squeeze(0)  # [1,D,H,W]

def resize_mask(mask: np.ndarray, target_shape):
    x = torch.from_numpy(mask.astype(np.float32))
    x = x.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=tuple(target_shape), mode="nearest")
    return x.squeeze(0)  # [1,D,H,W]

def load_preprocessed_volume(path, target_shape, clip_percentiles=(0.5, 99.5)):
    arr, affine, header = load_nifti(path)
    arr = robust_normalize(arr, *clip_percentiles)
    tensor = resize_volume(arr, target_shape)
    return tensor, {"original_shape": arr.shape, "affine": affine, "header": header}

def save_mask_like_reference(mask_dhw: np.ndarray, reference_path, output_path):
    """
    将模型输出 mask（D,H,W）resize 回参考影像原始空间并保存，
    affine/header 完全沿用 reference。
    """
    ref = nib.load(str(reference_path))
    target_xyz = ref.shape[:3]

    x = torch.from_numpy(mask_dhw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # 当前输入 mask 是 D,H,W；目标需要 Z,Y,X 后再转回 X,Y,Z
    x = F.interpolate(
        x, size=(target_xyz[2], target_xyz[1], target_xyz[0]),
        mode="nearest"
    )
    arr_zyx = x.squeeze().cpu().numpy()
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    arr_xyz = (arr_xyz > 0.5).astype(np.uint8)

    out = nib.Nifti1Image(arr_xyz, ref.affine, header=ref.header)
    out.set_data_dtype(np.uint8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output_path))


def check_nifti(path):
    """Return True only when a NIfTI file exists and can be read safely."""
    try:
        path = Path(path)
        if not path.is_file():
            return False
        img = nib.load(str(path))
        _ = img.shape
        _ = img.affine
        # Force data access so truncated/corrupt compressed files are caught early.
        _ = img.get_fdata(dtype=np.float32)
        return True
    except Exception:
        return False

def safe_load_preprocessed_volume(path, target_shape, clip_percentiles=(0.5, 99.5)):
    """Safe loader: returns (tensor, meta) or (None, None) on a broken file."""
    try:
        return load_preprocessed_volume(path, target_shape, clip_percentiles)
    except Exception as e:
        print(f"[NIFTI ERROR][SKIP] {path} | {type(e).__name__}: {e}")
        return None, None
