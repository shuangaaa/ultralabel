from pathlib import Path
import nibabel as nib
import numpy as np


def load_nifti_as_video(path: Path):
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    elif data.ndim == 4:
        data = np.squeeze(data)
        if data.ndim == 2:
            data = data[..., None]
    if data.ndim != 3 or min(data.shape[:2]) < 2:
        raise ValueError(f"Expected an H x W x T ultrasound sequence, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Sequence contains nonfinite intensities")
    return np.transpose(data, (2, 0, 1))


def normalize_to_uint8(video, pmin=1.0, pmax=99.0):
    video = np.asarray(video, dtype=np.float32)
    lo, hi = np.percentile(video, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(video, dtype=np.uint8)
    return (np.clip((video - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def gray_to_rgb(video):
    return np.repeat(np.asarray(video)[..., None], 3, axis=-1)
