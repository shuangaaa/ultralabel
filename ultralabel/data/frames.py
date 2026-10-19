from pathlib import Path
import re
import numpy as np
from PIL import Image


def frame_paths(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"frame directory does not exist: {directory}")
    def key(path):
        return [(0, int(v)) if v.isdigit() else (1, v.lower())
                for v in re.split(r"(\d+)", path.name)]
    paths = sorted((p for p in directory.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}), key=key)
    if not paths:
        raise ValueError(f"no image frames found: {directory}")
    return paths


def load_frame(path: str | Path, image_size=(460, 460)) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def resize_points(points, source_size, target_size):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite N x 2 array")
    sh, sw = source_size
    th, tw = target_size
    if min(sh, sw, th, tw) < 2:
        raise ValueError("image dimensions must be at least 2")
    return points * np.array([(tw - 1) / (sw - 1), (th - 1) / (sh - 1)])
