from pathlib import Path
import imageio.v3 as iio
import numpy as np


def load_video(path: Path):
    frames = iio.imread(path, plugin="FFMPEG")
    metadata = iio.immeta(path, plugin="FFMPEG")
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError("Expected RGB video frames")
    return frames[..., :3].astype(np.uint8), float(metadata.get("fps", 5.0))


def save_mp4(frames, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    iio.imwrite(path, frames, plugin="FFMPEG", fps=fps, codec="libx264",
                macro_block_size=1, pixelformat="yuv444p" if any(v % 2 for v in frames.shape[1:3]) else "yuv420p")
