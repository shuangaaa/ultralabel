import numpy as np


def build_overlay_video(frames, masks, color=(255, 0, 0), alpha=0.35):
    frames = np.asarray(frames, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)[..., None]
    if frames.shape[:-1] != masks.shape[:-1]:
        raise ValueError("Mask and video dimensions differ")
    result = frames * (1 - alpha * masks) + np.asarray(color) * alpha * masks
    return np.clip(result, 0, 255).astype(np.uint8)
