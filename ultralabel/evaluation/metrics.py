import numpy as np


def segmentation_metrics(prediction, reference):
    prediction, reference = np.asarray(prediction, bool), np.asarray(reference, bool)
    if prediction.shape != reference.shape or prediction.size == 0:
        raise ValueError("masks must have the same nonempty shape")
    intersection = np.logical_and(prediction, reference).sum()
    union = np.logical_or(prediction, reference).sum()
    total = prediction.sum() + reference.sum()
    return {"dice": float(2 * intersection / total) if total else 1.0,
            "iou": float(intersection / union) if union else 1.0}


def endpoint_error(prediction, reference, valid=None):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.shape[-1] != 2:
        raise ValueError("point arrays must share a shape ending in 2")
    error = np.linalg.norm(prediction - reference, axis=-1)
    if valid is not None:
        error = error[np.asarray(valid, dtype=bool)]
    if error.size == 0 or not np.isfinite(error).all():
        raise ValueError("no finite point errors to aggregate")
    return float(error.mean())
