import numpy as np


def frame_descriptors(frame, roi, bins=32):
    image = np.asarray(frame, dtype=float)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    roi = np.asarray(roi, dtype=bool)
    if image.shape != roi.shape or not roi.any() or not np.isfinite(image).all():
        raise ValueError("expected a finite image and nonempty ROI with the same shape")
    rows = np.flatnonzero(roi.any(axis=1))
    means = np.array([image[y, roi[y]].mean() for y in rows])
    attenuation = -float(np.polyfit(rows, means, 1)[0]) if len(rows) > 1 else 0.0
    values = image[roi]
    hist, _ = np.histogram(values, bins=bins)
    probability = hist[hist > 0] / hist.sum()
    entropy = float(-(probability * np.log2(probability)).sum())
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    return np.array([attenuation, entropy, iqr])


def temporal_stability(previous, current, roi):
    a, b = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    if a.ndim == 3:
        a, b = a.mean(axis=-1), b.mean(axis=-1)
    roi = np.asarray(roi, dtype=bool)
    if a.shape != b.shape or a.shape != roi.shape or not roi.any():
        raise ValueError("frames and ROI must share a shape and the ROI must be nonempty")
    a, b = a[roi], b[roi]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("frames must be finite")
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else 0.0


def build_qc_classifier(config):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from dataclasses import asdict
    return make_pipeline(StandardScaler(), RandomForestClassifier(**asdict(config)))
