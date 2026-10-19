import numpy as np


def cosine_similarity_grid(query, features, eps=1e-8):
    query = np.asarray(query, dtype=float)
    features = np.asarray(features, dtype=float)
    if features.ndim != 3 or query.shape != (features.shape[-1],):
        raise ValueError("expected a C-vector and H x W x C features")
    denominator = np.linalg.norm(features, axis=-1) * np.linalg.norm(query)
    return (features @ query) / np.maximum(denominator, eps)


def local_window(center, radius, shape):
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    x, y = np.rint(center).astype(int)
    h, w = shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("center lies outside the image")
    return (slice(max(0, y - radius), min(h, y + radius + 1)),
            slice(max(0, x - radius), min(w, x + radius + 1)))


def peak_soft_argmax(similarity, radius=35, beta=1.0):
    values = np.asarray(similarity, dtype=float)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all() or beta <= 0:
        raise ValueError("expected a finite 2D similarity map and positive beta")
    y, x = np.unravel_index(np.argmax(values), values.shape)
    ys, xs = local_window((x, y), radius, values.shape)
    logits = beta * values[ys, xs]
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    yy, xx = np.mgrid[ys, xs]
    return np.array([(weights * xx).sum(), (weights * yy).sum()])
