import numpy as np


def sampling_interval(reference_areas, radius_fraction=0.1):
    areas = np.asarray(reference_areas, dtype=float)
    if areas.size == 0 or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("reference areas must be finite and positive")
    if radius_fraction <= 0:
        raise ValueError("radius_fraction must be positive")
    return float(radius_fraction * np.sqrt(np.median(areas) / np.pi))


def sample_interior(mask, interval):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or interval <= 0:
        raise ValueError("expected a 2D mask and positive interval")
    y, x = np.meshgrid(np.arange(0, mask.shape[0], max(interval, 1.0)),
                       np.arange(0, mask.shape[1], max(interval, 1.0)), indexing="ij")
    xy = np.stack([x.ravel(), y.ravel()], axis=-1)
    ij = np.rint(xy).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, mask.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, mask.shape[0] - 1)
    return xy[mask[ij[:, 1], ij[:, 0]]]


def sample_contour(contour, interval, closed=True):
    points = np.asarray(contour, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or interval <= 0:
        raise ValueError("expected an N x 2 contour and positive interval")
    if closed and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, lengths > 1e-10]]
    if len(points) < 2:
        raise ValueError("contour has zero length")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    positions = np.arange(0, arc[-1], interval)
    return np.column_stack([np.interp(positions, arc, points[:, i]) for i in range(2)])
