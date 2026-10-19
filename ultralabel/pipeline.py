from dataclasses import dataclass
import numpy as np
from .config import UltraLabelConfig
from .models.extractor.base import FrozenFeatureExtractor, ImageGuidedUpsampler
from .models.tracker.base import PointTracker
from .reconstruction.base import PointToContour


@dataclass
class AnnotationResult:
    trajectories: np.ndarray
    masks: np.ndarray


class AnnotationPipeline:
    def __init__(self, extractor: FrozenFeatureExtractor, upsampler: ImageGuidedUpsampler,
                 tracker: PointTracker, reconstruction: PointToContour,
                 config: UltraLabelConfig | None = None):
        self.extractor = extractor
        self.upsampler = upsampler
        self.tracker = tracker
        self.reconstruction = reconstruction
        self.config = config or UltraLabelConfig()

    def _features(self, frame):
        low = self.extractor.extract_features(frame)
        guidance = self.upsampler.optimize_guidance(frame, low, self.config.upsampling.iterations)
        high = self.upsampler.upsample_features(low, guidance, self.config.image_size)
        return low, high

    def propagate(self, frames, reference_points, boundary_indices):
        frames = np.asarray(frames)
        points = np.asarray(reference_points, dtype=float)
        boundary_indices = np.asarray(boundary_indices, dtype=int)
        if frames.ndim != 4 or frames.shape[1:3] != self.config.image_size:
            raise ValueError("frames must have shape T x H x W x C matching image_size")
        if not 0 <= self.config.query_frame < len(frames):
            raise ValueError("query_frame is outside the sequence")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("reference_points must be a nonempty N x 2 array")
        if boundary_indices.ndim != 1 or boundary_indices.size == 0 or np.any(boundary_indices < 0) or np.any(boundary_indices >= len(points)):
            raise ValueError("boundary_indices must select reference points")
        reference = self._features(frames[self.config.query_frame])
        self.tracker.initialize_queries(*reference, points)
        trajectories, masks = [], []
        cfg = self.config.reconstruction
        for index, frame in enumerate(frames):
            predicted = points.copy() if index == self.config.query_frame else self.tracker.track_points(*self._features(frame))
            predicted = np.asarray(predicted)
            if predicted.shape != points.shape or not np.isfinite(predicted).all():
                raise ValueError("tracker returned invalid point coordinates")
            density = self.reconstruction.points_to_density(predicted, self.config.image_size, cfg.kde_bandwidth)
            contour = self.reconstruction.extract_contour(density, cfg.kde_threshold)
            contour = self.reconstruction.refine_contour(contour, predicted[boundary_indices], cfg.cpd_max_iterations, cfg.cpd_tolerance)
            mask = np.asarray(self.reconstruction.rasterize_contour(contour, self.config.image_size), dtype=bool)
            if mask.shape != self.config.image_size:
                raise ValueError("reconstruction returned an invalid mask shape")
            trajectories.append(predicted)
            masks.append(mask)
        return AnnotationResult(np.stack(trajectories), np.stack(masks))
