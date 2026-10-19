import numpy as np
from ultralabel.config import UltraLabelConfig
from ultralabel.pipeline import AnnotationPipeline


class FeatureExtractor:
    def extract_features(self, frame):
        return frame


class Upsampler:
    def __init__(self):
        self.iterations = []

    def optimize_guidance(self, frame, features, iterations):
        self.iterations.append(iterations)
        return None

    def upsample_features(self, features, guidance, output_size):
        return features


class Tracker:
    def initialize_queries(self, low_features, high_features, points):
        self.reference_value = float(low_features[0, 0, 0])
        self.points = points

    def track_points(self, low_features, high_features):
        return self.points + float(low_features[0, 0, 0]) - self.reference_value


class Reconstruction:
    def points_to_density(self, points, image_size, bandwidth):
        return np.zeros(image_size)

    def extract_contour(self, density, threshold):
        return np.array([[0, 0], [1, 0], [1, 1]])

    def refine_contour(self, contour, boundary_anchors, max_iterations, tolerance):
        return contour

    def rasterize_contour(self, contour, image_size):
        return np.zeros(image_size, bool)


def test_pipeline_uses_selected_reference_and_keeps_temporal_order():
    frames = np.stack([np.full((8, 8, 3), i, dtype=float) for i in range(3)])
    points = np.array([[3., 3.], [5., 5.]])
    upsampler = Upsampler()
    pipeline = AnnotationPipeline(FeatureExtractor(), upsampler, Tracker(), Reconstruction(),
                                 UltraLabelConfig(query_frame=1, image_size=(8, 8)))
    result = pipeline.propagate(frames, points, [0, 1])
    np.testing.assert_allclose(result.trajectories, np.stack([points-1, points, points+1]))
    assert result.masks.shape == (3, 8, 8)
    assert upsampler.iterations == [50, 50, 50]
