from abc import ABC, abstractmethod
import numpy as np


class SimilarityRefiner(ABC):
    @abstractmethod
    def refine_similarity(self, similarity: np.ndarray) -> np.ndarray:
        pass


class PointTracker(ABC):
    @abstractmethod
    def initialize_queries(self, low_features, high_features, points):
        pass

    @abstractmethod
    def track_points(self, low_features, high_features) -> np.ndarray:
        """Return N x 2 point coordinates in the resized image space."""
