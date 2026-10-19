from abc import ABC, abstractmethod
import numpy as np


class PointToContour(ABC):
    @abstractmethod
    def points_to_density(self, points, image_size, bandwidth) -> np.ndarray:
        pass

    @abstractmethod
    def extract_contour(self, density, threshold) -> np.ndarray:
        pass

    @abstractmethod
    def refine_contour(self, contour, boundary_anchors, max_iterations, tolerance) -> np.ndarray:
        pass

    @abstractmethod
    def rasterize_contour(self, contour, image_size) -> np.ndarray:
        pass
