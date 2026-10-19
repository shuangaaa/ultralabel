from abc import ABC, abstractmethod
import numpy as np


class FrozenFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Return an Hf x Wf x C descriptor grid."""


class ImageGuidedUpsampler(ABC):
    @abstractmethod
    def optimize_guidance(self, frame: np.ndarray, features: np.ndarray, iterations: int):
        pass

    @abstractmethod
    def upsample_features(self, features: np.ndarray, guidance, output_size) -> np.ndarray:
        pass
