from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ultralabel.config import UltraLabelConfig


@dataclass
class TrackingResult:
    masks: np.ndarray
    trajectories: np.ndarray | None = None


class TrackingBackend(ABC):
    @abstractmethod
    def track(self, session, request, config: UltraLabelConfig, job) -> TrackingResult:
        pass

    @abstractmethod
    def refine_frame(self, session, frame_index, edits, config: UltraLabelConfig) -> np.ndarray:
        pass
