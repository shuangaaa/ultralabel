from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class EncoderConfig:
    name: str = "dino_vitb8"
    patch_size: int = 8
    frozen: bool = True
    checkpoint: str | None = None


@dataclass(frozen=True)
class UpsamplingConfig:
    name: str = "upsample_anything"
    iterations: int = 50
    checkpoint: str | None = None


@dataclass(frozen=True)
class TrackingConfig:
    local_search_radius: int = 128
    soft_argmax_radius: int = 35
    beta: float = 1.0
    refiner_checkpoint: str | None = None


@dataclass(frozen=True)
class SamplingConfig:
    radius_fraction: float = 0.1
    include_boundary: bool = True
    include_interior: bool = True


@dataclass(frozen=True)
class ReconstructionConfig:
    kde_bandwidth: float = 20.0
    kde_threshold: float = 0.005
    cpd_transform: str = "similarity"
    cpd_max_iterations: int = 8
    cpd_tolerance: float = 1e-4


@dataclass(frozen=True)
class StudentConfig:
    name: str = "cotracker3"
    steps: int = 50000
    batch_size: int = 4
    optimizer: str = "AdamW"
    learning_rate: float = 5e-5
    scheduler: str = "cosine"
    loss: str = "huber"
    huber_delta: float = 1.0


@dataclass(frozen=True)
class QualityConfig:
    n_estimators: int = 400
    max_depth: int = 4
    min_samples_leaf: int = 10
    class_weight: str = "balanced_subsample"
    random_state: int = 42


SECTIONS = dict(encoder=EncoderConfig, upsampling=UpsamplingConfig,
                tracking=TrackingConfig, sampling=SamplingConfig,
                reconstruction=ReconstructionConfig, student=StudentConfig,
                quality_control=QualityConfig)


@dataclass(frozen=True)
class UltraLabelConfig:
    device: str = "cuda:0"
    query_frame: int = 0
    image_size: tuple[int, int] = (460, 460)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    upsampling: UpsamplingConfig = field(default_factory=UpsamplingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    quality_control: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self):
        if len(self.image_size) != 2 or min(self.image_size) < 2:
            raise ValueError("image_size must contain height and width of at least 2")
        if self.query_frame < 0:
            raise ValueError("query_frame must be nonnegative")
        for name in SECTIONS:
            for key, value in asdict(getattr(self, name)).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(value) or (value <= 0 and key != "random_state"):
                        raise ValueError(f"{name}.{key} must be positive and finite")
        if self.tracking.soft_argmax_radius > self.tracking.local_search_radius:
            raise ValueError("soft_argmax_radius exceeds local_search_radius")
        if not self.sampling.include_boundary and not self.sampling.include_interior:
            raise ValueError("enable boundary or interior sampling")
        if self.reconstruction.cpd_transform != "similarity":
            raise ValueError("cpd_transform must be similarity")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> UltraLabelConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    for name, cls in SECTIONS.items():
        if name in raw:
            raw[name] = cls(**raw[name])
    if "image_size" in raw:
        raw["image_size"] = tuple(raw["image_size"])
    return UltraLabelConfig(**raw)
