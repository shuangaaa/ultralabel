from pathlib import Path
import pytest
from ultralabel.config import load_config, UltraLabelConfig, TrackingConfig


def test_paper_config_has_typed_sections():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    assert config.image_size == (460, 460)
    assert config.student.learning_rate == 5e-5
    assert config.reconstruction.cpd_max_iterations == 8
    assert config.quality_control.class_weight == "balanced_subsample"


def test_invalid_search_window_is_rejected():
    with pytest.raises(ValueError, match="soft_argmax_radius"):
        UltraLabelConfig(tracking=TrackingConfig(local_search_radius=10))
