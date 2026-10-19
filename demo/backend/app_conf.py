from dataclasses import dataclass, field
from pathlib import Path
import os
from ultralabel.config import UltraLabelConfig


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "ULTRALABEL_DEMO_DATA", str(Path(__file__).resolve().parents[2] / ".demo_data"))))
    fps: int = 5
    method: UltraLabelConfig = field(default_factory=UltraLabelConfig)

    @property
    def uploads_dir(self):
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self):
        return self.data_dir / "outputs"


settings = Settings()
