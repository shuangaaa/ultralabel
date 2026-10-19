from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import json
import numpy as np


@dataclass
class SessionState:
    session_id: str
    filename: str
    sequence_path: Path
    directory: Path
    video_path: Path
    frames: np.ndarray
    fps: float
    query_frame_index: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reference_masks: np.ndarray | None = None
    query_points_pos: list = field(default_factory=list)
    query_points_neg: list = field(default_factory=list)
    gt_points_pos: list = field(default_factory=list)
    frame_edits: dict = field(default_factory=dict)
    job_id: str | None = None
    masks: np.ndarray | None = None
    trajectories: np.ndarray | None = None
    lock: RLock = field(default_factory=RLock)

    @property
    def num_frames(self):
        return len(self.frames)

    @property
    def height(self):
        return self.frames.shape[1]

    @property
    def width(self):
        return self.frames.shape[2]

    def metadata(self):
        return {"session_id": self.session_id, "filename": self.filename,
                "created_at": self.created_at, "num_frames": self.num_frames,
                "orig_width": self.width, "orig_height": self.height, "fps": self.fps,
                "query_frame_index": self.query_frame_index,
                "has_gt": self.reference_masks is not None,
                "has_predictions": self.masks is not None}

    def annotations(self):
        return {"coordinate_system": "normalized_xy", "query_frame_index": self.query_frame_index,
                "gt_pos": self.gt_points_pos, "user_pos": self.query_points_pos,
                "user_neg": self.query_points_neg, "frame_edits": self.frame_edits}

    def persist(self):
        with self.lock:
            target = self.directory / "session.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps({**self.metadata(), **self.annotations()}, indent=2))
            temporary.replace(target)


class SessionStore:
    def __init__(self):
        self._items = {}
        self._lock = RLock()

    def add(self, session):
        with self._lock:
            self._items[session.session_id] = session
        session.persist()
        return session

    def get(self, session_id):
        with self._lock:
            return self._items[session_id]


sessions = SessionStore()
