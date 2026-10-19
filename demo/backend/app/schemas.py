from pydantic import BaseModel, Field
from typing import Literal


class QueryPoint(BaseModel):
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)
    label: Literal[0, 1] = 1


class TrackRequest(BaseModel):
    use_cpd: bool = True
    foreground_class: int = Field(default=1, ge=1)


class FrameEditRequest(BaseModel):
    op: Literal["add", "remove", "apply", "noop"]
    x: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    y: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    use_cpd: bool = True
    max_remove_dist: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    apply: bool = False


class FrameEdits(BaseModel):
    adds: list[QueryPoint] = Field(default_factory=list)
    removes: list[QueryPoint] = Field(default_factory=list)
