import numpy as np
from ..io.video import save_mp4
from ..io.overlay import build_overlay_video


def run_tracking(backend, session, request, config, job):
    result = backend.track(session, request, config, job)
    masks = np.asarray(result.masks)
    if masks.shape != session.frames.shape[:3] or not np.isfinite(masks).all():
        raise ValueError("Tracking backend returned invalid masks")
    if not np.isin(masks, [0, 1]).all():
        raise ValueError("Tracking masks must be binary")
    trajectories = result.trajectories
    if trajectories is not None:
        trajectories = np.asarray(trajectories)
        if trajectories.ndim != 3 or trajectories.shape[0] != session.num_frames or trajectories.shape[-1] != 2 or not np.isfinite(trajectories).all():
            raise ValueError("Tracking backend returned invalid trajectories")
    overlay = session.directory / "overlay.mp4"
    save_mp4(build_overlay_video(session.frames, masks), overlay, session.fps)
    with session.lock:
        session.masks = masks.astype(bool)
        session.trajectories = trajectories
        np.savez_compressed(session.directory / "predictions.npz", masks=session.masks,
                            **({"trajectories": trajectories} if trajectories is not None else {}))
        session.persist()
    job.outputs = {"meta": f"/api/session/{session.session_id}/meta",
                   "overlay": f"/api/session/{session.session_id}/overlay",
                   "download": f"/api/session/{session.session_id}/download.zip"}
