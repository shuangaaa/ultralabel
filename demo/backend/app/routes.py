from pathlib import Path
from uuid import uuid4
import io
import json
import shutil
import zipfile
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from ultralabel.data.sampling import sample_interior, sampling_interval
from ..app_conf import settings
from ..core.task_manager import task_manager
from ..io.nifti import load_nifti_as_video, normalize_to_uint8, gray_to_rgb
from ..io.video import load_video, save_mp4
from ..io.rle import mask_to_rle
from ..io.overlay import build_overlay_video
from ..pipeline.runner import run_tracking
from .schemas import QueryPoint, TrackRequest, FrameEditRequest, FrameEdits
from .sessions import SessionState, sessions

router = APIRouter(prefix="/api")
NO_CACHE = {"Cache-Control": "no-store"}


def get_session(session_id):
    try:
        return sessions.get(session_id)
    except KeyError as error:
        raise HTTPException(404, "Session not found.") from error


def get_backend(request):
    backend = request.app.state.inference_backend
    if backend is None:
        raise HTTPException(409, "Tracking backend is not configured.")
    return backend


def check_frame(session, index):
    if not 0 <= index < session.num_frames:
        raise HTTPException(404, "Frame index is outside the sequence.")


async def save_upload(upload, directory, stem, allowed):
    filename = (upload.filename or "").lower()
    suffix = next((ending for ending in allowed if filename.endswith(ending)), None)
    if suffix is None:
        raise HTTPException(422, "Unsupported sequence format.")
    destination = directory / (stem + suffix)
    total = 0
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > 512 * 1024 * 1024:
                raise HTTPException(413, "Upload exceeds 512 MB.")
            stream.write(chunk)
    return destination


@router.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@router.post("/session/create", status_code=201)
async def create_session(request: Request, sequence: UploadFile = File(...),
                         gt: UploadFile | None = File(default=None)):
    session_id = uuid4().hex
    directory = Path(request.app.state.data_dir) / "sessions" / session_id
    directory.mkdir(parents=True)
    try:
        path = await save_upload(sequence, directory, "sequence",
                                 [".nii.gz", ".nii", ".mp4", ".mov", ".avi", ".webm"])
        video_path = directory / "video.mp4"
        if path.name.endswith((".nii", ".nii.gz")):
            frames = gray_to_rgb(normalize_to_uint8(load_nifti_as_video(path)))
            fps = settings.fps
        else:
            frames, fps = load_video(path)
        if len(frames) == 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frames or frame rate")
        reference = None
        gt_points = []
        if gt is not None:
            gt_path = await save_upload(gt, directory, "reference", [".nii.gz", ".nii"])
            reference = load_nifti_as_video(gt_path) > 0
            if reference.shape[1:] != frames.shape[1:3] or len(reference) not in (1, len(frames)):
                raise ValueError("Reference annotation and sequence dimensions differ")
            mask = reference[0]
            if mask.any():
                points = sample_interior(mask, sampling_interval([mask.sum()]))
                gt_points = [{"x": float(x / max(frames.shape[2] - 1, 1)),
                              "y": float(y / max(frames.shape[1] - 1, 1))} for x, y in points]
        save_mp4(frames, video_path, fps)
        session = SessionState(session_id=session_id, filename=sequence.filename or path.name,
                               sequence_path=path, directory=directory, video_path=video_path,
                               frames=frames, fps=fps, reference_masks=reference,
                               gt_points_pos=gt_points)
        sessions.add(session)
        return {**session.metadata(), "video_url": f"/api/session/{session_id}/video"}
    except Exception as error:
        shutil.rmtree(directory, ignore_errors=True)
        if isinstance(error, HTTPException):
            raise
        raise HTTPException(422, f"Could not read sequence: {error}") from error


@router.get("/session/{session_id}")
@router.get("/session/{session_id}/meta")
def read_session(session_id: str):
    session = get_session(session_id)
    with session.lock:
        return {**session.metadata(), "job_id": session.job_id}


@router.get("/session/{session_id}/video")
def video(session_id: str):
    return FileResponse(get_session(session_id).video_path, media_type="video/mp4")


@router.get("/session/{session_id}/frames/{frame_index}/image")
def frame_image(session_id: str, frame_index: int):
    session = get_session(session_id)
    check_frame(session, frame_index)
    output = io.BytesIO()
    Image.fromarray(session.frames[frame_index]).save(output, format="PNG")
    return Response(output.getvalue(), media_type="image/png", headers=NO_CACHE)


@router.get("/session/{session_id}/query_points")
def query_points(session_id: str):
    session = get_session(session_id)
    with session.lock:
        return {"session_id": session_id, **session.annotations()}


@router.post("/session/{session_id}/query_points/add")
def add_query_point(session_id: str, point: QueryPoint):
    session = get_session(session_id)
    with session.lock:
        target = session.query_points_pos if point.label else session.query_points_neg
        target.append({"x": point.x, "y": point.y})
        session.persist()
        return {"ok": True, "num_pos": len(session.query_points_pos),
                "num_neg": len(session.query_points_neg)}


@router.post("/session/{session_id}/query_points/reset")
def reset_query_points(session_id: str):
    session = get_session(session_id)
    with session.lock:
        session.query_points_pos.clear()
        session.query_points_neg.clear()
        session.persist()
    return {"ok": True}


@router.post("/session/{session_id}/track", status_code=202)
def track(session_id: str, payload: TrackRequest, request: Request):
    session = get_session(session_id)
    backend = get_backend(request)
    with session.lock:
        if session.job_id:
            previous = task_manager.get_job(session.job_id)
            if previous and previous.status in ("queued", "running"):
                raise HTTPException(409, "A tracking task is already active.")
        if not session.query_points_pos and not session.gt_points_pos:
            raise HTTPException(422, "Select reference points before tracking.")
        job = task_manager.create_job()
        session.job_id = job.job_id
        session.persist()
        task_manager.start(job, lambda item: run_tracking(backend, session, payload, settings.method, item))
    return {"ok": True, "job_id": job.job_id}


@router.get("/session/{session_id}/track/status")
def tracking_status(session_id: str):
    session = get_session(session_id)
    return task_manager.snapshot(session.job_id) if session.job_id else {"status": "idle"}


@router.get("/session/{session_id}/frames/{frame_index}/mask-rle")
def frame_mask(session_id: str, frame_index: int):
    session = get_session(session_id)
    check_frame(session, frame_index)
    with session.lock:
        mask, source = None, None
        if session.masks is not None:
            mask, source = session.masks[frame_index], "prediction"
        elif session.reference_masks is not None:
            if len(session.reference_masks) > 1:
                mask = session.reference_masks[frame_index]
            elif frame_index == session.query_frame_index:
                mask = session.reference_masks[0]
            if mask is not None:
                source = "reference"
        return JSONResponse({"session_id": session_id, "frame_index": frame_index,
                             "rle": mask_to_rle(mask) if mask is not None else None,
                             "source": source}, headers=NO_CACHE)


@router.post("/session/{session_id}/frames/{frame_index}/edits")
def save_frame_edits(session_id: str, frame_index: int, payload: FrameEdits):
    session = get_session(session_id)
    check_frame(session, frame_index)
    with session.lock:
        session.frame_edits[str(frame_index)] = payload.model_dump()
        session.persist()
    return {"ok": True, "frame_index": frame_index, "saved": True}


@router.get("/session/{session_id}/frames/{frame_index}/edits")
def read_frame_edits(session_id: str, frame_index: int):
    session = get_session(session_id)
    check_frame(session, frame_index)
    with session.lock:
        return session.frame_edits.get(str(frame_index), {"adds": [], "removes": []})


@router.post("/session/{session_id}/frames/{frame_index}/edit")
def edit_frame(session_id: str, frame_index: int, payload: FrameEditRequest, request: Request):
    session = get_session(session_id)
    check_frame(session, frame_index)
    with session.lock:
        edits = session.frame_edits.setdefault(str(frame_index), {"adds": [], "removes": []})
        if payload.op in ("add", "remove"):
            edits["adds" if payload.op == "add" else "removes"].append(
                {"x": payload.x, "y": payload.y, "label": 1 if payload.op == "add" else 0})
            session.persist()
        if not payload.apply and payload.op != "apply":
            return {"ok": True, "applied": False, "saved": True}
        backend = get_backend(request)
        if session.masks is None:
            raise HTTPException(409, "No predicted annotation is available for refinement.")
        config = {**edits, "use_cpd": payload.use_cpd, "max_remove_dist": payload.max_remove_dist}
        mask = np.asarray(backend.refine_frame(session, frame_index, config, settings.method))
        if mask.shape != (session.height, session.width) or not np.isin(mask, [0, 1]).all():
            raise HTTPException(502, "Refinement backend returned an invalid mask.")
        masks = session.masks.copy()
        masks[frame_index] = mask
        save_mp4(build_overlay_video(session.frames, masks), session.directory / "overlay.mp4", session.fps)
        session.masks = masks
        np.savez_compressed(session.directory / "predictions.npz", masks=masks,
                            **({"trajectories": session.trajectories} if session.trajectories is not None else {}))
        session.frame_edits.pop(str(frame_index), None)
        session.persist()
        return {"ok": True, "applied": True, "rle": mask_to_rle(mask)}


@router.get("/session/{session_id}/overlay")
def overlay(session_id: str):
    session = get_session(session_id)
    path = session.directory / "overlay.mp4"
    if session.masks is None or not path.exists():
        raise HTTPException(404, "No overlay is available.")
    return FileResponse(path, media_type="video/mp4", headers=NO_CACHE)


@router.get("/session/{session_id}/download.zip")
def download(session_id: str):
    session = get_session(session_id)
    output = io.BytesIO()
    with session.lock, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("meta.json", json.dumps(session.metadata(), indent=2))
        archive.writestr("annotations.json", json.dumps(session.annotations(), indent=2))
        for filename in ["predictions.npz", "overlay.mp4"]:
            path = session.directory / filename
            if session.masks is not None and path.exists():
                archive.write(path, filename)
        if session.reference_masks is not None:
            reference = io.BytesIO()
            np.savez_compressed(reference, reference_masks=session.reference_masks)
            archive.writestr("reference_annotations.npz", reference.getvalue())
    return Response(output.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="session_{session_id}.zip"'})
