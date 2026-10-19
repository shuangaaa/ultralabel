import gzip
import io
import json
import time
import zipfile
import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from pycocotools.mask import decode
from demo.backend.app.main import create_app


def nifti_bytes(data):
    return gzip.compress(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)).to_bytes())


@pytest.fixture
def uploaded(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    video = np.arange(16 * 20 * 3).reshape(16, 20, 3)
    reference = np.zeros((16, 20), np.float32)
    reference[4:12, 5:15] = 1
    response = client.post("/api/session/create", files={
        "sequence": ("clip.nii.gz", nifti_bytes(video)),
        "gt": ("reference.nii.gz", nifti_bytes(reference)),
    })
    assert response.status_code == 201, response.text
    return app, client, response.json(), reference


def test_upload_preview_and_query_point_persistence(uploaded):
    _, client, session, _ = uploaded
    sid = session["session_id"]
    assert (session["num_frames"], session["orig_width"], session["orig_height"]) == (3, 20, 16)
    assert client.get(session["video_url"]).headers["content-type"] == "video/mp4"
    frame = client.get(f"/api/session/{sid}/frames/1/image")
    assert Image.open(io.BytesIO(frame.content)).size == (20, 16)
    assert client.get(f"/api/session/{sid}/query_points").json()["gt_pos"]
    point = {"x": .25, "y": .75, "label": 1}
    assert client.post(f"/api/session/{sid}/query_points/add", json=point).status_code == 200
    assert client.get(f"/api/session/{sid}/query_points").json()["user_pos"] == [{"x":.25,"y":.75}]
    assert client.post(f"/api/session/{sid}/query_points/add", json={**point,"x":1.1}).status_code == 422
    client.post(f"/api/session/{sid}/query_points/reset")
    points = client.get(f"/api/session/{sid}/query_points").json()
    assert points["user_pos"] == [] and points["gt_pos"]


def test_reference_mask_is_identified_as_reference(uploaded):
    _, client, session, reference = uploaded
    sid = session["session_id"]
    body = client.get(f"/api/session/{sid}/frames/0/mask-rle").json()
    assert body["source"] == "reference"
    rle = body["rle"]; rle["counts"] = rle["counts"].encode()
    np.testing.assert_array_equal(decode(rle), reference)
    assert client.get(f"/api/session/{sid}/frames/1/mask-rle").json()["rle"] is None
    assert not client.get(f"/api/session/{sid}/meta").json()["has_predictions"]


def test_frame_edits_and_export_work_without_inference(uploaded):
    _, client, session, _ = uploaded
    sid = session["session_id"]
    edits = {"adds":[{"x":.3,"y":.4}],"removes":[{"x":.6,"y":.7}]}
    for _ in range(2):
        assert client.post(f"/api/session/{sid}/frames/1/edits", json=edits).status_code == 200
    stored = client.get(f"/api/session/{sid}/frames/1/edits").json()
    assert len(stored["adds"]) == len(stored["removes"]) == 1
    response = client.get(f"/api/session/{sid}/download.zip")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert {"meta.json","annotations.json","reference_annotations.npz"} == set(archive.namelist())
        assert json.loads(archive.read("annotations.json"))["frame_edits"]["1"] == stored
        assert not json.loads(archive.read("meta.json"))["has_predictions"]


def test_tracking_and_apply_require_a_backend(uploaded):
    _, client, session, _ = uploaded
    sid = session["session_id"]
    assert client.post(f"/api/session/{sid}/track", json={}).status_code == 409
    assert client.get(f"/api/session/{sid}/track/status").json() == {"status":"idle"}
    assert client.post(f"/api/session/{sid}/frames/0/edit", json={"op":"apply"}).status_code == 409
    assert not client.get(f"/api/session/{sid}/meta").json()["has_predictions"]


def test_invalid_uploads_and_unknown_sessions(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.post("/api/session/create", files={"sequence":("bad.nii.gz",b"invalid")})
    assert response.status_code == 422
    assert list((tmp_path / "sessions").iterdir()) == []
    assert client.get("/api/session/unknown").status_code == 404


def test_failed_backend_does_not_produce_prediction_outputs(uploaded):
    app, client, session, _ = uploaded
    class FailureBackend:
        def track(self, *args):
            raise RuntimeError("test backend error")
    app.state.inference_backend = FailureBackend()
    sid = session["session_id"]
    assert client.post(f"/api/session/{sid}/track", json={}).status_code == 202
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = client.get(f"/api/session/{sid}/track/status").json()
        if status["status"] == "failed":
            break
        time.sleep(.01)
    assert status["status"] == "failed"
    assert status["message"] == "test backend error"
    assert not client.get(f"/api/session/{sid}/meta").json()["has_predictions"]


def test_regular_video_upload_uses_the_same_session_flow(uploaded):
    _, client, session, _ = uploaded
    video = client.get(session["video_url"]).content
    response = client.post("/api/session/create", files={"sequence":("clip.mp4", video,"video/mp4")})
    assert response.status_code == 201, response.text
    item = response.json()
    assert item["num_frames"] == 3 and not item["has_gt"]
    assert (item["orig_width"], item["orig_height"]) == (20,16)
