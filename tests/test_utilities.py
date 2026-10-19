import numpy as np
import pytest
from PIL import Image
from ultralabel.data.frames import resize_points, frame_paths
from ultralabel.data.sampling import sample_contour, sample_interior, sampling_interval
from ultralabel.models.tracker.matching import cosine_similarity_grid, peak_soft_argmax
from ultralabel.evaluation.metrics import segmentation_metrics, endpoint_error
from ultralabel.quality.descriptors import frame_descriptors, temporal_stability


def test_coordinate_round_trip():
    points = np.array([[0., 0.], [99., 199.], [31.5, 89.5]])
    resized = resize_points(points, (200, 100), (460, 460))
    np.testing.assert_allclose(resize_points(resized, (460, 460), (200, 100)), points)


def test_frame_order_is_numeric(tmp_path):
    for name in ["frame10.png", "frame2.png", "frame1.png"]:
        Image.new("RGB", (10, 10)).save(tmp_path / name)
    assert [p.name for p in frame_paths(tmp_path)] == ["frame1.png", "frame2.png", "frame10.png"]


def test_closed_contour_sampling_and_interior():
    contour = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    sampled = sample_contour(contour, 2)
    np.testing.assert_allclose(sampled, [[0,0],[2,0],[4,0],[4,2],[4,4],[2,4],[0,4],[0,2]])
    mask = np.zeros((10, 10), bool); mask[2:8, 2:8] = True
    points = sample_interior(mask, sampling_interval([mask.sum()]))
    rounded = np.rint(points).astype(int)
    assert mask[rounded[:,1], rounded[:,0]].all()


def test_similarity_and_soft_argmax_at_border():
    grid = np.array([[[1.,0.], [0.,1.]], [[-1.,0.], [1.,1.]]])
    sim = cosine_similarity_grid([1.,0.], grid)
    np.testing.assert_allclose(sim, [[1,0],[-1,2**-0.5]])
    values = np.zeros((6,6)); values[0,0] = 1000
    np.testing.assert_allclose(peak_soft_argmax(values, radius=2), [0,0], atol=1e-8)


def test_metrics_empty_and_mismatched_masks():
    empty = np.zeros((4,4), bool)
    assert segmentation_metrics(empty, empty) == {"dice": 1., "iou": 1.}
    assert endpoint_error([[1,1]], [[4,5]]) == 5
    with pytest.raises(ValueError):segmentation_metrics(empty, empty[:2])


def test_quality_descriptors_known_depth_slope():
    frame = np.repeat(np.arange(8.,0.,-1)[:,None], 8, axis=1)
    roi = np.ones_like(frame, bool)
    descriptors = frame_descriptors(frame, roi)
    assert descriptors[0] == pytest.approx(1.)
    assert temporal_stability(frame, frame, roi) == pytest.approx(1.)
