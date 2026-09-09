"""Unit tests for utils/pose_roi.py.

These tests only need numpy/opencv/scikit-image (no MediaPipe), and simulate detectors
with plain Python callables, matching the injection points used by the real MediaPipe
adapters in utils/mediapipe_utils.py.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_roi import (  # noqa: E402
    crop_roi,
    extract_pose_roi_landmarks,
    face_roi_from_pose,
    interpolate_boxes,
    interpolate_landmarks,
    landmarks_crop_to_frame,
)


def make_pose_landmarks(face_points=None, nose=None, left_shoulder=None, right_shoulder=None, visible_idx=()):
    """Build a (33,2) xy array and (33,) visibility array for a fake Pose detection."""
    xy = np.zeros((33, 2), dtype=np.float64)
    visibility = np.zeros((33,), dtype=np.float64)
    if face_points:
        for idx, (x, y) in face_points.items():
            xy[idx] = [x, y]
            visibility[idx] = 1.0
    if nose is not None:
        xy[0] = nose
    if left_shoulder is not None:
        xy[11] = left_shoulder
    if right_shoulder is not None:
        xy[12] = right_shoulder
    for idx in visible_idx:
        visibility[idx] = 1.0
    return xy, visibility


class TestFaceRoiFromPose(unittest.TestCase):
    def test_valid_roi_from_face_landmarks(self):
        # 5 face points (>= min_face_points=4) spanning a 100x60 box centered at (200, 150).
        face_points = {
            0: (200, 120),   # nose, top
            2: (170, 150),   # left eye
            5: (230, 150),   # right eye
            9: (180, 180),   # mouth left
            10: (220, 180),  # mouth right
        }
        xy, visibility = make_pose_landmarks(face_points=face_points)

        box = face_roi_from_pose(xy, visibility, visibility_threshold=0.35)

        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        side = x2 - x1
        self.assertAlmostEqual(y2 - y1, side, places=3)  # square

        span_x = 230 - 170
        span_y = 180 - 120
        expected_side = 2.5 * max(span_x, span_y)
        self.assertAlmostEqual(side, expected_side, places=3)

        expected_cx = (170 + 230) / 2.0
        expected_cy = (120 + 180) / 2.0 - 0.15 * expected_side
        self.assertAlmostEqual((x1 + x2) / 2.0, expected_cx, places=3)
        self.assertAlmostEqual((y1 + y2) / 2.0, expected_cy, places=3)

    def test_fallback_to_nose_and_shoulders(self):
        # Only nose + shoulders visible, face points below threshold -> fallback path.
        xy, visibility = make_pose_landmarks(
            nose=(300, 100), left_shoulder=(250, 250), right_shoulder=(350, 250),
            visible_idx=(0, 11, 12),
        )

        box = face_roi_from_pose(xy, visibility, visibility_threshold=0.35)

        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        side = x2 - x1
        expected_side = 0.9 * 100.0  # shoulder distance
        self.assertAlmostEqual(side, expected_side, places=3)
        self.assertAlmostEqual((x1 + x2) / 2.0, 300.0, places=3)
        expected_cy = 100.0 - 0.05 * expected_side
        self.assertAlmostEqual((y1 + y2) / 2.0, expected_cy, places=3)

    def test_no_roi_when_nothing_visible(self):
        xy, visibility = make_pose_landmarks()
        box = face_roi_from_pose(xy, visibility, visibility_threshold=0.35)
        self.assertIsNone(box)

    def test_insufficient_face_points_and_missing_shoulder_gives_none(self):
        # Only 2 face points and one shoulder visible: neither strategy applies.
        xy, visibility = make_pose_landmarks(
            face_points={0: (200, 120), 2: (170, 150)},
            left_shoulder=(250, 250),
            visible_idx=(0, 2, 11),
        )
        box = face_roi_from_pose(xy, visibility, visibility_threshold=0.35)
        self.assertIsNone(box)


class TestInterpolateBoxes(unittest.TestCase):
    def test_interpolates_gap_and_extrapolates_edges(self):
        boxes = np.zeros((5, 4), dtype=np.float32)
        valid = np.array([False, True, False, True, False])
        boxes[1] = [0, 0, 10, 10]
        boxes[3] = [0, 0, 30, 30]

        filled = interpolate_boxes(boxes, valid)

        np.testing.assert_allclose(filled[1], [0, 0, 10, 10])
        np.testing.assert_allclose(filled[3], [0, 0, 30, 30])
        np.testing.assert_allclose(filled[2], [0, 0, 20, 20])  # midpoint interpolation
        np.testing.assert_allclose(filled[0], [0, 0, 10, 10])  # nearest valid (before first)
        np.testing.assert_allclose(filled[4], [0, 0, 30, 30])  # nearest valid (after last)

    def test_raises_when_nothing_valid(self):
        boxes = np.zeros((3, 4), dtype=np.float32)
        valid = np.zeros(3, dtype=bool)
        with self.assertRaises(ValueError):
            interpolate_boxes(boxes, valid)


class TestCropRoundTrip(unittest.TestCase):
    def test_crop_and_map_back_recovers_original_point(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = np.array([100.0, 50.0, 300.0, 250.0], dtype=np.float32)  # 200x200 square

        crop, tform = crop_roi(frame, box, out_size=512)
        self.assertEqual(crop.shape, (512, 512, 3))

        original_point = np.array([[150.0, 120.0, -0.01]])  # x, y, z
        crop_xy = tform(original_point[:, :2])
        crop_point = np.concatenate([crop_xy, original_point[:, 2:]], axis=1)

        recovered = landmarks_crop_to_frame(crop_point, tform)

        np.testing.assert_allclose(recovered[0, :2], original_point[0, :2], atol=1e-2)
        self.assertAlmostEqual(recovered[0, 2], -0.01, places=6)  # z untouched

    def test_crop_pads_out_of_bounds_region(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Box extends far outside the frame; must not raise and must return the requested size.
        box = np.array([-200.0, -200.0, 200.0, 200.0], dtype=np.float32)
        crop, _ = crop_roi(frame, box, out_size=64)
        self.assertEqual(crop.shape, (64, 64, 3))


class TestInterpolateLandmarks(unittest.TestCase):
    def test_fills_missing_frames_and_keeps_frame_count(self):
        t = 4
        landmarks = np.full((t, 478, 3), np.nan, dtype=np.float32)
        detected = np.array([True, False, False, True])
        landmarks[0] = 1.0
        landmarks[3] = 5.0

        filled = interpolate_landmarks(landmarks, detected)

        self.assertEqual(filled.shape, (t, 478, 3))
        self.assertFalse(np.isnan(filled).any())
        np.testing.assert_allclose(filled[0], 1.0)
        np.testing.assert_allclose(filled[3], 5.0)
        # linear interpolation between frame 0 (value 1) and frame 3 (value 5)
        np.testing.assert_allclose(filled[1], 1.0 + (5.0 - 1.0) * (1 / 3))
        np.testing.assert_allclose(filled[2], 1.0 + (5.0 - 1.0) * (2 / 3))

    def test_raises_when_never_detected(self):
        landmarks = np.full((3, 478, 3), np.nan, dtype=np.float32)
        detected = np.zeros(3, dtype=bool)
        with self.assertRaises(ValueError):
            interpolate_landmarks(landmarks, detected)


class TestExtractPoseRoiLandmarks(unittest.TestCase):
    def _frames(self, n):
        return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(n)]

    def test_full_pipeline_aligned_and_nan_free(self):
        n = 6
        frames = self._frames(n)
        face_points = {0: (300, 200), 2: (270, 230), 5: (330, 230), 9: (280, 260), 10: (320, 260)}

        # Pose fails on frame 2; FaceLandmarker fails on frame 4.
        pose_calls = {"n": -1}

        def detect_pose(frame):
            pose_calls["n"] += 1
            return None if pose_calls["n"] == 2 else make_pose_landmarks(face_points=face_points)

        call_count = {"n": -1}

        def detect_face(crop):
            call_count["n"] += 1
            i = call_count["n"]
            if i == 4:
                return None
            out = np.zeros((478, 3), dtype=np.float64)
            out[:, 0] = 256.0
            out[:, 1] = 256.0
            out[:, 2] = -0.02
            out[0, 0] = i  # tag the detection with the frame index for ordering checks
            return out

        result = extract_pose_roi_landmarks(frames, detect_pose, detect_face, roi_size=512)

        self.assertEqual(result["frame_count"], n)
        self.assertEqual(result["landmarks"].shape, (n, 478, 3))
        self.assertFalse(np.isnan(result["landmarks"]).any())
        self.assertEqual(result["pose_roi_interpolated"], 1)
        self.assertEqual(result["face_interpolated"], 1)
        self.assertFalse(result["pose_valid"][2])
        self.assertFalse(result["face_detected"][4])
        # frame ordering preserved: landmark[0,0] tag increases monotonically across
        # directly-detected frames.
        detected_idx = np.where(result["face_detected"])[0]
        tags = result["landmarks"][detected_idx, 0, 0]
        np.testing.assert_array_equal(tags, np.sort(tags))

    def test_raises_when_pose_never_valid(self):
        frames = self._frames(3)
        with self.assertRaises(RuntimeError):
            extract_pose_roi_landmarks(frames, lambda f: None, lambda c: np.zeros((478, 3)))

    def test_raises_when_face_never_detected(self):
        frames = self._frames(3)
        face_points = {0: (300, 200), 2: (270, 230), 5: (330, 230), 9: (280, 260), 10: (320, 260)}

        def detect_pose(frame):
            return make_pose_landmarks(face_points=face_points)

        with self.assertRaises(RuntimeError):
            extract_pose_roi_landmarks(frames, detect_pose, lambda c: None)


if __name__ == "__main__":
    unittest.main()
