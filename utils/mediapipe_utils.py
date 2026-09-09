import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.pose_roi import (
    crop_roi,
    extract_pose_roi_landmarks,
    face_roi_from_pose,
    landmarks_crop_to_frame,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FACE_LANDMARKER_PATH = os.path.join(_REPO_ROOT, "assets", "face_landmarker.task")

_face_landmarker_cache = {}
_pose_cache = {}


def resolve_face_landmarker_path(model_path=None):
    """Resolve the Face Landmarker model path relative to the repo root, not the cwd."""
    path = model_path or _DEFAULT_FACE_LANDMARKER_PATH
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MediaPipe Face Landmarker model not found at '{path}'. Download it with "
            "quick_install.sh, or place it at assets/face_landmarker.task."
        )
    return path


def get_face_landmarker(model_path=None):
    """Return a cached ``FaceLandmarker``. Blendshapes/transform matrices are disabled
    since SMIRK does not consume them."""
    resolved = resolve_face_landmarker_path(model_path)
    if resolved not in _face_landmarker_cache:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=resolved),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
        )
        _face_landmarker_cache[resolved] = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker_cache[resolved]


def get_pose_detector(static_image_mode=False):
    """Return a cached ``mp.solutions.pose.Pose`` instance.

    ``static_image_mode=False`` enables MediaPipe's temporal tracking across a video (with
    landmark smoothing), matching RGB2SMPLX's ``prepare.py`` configuration. Use
    ``static_image_mode=True`` for standalone images.
    """
    if static_image_mode not in _pose_cache:
        _pose_cache[static_image_mode] = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=2,
            smooth_landmarks=not static_image_mode,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _pose_cache[static_image_mode]


def make_face_detect_fn(face_landmarker):
    """Adapt a MediaPipe ``FaceLandmarker`` to the ``detect_face(bgr_image) -> (478,3)|None``
    signature expected by ``utils.pose_roi``."""

    def detect_face(image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = face_landmarker.detect(mp_image)
        if len(detection_result.face_landmarks) == 0:
            return None
        face_landmarks = detection_result.face_landmarks[0]
        landmarks_numpy = np.zeros((478, 3), dtype=np.float64)
        for i, landmark in enumerate(face_landmarks):
            landmarks_numpy[i] = [landmark.x * mp_image.width, landmark.y * mp_image.height, landmark.z]
        return landmarks_numpy

    return detect_face


def make_pose_detect_fn(pose_detector):
    """Adapt a MediaPipe ``Pose`` instance to the
    ``detect_pose(bgr_image) -> (xy(33,2), visibility(33,))|None`` signature."""

    def detect_pose(image_bgr):
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = pose_detector.process(image_rgb)
        if result.pose_landmarks is None:
            return None
        landmarks = result.pose_landmarks.landmark
        xy = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
        visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)
        return xy, visibility

    return detect_pose


def run_mediapipe(image, face_detection_mode="direct", model_path=None):
    """Detect the 478 SMIRK face landmarks in a single BGR image.

    face_detection_mode:
      - 'direct' (default, original SMIRK behavior): run FaceLandmarker on the full frame.
        Fails when the face occupies too few pixels.
      - 'pose_roi': locate the face with MediaPipe Pose first, then run FaceLandmarker on a
        zoomed-in crop around it. More robust to small/distant faces.
    """
    if face_detection_mode == "direct":
        detect_face = make_face_detect_fn(get_face_landmarker(model_path))
        landmarks = detect_face(image)
        if landmarks is None:
            print("No face detected")
        return landmarks

    if face_detection_mode == "pose_roi":
        return run_mediapipe_pose_roi_image(image, model_path=model_path)

    raise ValueError(f"Unknown face_detection_mode: {face_detection_mode!r}")


def run_mediapipe_pose_roi_image(image, model_path=None, roi_size=512):
    """Single-image variant of the pose_roi pipeline (used by ``demo.py``)."""
    detect_pose = make_pose_detect_fn(get_pose_detector(static_image_mode=True))
    detect_face = make_face_detect_fn(get_face_landmarker(model_path))

    pose_result = detect_pose(image)
    if pose_result is None:
        print("No person detected by MediaPipe Pose")
        return None

    xy, visibility = pose_result
    box = face_roi_from_pose(xy, visibility)
    if box is None:
        print("MediaPipe Pose could not localize a face ROI")
        return None

    crop, tform = crop_roi(image, box, out_size=roi_size)
    landmarks_crop = detect_face(crop)
    if landmarks_crop is None:
        print("No face detected")
        return None

    return landmarks_crop_to_frame(landmarks_crop, tform)


def _read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Video contains no decodable frames: {video_path}")
    return frames


def run_mediapipe_pose_roi_video(video_path_or_frames, model_path=None, roi_size=512, progress=False):
    """Run the Pose -> ROI -> FaceLandmarker pipeline over an entire video.

    ``video_path_or_frames`` is either a path to a video file, or an already-decoded list of
    BGR frames (as produced by repeated ``cv2.VideoCapture.read()``).

    Returns a dict (see ``utils.pose_roi.extract_pose_roi_landmarks``) with a ``(T, 478, 3)``
    landmark array temporally aligned with the frames (NaN-free: frames where FaceLandmarker
    failed are interpolated from neighboring frames), plus per-frame detection flags and
    counts for diagnostics.
    """
    if isinstance(video_path_or_frames, (str, os.PathLike)):
        frames = _read_video_frames(video_path_or_frames)
    else:
        frames = list(video_path_or_frames)

    detect_pose = make_pose_detect_fn(get_pose_detector(static_image_mode=False))
    detect_face = make_face_detect_fn(get_face_landmarker(model_path))

    return extract_pose_roi_landmarks(frames, detect_pose, detect_face, roi_size=roi_size, progress=progress)
