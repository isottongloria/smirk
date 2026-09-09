import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool
import argparse

# Allow running this script directly (`python datasets/preprocess_scripts/apply_mediapipe_to_dataset.py ...`
# from the repo root, as documented in readme.md) even though it imports the top-level `utils` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.mediapipe_utils import (
    get_face_landmarker,
    make_face_detect_fn,
    run_mediapipe_pose_roi_image,
    run_mediapipe_pose_roi_video,
)


def process_image(image_file, output_file, vis_file, face_detection_mode):
    image = cv2.imread(image_file)

    if face_detection_mode == 'pose_roi':
        landmarks_np = run_mediapipe_pose_roi_image(image)
    else:
        landmarks_np = make_face_detect_fn(get_face_landmarker())(image)

    if landmarks_np is None:
        return

    # Save landmarks
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, landmarks_np)

    # Save visualization if required
    if vis_file:
        for landmark in landmarks_np:
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)
        os.makedirs(os.path.dirname(vis_file), exist_ok=True)
        cv2.imwrite(vis_file, image)


def process_video(video_file, output_file, vis_file, face_detection_mode):
    if face_detection_mode == 'pose_roi':
        process_video_pose_roi(video_file, output_file, vis_file)
        return

    detect_face = make_face_detect_fn(get_face_landmarker())

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_landmarks = []

    if vis_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vis_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks_np = detect_face(frame)

        if landmarks_np is not None:
            frame_landmarks.append(landmarks_np)

            if vis_file:
                for landmark in landmarks_np:
                    cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)

        if vis_file:
            out.write(frame)

    cap.release()
    if vis_file:
        out.release()

    # Save video landmarks
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, np.array(frame_landmarks))


def process_video_pose_roi(video_file, output_file, vis_file):
    """Robust video landmark extraction: MediaPipe Pose tracking locates the face in every
    frame, then FaceLandmarker runs on a zoomed-in crop. Frames without a direct detection
    are interpolated, so exactly one 478x3 entry is saved per decoded frame (same .npy
    format as the 'direct' mode)."""
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not frames:
        np.save(output_file, np.zeros((0, 478, 3), dtype=np.float32))
        return

    result = run_mediapipe_pose_roi_video(frames)
    landmarks_all = result['landmarks']

    print(
        f"{video_file}: frames={result['frame_count']} "
        f"pose_roi_real={result['pose_roi_real']} pose_roi_interpolated={result['pose_roi_interpolated']} "
        f"face_real={result['face_real']} face_interpolated={result['face_interpolated']}"
    )

    np.save(output_file, landmarks_all)

    if vis_file:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vis_file, fourcc, fps, (width, height))
        for frame, landmarks in zip(frames, landmarks_all):
            for landmark in landmarks:
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)
            out.write(frame)
        out.release()


# Function to process a file
def process_file(root, file_name, input_dir, output_dir, vis_dir, face_detection_mode):
    input_path = os.path.join(root, file_name)
    rel_path = os.path.relpath(input_path, input_dir)
    output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.npy')
    vis_path = os.path.join(vis_dir, rel_path) if vis_dir else None

    if file_name.lower().endswith(('.jpg', '.png')):
        process_image(input_path, output_path, vis_path, face_detection_mode)
    elif file_name.lower().endswith(('.mp4', '.avi')):
        process_video(input_path, output_path, vis_path, face_detection_mode)


# Main processing function
def process_sample(payload):
    root, file_name, input_dir, output_dir, vis_dir, face_detection_mode = payload
    process_file(root, file_name, input_dir, output_dir, vis_dir, face_detection_mode)


def main():
    parser = argparse.ArgumentParser(description='Process images/videos with MediaPipe.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--vis_dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--num_processes', type=int, default=16, help='Number of processes to use for processing')
    parser.add_argument('--face-detection-mode', type=str, default='direct', choices=['direct', 'pose_roi'],
                         help="'direct' (default): run FaceLandmarker independently on every frame, as in the "
                              "original SMIRK release; frames with no detected face are dropped. 'pose_roi': "
                              "track the face across each video with MediaPipe Pose first, then run "
                              "FaceLandmarker on a zoomed-in crop; missing frames are interpolated so every "
                              "decoded frame gets a landmark entry and the .npy stays temporally aligned.")
    args = parser.parse_args()

    all_files = []

    for root, _, files in os.walk(args.input_dir):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.png', '.mp4', '.avi')):
                all_files.append((root, file_name, args.input_dir, args.output_dir, args.vis_dir, args.face_detection_mode))

    with Pool(args.num_processes) as pool:
        list(tqdm(pool.imap(process_sample, all_files), total=len(all_files)))


if __name__ == '__main__':
    main()
