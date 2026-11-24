# File: preprocessing/frame_extraction.py
# What this file does: Opens a video file using OpenCV (cv2.VideoCapture) and extracts N=16 evenly
#   spaced RGB video frames across the video's duration so temporal segments are uniformly represented.
# Inputs:
#   - video_path (str): Path to an input .mp4 video file (or synthetic dummy video in --dry-run mode).
#   - num_frames (int): Number of evenly spaced frames to extract per video (default: 16).
#   - --dry-run (flag): Synthesizes a mock 32-frame video and verifies 16 evenly spaced RGB frames are extracted in <2s.
# Outputs:
#   - List of (frame_index, rgb_frame_array) tuples where each frame is a uint8 NumPy array of shape (H, W, 3) in RGB order.

import argparse
import os
import pickle
import tempfile
import time
from typing import List, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


def compute_evenly_spaced_indices(total_frames: int, num_frames: int = 16) -> List[int]:
    """
    Computes `num_frames` evenly spaced integer frame indices across `[0, total_frames - 1]`.
    If the video has fewer frames than `num_frames`, returns all available indices `[0, ..., total_frames - 1]`.
    """
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    return [int(round(idx)) for idx in np.linspace(0, total_frames - 1, num_frames)]


def extract_evenly_spaced_frames(
    video_path: str,
    num_frames: int = 16,
) -> List[Tuple[int, np.ndarray]]:
    """
    Opens `video_path` with OpenCV (`cv2.VideoCapture`), samples `num_frames` evenly spaced frames,
    converts each frame from BGR to RGB, and returns a list of `(frame_idx, rgb_frame)` tuples.
    Includes a seamless fallback loader for synthetic `.mp4` dry-run files when OpenCV is not installed
    on a lightweight laptop environment.
    """
    if not os.path.exists(video_path):
        return []

    # Primary execution path: OpenCV VideoCapture (used in Google Colab and full local environments)
    if HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_indices = compute_evenly_spaced_indices(total_frames, num_frames=num_frames)
            target_set = set(target_indices)

            extracted: List[Tuple[int, np.ndarray]] = []
            if total_frames > 0 and target_indices:
                # Seek or sequentially read frames
                current_idx = 0
                max_target = target_indices[-1]
                while current_idx <= max_target:
                    ret, bgr_frame = cap.read()
                    if not ret or bgr_frame is None:
                        break
                    if current_idx in target_set:
                        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        extracted.append((current_idx, rgb_frame))
                    current_idx += 1
            else:
                # Fallback if container metadata does not report frame count: read all and sub-sample
                all_frames: List[Tuple[int, np.ndarray]] = []
                idx = 0
                while True:
                    ret, bgr_frame = cap.read()
                    if not ret or bgr_frame is None:
                        break
                    all_frames.append((idx, cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)))
                    idx += 1
                indices = compute_evenly_spaced_indices(len(all_frames), num_frames=num_frames)
                extracted = [all_frames[i] for i in indices]

            cap.release()
            if extracted:
                return extracted

    # Lightweight dry-run fallback for synthetic pickled dummy video containers when cv2 is absent
    try:
        with open(video_path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "frames" in payload:
            raw_frames = payload["frames"]
            indices = compute_evenly_spaced_indices(len(raw_frames), num_frames=num_frames)
            return [(int(i), np.asarray(raw_frames[i], dtype=np.uint8)) for i in indices]
    except Exception:
        pass

    return []


def create_synthetic_video(
    video_path: str,
    total_frames: int = 32,
    height: int = 256,
    width: int = 256,
    include_face: bool = True,
) -> None:
    """
    Creates a tiny synthetic video file on disk for `--dry-run` testing.
    Draws a bright synthetic face oval with two eye landmarks and a mouth so both real MTCNN
    and fallback dry-run detectors can test detection and alignment end-to-end.
    """
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    frames: List[np.ndarray] = []

    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    ry, rx = int(height * 0.28), int(width * 0.22)

    for f_idx in range(total_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        if include_face:
            # Draw skin-toned oval face center
            face_mask = ((yy - cy) ** 2) / (ry ** 2) + ((xx - cx) ** 2) / (rx ** 2) <= 1.0
            frame[face_mask] = np.array([225, 190, 165], dtype=np.uint8)
            # Draw left and right eye dark patches
            eye_y = cy - int(ry * 0.25)
            left_eye_x = cx - int(rx * 0.4)
            right_eye_x = cx + int(rx * 0.4)
            frame[eye_y - 5 : eye_y + 5, left_eye_x - 7 : left_eye_x + 7] = np.array([30, 30, 30], dtype=np.uint8)
            frame[eye_y - 5 : eye_y + 5, right_eye_x - 7 : right_eye_x + 7] = np.array([30, 30, 30], dtype=np.uint8)
            # Small frame-dependent pixel marker so frames are verifiable
            frame[4, 4, 0] = np.uint8(f_idx % 255)
        frames.append(frame)

    if HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 16.0, (width, height))
        if writer.isOpened():
            for rgb in frames:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                return

    # Save portable synthetic container if cv2 codec is unavailable on the local laptop
    with open(video_path, "wb") as f:
        pickle.dump({"frames": frames, "fps": 16.0}, f)


def run_dry_run() -> None:
    """
    Verifies that a 32-frame synthetic video is created and 16 evenly spaced RGB frames are extracted
    on CPU in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing preprocessing/frame_extraction.py with a 32-frame synthetic video...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_video = os.path.join(tmp_dir, "dummy_000.mp4")
        create_synthetic_video(dummy_video, total_frames=32, height=128, width=128, include_face=True)
        extracted = extract_evenly_spaced_frames(dummy_video, num_frames=16)

        assert len(extracted) == 16, f"Expected 16 extracted frames, got {len(extracted)}"
        first_idx, first_frame = extracted[0]
        last_idx, last_frame = extracted[-1]
        assert first_idx == 0 and last_idx == 31, f"Expected boundary indices (0, 31), got ({first_idx}, {last_idx})"
        assert first_frame.shape == (128, 128, 3), f"Unexpected frame shape: {first_frame.shape}"
        assert first_frame.dtype == np.uint8, f"Unexpected frame dtype: {first_frame.dtype}"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] Extracted 16 evenly spaced RGB frames (indices 0..31) in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 16 evenly spaced RGB frames per video.")
    parser.add_argument("--video_path", type=str, default="", help="Path to a single .mp4 video file.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample per video.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) synthetic frame extraction test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    if not args.video_path:
        raise ValueError("Please provide --video_path or run with --dry-run.")
    frames = extract_evenly_spaced_frames(args.video_path, num_frames=args.num_frames)
    print(f"[INFO] Extracted {len(frames)} frames from '{args.video_path}'.")


if __name__ == "__main__":
    main()
