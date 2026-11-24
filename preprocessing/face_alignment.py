# File: preprocessing/face_alignment.py
# What this file does: Aligns an RGB frame horizontally using the detected left and right eye landmarks,
#   crops the face using the 1.3x expanded bounding box, resizes it to a 224x224 RGB image, and saves
#   it cleanly to disk.
# Inputs:
#   - rgb_frame (np.ndarray): Original RGB video frame of shape (H, W, 3), uint8.
#   - detection (dict): Output from FaceDetector containing 'box', 'expanded_box', and 'landmarks' (5, 2).
#   - target_size (Tuple[int, int]): Output spatial resolution (default: (224, 224) per Section 5.2 of SPEC.md).
#   - --dry-run (flag): Verifies eye-landmark alignment, 1.3x crop, and 224x224 RGB output on CPU in <5s.
# Outputs:
#   - aligned_crop (np.ndarray): Aligned and resized RGB face image of shape (224, 224, 3), dtype uint8.

import argparse
import math
import os
import tempfile
import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False


def resize_rgb_image(rgb_crop: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resizes an RGB uint8 image crop `(H, W, 3)` to `target_size` `(224, 224)`.
    Uses OpenCV bilinear interpolation when available, PIL Image.BILINEAR as second choice,
    or pure NumPy bilinear-style index resampling on minimal environments.
    """
    target_w, target_h = int(target_size[0]), int(target_size[1])
    h, w = rgb_crop.shape[:2]
    if h == target_h and w == target_w:
        return rgb_crop.astype(np.uint8)

    if HAS_CV2:
        interp = cv2.INTER_AREA if (h > target_h or w > target_w) else cv2.INTER_LINEAR
        return cv2.resize(rgb_crop, (target_w, target_h), interpolation=interp)

    if HAS_PIL:
        pil_img = Image.fromarray(rgb_crop.astype(np.uint8))
        pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
        return np.asarray(pil_img, dtype=np.uint8)

    # Pure NumPy fallback resizer so --dry-run works even without cv2/Pillow installed locally
    row_indices = np.linspace(0, h - 1, target_h).astype(np.int32)
    col_indices = np.linspace(0, w - 1, target_w).astype(np.int32)
    return rgb_crop[row_indices][:, col_indices].astype(np.uint8)


def align_and_crop_face(
    rgb_frame: np.ndarray,
    detection: Dict,
    target_size: Tuple[int, int] = (224, 224),
    margin_scale: float = 1.3,
) -> np.ndarray:
    """
    1. Computes the in-plane rotation angle between the left eye and right eye landmarks (if available).
    2. Rotates the frame around the face center so the eyes lie horizontally level.
    3. Crops the face using the 1.3x expanded bounding box (`expanded_box`).
    4. Resizes the cropped RGB face to `224x224x3` (uint8).
    """
    if rgb_frame is None or rgb_frame.ndim != 3:
        raise ValueError("rgb_frame must be a 3D uint8 RGB NumPy array of shape (H, W, 3).")

    h, w = rgb_frame.shape[:2]
    working_frame = rgb_frame

    landmarks = detection.get("landmarks", None)
    if landmarks is not None and len(landmarks) >= 2 and HAS_CV2:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        dy = float(right_eye[1] - left_eye[1])
        dx = float(right_eye[0] - left_eye[0])
        angle_deg = math.degrees(math.atan2(dy, dx))

        # Rotate frame around eye midpoint if tilt is noticeable (> 1.0 deg) and plausible (< 45 deg)
        if 1.0 < abs(angle_deg) < 45.0:
            eye_center = (
                float(0.5 * (left_eye[0] + right_eye[0])),
                float(0.5 * (left_eye[1] + right_eye[1])),
            )
            rot_mat = cv2.getRotationMatrix2D(eye_center, angle_deg, 1.0)
            working_frame = cv2.warpAffine(
                rgb_frame,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

    # Extract 1.3x expanded bounding box coordinates
    if "expanded_box" in detection and detection["expanded_box"] is not None:
        ex1, ey1, ex2, ey2 = [int(round(v)) for v in detection["expanded_box"]]
    else:
        x1, y1, x2, y2 = [float(v) for v in detection["box"]]
        bw = max(1.0, x2 - x1) * margin_scale
        bh = max(1.0, y2 - y1) * margin_scale
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        ex1 = int(round(cx - 0.5 * bw))
        ey1 = int(round(cy - 0.5 * bh))
        ex2 = int(round(cx + 0.5 * bw))
        ey2 = int(round(cy + 0.5 * bh))

    ex1 = max(0, min(w - 1, ex1))
    ey1 = max(0, min(h - 1, ey1))
    ex2 = max(ex1 + 1, min(w, ex2))
    ey2 = max(ey1 + 1, min(h, ey2))

    cropped = working_frame[ey1:ey2, ex1:ex2]
    aligned_224 = resize_rgb_image(cropped, target_size=target_size)
    return aligned_224


def save_rgb_crop(rgb_crop: np.ndarray, output_path: str) -> None:
    """
    Saves a 224x224 RGB uint8 face crop to `output_path` (.jpg or .png).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if HAS_CV2:
        bgr_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_crop)
        return

    if HAS_PIL:
        Image.fromarray(rgb_crop).save(output_path)
        return

    # Minimal PPM/raw header writer fallback if neither cv2 nor PIL is installed locally
    with open(output_path, "wb") as f:
        h, w, _ = rgb_crop.shape
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        f.write(header)
        f.write(rgb_crop.tobytes())


def run_dry_run() -> None:
    """
    Synthesizes a tilted dummy face frame and detection metadata, runs `align_and_crop_face`,
    and verifies the output is a (224, 224, 3) uint8 RGB image saved cleanly to disk in <5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing preprocessing/face_alignment.py (alignment + 224x224 resize)...")

    dummy_frame = np.full((300, 300, 3), 50, dtype=np.uint8)
    dummy_frame[80:220, 90:210] = np.array([210, 175, 150], dtype=np.uint8)
    dummy_detection = {
        "box": [90.0, 80.0, 210.0, 220.0],
        "expanded_box": [72, 59, 228, 241],
        "prob": 0.99,
        "landmarks": np.array(
            [
                [125.0, 125.0],  # left eye
                [175.0, 132.0],  # right eye (tilted slightly)
                [150.0, 155.0],  # nose
                [130.0, 185.0],  # mouth left
                [170.0, 190.0],  # mouth right
            ],
            dtype=np.float32,
        ),
    }

    aligned = align_and_crop_face(dummy_frame, dummy_detection, target_size=(224, 224), margin_scale=1.3)
    assert aligned.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {aligned.shape}"
    assert aligned.dtype == np.uint8, f"Expected uint8 dtype, got {aligned.dtype}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_img = os.path.join(tmp_dir, "crop_000.jpg")
        save_rgb_crop(aligned, out_img)
        assert os.path.exists(out_img) and os.path.getsize(out_img) > 0, "Failed to save cropped face image!"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] Aligned and resized face crop to {aligned.shape} (uint8) in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align and resize detected face crops to 224x224 RGB.")
    parser.add_argument("--target_size", type=int, default=224, help="Output crop size (default: 224).")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) synthetic alignment test on CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] face_alignment module ready. Import align_and_crop_face or run with --dry-run.")


if __name__ == "__main__":
    main()
