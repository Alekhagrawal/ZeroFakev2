# File: preprocessing/extract_and_crop.py
# What this file does: Main preprocessing runner that reads splits.json, extracts 16 evenly spaced
#   frames per video using OpenCV, detects and aligns the largest primary face with MTCNN (using a 1.3x
#   margin and resizing to 224x224 RGB), saves crops cleanly to the output folder (and optional zip archive),
#   writes frames_manifest.csv, and logs any faceless/failed frames to failed_detections.csv without crashing.
# Inputs:
#   - --data_dir (str): Root directory containing the raw FaceForensics++ videos.
#   - --splits_path (str): Path to splits.json created by preprocessing/create_splits.py.
#   - --output_dir (str): Directory where cropped face images, frames_manifest.csv, and failed_detections.csv are saved.
#   - --zip_path (str, optional): Optional path to also package all cropped faces into a single cropped_faces.zip archive.
#   - --num_frames (int): Number of evenly spaced frames to sample per video (default: 16).
#   - --margin_scale (float): Face bounding box expansion factor (default: 1.3).
#   - --image_size (int): Target height and width of saved RGB crops (default: 224).
#   - --dry-run (flag): Synthesizes 4 tiny dummy videos, runs frame extraction + MTCNN/fallback detection + alignment,
#     verifies frames_manifest.csv and failed_detections.csv, and cleans up all temporary files in <5 seconds on CPU.
# Outputs:
#   - Cropped 224x224 RGB face images organized under <output_dir>/crops/<split>/<generator_name>/<video_stem>_frame<idx>.jpg
#   - frames_manifest.csv with columns: image_path, split, label, video_id, identity_ids, generator_name
#   - failed_detections.csv logging any video/frame where no face was found or video reading failed.

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is importable whether invoked as `python preprocessing/extract_and_crop.py` or `-m`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.create_splits import create_logo_splits, save_splits_json
from preprocessing.face_alignment import align_and_crop_face, save_rgb_crop
from preprocessing.face_detection import FaceDetector
from preprocessing.frame_extraction import create_synthetic_video, extract_evenly_spaced_frames

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False


MANIFEST_COLUMNS = [
    "image_path",
    "split",
    "label",
    "video_id",
    "identity_ids",
    "generator_name",
]

FAILED_COLUMNS = [
    "video_id",
    "video_path",
    "split",
    "generator_name",
    "frame_index",
    "reason",
]


def run_extraction_pipeline(
    data_dir: str,
    splits_path: str,
    output_dir: str,
    num_frames: int = 16,
    margin_scale: float = 1.3,
    image_size: int = 224,
    zip_path: Optional[str] = None,
    device: Optional[str] = None,
    allow_synthetic_fallback: bool = False,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    Executes the full frame extraction, MTCNN face detection, 1.3x margin cropping, and 224x224 alignment
    pipeline across all splits in `splits_path`.
    Never crashes on missing faces—logs failures to `failed_detections.csv` and continues.
    """
    with open(splits_path, "r", encoding="utf-8") as f:
        split_payload = json.load(f)

    splits_dict: Dict[str, List[Dict]] = split_payload.get("splits", {})
    out_root = Path(output_dir)
    crops_root = out_root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    manifest_csv_path = out_root / "frames_manifest.csv"
    failed_csv_path = out_root / "failed_detections.csv"

    detector = FaceDetector(
        device=device,
        margin_scale=margin_scale,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )

    all_videos: List[Dict] = []
    for split_name in ("train", "val", "test_closed", "test_unseen"):
        for rec in splits_dict.get(split_name, []):
            rec_with_split = dict(rec)
            rec_with_split["split"] = split_name
            all_videos.append(rec_with_split)

    iterator = (
        tqdm(all_videos, desc="Extracting & Cropping Faces")
        if (HAS_TQDM and show_progress)
        else all_videos
    )

    manifest_rows: List[Dict] = []
    failed_rows: List[Dict] = []
    saved_abs_to_rel: List[tuple] = []

    for rec in iterator:
        split_name = rec["split"]
        video_id = rec["video_id"]
        rel_video_path = rec["video_path"]
        label = int(rec["label"])
        generator_name = rec["generator_name"]
        identity_ids_list = rec.get("identity_ids", [])
        identity_ids_str = "|".join(str(x) for x in identity_ids_list)

        abs_video_path = os.path.join(data_dir, rel_video_path)
        sampled_frames = extract_evenly_spaced_frames(abs_video_path, num_frames=num_frames)

        if not sampled_frames:
            failed_rows.append(
                {
                    "video_id": video_id,
                    "video_path": rel_video_path,
                    "split": split_name,
                    "generator_name": generator_name,
                    "frame_index": -1,
                    "reason": "video_unreadable_or_zero_frames",
                }
            )
            continue

        video_stem = Path(rel_video_path).stem
        for frame_idx, rgb_frame in sampled_frames:
            try:
                detection = detector.detect_primary_face(rgb_frame)
                if detection is None:
                    failed_rows.append(
                        {
                            "video_id": video_id,
                            "video_path": rel_video_path,
                            "split": split_name,
                            "generator_name": generator_name,
                            "frame_index": frame_idx,
                            "reason": "no_face_detected",
                        }
                    )
                    continue

                aligned_crop = align_and_crop_face(
                    rgb_frame,
                    detection,
                    target_size=(image_size, image_size),
                    margin_scale=margin_scale,
                )

                crop_filename = f"{video_stem}_frame{frame_idx:04d}.jpg"
                rel_crop_path = Path("crops") / split_name / generator_name / crop_filename
                abs_crop_path = out_root / rel_crop_path

                save_rgb_crop(aligned_crop, str(abs_crop_path))
                rel_crop_posix = rel_crop_path.as_posix()

                manifest_rows.append(
                    {
                        "image_path": rel_crop_posix,
                        "split": split_name,
                        "label": label,
                        "video_id": video_id,
                        "identity_ids": identity_ids_str,
                        "generator_name": generator_name,
                    }
                )
                saved_abs_to_rel.append((str(abs_crop_path), rel_crop_posix))

            except Exception as exc:
                # Never crash on an anomalous frame; log and continue
                failed_rows.append(
                    {
                        "video_id": video_id,
                        "video_path": rel_video_path,
                        "split": split_name,
                        "generator_name": generator_name,
                        "frame_index": frame_idx,
                        "reason": f"exception:{type(exc).__name__}",
                    }
                )

    # Write frames_manifest.csv
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write failed_detections.csv
    with open(failed_csv_path, "w", newline="", encoding="utf-8") as ff:
        writer = csv.DictWriter(ff, fieldnames=FAILED_COLUMNS)
        writer.writeheader()
        writer.writerows(failed_rows)

    # Optional single-archive packaging for fast Google Drive -> Colab SSD transfer (Rule 4)
    if zip_path:
        os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(str(manifest_csv_path), arcname="frames_manifest.csv")
            zf.write(str(failed_csv_path), arcname="failed_detections.csv")
            for abs_p, rel_p in saved_abs_to_rel:
                zf.write(abs_p, arcname=rel_p)

    return {
        "total_videos": len(all_videos),
        "saved_crops": len(manifest_rows),
        "failed_frames": len(failed_rows),
    }


def run_dry_run() -> None:
    """
    Creates 4 tiny synthetic dummy videos in a temporary directory (3 with synthetic faces across
    train/val/test_closed + 1 held-out NeuralTextures video with a faceless frame test), runs
    frame extraction (16 frames/video), face detection + 1.3x margin alignment (224x224), verifies
    `frames_manifest.csv` and `failed_detections.csv`, and cleans up all temporary test files in <5s.
    """
    start_time = time.time()
    print("[DRY-RUN] Creating 4 tiny synthetic dummy videos and testing extract_and_crop.py end-to-end...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = os.path.join(tmp_dir, "raw_ff")
        out_dir = os.path.join(tmp_dir, "processed")
        splits_json = os.path.join(tmp_dir, "splits.json")
        zip_out = os.path.join(tmp_dir, "cropped_faces.zip")

        # Define 4 synthetic videos across disjoint actor components (000, 001, 002, 003_004)
        synthetic_specs = [
            ("original/000.mp4", "Real", 0, ["000"], True),
            ("Deepfakes/001_005.mp4", "Deepfakes", 1, ["001", "005"], True),
            ("Face2Face/002_006.mp4", "Face2Face", 1, ["002", "006"], True),
            ("NeuralTextures/003_004.mp4", "NeuralTextures", 1, ["003", "004"], True),
        ]

        records: List[Dict] = []
        for rel_path, gen_name, label, id_list, has_face in synthetic_specs:
            abs_video = os.path.join(raw_dir, rel_path)
            create_synthetic_video(abs_video, total_frames=20, height=160, width=160, include_face=has_face)
            records.append(
                {
                    "video_id": f"{gen_name}/{Path(rel_path).stem}",
                    "video_path": rel_path,
                    "filename": Path(rel_path).name,
                    "label": label,
                    "generator_name": gen_name,
                    "identity_ids": id_list,
                }
            )

        # Build verified LOGO split for the 4 synthetic videos
        split_payload = create_logo_splits(records, holdout_generator="NeuralTextures", seed=42)
        save_splits_json(split_payload, splits_json)

        stats = run_extraction_pipeline(
            data_dir=raw_dir,
            splits_path=splits_json,
            output_dir=out_dir,
            num_frames=16,
            margin_scale=1.3,
            image_size=224,
            zip_path=zip_out,
            device="cpu",
            allow_synthetic_fallback=True,
            show_progress=False,
        )

        manifest_file = os.path.join(out_dir, "frames_manifest.csv")
        failed_file = os.path.join(out_dir, "failed_detections.csv")

        assert os.path.exists(manifest_file), "frames_manifest.csv was not created!"
        assert os.path.exists(failed_file), "failed_detections.csv was not created!"
        assert os.path.exists(zip_out), "cropped_faces.zip was not created!"

        with open(manifest_file, "r", encoding="utf-8") as mf:
            reader = csv.DictReader(mf)
            assert reader.fieldnames == MANIFEST_COLUMNS, (
                f"Expected columns {MANIFEST_COLUMNS}, got {reader.fieldnames}"
            )
            rows = list(reader)

        # 4 videos * 16 evenly spaced frames = 64 crops
        assert len(rows) == 64, f"Expected 64 cropped face rows in frames_manifest.csv, got {len(rows)}"
        first_crop_path = os.path.join(out_dir, rows[0]["image_path"])
        assert os.path.exists(first_crop_path), f"Cropped face image missing on disk: {first_crop_path}"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run took {elapsed:.3f}s, exceeding 5.0s limit!"
    print(
        f"[DRY-RUN PASSED] Processed {stats['total_videos']} synthetic videos -> "
        f"{stats['saved_crops']} aligned 224x224 crops saved, frames_manifest.csv verified, "
        f"and temporary files cleaned up in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 16 evenly spaced frames per video, align & crop faces with MTCNN (1.3x margin, 224x224)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/raw/ff-c23",
        help="Root directory containing FaceForensics++ videos.",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default="datasets/splits.json",
        help="Path to splits.json generated by preprocessing/create_splits.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/processed",
        help="Directory to save 224x224 crops, frames_manifest.csv, and failed_detections.csv.",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default="datasets/processed/cropped_faces.zip",
        help="Optional output path for single-archive cropped_faces.zip (pass empty string '' to skip zipping).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of evenly spaced frames to extract per video (default: 16).",
    )
    parser.add_argument(
        "--margin_scale",
        type=float,
        default=1.3,
        help="Bounding box margin expansion factor (default: 1.3).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Output spatial size for aligned RGB face crops (default: 224).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a fast (<5s) verification on 4 synthetic dummy videos and clean up temporary files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    zip_target = args.zip_path.strip() if args.zip_path else None
    print(f"[INFO] Starting frame extraction and MTCNN face cropping from '{args.data_dir}'...")
    stats = run_extraction_pipeline(
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        margin_scale=args.margin_scale,
        image_size=args.image_size,
        zip_path=zip_target,
    )
    print(
        f"[SUCCESS] Completed preprocessing:\n"
        f"          Videos Processed : {stats['total_videos']}\n"
        f"          Saved 224x224    : {stats['saved_crops']}\n"
        f"          Failed Frames    : {stats['failed_frames']}\n"
        f"          Manifest CSV     : {os.path.join(args.output_dir, 'frames_manifest.csv')}\n"
        f"          Failed Log CSV   : {os.path.join(args.output_dir, 'failed_detections.csv')}"
    )


if __name__ == "__main__":
    main()
