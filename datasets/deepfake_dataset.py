# File: datasets/deepfake_dataset.py
# What this file does: Implements the ZeroFake dataset loader (`DeepfakeDataset`) reading cropped face
#   images from frames_manifest.csv and splits.json with standard ImageNet normalization, alongside the
#   `GeneratorBalancedBatchSampler` which guarantees that every training mini-batch contains 50% Real
#   crops (label 0) and 50% Fake crops (label 1) evenly distributed across all seen training generators.
# Inputs:
#   - manifest_path (str): Path to frames_manifest.csv produced by preprocessing/extract_and_crop.py.
#   - splits_path (str, optional): Path to splits.json to cross-verify zero held-out generator leakage.
#   - root_dir (str): Base folder containing the relative `image_path` face crops.
#   - split (str): Dataset partition to load ('train', 'val', 'test_closed', or 'test_unseen').
#   - batch_size (int): Mini-batch size for `GeneratorBalancedBatchSampler` (default: 64).
#   - --dry-run (flag): Synthesizes a mock manifest + face crops, builds `DeepfakeDataset` and
#     `GeneratorBalancedBatchSampler`, and asserts every batch has 50% Real and 50% evenly split Fake generators in <5s.
# Outputs:
#   - DeepfakeDataset item dictionary: {'image': (3, 224, 224) float32 tensor, 'label': int (0 or 1),
#     'generator_name': str, 'video_id': str, 'image_path': str}.
#   - GeneratorBalancedBatchSampler yielding lists of dataset indices of length `batch_size`.

import sys
from pathlib import Path

# Insert repository root at sys.path[0] so Colab's pre-installed Hugging Face 'datasets' package
# never shadows ZeroFake's local 'datasets.deepfake_dataset' module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, Sampler
    HAS_TORCH = True
except ImportError:
    torch = None
    Dataset = object
    Sampler = object
    HAS_TORCH = False

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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_rgb_image_224(image_path: str, image_size: int = 224) -> np.ndarray:
    """
    Loads an RGB face crop from disk and returns a `(image_size, image_size, 3)` uint8 NumPy array.
    Supports OpenCV, Pillow, and raw P6 PPM fallback files created during minimal dry-runs.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Cropped face image not found: {image_path}")

    if HAS_CV2:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (image_size, image_size):
                rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            return rgb.astype(np.uint8)

    if HAS_PIL:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            if rgb_img.size != (image_size, image_size):
                rgb_img = rgb_img.resize((image_size, image_size), Image.BILINEAR)
            return np.asarray(rgb_img, dtype=np.uint8)

    # Fallback reader for P6 PPM dry-run images when cv2 and PIL are absent locally
    with open(image_path, "rb") as f:
        data = f.read()
    if data.startswith(b"P6\n"):
        lines = data.split(b"\n", 3)
        w, h = [int(x) for x in lines[1].decode("ascii").split()]
        pixel_bytes = lines[3]
        arr = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(h, w, 3)
        return arr.copy()

    return np.zeros((image_size, image_size, 3), dtype=np.uint8)


def normalize_imagenet_chw(rgb_uint8: np.ndarray, augment: bool = False, rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Converts an `(H, W, 3)` uint8 RGB image in `[0, 255]` to a normalized `(3, H, W)` float32 tensor
    using standard ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
    Applies random horizontal flip when `augment=True` (for training split).
    """
    img = rgb_uint8
    if augment:
        coin = rng.random() if rng is not None else random.random()
        if coin < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])

    chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
    norm_chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return norm_chw.astype(np.float32)


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset reading `frames_manifest.csv` (and optionally validating against `splits.json`)
    for a target split (`'train'`, `'val'`, `'test_closed'`, or `'test_unseen'`).
    When `split == 'test_unseen'` and `include_closed_real_in_unseen=True`, automatically includes the
    `test_closed` Real face crops so unseen-generator ROC-AUC has authentic negative samples for evaluation.
    """

    def __init__(
        self,
        manifest_path: str,
        root_dir: Optional[str] = None,
        split: str = "train",
        splits_path: Optional[str] = None,
        image_size: int = 224,
        augment: Optional[bool] = None,
        include_closed_real_in_unseen: bool = True,
    ) -> None:
        super().__init__()
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"frames_manifest.csv not found at: {manifest_path}")

        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        self.root_dir = os.path.abspath(root_dir) if root_dir is not None else self.manifest_dir
        self.split = split.strip()
        self.image_size = int(image_size)
        self.augment = (self.split == "train") if augment is None else bool(augment)

        holdout_generator = None
        video_to_split: Dict[str, str] = {}
        has_authoritative_splits = False
        if splits_path and os.path.exists(splits_path):
            with open(splits_path, "r", encoding="utf-8") as sf:
                splits_payload = json.load(sf)
            split_meta = splits_payload.get("metadata", {})
            holdout_generator = split_meta.get("holdout_generator", None)
            for sp_name, sp_records in splits_payload.get("splits", {}).items():
                for rec in sp_records:
                    vid_key = str(rec.get("video_id", "")).strip()
                    if vid_key:
                        video_to_split[vid_key] = sp_name.strip()
                    gen_key = str(rec.get("generator_name", "")).strip()
                    fname_stem = Path(str(rec.get("filename", rec.get("video_path", "")))).stem
                    if gen_key and fname_stem:
                        video_to_split[f"{gen_key}/{fname_stem}"] = sp_name.strip()
            has_authoritative_splits = len(video_to_split) > 0

        self.samples: List[Dict] = []
        with open(self.manifest_path, "r", encoding="utf-8") as mf:
            reader = csv.DictReader(mf)
            for row in reader:
                row_vid = row["video_id"].strip()
                row_gen = row["generator_name"].strip()
                # When --splits_path is provided and populated (such as splits_holdout_faceswap.json),
                # assign each video's split strictly from splits_path (never falling back to the
                # pre-existing 'split' column inside frames_manifest.csv).
                if has_authoritative_splits:
                    alt_vid = f"{row_gen}/{Path(row_vid).stem}"
                    row_split = video_to_split.get(row_vid, video_to_split.get(alt_vid, None))
                    if row_split is None:
                        continue
                else:
                    row_split = row["split"].strip()

                row_label = int(row["label"])
                resolved_img_path = self._resolve_image_path(row["image_path"].strip())

                if row_split == self.split:
                    self.samples.append(
                        {
                            "image_path": resolved_img_path,
                            "split": row_split,
                            "label": row_label,
                            "video_id": row_vid,
                            "identity_ids": row.get("identity_ids", "").strip(),
                            "generator_name": row_gen,
                        }
                    )
                elif (
                    self.split == "test_unseen"
                    and include_closed_real_in_unseen
                    and row_split == "test_closed"
                    and row_label == 0
                ):
                    self.samples.append(
                        {
                            "image_path": resolved_img_path,
                            "split": "test_unseen",
                            "label": 0,
                            "video_id": row_vid,
                            "identity_ids": row.get("identity_ids", "").strip(),
                            "generator_name": row_gen,
                        }
                    )

        # Enforce Rule 3: Never allow the held-out unseen generator into train or val
        if holdout_generator and self.split in ("train", "val"):
            for s in self.samples:
                assert s["generator_name"] != holdout_generator, (
                    f"Rule 3 Violation: Held-out generator '{holdout_generator}' found in '{self.split}' split!"
                )

    def _resolve_image_path(self, raw_path: str) -> str:
        """
        Resolves an image path from `frames_manifest.csv` robustly across Colab and local disks:
          1. If `raw_path` exists verbatim, returns it directly.
          2. If `os.path.join(self.root_dir, raw_path)` or `os.path.join(self.manifest_dir, raw_path)` exists, returns it.
          3. If `raw_path` was saved with a prefix from another directory (e.g. `/content/processed_faces/crops/...`
             or `processed_faces/crops/...`), strips the prefix up to `crops/` and resolves relative to
             `self.manifest_dir` (i.e. `os.path.dirname(manifest_path)`) and `self.root_dir`.
        """
        norm_raw = raw_path.replace("\\", "/").strip()

        # 1. Verbatim check
        if os.path.exists(raw_path):
            return os.path.abspath(raw_path)

        # 2. Direct join relative to root_dir or os.path.dirname(manifest_path)
        rel_clean = norm_raw.lstrip("/")
        for base_dir in (self.root_dir, self.manifest_dir):
            cand = os.path.join(base_dir, rel_clean)
            if os.path.exists(cand):
                return os.path.abspath(cand)

        # 3. Strip legacy prefix like '/content/processed_faces/' or any prefix before 'crops/'
        candidate_suffixes = []
        for prefix in ("/content/processed_faces/", "content/processed_faces/", "processed_faces/", "/content/data/"):
            if norm_raw.startswith(prefix):
                candidate_suffixes.append(norm_raw[len(prefix) :].lstrip("/"))
        if "crops/" in norm_raw:
            crops_idx = norm_raw.index("crops/")
            candidate_suffixes.append(norm_raw[crops_idx:])
        candidate_suffixes.append(os.path.basename(norm_raw))

        for suffix in candidate_suffixes:
            for base_dir in (self.manifest_dir, self.root_dir):
                cand = os.path.join(base_dir, suffix)
                if os.path.exists(cand):
                    return os.path.abspath(cand)

        # Default fallback relative to os.path.dirname(manifest_path)
        fallback_suffix = candidate_suffixes[0] if candidate_suffixes else rel_clean
        return os.path.join(self.manifest_dir, fallback_suffix)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.samples[idx]
        abs_img_path = rec["image_path"] if os.path.exists(rec["image_path"]) else os.path.join(self.manifest_dir, rec["image_path"])
        rgb = load_rgb_image_224(abs_img_path, image_size=self.image_size)
        norm_chw = normalize_imagenet_chw(rgb, augment=self.augment)

        if HAS_TORCH:
            image_tensor = torch.from_numpy(norm_chw)
            label_tensor = torch.tensor(rec["label"], dtype=torch.long)
        else:
            image_tensor = norm_chw
            label_tensor = rec["label"]

        return {
            "image": image_tensor,
            "label": label_tensor,
            "generator_name": rec["generator_name"],
            "video_id": rec["video_id"],
            "image_path": rec["image_path"],
        }


class GeneratorBalancedBatchSampler(Sampler):
    """
    Generator-Balanced Mini-Batch Sampler (Section 2 & Rule 1 of AGENTS.md):
      Ensures every training mini-batch of size `B` contains:
        - 50% Real face crops (`B // 2` samples with `label == 0`)
        - 50% Fake face crops (`B - B // 2` samples with `label == 1`) divided equally across all
          seen training manipulation generators (e.g., Deepfakes, Face2Face, FaceSwap).
    """

    def __init__(
        self,
        dataset: DeepfakeDataset,
        batch_size: int = 64,
        num_batches: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        assert batch_size >= 2 and batch_size % 2 == 0, "batch_size must be an even integer >= 2."
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.half_batch = self.batch_size // 2
        self.seed = int(seed)
        self.epoch = 0

        self.real_indices: List[int] = []
        self.fake_by_generator: Dict[str, List[int]] = defaultdict(list)

        for idx, sample in enumerate(dataset.samples):
            if int(sample["label"]) == 0:
                self.real_indices.append(idx)
            else:
                self.fake_by_generator[sample["generator_name"]].append(idx)

        self.fake_generators = sorted(self.fake_by_generator.keys())
        if not self.real_indices:
            raise ValueError("GeneratorBalancedBatchSampler requires at least 1 Real sample (label 0).")
        if not self.fake_generators:
            raise ValueError("GeneratorBalancedBatchSampler requires at least 1 Fake generator (label 1).")

        if num_batches is not None:
            self.num_batches = int(num_batches)
        else:
            self.num_batches = max(1, len(dataset) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        num_gens = len(self.fake_generators)

        # Pre-shuffle index pools for smooth cycling without replacement bias
        real_pool = list(self.real_indices)
        rng.shuffle(real_pool)
        real_ptr = 0

        fake_pools: Dict[str, List[int]] = {}
        fake_ptrs: Dict[str, int] = {}
        for g in self.fake_generators:
            pool = list(self.fake_by_generator[g])
            rng.shuffle(pool)
            fake_pools[g] = pool
            fake_ptrs[g] = 0

        base_per_gen = self.half_batch // num_gens
        remainder = self.half_batch % num_gens

        for b_idx in range(self.num_batches):
            batch_indices: List[int] = []

            # 1. Sample 50% Real indices (half_batch)
            for _ in range(self.half_batch):
                if real_ptr >= len(real_pool):
                    rng.shuffle(real_pool)
                    real_ptr = 0
                batch_indices.append(real_pool[real_ptr])
                real_ptr += 1

            # 2. Sample 50% Fake indices divided equally across all seen training generators
            # Rotate which generator(s) receive the +1 remainder slot across batches so overall distribution is exact
            for g_offset in range(num_gens):
                g_name = self.fake_generators[(b_idx + g_offset) % num_gens]
                quota = base_per_gen + (1 if g_offset < remainder else 0)
                pool = fake_pools[g_name]
                ptr = fake_ptrs[g_name]

                for _ in range(quota):
                    if ptr >= len(pool):
                        rng.shuffle(pool)
                        ptr = 0
                    batch_indices.append(pool[ptr])
                    ptr += 1

                fake_ptrs[g_name] = ptr

            assert len(batch_indices) == self.batch_size
            rng.shuffle(batch_indices)
            yield batch_indices


def run_dry_run() -> None:
    """
    Creates a temporary dataset with Real crops and 3 Fake generators (Deepfakes, Face2Face, FaceSwap),
    initializes `DeepfakeDataset` and `GeneratorBalancedBatchSampler`, and verifies that every sampled
    batch of size 12 contains exactly 6 Real crops (50%) and 6 Fake crops (2 Deepfakes, 2 Face2Face, 2 FaceSwap).
    """
    start_time = time.time()
    print("[DRY-RUN] Testing datasets/deepfake_dataset.py (DeepfakeDataset + GeneratorBalancedBatchSampler)...")

    from preprocessing.face_alignment import save_rgb_crop

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "frames_manifest.csv")
        splits_path = os.path.join(tmp_dir, "splits.json")

        # Save a mock splits.json metadata block
        with open(splits_path, "w", encoding="utf-8") as sf:
            json.dump({"metadata": {"holdout_generator": "NeuralTextures"}, "splits": {}}, sf)

        generators = [
            ("Real", 0, 12),
            ("Deepfakes", 1, 8),
            ("Face2Face", 1, 8),
            ("FaceSwap", 1, 8),
        ]

        rows: List[Dict] = []
        dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        for gen_name, label, count in generators:
            for i in range(count):
                rel_img = f"crops/train/{gen_name}/vid_{i:03d}_frame0000.jpg"
                abs_img = os.path.join(tmp_dir, rel_img)
                save_rgb_crop(dummy_img, abs_img)
                rows.append(
                    {
                        "image_path": rel_img,
                        "split": "train",
                        "label": label,
                        "video_id": f"{gen_name}/vid_{i:03d}",
                        "identity_ids": f"{i:03d}",
                        "generator_name": gen_name,
                    }
                )

        with open(manifest_path, "w", encoding="utf-8") as mf:
            writer = csv.DictWriter(
                mf,
                fieldnames=["image_path", "split", "label", "video_id", "identity_ids", "generator_name"],
            )
            writer.writeheader()
            writer.writerows(rows)

        dataset = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=tmp_dir,
            split="train",
            splits_path=splits_path,
            image_size=224,
        )
        assert len(dataset) == 36, f"Expected 36 dataset items, got {len(dataset)}"

        sample0 = dataset[0]
        img_shape = tuple(sample0["image"].shape)
        assert img_shape == (3, 224, 224), f"Expected normalized image shape (3, 224, 224), got {img_shape}"

        # Batch size 12 -> 6 Real (50%) and 6 Fake (2 per seen training generator)
        sampler = GeneratorBalancedBatchSampler(dataset, batch_size=12, num_batches=4, seed=42)
        for batch_indices in sampler:
            assert len(batch_indices) == 12
            real_count = 0
            gen_counts: Dict[str, int] = defaultdict(int)
            for idx in batch_indices:
                rec = dataset.samples[idx]
                if rec["label"] == 0:
                    real_count += 1
                else:
                    gen_counts[rec["generator_name"]] += 1

            assert real_count == 6, f"Expected 6 Real crops (50%), got {real_count}"
            assert gen_counts["Deepfakes"] == 2 and gen_counts["Face2Face"] == 2 and gen_counts["FaceSwap"] == 2, (
                f"Expected equal Fake generator distribution (2 each), got {dict(gen_counts)}"
            )

        # Also verify custom --splits_path rotation (e.g. splits_holdout_faceswap.json) overrides
        # the pre-existing 'split' column in frames_manifest.csv:
        faceswap_splits_path = os.path.join(tmp_dir, "splits_holdout_faceswap.json")
        rot_splits = {"train": [], "val": [], "test_closed": [], "test_unseen": []}
        for r in rows:
            vid = r["video_id"]
            gen = r["generator_name"]
            if gen == "FaceSwap":
                rot_splits["test_unseen"].append({"video_id": vid, "generator_name": gen})
            else:
                rot_splits["train"].append({"video_id": vid, "generator_name": gen})
        with open(faceswap_splits_path, "w", encoding="utf-8") as fsf:
            json.dump({"metadata": {"holdout_generator": "FaceSwap"}, "splits": rot_splits}, fsf)

        train_rot_ds = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=tmp_dir,
            split="train",
            splits_path=faceswap_splits_path,
            image_size=224,
        )
        unseen_rot_ds = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=tmp_dir,
            split="test_unseen",
            splits_path=faceswap_splits_path,
            image_size=224,
            include_closed_real_in_unseen=False,
        )
        assert all(s["generator_name"] != "FaceSwap" for s in train_rot_ds.samples), (
            "FaceSwap leaked into train split under splits_holdout_faceswap.json!"
        )
        assert len(unseen_rot_ds) == 8 and all(s["generator_name"] == "FaceSwap" for s in unseen_rot_ds.samples), (
            "test_unseen did not load FaceSwap samples strictly from splits_holdout_faceswap.json!"
        )

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] DeepfakeDataset (3, 224, 224) + GeneratorBalancedBatchSampler "
        f"(50% Real, 50% evenly split across Deepfakes/Face2Face/FaceSwap) verified in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZeroFake dataset loader and generator-balanced batch sampler.")
    parser.add_argument("--manifest_path", type=str, default="datasets/processed/frames_manifest.csv")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dataset & sampler verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] DeepfakeDataset and GeneratorBalancedBatchSampler ready. Run with --dry-run to test.")


if __name__ == "__main__":
    main()
