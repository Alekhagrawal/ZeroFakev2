# File: evaluation/robustness.py
# What this file does: Stress-tests a trained ZeroFake checkpoint on degraded versions of 'test_closed'
#   and 'test_unseen' by applying 11 real-world visual perturbations on the fly in memory (never
#   overwriting saved face images on disk):
#     0. Clean (no distortion baseline)
#     1. JPEG compression at quality levels 90, 75, and 50
#     2. Gaussian blur with kernel sizes 3x3 (sigma=1.0) and 5x5 (sigma=2.0)
#     3. Gaussian sensor noise (std=0.03 and std=0.06)
#     4. Downsampling to 112x112 and upsampling back to 224x224
#     5. Brightness shifts (+20% and -20%) and contrast shifts (+20% and -20%)
#   Saves all degradation results (Clean vs. each distortion level, reporting ROC-AUC, EER, F1,
#   Selective Accuracy, and Coverage) into results/robustness_summary.csv.
# Inputs:
#   - --checkpoint (str): Path to trained ZeroFake checkpoint (.pt).
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --output_csv (str): Output CSV path (default: results/robustness_summary.csv).
#   - --batch_size (int): Evaluation mini-batch size (default: 64).
#   - --dry-run (flag): Synthesizes a dummy checkpoint and mock test_closed/test_unseen batches on CPU,
#     runs all in-memory distortion operators, and verifies results/robustness_summary.csv in <5 seconds.
# Outputs:
#   - results/robustness_summary.csv containing per-split and per-perturbation ROC-AUC, EER, F1,
#     Selective Accuracy, and Coverage.

import sys
from pathlib import Path

# Insert repository root at sys.path[0] so Colab's pre-installed Hugging Face 'datasets' package
# never shadows ZeroFake's local 'datasets.deepfake_dataset' module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if "datasets" in sys.modules and not hasattr(sys.modules["datasets"], "__path__"):
    del sys.modules["datasets"]

import argparse
import csv
import io
import os
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import IMAGENET_MEAN, IMAGENET_STD, DeepfakeDataset
from evaluation.closed_set import (
    compute_binary_classification_metrics,
    load_model_and_scorer_from_checkpoint,
)
from evaluation.open_set import compute_selective_open_set_metrics
from models.open_set import OpenSetUncertaintyScorer
from models.zerofake_net import ZeroFakeNet
from training.train import cast_outputs_to_float32, get_model_state, save_checkpoint_file

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
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


ROBUSTNESS_CSV_COLUMNS = [
    "split",
    "perturbation_family",
    "perturbation_name",
    "severity_param",
    "roc_auc",
    "eer",
    "f1",
    "accuracy",
    "selective_accuracy",
    "coverage",
    "coverage_pct",
    "tau_reject",
]


def denormalize_batch_to_uint8_hwc(batch_chw: np.ndarray) -> np.ndarray:
    """
    Converts an ImageNet-normalized batch `(B, 3, H, W)` float32 tensor in memory back into
    `(B, H, W, 3)` RGB uint8 values in `[0, 255]` so spatial/codec perturbations can be applied on the fly.
    """
    mean = IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std = IMAGENET_STD.reshape(1, 3, 1, 1)
    rgb_01 = np.clip(batch_chw * std + mean, 0.0, 1.0)
    rgb_hwc = np.transpose(rgb_01, (0, 2, 3, 1))
    return np.round(rgb_hwc * 255.0).astype(np.uint8)


def normalize_uint8_hwc_to_batch_chw(rgb_hwc_uint8: np.ndarray) -> np.ndarray:
    """
    Converts a perturbed `(B, H, W, 3)` uint8 RGB batch back into an ImageNet-normalized `(B, 3, H, W)` float32 array.
    """
    chw_01 = np.transpose(rgb_hwc_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2))
    mean = IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std = IMAGENET_STD.reshape(1, 3, 1, 1)
    return ((chw_01 - mean) / std).astype(np.float32)


def apply_jpeg_compression_in_memory(rgb_uint8: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    1. JPEG compression at quality levels 90, 75, and 50 applied strictly in RAM buffer.
    Uses OpenCV `imencode('.jpg')`, Pillow `BytesIO`, or an 8x8 DCT block quantization approximation fallback.
    """
    q = int(np.clip(quality, 5, 100))
    if HAS_CV2:
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec_bgr is not None:
                return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    if HAS_PIL:
        bio = io.BytesIO()
        Image.fromarray(rgb_uint8).save(bio, format="JPEG", quality=q)
        bio.seek(0)
        with Image.open(bio) as dec_img:
            return np.asarray(dec_img.convert("RGB"), dtype=np.uint8)

    # Pure-NumPy block quantization fallback for minimal environments without cv2/Pillow
    step = max(1.0, (100.0 - float(q)) * 0.35)
    quantized = np.round(rgb_uint8.astype(np.float32) / step) * step
    return np.clip(quantized, 0.0, 255.0).astype(np.uint8)


def _make_gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    half = ksize // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    kern = np.exp(-(x ** 2) / (2.0 * (sigma ** 2)))
    return kern / np.sum(kern)


def apply_gaussian_blur_in_memory(rgb_uint8: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    2. Gaussian blur with kernel sizes 3x3 (sigma=1.0) and 5x5 (sigma=2.0) in memory.
    """
    if HAS_CV2:
        return cv2.GaussianBlur(rgb_uint8, (int(ksize), int(ksize)), sigmaX=float(sigma), sigmaY=float(sigma))

    # Separable 1D Gaussian convolution in pure NumPy
    k1d = _make_gaussian_kernel_1d(int(ksize), float(sigma))
    pad = int(ksize) // 2
    img_f = rgb_uint8.astype(np.float32)

    # Horizontal pass
    padded_h = np.pad(img_f, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    horiz = np.zeros_like(img_f)
    for i, w_val in enumerate(k1d):
        horiz += w_val * padded_h[:, i : i + img_f.shape[1], :]

    # Vertical pass
    padded_v = np.pad(horiz, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    vert = np.zeros_like(img_f)
    for i, w_val in enumerate(k1d):
        vert += w_val * padded_v[i : i + img_f.shape[0], :, :]

    return np.clip(np.round(vert), 0.0, 255.0).astype(np.uint8)


def apply_gaussian_noise_in_memory(
    rgb_uint8: np.ndarray,
    std: float = 0.03,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    3. Gaussian sensor noise with `std=0.03` and `std=0.06` (in normalized `[0, 1]` pixel range) in memory.
    """
    gen = rng if rng is not None else np.random.default_rng(42)
    img_01 = rgb_uint8.astype(np.float32) / 255.0
    noise = gen.normal(0.0, float(std), size=img_01.shape).astype(np.float32)
    noisy_01 = np.clip(img_01 + noise, 0.0, 1.0)
    return np.round(noisy_01 * 255.0).astype(np.uint8)


def apply_downsample_112_upsample_224(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    4. Downsamples face crop to `112x112` and upsamples back to `224x224` in memory.
    """
    h, w = rgb_uint8.shape[:2]
    low_h, low_w = max(1, h // 2), max(1, w // 2)

    if HAS_CV2:
        small = cv2.resize(rgb_uint8, (low_w, low_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if HAS_PIL:
        pil_img = Image.fromarray(rgb_uint8)
        small = pil_img.resize((low_w, low_h), Image.BILINEAR)
        restored = small.resize((w, h), Image.BILINEAR)
        return np.asarray(restored, dtype=np.uint8)

    # Pure-NumPy 2x2 average pooling downsample -> 2x2 repeat upsample
    small = 0.25 * (
        rgb_uint8[0::2, 0::2].astype(np.float32)
        + rgb_uint8[1::2, 0::2].astype(np.float32)
        + rgb_uint8[0::2, 1::2].astype(np.float32)
        + rgb_uint8[1::2, 1::2].astype(np.float32)
    )
    restored = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    return np.clip(np.round(restored[:h, :w]), 0.0, 255.0).astype(np.uint8)


def apply_brightness_contrast_shift(
    rgb_uint8: np.ndarray,
    brightness_delta: float = 0.0,
    contrast_factor: float = 1.0,
) -> np.ndarray:
    """
    5. Brightness and contrast shifts (+/- 20%) in memory:
       - `brightness_delta` in `{-0.20, +0.20}` shifts pixel intensity in `[0, 1]` by `+/- 0.20`.
       - `contrast_factor` in `{0.80, 1.20}` scales pixel contrast around the per-image mean by `+/- 20%`.
    """
    img_01 = rgb_uint8.astype(np.float32) / 255.0
    if contrast_factor != 1.0:
        mean_val = np.mean(img_01, axis=(0, 1), keepdims=True)
        img_01 = (img_01 - mean_val) * float(contrast_factor) + mean_val
    if brightness_delta != 0.0:
        img_01 = img_01 + float(brightness_delta)
    shifted_01 = np.clip(img_01, 0.0, 1.0)
    return np.round(shifted_01 * 255.0).astype(np.uint8)


def get_perturbation_suite() -> List[Tuple[str, str, str, Callable[[np.ndarray], np.ndarray]]]:
    """
    Returns the complete list of 13 evaluation conditions (Clean + 12 distortion levels across the 5 families):
      `(perturbation_family, perturbation_name, severity_param, transform_fn)`
    """
    rng_03 = np.random.default_rng(303)
    rng_06 = np.random.default_rng(606)

    return [
        ("clean", "clean", "none", lambda img: img),
        ("jpeg", "jpeg_q90", "quality=90", lambda img: apply_jpeg_compression_in_memory(img, quality=90)),
        ("jpeg", "jpeg_q75", "quality=75", lambda img: apply_jpeg_compression_in_memory(img, quality=75)),
        ("jpeg", "jpeg_q50", "quality=50", lambda img: apply_jpeg_compression_in_memory(img, quality=50)),
        ("gaussian_blur", "blur_3x3_s1.0", "3x3,sigma=1.0", lambda img: apply_gaussian_blur_in_memory(img, ksize=3, sigma=1.0)),
        ("gaussian_blur", "blur_5x5_s2.0", "5x5,sigma=2.0", lambda img: apply_gaussian_blur_in_memory(img, ksize=5, sigma=2.0)),
        ("gaussian_noise", "noise_std0.03", "std=0.03", lambda img: apply_gaussian_noise_in_memory(img, std=0.03, rng=rng_03)),
        ("gaussian_noise", "noise_std0.06", "std=0.06", lambda img: apply_gaussian_noise_in_memory(img, std=0.06, rng=rng_06)),
        ("resize", "resize_112_to_224", "112x112->224x224", apply_downsample_112_upsample_224),
        ("brightness_contrast", "brightness_plus20", "brightness=+20%", lambda img: apply_brightness_contrast_shift(img, brightness_delta=0.20)),
        ("brightness_contrast", "brightness_minus20", "brightness=-20%", lambda img: apply_brightness_contrast_shift(img, brightness_delta=-0.20)),
        ("brightness_contrast", "contrast_plus20", "contrast=+20%", lambda img: apply_brightness_contrast_shift(img, contrast_factor=1.20)),
        ("brightness_contrast", "contrast_minus20", "contrast=-20%", lambda img: apply_brightness_contrast_shift(img, contrast_factor=0.80)),
    ]


def perturb_normalized_batch_in_memory(
    batch_images: Any,
    transform_fn: Callable[[np.ndarray], np.ndarray],
) -> Any:
    """
    Takes a batch of ImageNet-normalized images `(B, 3, 224, 224)` (Tensor or ndarray),
    denormalizes to uint8 RGB in memory, applies `transform_fn` to each frame, and re-normalizes.
    Never modifies any file on disk.
    """
    if HAS_TORCH and isinstance(batch_images, torch.Tensor):
        orig_device = batch_images.device
        arr_chw = batch_images.detach().cpu().float().numpy()
        is_tensor = True
    else:
        arr_chw = np.asarray(batch_images, dtype=np.float32)
        orig_device = None
        is_tensor = False

    hwc_uint8 = denormalize_batch_to_uint8_hwc(arr_chw)
    perturbed_list = [transform_fn(hwc_uint8[i]) for i in range(hwc_uint8.shape[0])]
    perturbed_hwc = np.stack(perturbed_list, axis=0)
    norm_chw = normalize_uint8_hwc_to_batch_chw(perturbed_hwc)

    if is_tensor and HAS_TORCH:
        return torch.from_numpy(norm_chw).to(orig_device)
    return norm_chw


def evaluate_loader_under_perturbation(
    model: ZeroFakeNet,
    data_loader: Iterable,
    scorer: OpenSetUncertaintyScorer,
    transform_fn: Callable[[np.ndarray], np.ndarray],
    device: str = "cpu",
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Evaluates `model` and `scorer` across `data_loader` after applying `transform_fn` on the fly in memory.
    Returns ROC-AUC, EER, F1, Accuracy, Selective Accuracy, and Coverage.
    """
    if HAS_TORCH and hasattr(model, "eval"):
        model.eval()

    amp_enabled = bool(use_amp and HAS_TORCH and str(device).startswith("cuda") and torch.cuda.is_available())

    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_u_scores: List[np.ndarray] = []

    for batch in data_loader:
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch[0], batch[1]

        perturbed_images = perturb_normalized_batch_in_memory(images, transform_fn)

        if HAS_TORCH and isinstance(perturbed_images, torch.Tensor):
            images_dev = perturbed_images.to(device, non_blocking=True)
            with torch.no_grad():
                if amp_enabled:
                    with torch.amp.autocast("cuda"):
                        raw_out = model(images_dev)
                else:
                    raw_out = model(images_dev)
                outputs = cast_outputs_to_float32(raw_out)

            y_np = labels.detach().cpu().numpy().reshape(-1).astype(np.int64)
            scores_np = outputs["evidence_fake"].detach().cpu().numpy().reshape(-1).astype(np.float64)
        else:
            x_np = np.asarray(perturbed_images, dtype=np.float32)
            y_np = np.asarray(labels, dtype=np.int64).reshape(-1)
            outputs = model(x_np)
            scores_np = np.asarray(outputs["evidence_fake"], dtype=np.float64).reshape(-1)

        pred_open = scorer.predict(outputs, open_set_enabled=True)
        u_np = np.asarray(pred_open["u_score"], dtype=np.float64).reshape(-1)

        all_labels.append(y_np)
        all_scores.append(scores_np)
        all_u_scores.append(u_np)

    y_all = np.concatenate(all_labels, axis=0)
    s_all = np.concatenate(all_scores, axis=0)
    u_all = np.concatenate(all_u_scores, axis=0)

    bin_metrics = compute_binary_classification_metrics(y_all, s_all, threshold=0.5)
    open_metrics = compute_selective_open_set_metrics(
        labels=y_all,
        scores=s_all,
        u_scores=u_all,
        tau_reject=scorer.tau_reject,
    )

    return {
        "roc_auc": round(bin_metrics["roc_auc"], 4),
        "eer": round(bin_metrics["eer"], 4),
        "f1": round(bin_metrics["f1"], 4),
        "accuracy": round(bin_metrics["accuracy"], 4),
        "selective_accuracy": round(open_metrics["selective_accuracy"], 4),
        "coverage": round(open_metrics["coverage"], 4),
        "coverage_pct": round(open_metrics["coverage_pct"], 2),
        "tau_reject": round(float(scorer.tau_reject), 4) if np.isfinite(scorer.tau_reject) else float("inf"),
    }


def run_robustness_suite(
    checkpoint_path: str,
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    output_csv: str = "results/robustness_summary.csv",
    batch_size: int = 64,
    test_closed_loader: Optional[Iterable] = None,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Runs all 13 degradation conditions (Clean + JPEG Q90/75/50 + Blur 3x3/5x5 + Noise 0.03/0.06 +
    Downsample 112x112->224x224 + Brightness/Contrast +/-20%) on both `test_closed` and `test_unseen`,
    and saves the complete table to `results/robustness_summary.csv`.
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    model, scorer, _ = load_model_and_scorer_from_checkpoint(checkpoint_path, device=device)

    if test_closed_loader is None or test_unseen_loader is None:
        manifest_path = os.path.join(data_dir, "frames_manifest.csv")
        resolved_splits = splits_path if splits_path else os.path.join(data_dir, "splits.json")
        if test_closed_loader is None:
            ds_closed = DeepfakeDataset(
                manifest_path=manifest_path,
                root_dir=data_dir,
                split="test_closed",
                splits_path=resolved_splits,
                augment=False,
            )
            test_closed_loader = DataLoader(ds_closed, batch_size=batch_size, shuffle=False, num_workers=2)
        if test_unseen_loader is None:
            ds_unseen = DeepfakeDataset(
                manifest_path=manifest_path,
                root_dir=data_dir,
                split="test_unseen",
                splits_path=resolved_splits,
                augment=False,
                include_closed_real_in_unseen=True,
            )
            test_unseen_loader = DataLoader(ds_unseen, batch_size=batch_size, shuffle=False, num_workers=2)

    perturbations = get_perturbation_suite()
    rows: List[Dict[str, Any]] = []

    for split_name, loader in (("test_closed", test_closed_loader), ("test_unseen", test_unseen_loader)):
        for family, p_name, sev_param, fn in perturbations:
            res = evaluate_loader_under_perturbation(
                model=model,
                data_loader=loader,
                scorer=scorer,
                transform_fn=fn,
                device=device,
            )
            row = {
                "split": split_name,
                "perturbation_family": family,
                "perturbation_name": p_name,
                "severity_param": sev_param,
                **res,
            }
            rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROBUSTNESS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def run_dry_run() -> None:
    """
    Synthesizes a dummy checkpoint and mock `test_closed` and `test_unseen` batches on CPU,
    runs all 13 degradation conditions (Clean + 12 perturbations across 5 families) on both splits,
    and verifies `results/robustness_summary.csv` in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing evaluation/robustness.py (Clean + JPEG 90/75/50 + Blur + Noise + 112x112 + Brightness/Contrast)...")

    net = ZeroFakeNet(mode="zerofake", pretrained=False)
    scorer = OpenSetUncertaintyScorer()

    rng = np.random.default_rng(55)
    val_batches = []
    closed_batches = []
    unseen_batches = []

    for _ in range(1):
        imgs_v = rng.normal(0.0, 1.0, size=(6, 3, 224, 224)).astype(np.float32)
        imgs_c = rng.normal(0.0, 1.0, size=(6, 3, 224, 224)).astype(np.float32)
        imgs_u = rng.normal(0.2, 1.1, size=(6, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        if HAS_TORCH:
            val_batches.append({"image": torch.from_numpy(imgs_v), "label": torch.from_numpy(lbls), "split": "val"})
            closed_batches.append({"image": torch.from_numpy(imgs_c), "label": torch.from_numpy(lbls)})
            unseen_batches.append({"image": torch.from_numpy(imgs_u), "label": torch.from_numpy(lbls)})
        else:
            val_batches.append({"image": imgs_v, "label": lbls, "split": "val"})
            closed_batches.append({"image": imgs_c, "label": lbls})
            unseen_batches.append({"image": imgs_u, "label": lbls})

    scorer.calibrate(val_batches, model=net, device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "best_model.pt")
        out_csv = os.path.join(tmp_dir, "results", "robustness_summary.csv")
        save_checkpoint_file(
            {
                "epoch": 14,
                "mode": "zerofake",
                "holdout_generator": "NeuralTextures",
                "model_state_dict": get_model_state(net),
                "open_set_state": scorer.state_dict(),
                "config": {"embedding_dim": 128, "num_fake_prototypes": 6},
            },
            ckpt_path,
        )

        rows = run_robustness_suite(
            checkpoint_path=ckpt_path,
            output_csv=out_csv,
            test_closed_loader=closed_batches,
            test_unseen_loader=unseen_batches,
            device="cpu",
        )

        assert os.path.exists(out_csv), "robustness_summary.csv was not created!"
        # 2 splits * 13 conditions = 26 rows
        assert len(rows) == 26, f"Expected 26 rows (2 splits x 13 conditions), got {len(rows)}"
        with open(out_csv, "r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            assert reader.fieldnames == ROBUSTNESS_CSV_COLUMNS

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Evaluated {len(rows)} split-perturbation combinations "
        f"(test_closed & test_unseen x 13 conditions) and saved robustness_summary.csv in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZeroFake in-memory robustness stress tests.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints/zerofake/best_model.pt",
        help="Path to trained ZeroFake checkpoint (.pt).",
    )
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Path to dataset directory.")
    parser.add_argument("--splits_path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/robustness_summary.csv",
        help="Output CSV path (default: results/robustness_summary.csv).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dry-run verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    rows = run_robustness_suite(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
    )
    print(f"[SUCCESS] Saved {len(rows)}-row robustness stress-test summary to '{args.output_csv}':")
    for r in rows:
        print(
            f"  [{r['split']:<11}] {r['perturbation_name']:<18} ({r['severity_param']:<17}) | "
            f"ROC-AUC={r['roc_auc']:.4f} | SelAcc={r['selective_accuracy']:.4f} | Cov={r['coverage_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
