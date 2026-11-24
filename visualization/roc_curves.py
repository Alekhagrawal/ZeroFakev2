# File: visualization/roc_curves.py
# What this file does: Plots overlaid Receiver Operating Characteristic (ROC) curves comparing all four
#   ablation models (Baseline vs. Contrastive vs. Single-Prototype vs. Full ZeroFake) on the held-out
#   unseen generator ('test_unseen'), displaying each model's exact ROC-AUC value in the legend and
#   saving a publication-quality 300-DPI PNG chart into results/plots/roc_comparison.png.
# Inputs:
#   - --checkpoints_dir (str): Directory containing baseline/, contrastive/, single_proto/, and zerofake/ checkpoints.
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --output_path (str): Output PNG path (default: results/plots/roc_comparison.png).
#   - --dry-run (flag): Synthesizes prediction scores for all 4 modes on CPU and saves a 300-DPI PNG chart in <5s.
# Outputs:
#   - High-resolution 300-DPI PNG image saved at results/plots/roc_comparison.png.

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
import os
import struct
import tempfile
import time
import zlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from evaluation.ablation import resolve_checkpoint_for_mode
from evaluation.unseen_generator import evaluate_unseen_generator
from models.zerofake_net import VALID_MODES
from training.validation import compute_roc_auc_and_eer

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MPL = False

try:
    from sklearn.metrics import roc_curve
    HAS_SKLEARN = True
except ImportError:
    roc_curve = None
    HAS_SKLEARN = False


MODE_DISPLAY_NAMES = {
    "baseline": "Baseline (Config A)",
    "contrastive": "Contrastive (Config B)",
    "single_proto": "Single-Prototype (Config C)",
    "zerofake": "Full ZeroFake (Config D/E)",
}

MODE_COLORS = {
    "baseline": "#7f8c8d",
    "contrastive": "#2980b9",
    "single_proto": "#e67e22",
    "zerofake": "#c0392b",
}


def compute_fpr_tpr_curve(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes `(fpr, tpr, auc_val)` arrays for plotting an ROC curve.
    Uses `scikit-learn` when available, or an exact pure-NumPy implementation.
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    auc_val, _ = compute_roc_auc_and_eer(y_true, y_score)

    if HAS_SKLEARN and roc_curve is not None and len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        return fpr, tpr, auc_val

    desc_idx = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[desc_idx]
    score_sorted = y_score[desc_idx]

    distinct_indices = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)

    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]

    tpr = tps / max(1.0, float(tps[-1]))
    fpr = fps / max(1.0, float(fps[-1]))
    return fpr, tpr, auc_val


def save_valid_300dpi_png_fallback(rgb_array: np.ndarray, output_path: str, dpi: int = 300) -> None:
    """
    Writes a valid standard PNG file with a 300-DPI `pHYs` chunk using pure Python `zlib`/`struct`
    whenever `matplotlib` is not installed on a minimal local laptop environment.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    h, w, _ = rgb_array.shape
    ppm = int(round(dpi / 0.0254))  # pixels per meter (300 DPI = 11811 ppm)

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    phys = struct.pack(">IIB", ppm, ppm, 1)

    # Filter byte 0 prepended to each scanline
    raw_scanlines = b"".join(b"\x00" + rgb_array[row].tobytes() for row in range(h))
    idat = zlib.compress(raw_scanlines, level=6)

    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + make_chunk(b"IHDR", ihdr)
        + make_chunk(b"pHYs", phys)
        + make_chunk(b"IDAT", idat)
        + make_chunk(b"IEND", b"")
    )
    with open(output_path, "wb") as f:
        f.write(png_bytes)


def plot_overlaid_roc_curves(
    mode_predictions: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "results/plots/roc_comparison.png",
    holdout_generator: str = "NeuralTextures",
    dpi: int = 300,
) -> Dict[str, float]:
    """
    Plots overlaid ROC curves for all provided ablation modes (`baseline`, `contrastive`, `single_proto`,
    `zerofake`) with ROC-AUC values in the legend, and saves a 300-DPI PNG chart to `output_path`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    auc_summary: Dict[str, float] = {}

    if HAS_MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=dpi)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color="#bdc3c7", label="Random Chance (AUC = 0.5000)")

        for mode_key in VALID_MODES:
            if mode_key not in mode_predictions:
                continue
            item = mode_predictions[mode_key]
            fpr, tpr, auc_val = compute_fpr_tpr_curve(item["labels"], item["scores"])
            auc_summary[mode_key] = auc_val

            display_name = MODE_DISPLAY_NAMES.get(mode_key, mode_key)
            color = MODE_COLORS.get(mode_key, "#2c3e50")
            lw = 2.8 if mode_key == "zerofake" else 2.0
            ax.plot(fpr, tpr, linewidth=lw, color=color, label=f"{display_name} (AUC = {auc_val:.4f})")

        ax.set_title(
            f"Unseen-Generator ROC Comparison (Held-Out: {holdout_generator})",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return auc_summary

    # Pure-NumPy 300-DPI canvas fallback when matplotlib is absent locally
    canvas = np.full((600, 750, 3), 255, dtype=np.uint8)
    for mode_key in VALID_MODES:
        if mode_key not in mode_predictions:
            continue
        item = mode_predictions[mode_key]
        fpr, tpr, auc_val = compute_fpr_tpr_curve(item["labels"], item["scores"])
        auc_summary[mode_key] = auc_val
        xs = np.clip((fpr * 600 + 75).astype(int), 0, 749)
        ys = np.clip((525 - tpr * 450).astype(int), 0, 599)
        canvas[ys, xs] = np.array([192, 57, 43], dtype=np.uint8)

    save_valid_300dpi_png_fallback(canvas, output_path, dpi=dpi)
    return auc_summary


def generate_roc_comparison_from_checkpoints(
    checkpoints_dir: str = "/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    output_path: str = "results/plots/roc_comparison.png",
    batch_size: int = 64,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluates all available ablation checkpoints (`baseline`, `contrastive`, `single_proto`, `zerofake`)
    on `test_unseen` and plots their overlaid ROC curves at 300 DPI.
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    if test_unseen_loader is None:
        manifest_path = os.path.join(data_dir, "frames_manifest.csv")
        resolved_splits = splits_path if splits_path else os.path.join(data_dir, "splits.json")
        ds_unseen = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=data_dir,
            split="test_unseen",
            splits_path=resolved_splits,
            augment=False,
            include_closed_real_in_unseen=True,
        )
        test_unseen_loader = DataLoader(ds_unseen, batch_size=batch_size, shuffle=False, num_workers=2)

    mode_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    holdout_name = "NeuralTextures"

    for mode in VALID_MODES:
        ckpt_path = resolve_checkpoint_for_mode(checkpoints_dir, mode)
        if ckpt_path is None:
            continue
        res = evaluate_unseen_generator(
            checkpoint_path=ckpt_path,
            data_dir=data_dir,
            splits_path=splits_path,
            batch_size=batch_size,
            test_unseen_loader=test_unseen_loader,
            device=device,
        )
        holdout_name = res.get("holdout_generator", holdout_name)
        raw = res["raw_outputs"]
        mode_predictions[mode] = {
            "labels": raw["labels"],
            "scores": raw["scores"],
        }

    return plot_overlaid_roc_curves(
        mode_predictions=mode_predictions,
        output_path=output_path,
        holdout_generator=holdout_name,
        dpi=300,
    )


def run_dry_run() -> None:
    """
    Synthesizes unseen-generator prediction scores for all 4 ablation modes (`baseline`, `contrastive`,
    `single_proto`, `zerofake`) on CPU and verifies that a 300-DPI PNG ROC plot is saved in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing visualization/roc_curves.py (4-mode overlaid ROC plot @ 300 DPI)...")

    rng = np.random.default_rng(42)
    labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    synth_preds = {
        "baseline": {
            "labels": labels,
            "scores": np.clip(labels * 0.35 + rng.uniform(0.1, 0.6, size=100), 0.0, 1.0),
        },
        "contrastive": {
            "labels": labels,
            "scores": np.clip(labels * 0.45 + rng.uniform(0.05, 0.55, size=100), 0.0, 1.0),
        },
        "single_proto": {
            "labels": labels,
            "scores": np.clip(labels * 0.55 + rng.uniform(0.05, 0.45, size=100), 0.0, 1.0),
        },
        "zerofake": {
            "labels": labels,
            "scores": np.clip(labels * 0.70 + rng.uniform(0.02, 0.30, size=100), 0.0, 1.0),
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = os.path.join(tmp_dir, "plots", "roc_comparison.png")
        aucs = plot_overlaid_roc_curves(
            mode_predictions=synth_preds,
            output_path=out_png,
            holdout_generator="NeuralTextures",
            dpi=300,
        )
        assert os.path.exists(out_png) and os.path.getsize(out_png) > 100, "ROC PNG file was not created!"
        assert len(aucs) == 4 and aucs["zerofake"] >= aucs["baseline"]

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Saved 300-DPI overlaid ROC curve PNG "
        f"(Baseline AUC={aucs['baseline']:.4f} vs ZeroFake AUC={aucs['zerofake']:.4f}) in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 300-DPI overlaid ROC curves across all 4 ZeroFake modes.")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
        help="Root directory containing per-mode checkpoints.",
    )
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Path to dataset directory.")
    parser.add_argument("--splits_path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/plots/roc_comparison.png",
        help="Output 300-DPI PNG path (default: results/plots/roc_comparison.png).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dry-run verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    aucs = generate_roc_comparison_from_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    print(f"[SUCCESS] Saved 300-DPI ROC comparison chart to '{args.output_path}': {aucs}")


if __name__ == "__main__":
    main()
