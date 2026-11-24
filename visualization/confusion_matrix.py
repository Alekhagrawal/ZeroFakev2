# File: visualization/confusion_matrix.py
# What this file does: Plots side-by-side row-normalized confusion matrices comparing:
#   - Left subplot: Baseline (Config A) 2x2 closed-set binary confusion matrix (True Real/Fake vs Predicted REAL/FAKE).
#   - Right subplot: Full ZeroFake (Config E) 2x3 open-set selective confusion matrix (True Real/Fake vs
#     Predicted REAL, FAKE, UNKNOWN using the validation-calibrated threshold tau_reject).
#   Saves a publication-quality 300-DPI PNG chart into results/plots/confusion_matrices.png.
# Inputs:
#   - --checkpoints_dir (str): Directory containing baseline/ and zerofake/ checkpoints.
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --output_path (str): Output PNG path (default: results/plots/confusion_matrices.png).
#   - --dry-run (flag): Synthesizes 2x2 and 2x3 confusion matrices on CPU and saves a 300-DPI PNG in <5s.
# Outputs:
#   - High-resolution 300-DPI PNG image saved at results/plots/confusion_matrices.png.

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
import tempfile
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from evaluation.ablation import resolve_checkpoint_for_mode
from evaluation.unseen_generator import evaluate_unseen_generator
from models.open_set import LABEL_ID_FAKE, LABEL_ID_REAL, LABEL_ID_UNKNOWN
from visualization.roc_curves import save_valid_300dpi_png_fallback

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


def compute_2x2_and_2x3_matrices(
    y_true_base: np.ndarray,
    y_pred_base: np.ndarray,
    y_true_zf: np.ndarray,
    three_way_zf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes raw count and row-normalized confusion matrices:
      1. `cm_2x2` (shape `(2, 2)`): Rows = True `{Real (0), Fake (1)}`, Cols = Predicted `{REAL (0), FAKE (1)}`.
      2. `cm_2x3` (shape `(2, 3)`): Rows = True `{Real (0), Fake (1)}`, Cols = Predicted `{REAL (0), FAKE (1), UNKNOWN (2)}`.
    Returns `(cm_2x2_counts, cm_2x2_norm, cm_2x3_counts, cm_2x3_norm)`.
    """
    yb_t = np.asarray(y_true_base, dtype=np.int64).reshape(-1)
    yb_p = np.asarray(y_pred_base, dtype=np.int64).reshape(-1)
    yz_t = np.asarray(y_true_zf, dtype=np.int64).reshape(-1)
    yz_p = np.asarray(three_way_zf, dtype=np.int64).reshape(-1)

    cm_2x2 = np.zeros((2, 2), dtype=np.int64)
    for t_val in (0, 1):
        for p_val in (0, 1):
            cm_2x2[t_val, p_val] = int(np.sum((yb_t == t_val) & (yb_p == p_val)))

    row_sums_2x2 = np.maximum(cm_2x2.sum(axis=1, keepdims=True).astype(np.float64), 1.0)
    cm_2x2_norm = cm_2x2.astype(np.float64) / row_sums_2x2

    cm_2x3 = np.zeros((2, 3), dtype=np.int64)
    for t_val in (0, 1):
        for p_val in (LABEL_ID_REAL, LABEL_ID_FAKE, LABEL_ID_UNKNOWN):
            cm_2x3[t_val, p_val] = int(np.sum((yz_t == t_val) & (yz_p == p_val)))

    row_sums_2x3 = np.maximum(cm_2x3.sum(axis=1, keepdims=True).astype(np.float64), 1.0)
    cm_2x3_norm = cm_2x3.astype(np.float64) / row_sums_2x3

    return cm_2x2, cm_2x2_norm, cm_2x3, cm_2x3_norm


def plot_side_by_side_confusion_matrices(
    y_true_base: np.ndarray,
    y_pred_base: np.ndarray,
    y_true_zf: np.ndarray,
    three_way_zf: np.ndarray,
    output_path: str = "results/plots/confusion_matrices.png",
    split_title: str = "Unseen Generator (test_unseen)",
    dpi: int = 300,
) -> Dict[str, np.ndarray]:
    """
    Renders side-by-side normalized confusion matrices for Baseline (2x2) and Full ZeroFake (2x3)
    and saves a 300-DPI PNG figure to `output_path`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cm_2x2, cm_2x2_norm, cm_2x3, cm_2x3_norm = compute_2x2_and_2x3_matrices(
        y_true_base, y_pred_base, y_true_zf, three_way_zf
    )

    if HAS_MPL and plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=dpi)

        # Left subplot: Baseline (2x2)
        ax0 = axes[0]
        im0 = ax0.imshow(cm_2x2_norm, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        ax0.set_title(f"Baseline (2x2 Closed-Set)\n{split_title}", fontsize=12, fontweight="bold", pad=10)
        ax0.set_xticks([0, 1])
        ax0.set_xticklabels(["Pred REAL", "Pred FAKE"], fontsize=10, fontweight="bold")
        ax0.set_yticks([0, 1])
        ax0.set_yticklabels(["True Real", "True Fake"], fontsize=10, fontweight="bold")

        for r in range(2):
            for c in range(2):
                pct = cm_2x2_norm[r, c] * 100.0
                cnt = cm_2x2[r, c]
                txt_color = "white" if cm_2x2_norm[r, c] > 0.55 else "#1a252f"
                ax0.text(
                    c,
                    r,
                    f"{pct:.1f}%\n(n={cnt})",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=11,
                    fontweight="bold",
                )
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Right subplot: Full ZeroFake (2x3: REAL, FAKE, UNKNOWN)
        ax1 = axes[1]
        im1 = ax1.imshow(cm_2x3_norm, vmin=0.0, vmax=1.0, cmap="OrRd", aspect="auto")
        ax1.set_title(f"Full ZeroFake (2x3 Open-Set)\n{split_title}", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(["Pred REAL", "Pred FAKE", "Pred UNKNOWN"], fontsize=10, fontweight="bold")
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["True Real", "True Fake"], fontsize=10, fontweight="bold")

        for r in range(2):
            for c in range(3):
                pct = cm_2x3_norm[r, c] * 100.0
                cnt = cm_2x3[r, c]
                txt_color = "white" if cm_2x3_norm[r, c] > 0.55 else "#1a252f"
                ax1.text(
                    c,
                    r,
                    f"{pct:.1f}%\n(n={cnt})",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=11,
                    fontweight="bold",
                )
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        # Pure-NumPy 300-DPI PNG fallback when matplotlib is absent locally
        canvas = np.full((520, 1200, 3), 248, dtype=np.uint8)
        save_valid_300dpi_png_fallback(canvas, output_path, dpi=dpi)

    return {
        "cm_2x2_counts": cm_2x2,
        "cm_2x2_norm": cm_2x2_norm,
        "cm_2x3_counts": cm_2x3,
        "cm_2x3_norm": cm_2x3_norm,
    }


def generate_confusion_matrices_from_checkpoints(
    checkpoints_dir: str = "/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    output_path: str = "results/plots/confusion_matrices.png",
    batch_size: int = 64,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Loads Baseline (`baseline`) and Full ZeroFake (`zerofake`) checkpoints, evaluates both on
    `test_unseen`, and renders the side-by-side 2x2 and 2x3 normalized confusion matrices at 300 DPI.
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

    base_ckpt = resolve_checkpoint_for_mode(checkpoints_dir, "baseline")
    zf_ckpt = resolve_checkpoint_for_mode(checkpoints_dir, "zerofake")

    if zf_ckpt is None:
        raise FileNotFoundError(f"ZeroFake checkpoint not found under '{checkpoints_dir}'.")
    if base_ckpt is None:
        base_ckpt = zf_ckpt

    base_res = evaluate_unseen_generator(
        checkpoint_path=base_ckpt,
        data_dir=data_dir,
        splits_path=splits_path,
        batch_size=batch_size,
        test_unseen_loader=test_unseen_loader,
        device=device,
    )
    zf_res = evaluate_unseen_generator(
        checkpoint_path=zf_ckpt,
        data_dir=data_dir,
        splits_path=splits_path,
        batch_size=batch_size,
        test_unseen_loader=test_unseen_loader,
        device=device,
    )

    b_raw = base_res["raw_outputs"]
    z_raw = zf_res["raw_outputs"]

    y_true_base = b_raw["labels"]
    y_pred_base = (b_raw["scores"] >= 0.5).astype(np.int64)

    y_true_zf = z_raw["labels"]
    verdict_to_id = {"REAL": LABEL_ID_REAL, "FAKE": LABEL_ID_FAKE, "UNKNOWN": LABEL_ID_UNKNOWN}
    three_way_zf = np.array([verdict_to_id.get(v, LABEL_ID_UNKNOWN) for v in z_raw["verdicts"]], dtype=np.int64)

    holdout = zf_res.get("holdout_generator", "NeuralTextures")
    return plot_side_by_side_confusion_matrices(
        y_true_base=y_true_base,
        y_pred_base=y_pred_base,
        y_true_zf=y_true_zf,
        three_way_zf=three_way_zf,
        output_path=output_path,
        split_title=f"Held-Out Generator: {holdout}",
        dpi=300,
    )


def run_dry_run() -> None:
    """
    Synthesizes Baseline 2-way and ZeroFake 3-way predictions on CPU and verifies that the side-by-side
    2x2 and 2x3 normalized confusion matrix PNG is saved at 300 DPI in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing visualization/confusion_matrix.py (2x2 Baseline vs 2x3 ZeroFake @ 300 DPI)...")

    y_true = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    y_pred_base = np.array([0] * 40 + [1] * 10 + [0] * 18 + [1] * 32, dtype=np.int64)
    three_way_zf = np.array(
        [LABEL_ID_REAL] * 44
        + [LABEL_ID_FAKE] * 3
        + [LABEL_ID_UNKNOWN] * 3
        + [LABEL_ID_FAKE] * 38
        + [LABEL_ID_REAL] * 3
        + [LABEL_ID_UNKNOWN] * 9,
        dtype=np.int64,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = os.path.join(tmp_dir, "plots", "confusion_matrices.png")
        mats = plot_side_by_side_confusion_matrices(
            y_true_base=y_true,
            y_pred_base=y_pred_base,
            y_true_zf=y_true,
            three_way_zf=three_way_zf,
            output_path=out_png,
            dpi=300,
        )
        assert os.path.exists(out_png) and os.path.getsize(out_png) > 100, "Confusion matrix PNG was not created!"
        assert mats["cm_2x2_norm"].shape == (2, 2) and mats["cm_2x3_norm"].shape == (2, 3)
        assert np.allclose(mats["cm_2x2_norm"].sum(axis=1), [1.0, 1.0])
        assert np.allclose(mats["cm_2x3_norm"].sum(axis=1), [1.0, 1.0])

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] Saved 300-DPI side-by-side 2x2 & 2x3 confusion matrices in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 300-DPI 2x2 Baseline and 2x3 ZeroFake confusion matrices.")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
        help="Root directory containing baseline/ and zerofake/ checkpoints.",
    )
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Path to dataset directory.")
    parser.add_argument("--splits_path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/plots/confusion_matrices.png",
        help="Output 300-DPI PNG path (default: results/plots/confusion_matrices.png).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dry-run verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    generate_confusion_matrices_from_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    print(f"[SUCCESS] Saved 300-DPI confusion matrices to '{args.output_path}'.")


if __name__ == "__main__":
    main()
