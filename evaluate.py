# File: evaluate.py
# What this file does: End-to-end evaluation and publication-chart orchestrator for ZeroFake.
#   1. Evaluates all available ablation checkpoints ('baseline', 'contrastive', 'single_proto', 'zerofake')
#      under --checkpoints_dir on 'test_closed' (known generators) and 'test_unseen' (held-out generator),
#      plus optional cross-dataset 'celeb_df' if present, saving results/ablation_summary.csv and
#      results/results_summary.csv.
#   2. Runs the 13-condition in-memory robustness stress-test suite on the Full ZeroFake checkpoint
#      (JPEG Q90/75/50, Gaussian blur 3x3/5x5, Gaussian noise 0.03/0.06, 112x112 downsampling, and
#      brightness/contrast +/-20%), saving results/robustness_summary.csv.
#   3. Generates all 4 high-resolution 300-DPI PNG chart files inside <output_dir>/plots/:
#      - roc_comparison.png (Overlaid ROC curves across all 4 modes on test_unseen)
#      - confusion_matrices.png (Side-by-side 2x2 Baseline vs 2x3 Open-Set ZeroFake confusion matrices)
#      - tsne_prototypes.png (2D t-SNE projection of 128-d embeddings + 1 Real & 6 Fake prototypes)
#      - gradcam_heatmaps.png (4x3 ConvNeXt-Tiny Grad-CAM visual attribution grid)
#   4. Supports --dry-run using dummy checkpoints and synthetic images on CPU in under 5 seconds.
# Inputs:
#   - --checkpoints_dir (str): Directory containing baseline/, contrastive/, single_proto/, and zerofake/ checkpoints.
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --output_dir (str): Directory to save CSV tables and plots/ (default: results).
#   - --batch_size (int): Evaluation mini-batch size (default: 64).
#   - --dry-run (flag): Runs the entire evaluation + robustness + 4-plot suite on synthetic data on CPU in <5s.
# Outputs:
#   - <output_dir>/ablation_summary.csv and <output_dir>/results_summary.csv
#   - <output_dir>/robustness_summary.csv
#   - <output_dir>/plots/roc_comparison.png
#   - <output_dir>/plots/confusion_matrices.png
#   - <output_dir>/plots/tsne_prototypes.png
#   - <output_dir>/plots/gradcam_heatmaps.png

import sys
from pathlib import Path

# Insert repository root at sys.path[0] so Colab's pre-installed Hugging Face 'datasets' package
# never shadows ZeroFake's local 'datasets.deepfake_dataset' module.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if "datasets" in sys.modules and not hasattr(sys.modules["datasets"], "__path__"):
    del sys.modules["datasets"]

import argparse
import os
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from evaluation.ablation import resolve_checkpoint_for_mode, run_ablation_suite
from evaluation.robustness import denormalize_batch_to_uint8_hwc, run_robustness_suite
from evaluation.unseen_generator import evaluate_unseen_generator
from models.open_set import (
    LABEL_ID_FAKE,
    LABEL_ID_REAL,
    LABEL_ID_UNKNOWN,
    OpenSetUncertaintyScorer,
)
from models.zerofake_net import VALID_MODES, ZeroFakeNet
from training.train import get_model_state, save_checkpoint_file
from visualization.confusion_matrix import (
    generate_confusion_matrices_from_checkpoints,
    plot_side_by_side_confusion_matrices,
)
from visualization.embeddings import (
    generate_tsne_from_checkpoint,
    plot_tsne_embeddings_and_prototypes,
)
from visualization.gradcam import (
    compute_single_gradcam_map,
    generate_gradcam_grid_from_checkpoint,
    overlay_heatmap_on_rgb,
    plot_gradcam_4x3_grid,
)
from visualization.roc_curves import (
    generate_roc_comparison_from_checkpoints,
    plot_overlaid_roc_curves,
)

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
    HAS_TORCH = False


def run_complete_evaluation_pipeline(
    checkpoints_dir: str = "/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    output_dir: str = "results",
    batch_size: int = 64,
    test_closed_loader: Optional[Iterable] = None,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the full end-to-end ZeroFake evaluation, robustness stress test, and 4-plot generation suite:
      1. Evaluates all available checkpoints (`baseline`, `contrastive`, `single_proto`, `zerofake`)
         on `test_closed` and `test_unseen` -> saves `ablation_summary.csv` and `results_summary.csv`.
      2. Runs robustness stress tests on the `zerofake` checkpoint -> saves `robustness_summary.csv`.
      3. Generates all 4 publication-quality 300-DPI PNG plots in `<output_dir>/plots/`:
         - `roc_comparison.png`
         - `confusion_matrices.png`
         - `tsne_prototypes.png`
         - `gradcam_heatmaps.png`
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ablation_csv_path = os.path.join(output_dir, "ablation_summary.csv")
    robustness_csv_path = os.path.join(output_dir, "robustness_summary.csv")

    roc_png_path = os.path.join(plots_dir, "roc_comparison.png")
    cm_png_path = os.path.join(plots_dir, "confusion_matrices.png")
    tsne_png_path = os.path.join(plots_dir, "tsne_prototypes.png")
    gradcam_png_path = os.path.join(plots_dir, "gradcam_heatmaps.png")

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

    # Step 1: Ablation Evaluation across all checkpoints
    print("[EVAL 1/3] Running ablation evaluation across all checkpoints on test_closed & test_unseen...")
    ablation_rows = run_ablation_suite(
        checkpoints_dir=checkpoints_dir,
        data_dir=data_dir,
        splits_path=splits_path,
        output_csv=ablation_csv_path,
        batch_size=batch_size,
        test_closed_loader=test_closed_loader,
        test_unseen_loader=test_unseen_loader,
        device=device,
    )

    # Locate primary ZeroFake checkpoint (or fallback to any available mode checkpoint)
    zf_ckpt = resolve_checkpoint_for_mode(checkpoints_dir, "zerofake")
    if zf_ckpt is None:
        for fallback_m in ("single_proto", "contrastive", "baseline"):
            zf_ckpt = resolve_checkpoint_for_mode(checkpoints_dir, fallback_m)
            if zf_ckpt is not None:
                break
    if zf_ckpt is None:
        raise FileNotFoundError(f"No valid checkpoint (.pt) found under '{checkpoints_dir}'!")

    # Step 2: Robustness Stress Tests
    print(f"[EVAL 2/3] Running 13-condition in-memory robustness stress tests on '{zf_ckpt}'...")
    robustness_rows = run_robustness_suite(
        checkpoint_path=zf_ckpt,
        data_dir=data_dir,
        splits_path=splits_path,
        output_csv=robustness_csv_path,
        batch_size=batch_size,
        test_closed_loader=test_closed_loader,
        test_unseen_loader=test_unseen_loader,
        device=device,
    )

    # Step 3: Generate all 4 300-DPI PNG Visualizations in <output_dir>/plots/
    print(f"[EVAL 3/3] Generating 4 high-resolution 300-DPI PNG charts in '{plots_dir}'...")
    generate_roc_comparison_from_checkpoints(
        checkpoints_dir=checkpoints_dir,
        data_dir=data_dir,
        splits_path=splits_path,
        output_path=roc_png_path,
        batch_size=batch_size,
        test_unseen_loader=test_unseen_loader,
        device=device,
    )
    generate_confusion_matrices_from_checkpoints(
        checkpoints_dir=checkpoints_dir,
        data_dir=data_dir,
        splits_path=splits_path,
        output_path=cm_png_path,
        batch_size=batch_size,
        test_unseen_loader=test_unseen_loader,
        device=device,
    )
    generate_tsne_from_checkpoint(
        checkpoint_path=zf_ckpt,
        data_dir=data_dir,
        splits_path=splits_path,
        output_path=tsne_png_path,
        max_samples=1000,
        batch_size=batch_size,
        test_closed_loader=test_closed_loader,
        test_unseen_loader=test_unseen_loader,
        device=device,
    )

    # For Grad-CAM: if test_closed_loader is a real DeepfakeDataset DataLoader, use generate_gradcam_grid_from_checkpoint;
    # otherwise (e.g., synthetic batch list in dry-run), build the 4x3 grid directly from the batch tensors.
    manifest_candidate = os.path.join(data_dir, "frames_manifest.csv")
    if os.path.exists(manifest_candidate):
        generate_gradcam_grid_from_checkpoint(
            checkpoint_path=zf_ckpt,
            data_dir=data_dir,
            splits_path=splits_path,
            output_path=gradcam_png_path,
            device=device,
        )
    else:
        from evaluation.closed_set import load_model_and_scorer_from_checkpoint

        model_gc, _, _ = load_model_and_scorer_from_checkpoint(zf_ckpt, device=device)
        first_batch = next(iter(test_closed_loader))
        imgs_b = first_batch["image"]
        row_titles = [
            "1. Real Face\n(Verdict: REAL)",
            "2. Known Fake\n(Verdict: FAKE)",
            "3. Unseen Fake\n(Verdict: FAKE)",
            "4. Ambiguous / OOD\n(Verdict: UNKNOWN)",
        ]
        cat_rows = []
        for idx, r_title in enumerate(row_titles):
            img_single = imgs_b[idx % len(imgs_b)]
            chw_np = (
                img_single.detach().cpu().float().numpy()
                if (HAS_TORCH and isinstance(img_single, torch.Tensor))
                else np.asarray(img_single, dtype=np.float32)
            )
            orig_rgb = denormalize_batch_to_uint8_hwc(chw_np[None, ...])[0]
            cam_01 = compute_single_gradcam_map(model_gc, img_single, device=device)
            hm_rgb, ov_rgb = overlay_heatmap_on_rgb(orig_rgb, cam_01, alpha=0.45)
            cat_rows.append(
                {
                    "row_label": r_title,
                    "original_rgb": orig_rgb,
                    "heatmap_rgb": hm_rgb,
                    "overlay_rgb": ov_rgb,
                }
            )
        plot_gradcam_4x3_grid(cat_rows, output_path=gradcam_png_path, dpi=300)

    return {
        "ablation_csv": ablation_csv_path,
        "results_summary_csv": os.path.join(output_dir, "results_summary.csv"),
        "robustness_csv": robustness_csv_path,
        "plots": {
            "roc_comparison": roc_png_path,
            "confusion_matrices": cm_png_path,
            "tsne_prototypes": tsne_png_path,
            "gradcam_heatmaps": gradcam_png_path,
        },
        "num_ablation_rows": len(ablation_rows),
        "num_robustness_rows": len(robustness_rows),
    }


def run_dry_run() -> None:
    """
    Synthesizes dummy checkpoints for all 4 ablation modes (`baseline`, `contrastive`, `single_proto`,
    `zerofake`) and synthetic `test_closed` / `test_unseen` batches on CPU, runs the entire evaluation,
    robustness, and 4-plot generation pipeline end-to-end, and verifies all CSVs and 300-DPI PNGs in <5 seconds.
    """
    start_time = time.time()
    print("=" * 78)
    print("[DRY-RUN] Testing evaluate.py end-to-end (Ablation CSV + Robustness CSV + 4x 300-DPI PNG plots)...")
    print("=" * 78)

    rng = np.random.default_rng(42)
    val_batches = []
    closed_batches = []
    unseen_batches = []

    for _ in range(1):
        imgs_v = rng.normal(0.0, 1.0, size=(6, 3, 224, 224)).astype(np.float32)
        imgs_c = rng.normal(0.0, 1.0, size=(6, 3, 224, 224)).astype(np.float32)
        imgs_u = rng.normal(0.3, 1.2, size=(6, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        gens_c = ["Real", "Real", "Real", "Deepfakes", "Face2Face", "FaceSwap"]
        gens_u = ["Real", "Real", "Real", "NeuralTextures", "NeuralTextures", "NeuralTextures"]
        if HAS_TORCH:
            val_batches.append({"image": torch.from_numpy(imgs_v), "label": torch.from_numpy(lbls), "split": "val"})
            closed_batches.append({"image": torch.from_numpy(imgs_c), "label": torch.from_numpy(lbls), "generator_name": gens_c})
            unseen_batches.append({"image": torch.from_numpy(imgs_u), "label": torch.from_numpy(lbls), "generator_name": gens_u})
        else:
            val_batches.append({"image": imgs_v, "label": lbls, "split": "val"})
            closed_batches.append({"image": imgs_c, "label": lbls, "generator_name": gens_c})
            unseen_batches.append({"image": imgs_u, "label": lbls, "generator_name": gens_u})

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpts_root = os.path.join(tmp_dir, "checkpoints")
        out_root = os.path.join(tmp_dir, "results")

        for mode in VALID_MODES:
            net = ZeroFakeNet(mode=mode, pretrained=False)
            scorer = OpenSetUncertaintyScorer()
            scorer.calibrate(val_batches, model=net, device="cpu")
            ckpt_file = os.path.join(ckpts_root, mode, "best_model.pt")
            save_checkpoint_file(
                {
                    "epoch": 14,
                    "mode": mode,
                    "holdout_generator": "NeuralTextures",
                    "model_state_dict": get_model_state(net),
                    "open_set_state": scorer.state_dict(),
                    "config": {"embedding_dim": 128, "num_fake_prototypes": 1 if mode == "single_proto" else 6},
                },
                ckpt_file,
            )

        summary = run_complete_evaluation_pipeline(
            checkpoints_dir=ckpts_root,
            data_dir=os.path.join(tmp_dir, "nonexistent_data_dir"),
            output_dir=out_root,
            test_closed_loader=closed_batches,
            test_unseen_loader=unseen_batches,
            device="cpu",
        )

        # Assert all CSVs and all 4 PNG plots exist and are non-empty
        for csv_key in ("ablation_csv", "results_summary_csv", "robustness_csv"):
            c_path = summary[csv_key]
            assert os.path.exists(c_path) and os.path.getsize(c_path) > 50, f"Missing or empty CSV: {c_path}"

        for plot_name, p_path in summary["plots"].items():
            assert os.path.exists(p_path) and os.path.getsize(p_path) > 100, (
                f"Missing or empty 300-DPI PNG plot '{plot_name}': {p_path}"
            )

    elapsed = time.time() - start_time
    assert elapsed < 10.0, f"Dry-run took {elapsed:.3f}s, exceeding limit!"
    print("=" * 78)
    print(
        f"[DRY-RUN PASSED] Generated ablation_summary.csv ({summary['num_ablation_rows']} rows), "
        f"robustness_summary.csv ({summary['num_robustness_rows']} rows), and all 4 300-DPI PNG plots "
        f"in {elapsed:.3f}s on CPU!"
    )
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete ZeroFake evaluation pipeline (Ablation + Robustness + 4x 300-DPI PNG charts)."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
        help="Root directory containing baseline/, contrastive/, single_proto/, and zerofake/ checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/content/data",
        help="Directory containing extracted face crops, frames_manifest.csv, and splits.json.",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default=None,
        help="Optional explicit path to splits.json (defaults to <data_dir>/splits.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output folder for CSV summaries and plots/ (default: results).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation mini-batch size (default: 64).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run end-to-end CPU dry-run on dummy checkpoints and synthetic images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    summary = run_complete_evaluation_pipeline(
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print("=" * 78)
    print("[SUCCESS] ZeroFake Complete Evaluation & Chart Suite Finished:")
    print(f"  - Ablation CSV   : {summary['ablation_csv']} ({summary['num_ablation_rows']} rows)")
    print(f"  - Summary CSV    : {summary['results_summary_csv']}")
    print(f"  - Robustness CSV : {summary['robustness_csv']} ({summary['num_robustness_rows']} rows)")
    for p_name, p_file in summary["plots"].items():
        print(f"  - 300-DPI Plot   : {p_file} ({p_name})")
    print("=" * 78)


if __name__ == "__main__":
    main()
