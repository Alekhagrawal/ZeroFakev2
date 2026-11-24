# File: evaluation/ablation.py
# What this file does: Loops through all saved ZeroFake ablation checkpoints ('baseline', 'contrastive',
#   'single_proto', and 'zerofake'—evaluating both Config D closed-set forced binary and Config E full
#   open-set selective rejection), computes all closed-set, unseen-generator, and open-set rejection metrics,
#   and saves a clean comparison table in results/ablation_summary.csv (and results/results_summary.csv).
# Inputs:
#   - --checkpoints_dir (str): Root directory containing per-mode checkpoint subfolders
#     (e.g., /content/drive/MyDrive/ZeroFake_Storage/checkpoints).
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --output_csv (str): Output CSV path (default: results/ablation_summary.csv).
#   - --batch_size (int): Evaluation mini-batch size (default: 64).
#   - --dry-run (flag): Synthesizes dummy checkpoints for all 4 modes on CPU, evaluates them on synthetic
#     test_closed and test_unseen splits, and verifies results/ablation_summary.csv generation in <5s.
# Outputs:
#   - results/ablation_summary.csv (and results/results_summary.csv) containing the complete comparison table.

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
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from evaluation.closed_set import evaluate_closed_set
from evaluation.open_set import compute_selective_open_set_metrics
from evaluation.unseen_generator import evaluate_unseen_generator
from models.open_set import OpenSetUncertaintyScorer
from models.zerofake_net import VALID_MODES, ZeroFakeNet
from training.train import get_model_state, save_checkpoint_file

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
    HAS_TORCH = False


ABLATION_CSV_COLUMNS = [
    "config_id",
    "mode",
    "holdout_generator",
    "closed_accuracy",
    "closed_precision",
    "closed_recall",
    "closed_f1",
    "closed_roc_auc",
    "closed_eer",
    "unseen_accuracy",
    "unseen_precision",
    "unseen_recall",
    "unseen_f1",
    "unseen_roc_auc",
    "unseen_eer",
    "unseen_fake_auc",
    "unseen_fake_recall",
    "tau_reject",
    "coverage",
    "selective_accuracy",
    "known_real_rejection_rate",
    "known_fake_rejection_rate",
    "unseen_fake_rejection_rate",
    "ood_auroc",
    "ood_aupr",
]

MODE_TO_CONFIG_LABEL = {
    "baseline": "Config A (Baseline)",
    "contrastive": "Config B (Contrastive)",
    "single_proto": "Config C (Single-Prototype)",
    "zerofake_closed": "Config D (ZeroFake 100% Cov)",
    "zerofake": "Config E (Full ZeroFake Open-Set)",
}


def resolve_checkpoint_for_mode(checkpoints_dir: str, mode: str) -> Optional[str]:
    """
    Locates `best_model.pt` (or `last_checkpoint.pt`) for a given ablation `mode` inside `checkpoints_dir`.
    Supports both `<checkpoints_dir>/<mode>/best_model.pt` and `<checkpoints_dir>/best_model_<mode>.pt`.
    """
    candidates = [
        os.path.join(checkpoints_dir, mode, "best_model.pt"),
        os.path.join(checkpoints_dir, mode, "last_checkpoint.pt"),
        os.path.join(checkpoints_dir, f"best_model_{mode}.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_ablation_suite(
    checkpoints_dir: str = "/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    output_csv: str = "results/ablation_summary.csv",
    batch_size: int = 64,
    test_closed_loader: Optional[Iterable] = None,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Loops through all saved checkpoints (`baseline`, `contrastive`, `single_proto`, `zerofake`),
    evaluates each on `test_closed` and `test_unseen` plus open-set selective prediction, and
    writes a clean comparison table to `output_csv` (and `results/results_summary.csv`).
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

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

    rows: List[Dict[str, Any]] = []

    for mode in VALID_MODES:
        ckpt_path = resolve_checkpoint_for_mode(checkpoints_dir, mode)
        if ckpt_path is None:
            print(f"[WARNING] No checkpoint found for mode='{mode}' under '{checkpoints_dir}'. Skipping.")
            continue

        closed_res = evaluate_closed_set(
            checkpoint_path=ckpt_path,
            data_dir=data_dir,
            splits_path=splits_path,
            batch_size=batch_size,
            test_closed_loader=test_closed_loader,
            device=device,
        )
        unseen_res = evaluate_unseen_generator(
            checkpoint_path=ckpt_path,
            data_dir=data_dir,
            splits_path=splits_path,
            batch_size=batch_size,
            test_unseen_loader=test_unseen_loader,
            device=device,
        )

        c_raw = closed_res["raw_outputs"]
        u_raw = unseen_res["raw_outputs"]
        tau_val = float(closed_res["tau_reject"])

        open_res = compute_selective_open_set_metrics(
            labels=c_raw["labels"],
            scores=c_raw["scores"],
            u_scores=c_raw["u_scores"],
            tau_reject=tau_val,
            unseen_u_scores=u_raw["u_scores"],
            unseen_labels=u_raw["labels"],
            unseen_scores=u_raw["scores"],
        )

        # For full 'zerofake', emit both Config D (100% coverage closed-set) and Config E (calibrated open-set)
        if mode == "zerofake":
            row_d = {
                "config_id": MODE_TO_CONFIG_LABEL["zerofake_closed"],
                "mode": "zerofake_closed",
                "holdout_generator": unseen_res["holdout_generator"],
                "closed_accuracy": round(closed_res["accuracy"], 4),
                "closed_precision": round(closed_res["precision"], 4),
                "closed_recall": round(closed_res["recall"], 4),
                "closed_f1": round(closed_res["f1"], 4),
                "closed_roc_auc": round(closed_res["roc_auc"], 4),
                "closed_eer": round(closed_res["eer"], 4),
                "unseen_accuracy": round(unseen_res["accuracy"], 4),
                "unseen_precision": round(unseen_res["precision"], 4),
                "unseen_recall": round(unseen_res["recall"], 4),
                "unseen_f1": round(unseen_res["f1"], 4),
                "unseen_roc_auc": round(unseen_res["roc_auc"], 4),
                "unseen_eer": round(unseen_res["eer"], 4),
                "unseen_fake_auc": round(unseen_res["unseen_fake_auc"], 4),
                "unseen_fake_recall": round(unseen_res["unseen_fake_recall"], 4),
                "tau_reject": float("inf"),
                "coverage": 1.0,
                "selective_accuracy": round(closed_res["accuracy"], 4),
                "known_real_rejection_rate": 0.0,
                "known_fake_rejection_rate": 0.0,
                "unseen_fake_rejection_rate": 0.0,
                "ood_auroc": round(open_res["ood_auroc"], 4),
                "ood_aupr": round(open_res["ood_aupr"], 4),
            }
            rows.append(row_d)

        row_entry = {
            "config_id": MODE_TO_CONFIG_LABEL[mode],
            "mode": mode,
            "holdout_generator": unseen_res["holdout_generator"],
            "closed_accuracy": round(closed_res["accuracy"], 4),
            "closed_precision": round(closed_res["precision"], 4),
            "closed_recall": round(closed_res["recall"], 4),
            "closed_f1": round(closed_res["f1"], 4),
            "closed_roc_auc": round(closed_res["roc_auc"], 4),
            "closed_eer": round(closed_res["eer"], 4),
            "unseen_accuracy": round(unseen_res["accuracy"], 4),
            "unseen_precision": round(unseen_res["precision"], 4),
            "unseen_recall": round(unseen_res["recall"], 4),
            "unseen_f1": round(unseen_res["f1"], 4),
            "unseen_roc_auc": round(unseen_res["roc_auc"], 4),
            "unseen_eer": round(unseen_res["eer"], 4),
            "unseen_fake_auc": round(unseen_res["unseen_fake_auc"], 4),
            "unseen_fake_recall": round(unseen_res["unseen_fake_recall"], 4),
            "tau_reject": round(tau_val, 4) if np.isfinite(tau_val) else float("inf"),
            "coverage": round(open_res["coverage"], 4),
            "selective_accuracy": round(open_res["selective_accuracy"], 4),
            "known_real_rejection_rate": round(open_res["known_real_rejection_rate"], 4),
            "known_fake_rejection_rate": round(open_res["known_fake_rejection_rate"], 4),
            "unseen_fake_rejection_rate": round(open_res["unseen_fake_rejection_rate"], 4),
            "ood_auroc": round(open_res["ood_auroc"], 4),
            "ood_aupr": round(open_res["ood_aupr"], 4),
        }
        rows.append(row_entry)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=ABLATION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Also mirror to results/results_summary.csv if output_csv is results/ablation_summary.csv
    alt_summary_csv = os.path.join(os.path.dirname(os.path.abspath(output_csv)), "results_summary.csv")
    if os.path.abspath(output_csv) != os.path.abspath(alt_summary_csv):
        shutil.copy2(output_csv, alt_summary_csv)

    return rows


def run_dry_run() -> None:
    """
    Synthesizes dummy checkpoints for all 4 ablation modes (`baseline`, `contrastive`, `single_proto`,
    `zerofake`), evaluates them on synthetic `test_closed` and `test_unseen` batches on CPU, and verifies
    that `results/ablation_summary.csv` is written with Configs A through E in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing evaluation/ablation.py across all 4 checkpoints -> results/ablation_summary.csv...")

    rng = np.random.default_rng(123)
    val_batches = []
    closed_batches = []
    unseen_batches = []
    for _ in range(2):
        imgs_v = rng.normal(0.0, 1.0, size=(6, 3, 224, 224)).astype(np.float32)
        imgs_c = rng.normal(0.0, 1.0, size=(6, 3, 224, 224)).astype(np.float32)
        imgs_u = rng.normal(0.3, 1.2, size=(6, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        if HAS_TORCH:
            val_batches.append({"image": torch.from_numpy(imgs_v), "label": torch.from_numpy(lbls), "split": "val"})
            closed_batches.append({"image": torch.from_numpy(imgs_c), "label": torch.from_numpy(lbls)})
            unseen_batches.append({"image": torch.from_numpy(imgs_u), "label": torch.from_numpy(lbls)})
        else:
            val_batches.append({"image": imgs_v, "label": lbls, "split": "val"})
            closed_batches.append({"image": imgs_c, "label": lbls})
            unseen_batches.append({"image": imgs_u, "label": lbls})

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpts_root = os.path.join(tmp_dir, "checkpoints")
        out_csv = os.path.join(tmp_dir, "results", "ablation_summary.csv")

        for mode in VALID_MODES:
            net = ZeroFakeNet(mode=mode, pretrained=False)
            scorer = OpenSetUncertaintyScorer()
            scorer.calibrate(val_batches, model=net, device="cpu")
            mode_ckpt = os.path.join(ckpts_root, mode, "best_model.pt")
            save_checkpoint_file(
                {
                    "epoch": 14,
                    "mode": mode,
                    "holdout_generator": "NeuralTextures",
                    "model_state_dict": get_model_state(net),
                    "open_set_state": scorer.state_dict(),
                    "config": {"embedding_dim": 128, "num_fake_prototypes": 1 if mode == "single_proto" else 6},
                },
                mode_ckpt,
            )

        rows = run_ablation_suite(
            checkpoints_dir=ckpts_root,
            output_csv=out_csv,
            test_closed_loader=closed_batches,
            test_unseen_loader=unseen_batches,
            device="cpu",
        )

        assert os.path.exists(out_csv), "ablation_summary.csv was not created!"
        assert len(rows) == 5, f"Expected 5 ablation rows (Configs A through E), got {len(rows)}"
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ABLATION_CSV_COLUMNS

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Evaluated all 4 saved checkpoints -> generated 5 ablation configs (Configs A-E) "
        f"in ablation_summary.csv in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZeroFake ablation evaluation and save results/ablation_summary.csv.")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints",
        help="Root directory containing baseline/, contrastive/, single_proto/, and zerofake/ checkpoints.",
    )
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Path to dataset directory.")
    parser.add_argument("--splits_path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/ablation_summary.csv",
        help="Output CSV file path (default: results/ablation_summary.csv).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dry-run verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    rows = run_ablation_suite(
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
    )
    print(f"[SUCCESS] Saved {len(rows)}-row ablation comparison table to '{args.output_csv}':")
    for r in rows:
        print(
            f"  {r['config_id']:<34} | Closed AUC={r['closed_roc_auc']:.4f} | "
            f"Unseen AUC={r['unseen_roc_auc']:.4f} | Cov={r['coverage']*100:.1f}% | "
            f"SelAcc={r['selective_accuracy']:.4f} | OOD_AUROC={r['ood_auroc']:.4f}"
        )


if __name__ == "__main__":
    main()
