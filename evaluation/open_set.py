# File: evaluation/open_set.py
# What this file does: Evaluates selective 3-way open-set prediction ('REAL', 'FAKE', 'UNKNOWN') using
#   ONLY the saved validation threshold tau_reject inside the checkpoint (strictly adhering to Rule 3:
#   never re-tuning on test data). Computes Coverage (% accepted as REAL or FAKE), Selective Accuracy
#   on accepted samples, Known-Real Rejection Rate, Known-Fake / Unseen-Fake Rejection Rate, and OOD
#   AUROC / AUPR (measuring how well uncertainty U(x) separates unseen-generator samples from known samples).
# Inputs:
#   - --checkpoint (str): Path to a saved checkpoint (.pt) containing calibrated open_set_state.
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --batch_size (int): Evaluation mini-batch size (default: 64).
#   - --dry-run (flag): Synthesizes known (test_closed) and unseen (test_unseen) outputs on CPU and
#     verifies Coverage, Selective Accuracy, Rejection Rates, and OOD AUROC/AUPR in under 5 seconds.
# Outputs:
#   - Dictionary containing 'tau_reject', 'coverage', 'selective_accuracy', 'known_real_rejection_rate',
#     'known_fake_rejection_rate', 'unseen_fake_rejection_rate', 'ood_auroc', and 'ood_aupr'.

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
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from evaluation.closed_set import (
    collect_split_inference_outputs,
    load_model_and_scorer_from_checkpoint,
)
from models.open_set import OpenSetUncertaintyScorer
from models.zerofake_net import ZeroFakeNet
from training.train import get_model_state, save_checkpoint_file
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
    from sklearn.metrics import average_precision_score
    HAS_SKLEARN = True
except ImportError:
    average_precision_score = None
    HAS_SKLEARN = False


def compute_aupr(binary_ood_labels: np.ndarray, u_scores: np.ndarray) -> float:
    """
    Computes Area Under the Precision-Recall Curve (`AUPR`) given binary OOD labels `{0=Known, 1=Unseen/OOD}`
    and continuous uncertainty scores `U(x)`.
    Uses `scikit-learn` when available, or an exact pure-NumPy step-wise precision-recall integral fallback.
    """
    y_true = np.asarray(binary_ood_labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(u_scores, dtype=np.float64).reshape(-1)

    if len(y_true) == 0 or np.sum(y_true == 1) == 0:
        return 0.0

    if HAS_SKLEARN and average_precision_score is not None:
        return float(average_precision_score(y_true, y_score))

    # Pure-NumPy Average Precision (AUPR) calculation
    desc_idx = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[desc_idx]
    tps = np.cumsum(y_sorted == 1).astype(np.float64)
    fps = np.cumsum(y_sorted == 0).astype(np.float64)

    total_pos = float(np.sum(y_true == 1))
    precisions = tps / np.maximum(tps + fps, 1.0)
    recalls = tps / total_pos

    # Step-wise integration over recall increments at positive indices
    pos_indices = np.where(y_sorted == 1)[0]
    if len(pos_indices) == 0:
        return 0.0
    return float(np.mean(precisions[pos_indices]))


def compute_selective_open_set_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    u_scores: np.ndarray,
    tau_reject: float,
    unseen_u_scores: Optional[np.ndarray] = None,
    unseen_labels: Optional[np.ndarray] = None,
    unseen_scores: Optional[np.ndarray] = None,
    fake_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes open-set selective prediction metrics using ONLY the saved validation threshold `tau_reject`:
      1. `coverage`: Percentage (in [0, 1] and % in [0, 100]) of samples accepted (`U(x) <= tau_reject`).
      2. `selective_accuracy`: Binary classification accuracy strictly on accepted samples (`U(x) <= tau_reject`).
      3. `known_real_rejection_rate`: Fraction of known Real samples (`y == 0`) flagged as `'UNKNOWN'` (`U(x) > tau_reject`).
      4. `known_fake_rejection_rate`: Fraction of known Fake samples (`y == 1`) flagged as `'UNKNOWN'` (`U(x) > tau_reject`).
      5. `unseen_fake_rejection_rate`: Fraction of unseen-generator Fake samples flagged as `'UNKNOWN'`.
      6. `ood_auroc` & `ood_aupr`: How effectively uncertainty `U(x)` distinguishes unseen-generator fake
         samples (OOD label 1) from known closed-set samples (in-distribution label 0).
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    u_arr = np.asarray(u_scores, dtype=np.float64).reshape(-1)
    tau = float(tau_reject)

    n = len(y_true)
    if n == 0:
        return {
            "tau_reject": tau,
            "coverage": 0.0,
            "coverage_pct": 0.0,
            "selective_accuracy": 0.0,
            "known_real_rejection_rate": 0.0,
            "known_fake_rejection_rate": 0.0,
            "unseen_fake_rejection_rate": 0.0,
            "unseen_selective_accuracy": 0.0,
            "unseen_coverage": 0.0,
            "ood_auroc": 0.5,
            "ood_aupr": 0.5,
        }

    accepted_mask = u_arr <= tau
    rejected_mask = ~accepted_mask

    coverage = float(np.mean(accepted_mask))
    if np.any(accepted_mask):
        y_pred_acc = (y_score[accepted_mask] >= fake_threshold).astype(np.int64)
        selective_accuracy = float(np.mean(y_pred_acc == y_true[accepted_mask]))
    else:
        selective_accuracy = 0.0

    real_mask = y_true == 0
    fake_mask = y_true == 1

    known_real_rejection_rate = float(np.mean(rejected_mask[real_mask])) if np.any(real_mask) else 0.0
    known_fake_rejection_rate = float(np.mean(rejected_mask[fake_mask])) if np.any(fake_mask) else 0.0

    # If unseen-generator samples are provided, compute unseen rejection rate, unseen selective accuracy, and OOD AUROC/AUPR
    if unseen_u_scores is not None and len(unseen_u_scores) > 0:
        u_unseen = np.asarray(unseen_u_scores, dtype=np.float64).reshape(-1)
        if unseen_labels is not None:
            y_unseen = np.asarray(unseen_labels, dtype=np.int64).reshape(-1)
            unseen_fake_only = u_unseen[y_unseen == 1] if np.any(y_unseen == 1) else u_unseen
        else:
            y_unseen = np.ones_like(u_unseen, dtype=np.int64)
            unseen_fake_only = u_unseen

        unseen_fake_rejection_rate = float(np.mean(unseen_fake_only > tau)) if len(unseen_fake_only) > 0 else 0.0
        unseen_acc_mask = u_unseen <= tau
        unseen_coverage = float(np.mean(unseen_acc_mask))
        if unseen_scores is not None and np.any(unseen_acc_mask):
            s_unseen = np.asarray(unseen_scores, dtype=np.float64).reshape(-1)
            unseen_preds = (s_unseen[unseen_acc_mask] >= fake_threshold).astype(np.int64)
            unseen_selective_accuracy = float(np.mean(unseen_preds == y_unseen[unseen_acc_mask]))
        else:
            unseen_selective_accuracy = selective_accuracy

        # OOD detection: In-Distribution (known test_closed = 0) vs Out-Of-Distribution (unseen fake = 1)
        ood_y = np.concatenate(
            [np.zeros(len(u_arr), dtype=np.int64), np.ones(len(unseen_fake_only), dtype=np.int64)],
            axis=0,
        )
        ood_u = np.concatenate([u_arr, unseen_fake_only], axis=0)
        ood_auroc, _ = compute_roc_auc_and_eer(ood_y, ood_u)
        ood_aupr = compute_aupr(ood_y, ood_u)
    else:
        unseen_fake_rejection_rate = known_fake_rejection_rate
        unseen_selective_accuracy = selective_accuracy
        unseen_coverage = coverage
        ood_auroc, _ = compute_roc_auc_and_eer(y_true, u_arr)
        ood_aupr = compute_aupr(y_true, u_arr)

    return {
        "tau_reject": tau,
        "coverage": coverage,
        "coverage_pct": coverage * 100.0,
        "selective_accuracy": selective_accuracy,
        "known_real_rejection_rate": known_real_rejection_rate,
        "known_fake_rejection_rate": known_fake_rejection_rate,
        "unseen_fake_rejection_rate": unseen_fake_rejection_rate,
        "unseen_selective_accuracy": unseen_selective_accuracy,
        "unseen_coverage": unseen_coverage,
        "ood_auroc": ood_auroc,
        "ood_aupr": ood_aupr,
    }


def evaluate_open_set(
    checkpoint_path: str,
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    batch_size: int = 64,
    test_closed_loader: Optional[Iterable] = None,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates open-set selective prediction using strictly the validation-calibrated `tau_reject`
    stored inside `checkpoint_path` (Rule 3: never re-calibrates on test data).
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    model, scorer, ckpt = load_model_and_scorer_from_checkpoint(checkpoint_path, device=device)
    assert scorer.is_calibrated and np.isfinite(scorer.tau_reject), (
        f"Checkpoint '{checkpoint_path}' does not contain a finite validation-calibrated tau_reject!"
    )

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

    closed_out = collect_split_inference_outputs(
        model=model,
        data_loader=test_closed_loader,
        open_set_scorer=scorer,
        device=device,
    )
    unseen_out = collect_split_inference_outputs(
        model=model,
        data_loader=test_unseen_loader,
        open_set_scorer=scorer,
        device=device,
    )

    metrics = compute_selective_open_set_metrics(
        labels=closed_out["labels"],
        scores=closed_out["scores"],
        u_scores=closed_out["u_scores"],
        tau_reject=scorer.tau_reject,
        unseen_u_scores=unseen_out["u_scores"],
        unseen_labels=unseen_out["labels"],
        unseen_scores=unseen_out["scores"],
    )
    metrics["mode"] = model.mode
    metrics["holdout_generator"] = str(ckpt.get("holdout_generator", "NeuralTextures"))
    return metrics


def run_dry_run() -> None:
    """
    Synthesizes a dummy checkpoint with a validation-calibrated `tau_reject`, runs `evaluate_open_set()`
    on synthetic `test_closed` and `test_unseen` batches on CPU, and verifies Coverage, Selective Accuracy,
    Known-Real/Known-Fake Rejection Rates, and OOD AUROC/AUPR in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing evaluation/open_set.py (val-locked tau_reject, Coverage, Selective Acc, OOD AUROC/AUPR)...")

    net = ZeroFakeNet(mode="zerofake", pretrained=False)
    scorer = OpenSetUncertaintyScorer(alpha=0.4, beta=0.3, gamma=0.3, target_val_coverage=0.95)

    rng = np.random.default_rng(77)
    val_batches = []
    closed_batches = []
    unseen_batches = []

    for i in range(3):
        imgs_v = rng.normal(0.0, 1.0, size=(8, 3, 224, 224)).astype(np.float32)
        imgs_c = rng.normal(0.0, 1.0, size=(8, 3, 224, 224)).astype(np.float32)
        imgs_u = rng.normal(0.5, 1.5, size=(8, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        if HAS_TORCH:
            val_batches.append({"image": torch.from_numpy(imgs_v), "label": torch.from_numpy(lbls), "split": "val"})
            closed_batches.append({"image": torch.from_numpy(imgs_c), "label": torch.from_numpy(lbls)})
            unseen_batches.append({"image": torch.from_numpy(imgs_u), "label": torch.from_numpy(lbls)})
        else:
            val_batches.append({"image": imgs_v, "label": lbls, "split": "val"})
            closed_batches.append({"image": imgs_c, "label": lbls})
            unseen_batches.append({"image": imgs_u, "label": lbls})

    # Calibrate strictly on val_batches before saving checkpoint
    scorer.calibrate(val_batches, model=net, device="cpu")
    locked_tau = scorer.tau_reject

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "best_model.pt")
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
        res = evaluate_open_set(
            checkpoint_path=ckpt_path,
            test_closed_loader=closed_batches,
            test_unseen_loader=unseen_batches,
            device="cpu",
        )

    assert abs(res["tau_reject"] - locked_tau) < 1e-7, "Rule 3 Violation: tau_reject was modified during test evaluation!"
    for k in (
        "coverage",
        "selective_accuracy",
        "known_real_rejection_rate",
        "known_fake_rejection_rate",
        "unseen_fake_rejection_rate",
        "ood_auroc",
        "ood_aupr",
    ):
        assert k in res and np.isfinite(res[k]) and 0.0 <= res[k] <= 1.0, f"Invalid metric {k}: {res.get(k)}"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Open-Set (locked tau_reject={res['tau_reject']:.4f}) -> "
        f"Coverage={res['coverage_pct']:.1f}%, SelectiveAcc={res['selective_accuracy']:.4f}, "
        f"KnownRealRej={res['known_real_rejection_rate']:.4f}, KnownFakeRej={res['known_fake_rejection_rate']:.4f}, "
        f"OOD_AUROC={res['ood_auroc']:.4f}, OOD_AUPR={res['ood_aupr']:.4f} in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ZeroFake open-set selective prediction using saved val tau_reject.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint .pt.")
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Path to dataset directory.")
    parser.add_argument("--splits_path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dry-run verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    res = evaluate_open_set(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        batch_size=args.batch_size,
    )
    print(
        f"[OPEN-SET RESULTS] mode={res['mode']} | tau_reject={res['tau_reject']:.4f} | "
        f"Coverage={res['coverage_pct']:.1f}% | SelectiveAcc={res['selective_accuracy']:.4f} | "
        f"KnownRealRej={res['known_real_rejection_rate']:.4f} | KnownFakeRej={res['known_fake_rejection_rate']:.4f} | "
        f"UnseenFakeRej={res['unseen_fake_rejection_rate']:.4f} | "
        f"OOD_AUROC={res['ood_auroc']:.4f} | OOD_AUPR={res['ood_aupr']:.4f}"
    )


if __name__ == "__main__":
    main()
