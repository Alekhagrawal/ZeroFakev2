# File: training/validation.py
# What this file does: Evaluates the ZeroFake model strictly on the validation split ('val')—never
#   touching test_closed, test_unseen, or Celeb-DF v2—to compute validation losses (L_total, L_cls,
#   L_con, L_proto), ROC-AUC, and Equal Error Rate (EER). Also executes OpenSetUncertaintyScorer.calibrate()
#   on the validation split to lock in Z-score scalers and the 95% coverage threshold tau_reject.
# Inputs:
#   - model (ZeroFakeNet): Trained or in-training ZeroFake neural network.
#   - val_loader (iterable): DataLoader yielding validation split batches {'image': ..., 'label': ...}.
#   - criterion (ZeroFakeTotalLoss): Combined loss module.
#   - open_set_scorer (OpenSetUncertaintyScorer, optional): Uncertainty scorer to calibrate on val.
#   - --dry-run (flag): Runs validation metric computation (loss, ROC-AUC, EER) and open-set calibration on CPU in <5s.
# Outputs:
#   - Dictionary containing 'val_loss', 'val_loss_cls', 'val_loss_con', 'val_loss_proto', 'roc_auc',
#     'eer', 'f1', and optional 'open_set_calibration' stats ('tau_reject', 'val_coverage').

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from losses.total_loss import ZeroFakeTotalLoss
from models.open_set import OpenSetUncertaintyScorer
from models.zerofake_net import ZeroFakeNet

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAS_SKLEARN = True
except ImportError:
    roc_auc_score = None
    roc_curve = None
    HAS_SKLEARN = False


def compute_roc_auc_and_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Area Under the ROC Curve (`roc_auc`) and Equal Error Rate (`eer`, where False Acceptance
    Rate equals False Rejection Rate) given binary labels `y in {0, 1}` and continuous fake scores in `[0, 1]`.
    Works both with `scikit-learn` and via an exact pure-NumPy trapezoidal ROC implementation when
    `scikit-learn` is not installed locally.
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.5

    if HAS_SKLEARN:
        auc_val = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer_val = float(0.5 * (fpr[eer_idx] + fnr[eer_idx]))
        return auc_val, eer_val

    # Pure-NumPy exact ROC curve & EER fallback
    desc_idx = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[desc_idx]
    score_sorted = y_score[desc_idx]

    distinct_indices = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)

    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]

    total_pos = max(1.0, float(tps[-1]))
    total_neg = max(1.0, float(fps[-1]))

    tpr = tps / total_pos
    fpr = fps / total_neg
    fnr = 1.0 - tpr

    # Trapezoidal integration for ROC-AUC
    auc_val = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer_val = float(0.5 * (fpr[eer_idx] + fnr[eer_idx]))
    return auc_val, eer_val


def compute_binary_f1(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes binary F1-score at the operating threshold (default 0.5).
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_pred = (np.asarray(scores, dtype=np.float64).reshape(-1) >= threshold).astype(np.int64)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    denom = 2.0 * tp + fp + fn
    return float((2.0 * tp) / denom) if denom > 0 else 0.0


def _assert_val_only_split(val_loader: Iterable) -> None:
    """
    Enforces Rule 3 of AGENTS.md: verifies that `val_loader` is strictly the `'val'` split and
    never `'test_unseen'`, `'test_closed'`, or `'celeb_df'`.
    """
    dataset_obj = getattr(val_loader, "dataset", None)
    if dataset_obj is not None and hasattr(dataset_obj, "split"):
        split_name = str(getattr(dataset_obj, "split")).lower()
        assert split_name == "val", (
            f"Rule 3 Violation: training/validation.py must ONLY evaluate on 'val' split, "
            f"received split='{split_name}'!"
        )


def evaluate_validation_split(
    model: ZeroFakeNet,
    val_loader: Iterable,
    criterion: Optional[ZeroFakeTotalLoss] = None,
    open_set_scorer: Optional[OpenSetUncertaintyScorer] = None,
    calibrate_open_set: bool = True,
    device: str = "cpu",
    use_amp: bool = False,
    train_loader: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """
    Evaluates `model` strictly on `val_loader`:
      1. Computes average validation losses (`val_loss`, `val_loss_cls`, `val_loss_con`, `val_loss_proto`).
      2. Computes validation `roc_auc`, `eer`, and `f1`.
      3. When `open_set_scorer` is provided and `calibrate_open_set=True`, runs
         `open_set_scorer.calibrate(cached_val_outputs)` strictly on the validation outputs to lock in
         Z-score scalers and `tau_reject` at 95% validation coverage.
    """
    _assert_val_only_split(val_loader)

    if criterion is None:
        criterion = ZeroFakeTotalLoss(mode=getattr(model, "mode", "zerofake"))

    if HAS_TORCH and hasattr(model, "eval"):
        model.eval()

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    con_loss_sum = 0.0
    proto_loss_sum = 0.0
    num_batches = 0

    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    cached_val_outputs: List[Dict[str, Any]] = []

    amp_enabled = bool(use_amp and HAS_TORCH and str(device).startswith("cuda") and torch.cuda.is_available())

    for batch in val_loader:
        if isinstance(batch, dict):
            if "split" in batch:
                sp = batch["split"]
                if isinstance(sp, str):
                    assert sp.lower() == "val", f"Rule 3 Violation: non-val batch split '{sp}' in validation!"
                elif isinstance(sp, (list, tuple)):
                    for s in sp:
                        assert str(s).lower() == "val", f"Rule 3 Violation: non-val sample '{s}' in validation!"
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch[0], batch[1]

        if HAS_TORCH and isinstance(images, torch.Tensor):
            images_dev = images.to(device)
            labels_dev = labels.to(device)
            with torch.no_grad():
                if amp_enabled:
                    with torch.amp.autocast("cuda"):
                        raw_outputs = model(images_dev)
                else:
                    raw_outputs = model(images_dev)

                # Cast all floating-point tensors in outputs to float32 outside autocast before
                # computing validation losses or open-set uncertainty U(x)
                outputs = {
                    k: (v.float() if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v)
                    for k, v in raw_outputs.items()
                }
                loss_dict = criterion(outputs, labels_dev)

            total_loss_sum += float(loss_dict["loss_total"].item())
            cls_loss_sum += float(loss_dict["loss_cls"].item())
            con_loss_sum += float(loss_dict["loss_con"].item())
            proto_loss_sum += float(loss_dict["loss_proto"].item())

            y_np = labels_dev.detach().cpu().numpy().reshape(-1)
            scores_np = outputs["evidence_fake"].detach().cpu().float().numpy().reshape(-1)

            cached_batch = {
                "embeddings": outputs["embeddings"].detach().cpu().float(),
                "probs": outputs["probs"].detach().cpu().float(),
                "d_real": outputs["d_real"].detach().cpu().float(),
                "d_fake": outputs["d_fake"].detach().cpu().float(),
                "p_real_norm": outputs["p_real_norm"].detach().cpu().float(),
                "p_fake_norm": outputs["p_fake_norm"].detach().cpu().float(),
                "evidence_fake": outputs["evidence_fake"].detach().cpu().float(),
                "label": labels_dev.detach().cpu(),
                "split": "val",
            }
        else:
            images_np = np.asarray(images, dtype=np.float32)
            labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
            outputs = model(images_np)
            loss_dict = criterion(outputs, labels_np)

            total_loss_sum += float(loss_dict["loss_total"])
            cls_loss_sum += float(loss_dict["loss_cls"])
            con_loss_sum += float(loss_dict["loss_con"])
            proto_loss_sum += float(loss_dict["loss_proto"])

            y_np = labels_np
            scores_np = np.asarray(outputs["evidence_fake"], dtype=np.float64).reshape(-1)

            cached_batch = dict(outputs)
            cached_batch["label"] = labels_np
            cached_batch["split"] = "val"

        all_labels.append(y_np)
        all_scores.append(scores_np)
        cached_val_outputs.append(cached_batch)
        num_batches += 1

    if num_batches == 0:
        raise ValueError("val_loader yielded zero batches in evaluate_validation_split()!")

    y_all = np.concatenate(all_labels, axis=0)
    s_all = np.concatenate(all_scores, axis=0)

    roc_auc, eer = compute_roc_auc_and_eer(y_all, s_all)
    f1 = compute_binary_f1(y_all, s_all, threshold=0.5)

    cal_stats: Dict[str, Any] = {}
    if open_set_scorer is not None and calibrate_open_set:
        cal_stats = open_set_scorer.calibrate(
            cached_val_outputs,
            model=model,
            device=device,
            train_loader=train_loader,
        )

    return {
        "val_loss": total_loss_sum / num_batches,
        "val_loss_cls": cls_loss_sum / num_batches,
        "val_loss_con": con_loss_sum / num_batches,
        "val_loss_proto": proto_loss_sum / num_batches,
        "roc_auc": roc_auc,
        "eer": eer,
        "f1": f1,
        "num_samples": int(len(y_all)),
        "open_set_calibration": cal_stats,
    }


def run_dry_run() -> None:
    """
    Runs a fast (<5s) CPU dry-run of `evaluate_validation_split()` on synthetic validation batches,
    verifying loss calculation, ROC-AUC, EER, and 95%-coverage open-set threshold calibration.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing training/validation.py (val loss, ROC-AUC, EER, and open_set.calibrate)...")

    net = ZeroFakeNet(mode="zerofake", pretrained=False)
    criterion = ZeroFakeTotalLoss(mode="zerofake")
    scorer = OpenSetUncertaintyScorer(alpha=0.4, beta=0.3, gamma=0.3, target_val_coverage=0.95)

    rng = np.random.default_rng(42)
    val_batches = []
    for _ in range(4):
        imgs = rng.normal(0.0, 1.0, size=(10, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        if HAS_TORCH:
            val_batches.append(
                {
                    "image": torch.from_numpy(imgs),
                    "label": torch.from_numpy(lbls),
                    "split": "val",
                }
            )
        else:
            val_batches.append({"image": imgs, "label": lbls, "split": "val"})

    metrics = evaluate_validation_split(
        model=net,
        val_loader=val_batches,
        criterion=criterion,
        open_set_scorer=scorer,
        calibrate_open_set=True,
        device="cpu",
        use_amp=False,
    )

    assert np.isfinite(metrics["val_loss"])
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0
    assert scorer.is_calibrated and np.isfinite(scorer.tau_reject)

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] val_loss={metrics['val_loss']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
        f"EER={metrics['eer']:.4f} | tau_reject={scorer.tau_reject:.4f} "
        f"(coverage={scorer.val_coverage_achieved*100:.1f}%) in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ZeroFake on validation split & calibrate open-set threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU validation & calibration check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] training/validation.py ready. Import evaluate_validation_split or run with --dry-run.")


if __name__ == "__main__":
    main()
