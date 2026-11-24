# File: evaluation/closed_set.py
# What this file does: Evaluates a trained ZeroFake checkpoint on the 'test_closed' split (known manipulation
#   generators + authentic test videos) and computes closed-set binary detection metrics: Accuracy, Precision,
#   Recall, F1-score, ROC-AUC, and Equal Error Rate (EER).
# Inputs:
#   - --checkpoint (str): Path to a saved checkpoint (.pt) containing model_state_dict, mode, and open_set_state.
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --batch_size (int): Evaluation mini-batch size (default: 64).
#   - --dry-run (flag): Synthesizes a dummy checkpoint and mock test_closed batches on CPU and verifies all
#     6 closed-set metrics (Accuracy, Precision, Recall, F1, ROC-AUC, EER) in under 5 seconds.
# Outputs:
#   - Dictionary of closed-set detection metrics ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'eer',
#     'num_samples', 'num_real', 'num_fake') plus raw per-sample arrays for downstream open-set/plotting tools.

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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from models.open_set import OpenSetUncertaintyScorer
from models.zerofake_net import VALID_MODES, ZeroFakeNet
from training.train import (
    cast_outputs_to_float32,
    get_model_state,
    load_checkpoint_file,
    save_checkpoint_file,
    set_model_state,
)
from training.validation import compute_roc_auc_and_eer

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
    HAS_TORCH = False


def compute_binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes binary detection metrics on `labels` in `{0, 1}` (0=Real, 1=Fake) and continuous
    fake likelihood `scores` in `[0, 1]`:
      - `accuracy`:  (TP + TN) / N
      - `precision`: TP / (TP + FP)
      - `recall`:    TP / (TP + FN)
      - `f1`:        2 * TP / (2 * TP + FP + FN)
      - `roc_auc`:   Area Under the ROC Curve
      - `eer`:       Equal Error Rate (where FAR == FRR)
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    assert y_true.shape == y_score.shape, f"Shape mismatch: labels {y_true.shape} vs scores {y_score.shape}"

    n = len(y_true)
    if n == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "eer": 0.5,
            "num_samples": 0,
            "num_real": 0,
            "num_fake": 0,
        }

    y_pred = (y_score >= float(threshold)).astype(np.int64)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = float((tp + tn) / n)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_denom = 2.0 * tp + fp + fn
    f1 = float((2.0 * tp) / f1_denom) if f1_denom > 0 else 0.0

    roc_auc, eer = compute_roc_auc_and_eer(y_true, y_score)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "eer": eer,
        "num_samples": int(n),
        "num_real": int(np.sum(y_true == 0)),
        "num_fake": int(np.sum(y_true == 1)),
    }


def load_model_and_scorer_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    override_mode: Optional[str] = None,
    pretrained_backbone: bool = False,
) -> Tuple[ZeroFakeNet, OpenSetUncertaintyScorer, Dict[str, Any]]:
    """
    Loads a saved ZeroFake checkpoint (.pt), reconstructs `ZeroFakeNet` in the checkpoint's saved
    ablation mode (`'baseline'`, `'contrastive'`, `'single_proto'`, or `'zerofake'`), and restores
    the validation-calibrated `OpenSetUncertaintyScorer` state without re-tuning on test data.
    """
    ckpt = load_checkpoint_file(checkpoint_path, map_location=device)
    mode = (override_mode or ckpt.get("mode", "zerofake")).strip().lower()
    if mode not in VALID_MODES:
        mode = "zerofake"

    cfg = ckpt.get("config", {})
    embed_dim = int(cfg.get("embedding_dim", 128))
    num_fake_prototypes = int(cfg.get("num_fake_prototypes", 6))

    model = ZeroFakeNet(
        mode=mode,
        pretrained=pretrained_backbone,
        embed_dim=embed_dim,
        num_fake_prototypes=num_fake_prototypes,
    )
    if "model_state_dict" in ckpt:
        set_model_state(model, ckpt["model_state_dict"])

    if HAS_TORCH and hasattr(model, "to"):
        model = model.to(device)
        model.eval()

    scorer = OpenSetUncertaintyScorer(
        alpha=float(cfg.get("alpha", 0.4)),
        beta=float(cfg.get("beta", 0.3)),
        gamma=float(cfg.get("gamma", 0.3)),
        target_val_coverage=float(cfg.get("target_coverage", 0.95)),
        embed_dim=embed_dim,
    )
    if "open_set_state" in ckpt and isinstance(ckpt["open_set_state"], dict):
        scorer.load_state_dict(ckpt["open_set_state"])

    return model, scorer, ckpt


def collect_split_inference_outputs(
    model: ZeroFakeNet,
    data_loader: Iterable,
    open_set_scorer: Optional[OpenSetUncertaintyScorer] = None,
    device: str = "cpu",
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Runs forward inference across `data_loader` (using CUDA fp16 autocast for the forward pass and casting
    all outputs to float32 outside autocast) and collects labels, fake evidence scores, probabilities,
    embeddings, generator names, and open-set uncertainty scores `U(x)`.
    """
    if HAS_TORCH and hasattr(model, "eval"):
        model.eval()

    amp_enabled = bool(use_amp and HAS_TORCH and str(device).startswith("cuda") and torch.cuda.is_available())

    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_embeddings: List[np.ndarray] = []
    all_d_real: List[np.ndarray] = []
    all_d_fake: List[np.ndarray] = []
    all_u_scores: List[np.ndarray] = []
    all_verdicts: List[str] = []
    all_generators: List[str] = []

    for batch in data_loader:
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
            gens = batch.get("generator_name", None)
        else:
            images, labels = batch[0], batch[1]
            gens = None

        if HAS_TORCH and isinstance(images, torch.Tensor):
            images_dev = images.to(device, non_blocking=True)
            with torch.no_grad():
                if amp_enabled:
                    with torch.amp.autocast("cuda"):
                        raw_out = model(images_dev)
                else:
                    raw_out = model(images_dev)
                outputs = cast_outputs_to_float32(raw_out)

            y_np = labels.detach().cpu().numpy().reshape(-1).astype(np.int64)
            scores_np = outputs["evidence_fake"].detach().cpu().numpy().reshape(-1).astype(np.float64)
            probs_np = outputs["probs"].detach().cpu().numpy().reshape(-1).astype(np.float64)
            emb_np = outputs["embeddings"].detach().cpu().numpy().astype(np.float32)
            dr_np = outputs["d_real"].detach().cpu().numpy().reshape(-1).astype(np.float32)
            df_np = outputs["d_fake"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            x_np = np.asarray(images, dtype=np.float32)
            y_np = np.asarray(labels, dtype=np.int64).reshape(-1)
            outputs = model(x_np)
            scores_np = np.asarray(outputs["evidence_fake"], dtype=np.float64).reshape(-1)
            probs_np = np.asarray(outputs["probs"], dtype=np.float64).reshape(-1)
            emb_np = np.asarray(outputs["embeddings"], dtype=np.float32)
            dr_np = np.asarray(outputs["d_real"], dtype=np.float32).reshape(-1)
            df_np = np.asarray(outputs["d_fake"], dtype=np.float32).reshape(-1)

        if open_set_scorer is not None:
            pred_open = open_set_scorer.predict(outputs, open_set_enabled=True)
            all_u_scores.append(np.asarray(pred_open["u_score"], dtype=np.float64).reshape(-1))
            all_verdicts.extend(pred_open["verdicts"])
        else:
            all_u_scores.append(np.zeros_like(scores_np))
            all_verdicts.extend(["FAKE" if s >= 0.5 else "REAL" for s in scores_np])

        if gens is not None:
            if isinstance(gens, (list, tuple)):
                all_generators.extend([str(g) for g in gens])
            elif isinstance(gens, np.ndarray):
                all_generators.extend([str(g) for g in gens.tolist()])
            else:
                all_generators.extend([str(gens)] * len(y_np))
        else:
            all_generators.extend(["Real" if y_val == 0 else "Fake" for y_val in y_np])

        all_labels.append(y_np)
        all_scores.append(scores_np)
        all_probs.append(probs_np)
        all_embeddings.append(emb_np)
        all_d_real.append(dr_np)
        all_d_fake.append(df_np)

    if not all_labels:
        raise ValueError("DataLoader yielded zero batches during evaluation!")

    return {
        "labels": np.concatenate(all_labels, axis=0),
        "scores": np.concatenate(all_scores, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "embeddings": np.concatenate(all_embeddings, axis=0),
        "d_real": np.concatenate(all_d_real, axis=0),
        "d_fake": np.concatenate(all_d_fake, axis=0),
        "u_scores": np.concatenate(all_u_scores, axis=0),
        "verdicts": all_verdicts,
        "generators": all_generators,
    }


def evaluate_closed_set(
    checkpoint_path: str,
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    batch_size: int = 64,
    test_closed_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates `checkpoint_path` on the `'test_closed'` split (known generators) and returns
    Accuracy, Precision, Recall, F1-score, ROC-AUC, and Equal Error Rate (EER).
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    model, scorer, ckpt = load_model_and_scorer_from_checkpoint(checkpoint_path, device=device)

    if test_closed_loader is None:
        manifest_path = os.path.join(data_dir, "frames_manifest.csv")
        resolved_splits = splits_path if splits_path else os.path.join(data_dir, "splits.json")
        dataset = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=data_dir,
            split="test_closed",
            splits_path=resolved_splits,
            augment=False,
        )
        if HAS_TORCH:
            test_closed_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            raise RuntimeError("PyTorch DataLoader required when test_closed_loader is not pre-supplied.")

    collected = collect_split_inference_outputs(
        model=model,
        data_loader=test_closed_loader,
        open_set_scorer=scorer,
        device=device,
    )
    metrics = compute_binary_classification_metrics(collected["labels"], collected["scores"], threshold=0.5)
    metrics["mode"] = model.mode
    metrics["split"] = "test_closed"
    metrics["tau_reject"] = float(scorer.tau_reject)
    metrics["raw_outputs"] = collected
    return metrics


def run_dry_run() -> None:
    """
    Synthesizes a dummy checkpoint and synthetic `test_closed` batches on CPU, runs `evaluate_closed_set()`,
    and verifies Accuracy, Precision, Recall, F1-score, ROC-AUC, and EER in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing evaluation/closed_set.py (test_closed Accuracy, Precision, Recall, F1, ROC-AUC, EER)...")

    net = ZeroFakeNet(mode="zerofake", pretrained=False)
    scorer = OpenSetUncertaintyScorer()
    scorer.tau_reject = 1.05
    scorer.is_calibrated = True

    rng = np.random.default_rng(42)
    mock_batches = []
    for _ in range(3):
        imgs = rng.normal(0.0, 1.0, size=(8, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        gens = ["Real"] * 4 + ["Deepfakes", "Face2Face", "FaceSwap", "Deepfakes"]
        if HAS_TORCH:
            mock_batches.append({"image": torch.from_numpy(imgs), "label": torch.from_numpy(lbls), "generator_name": gens})
        else:
            mock_batches.append({"image": imgs, "label": lbls, "generator_name": gens})

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "best_model.pt")
        save_checkpoint_file(
            {
                "epoch": 14,
                "mode": "zerofake",
                "model_state_dict": get_model_state(net),
                "open_set_state": scorer.state_dict(),
                "config": {"embedding_dim": 128, "num_fake_prototypes": 6},
            },
            ckpt_path,
        )
        res = evaluate_closed_set(checkpoint_path=ckpt_path, test_closed_loader=mock_batches, device="cpu")

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "eer"):
        assert key in res and np.isfinite(res[key]) and 0.0 <= res[key] <= 1.0, f"Invalid {key}: {res.get(key)}"
    assert res["num_samples"] == 24

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] test_closed -> Acc={res['accuracy']:.4f}, Prec={res['precision']:.4f}, "
        f"Rec={res['recall']:.4f}, F1={res['f1']:.4f}, AUC={res['roc_auc']:.4f}, EER={res['eer']:.4f} "
        f"in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ZeroFake checkpoint on test_closed split.")
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

    res = evaluate_closed_set(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        batch_size=args.batch_size,
    )
    print(
        f"[CLOSED-SET RESULTS] mode={res['mode']} | N={res['num_samples']} | "
        f"Acc={res['accuracy']:.4f} | Prec={res['precision']:.4f} | Rec={res['recall']:.4f} | "
        f"F1={res['f1']:.4f} | ROC-AUC={res['roc_auc']:.4f} | EER={res['eer']:.4f}"
    )


if __name__ == "__main__":
    main()
