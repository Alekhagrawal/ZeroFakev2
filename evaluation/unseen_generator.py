# File: evaluation/unseen_generator.py
# What this file does: Evaluates a trained ZeroFake checkpoint on the held-out Leave-One-Generator-Out
#   (LOGO) split ('test_unseen', e.g., NeuralTextures or FaceSwap paired with authentic test videos)
#   and computes Accuracy, Precision, Recall, F1-score, ROC-AUC, Equal Error Rate (EER), plus explicit
#   unseen-generator fake detection AUC and unseen-generator fake recall.
# Inputs:
#   - --checkpoint (str): Path to a saved checkpoint (.pt).
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --batch_size (int): Evaluation mini-batch size (default: 64).
#   - --dry-run (flag): Synthesizes a dummy checkpoint and mock test_unseen batches on CPU and verifies
#     all binary and unseen-generator detection metrics in under 5 seconds.
# Outputs:
#   - Dictionary containing 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'eer',
#     'unseen_fake_auc', 'unseen_fake_recall', 'holdout_generator', and 'raw_outputs'.

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
from typing import Any, Dict, Iterable, Optional

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from evaluation.closed_set import (
    collect_split_inference_outputs,
    compute_binary_classification_metrics,
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


def evaluate_unseen_generator(
    checkpoint_path: str,
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    batch_size: int = 64,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates `checkpoint_path` on the held-out unseen generator split (`'test_unseen'`).
    Computes the full binary detection suite (Accuracy, Precision, Recall, F1-score, ROC-AUC, EER)
    plus explicit `unseen_fake_auc` and `unseen_fake_recall` measuring zero-day fake detection rate.
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    model, scorer, ckpt = load_model_and_scorer_from_checkpoint(checkpoint_path, device=device)
    holdout_generator = str(ckpt.get("holdout_generator", "NeuralTextures"))

    if test_unseen_loader is None:
        manifest_path = os.path.join(data_dir, "frames_manifest.csv")
        resolved_splits = splits_path if splits_path else os.path.join(data_dir, "splits.json")
        dataset = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=data_dir,
            split="test_unseen",
            splits_path=resolved_splits,
            augment=False,
            include_closed_real_in_unseen=True,
        )
        if HAS_TORCH:
            test_unseen_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            raise RuntimeError("PyTorch DataLoader required when test_unseen_loader is not pre-supplied.")

    collected = collect_split_inference_outputs(
        model=model,
        data_loader=test_unseen_loader,
        open_set_scorer=scorer,
        device=device,
    )

    labels = collected["labels"]
    scores = collected["scores"]

    metrics = compute_binary_classification_metrics(labels, scores, threshold=0.5)

    # Explicit unseen-generator fake detection AUC and recall
    fake_mask = labels == 1
    unseen_fake_recall = float(np.mean(scores[fake_mask] >= 0.5)) if np.any(fake_mask) else 0.0
    unseen_fake_auc, _ = compute_roc_auc_and_eer(labels, scores)

    metrics["unseen_fake_auc"] = unseen_fake_auc
    metrics["unseen_fake_recall"] = unseen_fake_recall
    metrics["holdout_generator"] = holdout_generator
    metrics["mode"] = model.mode
    metrics["split"] = "test_unseen"
    metrics["tau_reject"] = float(scorer.tau_reject)
    metrics["raw_outputs"] = collected
    return metrics


def run_dry_run() -> None:
    """
    Synthesizes a dummy checkpoint and mock `test_unseen` batches (Real + held-out NeuralTextures) on CPU,
    runs `evaluate_unseen_generator()`, and verifies all binary and unseen-generator metrics in <5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing evaluation/unseen_generator.py (test_unseen binary metrics + unseen fake AUC/recall)...")

    net = ZeroFakeNet(mode="zerofake", pretrained=False)
    scorer = OpenSetUncertaintyScorer()
    scorer.tau_reject = 1.05
    scorer.is_calibrated = True

    rng = np.random.default_rng(99)
    mock_batches = []
    for _ in range(3):
        imgs = rng.normal(0.0, 1.0, size=(8, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        gens = ["Real"] * 4 + ["NeuralTextures"] * 4
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
                "holdout_generator": "NeuralTextures",
                "model_state_dict": get_model_state(net),
                "open_set_state": scorer.state_dict(),
                "config": {"embedding_dim": 128, "num_fake_prototypes": 6},
            },
            ckpt_path,
        )
        res = evaluate_unseen_generator(checkpoint_path=ckpt_path, test_unseen_loader=mock_batches, device="cpu")

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "eer", "unseen_fake_auc", "unseen_fake_recall"):
        assert key in res and np.isfinite(res[key]) and 0.0 <= res[key] <= 1.0, f"Invalid {key}: {res.get(key)}"
    assert res["holdout_generator"] == "NeuralTextures"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] test_unseen ({res['holdout_generator']}) -> "
        f"AUC={res['unseen_fake_auc']:.4f}, EER={res['eer']:.4f}, F1={res['f1']:.4f}, "
        f"UnseenFakeRecall={res['unseen_fake_recall']:.4f} in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ZeroFake checkpoint on test_unseen LOGO split.")
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

    res = evaluate_unseen_generator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        batch_size=args.batch_size,
    )
    print(
        f"[UNSEEN-GENERATOR RESULTS] mode={res['mode']} | holdout={res['holdout_generator']} | "
        f"N={res['num_samples']} | Acc={res['accuracy']:.4f} | Prec={res['precision']:.4f} | "
        f"Rec={res['recall']:.4f} | F1={res['f1']:.4f} | ROC-AUC={res['roc_auc']:.4f} | "
        f"EER={res['eer']:.4f} | UnseenFakeAUC={res['unseen_fake_auc']:.4f} | "
        f"UnseenFakeRecall={res['unseen_fake_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
