# File: losses/classification.py
# What this file does: Computes the Binary Cross-Entropy classification loss (L_cls) using
#   BCEWithLogitsLoss on raw classifier logits so training remains numerically stable under
#   PyTorch fp16 Automatic Mixed Precision (AMP).
# Inputs:
#   - logits (torch.Tensor or np.ndarray): Raw binary fake logits of shape (B, 1) or (B,).
#   - labels (torch.Tensor or np.ndarray): Ground-truth binary labels y in {0, 1} of shape (B,) or (B, 1).
#   - --dry-run (flag): Tests L_cls forward and backward/value correctness on CPU in <5s.
# Outputs:
#   - loss_cls (torch.Tensor or float): Scalar Binary Cross-Entropy loss L_cls.

import argparse
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


if HAS_TORCH:

    class BinaryClassificationLoss(nn.Module):
        """
        Numerically stable Binary Cross-Entropy loss (`L_cls`) computed directly from raw logits
        using `nn.BCEWithLogitsLoss()`, which combines a Sigmoid layer and the BCELoss in a single
        log-sum-exp stabilized kernel safe for `fp16` mixed precision.
        """

        def __init__(self) -> None:
            super().__init__()
            self.criterion = nn.BCEWithLogitsLoss()

        def forward(self, logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
            """
            Computes `L_cls` given `logits` of shape `(B, 1)` or `(B,)` and `labels` of shape `(B,)` or `(B, 1)`.
            """
            logits_flat = logits.view(-1).float()
            targets_flat = labels.view(-1).float()
            assert logits_flat.shape == targets_flat.shape, (
                f"BinaryClassificationLoss shape mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}"
            )
            loss = self.criterion(logits_flat, targets_flat)
            assert torch.isfinite(loss), "BinaryClassificationLoss produced non-finite value!"
            return loss

else:

    class BinaryClassificationLoss:
        """
        NumPy dry-run fallback for numerically stable BCEWithLogitsLoss.
        Formula: max(x, 0) - x * y + log(1 + exp(-|x|))
        """

        def __init__(self) -> None:
            pass

        def __call__(self, logits: np.ndarray, labels: np.ndarray) -> float:
            return self.forward(logits, labels)

        def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
            x = np.asarray(logits, dtype=np.float64).reshape(-1)
            y = np.asarray(labels, dtype=np.float64).reshape(-1)
            assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
            loss_vec = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
            loss_val = float(np.mean(loss_vec))
            assert np.isfinite(loss_val), "BinaryClassificationLoss produced non-finite value!"
            return loss_val


def run_dry_run() -> None:
    """
    Tests `BinaryClassificationLoss` on synthetic dummy logits and labels on CPU in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing losses/classification.py (BCEWithLogitsLoss L_cls)...")

    loss_fn = BinaryClassificationLoss()
    if HAS_TORCH:
        logits = torch.tensor([[2.0], [-2.0], [1.5], [-1.5]], requires_grad=True)
        labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        loss_scalar = float(loss.item())
    else:
        logits = np.array([[2.0], [-2.0], [1.5], [-1.5]], dtype=np.float32)
        labels = np.array([1, 0, 1, 0], dtype=np.int64)
        loss_scalar = loss_fn(logits, labels)

    assert 0.0 < loss_scalar < 1.0, f"Unexpected loss value for well-aligned logits: {loss_scalar}"
    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] L_cls={loss_scalar:.4f} verified in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary classification loss (L_cls) via BCEWithLogitsLoss.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU loss verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] BinaryClassificationLoss ready. Import from losses.classification or run with --dry-run.")


if __name__ == "__main__":
    main()
