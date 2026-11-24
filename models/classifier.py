# File: models/classifier.py
# What this file does: Implements the binary classification head that maps 768-dimensional ConvNeXt-Tiny
#   backbone features h to a raw binary fake logit s and sigmoid fake probability p = P(y=1 | x) in [0, 1].
# Inputs:
#   - h (torch.Tensor or np.ndarray): Backbone feature tensor of shape (B, 768).
#   - --dry-run (flag): Passes dummy (8, 768) features on CPU and verifies logits (8, 1) and probabilities in [0, 1].
# Outputs:
#   - logits (torch.Tensor or np.ndarray): Raw binary classification logits of shape (B, 1).
#   - probs (torch.Tensor or np.ndarray): Sigmoid fake probabilities p = sigma(logits) of shape (B, 1) in [0, 1].

import argparse
import time
from typing import Tuple

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

    class BinaryClassifierHead(nn.Module):
        """
        Linear classification head mapping 768-d backbone features `h` to a binary fake logit `s`
        and fake probability `p = sigma(s) in [0, 1]` (where `y = 0` is Real and `y = 1` is Fake).
        """

        def __init__(self, in_dim: int = 768, dropout_p: float = 0.2) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.dropout = nn.Dropout(p=float(dropout_p)) if dropout_p > 0 else nn.Identity()
            self.fc = nn.Linear(self.in_dim, 1)

        def forward(self, h: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Forward pass:
              - Asserts `h` has shape `(B, 768)`.
              - Returns `(logits, probs)` each of shape `(B, 1)`.
            """
            assert h.ndim == 2 and h.shape[1] == self.in_dim, (
                f"BinaryClassifierHead expected input shape (B, {self.in_dim}), got {tuple(h.shape)}"
            )
            logits = self.fc(self.dropout(h))
            probs = torch.sigmoid(logits)
            assert logits.shape == (h.shape[0], 1) and probs.shape == (h.shape[0], 1), (
                f"BinaryClassifierHead expected output shapes (B, 1), got {tuple(logits.shape)} and {tuple(probs.shape)}"
            )
            return logits, probs

else:

    class BinaryClassifierHead:
        """
        NumPy dry-run fallback for BinaryClassifierHead (768 -> 1 linear logit + sigmoid probability).
        """

        def __init__(self, in_dim: int = 768, dropout_p: float = 0.0) -> None:
            self.in_dim = int(in_dim)
            rng = np.random.default_rng(202)
            self.w = rng.normal(0.0, 0.05, size=(self.in_dim, 1)).astype(np.float32)
            self.b = np.zeros((1,), dtype=np.float32)

        def __call__(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return self.forward(h)

        def forward(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            assert h.ndim == 2 and h.shape[1] == self.in_dim, (
                f"BinaryClassifierHead expected input shape (B, {self.in_dim}), got {h.shape}"
            )
            logits = (h @ self.w + self.b).astype(np.float32)
            probs = (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(np.float32)
            assert logits.shape == (h.shape[0], 1) and probs.shape == (h.shape[0], 1)
            return logits, probs


def run_dry_run() -> None:
    """
    Verifies that BinaryClassifierHead maps (8, 768) features to (8, 1) logits and probabilities in [0, 1] in <5s.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing models/classifier.py (768 -> 1 binary logit & fake probability p)...")

    clf = BinaryClassifierHead(in_dim=768)
    if HAS_TORCH:
        h = torch.randn(8, 768)
        logits, probs = clf(h)
        assert tuple(logits.shape) == (8, 1) and tuple(probs.shape) == (8, 1)
        assert bool((probs >= 0.0).all() and (probs <= 1.0).all()), "Probabilities must lie in [0, 1]!"
    else:
        h = np.random.randn(8, 768).astype(np.float32)
        logits, probs = clf(h)
        assert logits.shape == (8, 1) and probs.shape == (8, 1)
        assert bool((probs >= 0.0).all() and (probs <= 1.0).all()), "Probabilities must lie in [0, 1]!"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] BinaryClassifierHead mapped (8, 768) -> logits (8, 1) & p in [0, 1] in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary classification head mapping 768-d features to fake prob p.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU shape verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] BinaryClassifierHead ready. Import from models.classifier or run with --dry-run.")


if __name__ == "__main__":
    main()
