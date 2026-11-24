# File: losses/prototype_loss.py
# What this file does: Computes the Multi-Prototype distance loss (L_proto) that pulls each sample's
#   L2-normalized embedding z closer to its nearest same-class prototype (d_real(z) for Real label 0,
#   d_fake(z) = min_j d(z, p_fake_j) for Fake label 1), pushes opposite-class prototypes apart by a
#   cosine margin m, and regularizes the K Fake prototypes to remain diverse and non-collapsed.
# Inputs:
#   - d_real (torch.Tensor or np.ndarray): Cosine distances to the Real prototype of shape (B,).
#   - d_fake (torch.Tensor or np.ndarray): Minimum cosine distances to the K Fake prototypes of shape (B,).
#   - labels (torch.Tensor or np.ndarray): Ground-truth binary labels y in {0, 1} of shape (B,).
#   - p_fake_norm (torch.Tensor or np.ndarray, optional): L2-normalized Fake prototypes of shape (K, 128).
#   - --dry-run (flag): Verifies L_proto computation and gradient flow on CPU in <5s.
# Outputs:
#   - loss_proto (torch.Tensor or float): Scalar Multi-Prototype distance loss L_proto.

import argparse
import time
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


if HAS_TORCH:

    class PrototypeDistanceLoss(nn.Module):
        """
        Multi-Prototype Distance Loss (`L_proto`, Section 2.3 of SPEC.md):
          1. Same-Class Pull Term:
             `L_pull = mean_i [ (1 - y_i) * d_real(z_i) + y_i * d_fake(z_i) ]`
             where `d_fake(z_i) = min_{j=1..K} d(z_i, p_fake_j)`.
          2. Opposite-Class Margin Separation Term (optional margin `m`, default 0.5):
             `L_sep  = mean_i [ (1 - y_i) * max(0, m - d_fake(z_i)) + y_i * max(0, m - d_real(z_i)) ]`
          3. Fake-Prototype Diversity Regularizer (when `K > 1` and `p_fake_norm` is provided):
             Penalizes positive pairwise cosine similarity among the `K` Fake prototypes so they capture
             distinct manipulation families rather than collapsing onto a single vector.
        """

        def __init__(self, margin: float = 0.5, lambda_div: float = 0.1) -> None:
            super().__init__()
            self.margin = float(margin)
            self.lambda_div = float(lambda_div)

        def forward(
            self,
            d_real: "torch.Tensor",
            d_fake: "torch.Tensor",
            labels: "torch.Tensor",
            p_fake_norm: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            device_type = "cuda" if d_real.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                d_r = d_real.float().view(-1)
                d_f = d_fake.float().view(-1)
                y = labels.float().view(-1)
                p_fake_f32 = p_fake_norm.float() if p_fake_norm is not None else None

                assert d_r.shape == y.shape and d_f.shape == y.shape, (
                    f"PrototypeDistanceLoss shape mismatch: d_real={tuple(d_r.shape)}, "
                    f"d_fake={tuple(d_f.shape)}, labels={tuple(y.shape)}"
                )

                # 1. Pull embedding z toward its same-class prototype (d_real for label 0, d_fake for label 1)
                pull_loss = (1.0 - y) * d_r + y * d_f

                # 2. Margin separation from opposite-class prototype
                if self.margin > 0.0:
                    sep_real = F.relu(self.margin - d_f)
                    sep_fake = F.relu(self.margin - d_r)
                    sep_loss = (1.0 - y) * sep_real + y * sep_fake
                    base_loss = (pull_loss + sep_loss).mean()
                else:
                    base_loss = pull_loss.mean()

                # 3. Fake prototype diversity regularization (when K > 1)
                if p_fake_f32 is not None and p_fake_f32.ndim == 2 and p_fake_f32.shape[0] > 1 and self.lambda_div > 0.0:
                    k = p_fake_f32.shape[0]
                    pf = F.normalize(p_fake_f32, p=2, dim=-1, eps=1e-8)
                    sim_mat = torch.matmul(pf, pf.t())
                    triu_mask = torch.triu(torch.ones((k, k), dtype=torch.bool, device=pf.device), diagonal=1)
                    pairwise_sims = F.relu(sim_mat[triu_mask])
                    div_loss = pairwise_sims.mean() if pairwise_sims.numel() > 0 else (base_loss * 0.0)
                    total_proto_loss = base_loss + self.lambda_div * div_loss
                else:
                    total_proto_loss = base_loss

                assert torch.isfinite(total_proto_loss), "PrototypeDistanceLoss produced non-finite value!"
                return total_proto_loss

else:

    class PrototypeDistanceLoss:
        """
        NumPy dry-run fallback for PrototypeDistanceLoss (L_proto).
        """

        def __init__(self, margin: float = 0.5, lambda_div: float = 0.1) -> None:
            self.margin = float(margin)
            self.lambda_div = float(lambda_div)

        def __call__(
            self,
            d_real: np.ndarray,
            d_fake: np.ndarray,
            labels: np.ndarray,
            p_fake_norm: Optional[np.ndarray] = None,
        ) -> float:
            return self.forward(d_real, d_fake, labels, p_fake_norm=p_fake_norm)

        def forward(
            self,
            d_real: np.ndarray,
            d_fake: np.ndarray,
            labels: np.ndarray,
            p_fake_norm: Optional[np.ndarray] = None,
        ) -> float:
            d_r = np.asarray(d_real, dtype=np.float64).reshape(-1)
            d_f = np.asarray(d_fake, dtype=np.float64).reshape(-1)
            y = np.asarray(labels, dtype=np.float64).reshape(-1)
            assert d_r.shape == y.shape and d_f.shape == y.shape

            pull_loss = (1.0 - y) * d_r + y * d_f
            if self.margin > 0.0:
                sep_loss = (1.0 - y) * np.maximum(0.0, self.margin - d_f) + y * np.maximum(0.0, self.margin - d_r)
                base_loss = float(np.mean(pull_loss + sep_loss))
            else:
                base_loss = float(np.mean(pull_loss))

            if p_fake_norm is not None and p_fake_norm.ndim == 2 and p_fake_norm.shape[0] > 1 and self.lambda_div > 0.0:
                k = p_fake_norm.shape[0]
                norms = np.linalg.norm(p_fake_norm, ord=2, axis=-1, keepdims=True)
                pf = p_fake_norm / np.maximum(norms, 1e-8)
                sim_mat = pf @ pf.T
                iu = np.triu_indices(k, k=1)
                div_loss = float(np.mean(np.maximum(0.0, sim_mat[iu])))
                total_proto_loss = base_loss + self.lambda_div * div_loss
            else:
                total_proto_loss = base_loss

            assert np.isfinite(total_proto_loss), "PrototypeDistanceLoss produced non-finite value!"
            return float(total_proto_loss)


def run_dry_run() -> None:
    """
    Tests `PrototypeDistanceLoss` on synthetic cosine distances and K=6 Fake prototypes on CPU in <5s.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing losses/prototype_loss.py (L_proto pull + margin + diversity)...")

    proto_fn = PrototypeDistanceLoss(margin=0.5, lambda_div=0.1)
    if HAS_TORCH:
        d_real = torch.tensor([0.1, 0.2, 0.8, 0.9], requires_grad=True)
        d_fake = torch.tensor([0.7, 0.8, 0.15, 0.1], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        p_fake = F.normalize(torch.randn(6, 128, requires_grad=True), p=2, dim=-1)
        loss = proto_fn(d_real, d_fake, labels, p_fake_norm=p_fake)
        loss.backward()
        assert d_real.grad is not None and d_fake.grad is not None
        loss_val = float(loss.item())
    else:
        d_real = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        d_fake = np.array([0.7, 0.8, 0.15, 0.1], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        p_fake = np.random.randn(6, 128).astype(np.float32)
        loss_val = proto_fn(d_real, d_fake, labels, p_fake_norm=p_fake)

    assert loss_val > 0.0 and np.isfinite(loss_val)
    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] L_proto={loss_val:.4f} verified in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Prototype distance loss (L_proto).")
    parser.add_argument("--margin", type=float, default=0.5, help="Opposite-class prototype separation margin.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU prototype loss verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] PrototypeDistanceLoss ready. Import from losses.prototype_loss or run with --dry-run.")


if __name__ == "__main__":
    main()
