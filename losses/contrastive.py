# File: losses/contrastive.py
# What this file does: Implements Multi-Generator Supervised Contrastive Loss (L_con) with temperature
#   tau=0.07 on L2-normalized 128-d embeddings z. Pulls all authentic (Real) embeddings together and
#   aligns Fake embeddings across different manipulation generators, with explicit safeguards against
#   batches or anchors having zero positive pairs so it never produces NaN or Inf.
# Inputs:
#   - z (torch.Tensor or np.ndarray): L2-normalized embedding batch of shape (B, 128).
#   - labels (torch.Tensor or np.ndarray): Binary class labels y in {0, 1} of shape (B,).
#   - temperature (float): Contrastive temperature tau (default: 0.07 per Section 2.2 of SPEC.md).
#   - --dry-run (flag): Verifies L_con on both balanced batches and degenerate zero-positive-pair batches in <5s.
# Outputs:
#   - loss_con (torch.Tensor or float): Scalar Multi-Generator Supervised Contrastive Loss L_con (guaranteed finite).

import argparse
import time

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

    class SupervisedContrastiveLoss(nn.Module):
        """
        Multi-Generator Supervised Contrastive Loss (`L_con`) with temperature `tau = 0.07`.
        Formula (Section 2.2 of SPEC.md):
          L_con = (1 / |I_valid|) * sum_{i in I_valid} (-1 / |P(i)|) * sum_{p in P(i)}
                  log( exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau) )
        Safeguards:
          1. Forces fp32 computation inside the loss and subtracts row-wise maximum logits for log-sum-exp stability.
          2. Masks out anchors `i` with `|P(i)| == 0`; if the entire batch has zero positive pairs (e.g., B <= 1 or
             singleton classes), returns a clean `0.0` tensor tied to `z` (zero gradient, never NaN).
        """

        def __init__(self, temperature: float = 0.07) -> None:
            super().__init__()
            assert temperature > 0.0, "Contrastive temperature tau must be > 0."
            self.temperature = float(temperature)

        def forward(self, z: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
            assert z.ndim == 2, f"SupervisedContrastiveLoss expected 2D embeddings (B, D), got {tuple(z.shape)}"
            device_type = "cuda" if z.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                z_f32 = z.float()
                b = z_f32.shape[0]
                y = labels.view(-1)
                assert y.shape[0] == b, f"Label count {y.shape[0]} does not match batch size {b}!"

                if b <= 1:
                    return z_f32.sum() * 0.0

                # Compute strictly in fp32 for mixed-precision numerical stability
                z_norm = F.normalize(z_f32, p=2, dim=-1, eps=1e-8)
                sim_matrix = torch.matmul(z_norm, z_norm.t()) / self.temperature  # (B, B) in float32

                # Mask out self-comparisons A(i) = {1..B} \ {i}
                device = z_f32.device
                self_mask = torch.eye(b, dtype=torch.bool, device=device)
                non_self_mask = ~self_mask

                # Positive pair mask P(i) = {p in A(i) : y_p == y_i}
                y_col = y.unsqueeze(1)
                pos_mask = (y_col == y_col.t()) & non_self_mask
                pos_counts = pos_mask.sum(dim=1)  # (B,)

                # Safeguard: if zero positive pairs exist in the entire batch, return 0.0 without NaN
                valid_anchors = pos_counts > 0
                if not bool(valid_anchors.any()):
                    return z_f32.sum() * 0.0

                # Numerical stability: subtract max over non-self entries per row using dtype-safe finfo.min
                mask_fill_val = torch.finfo(sim_matrix.dtype).min
                masked_for_max = sim_matrix.masked_fill(self_mask, mask_fill_val)
                row_max = masked_for_max.max(dim=1, keepdim=True).values.detach()
                logits = sim_matrix - row_max

                exp_logits = torch.exp(logits) * non_self_mask.float()
                log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
                log_prob = logits - log_denom  # (B, B)

                # Average log_prob over positive pairs P(i) for valid anchors only
                pos_log_prob_sum = (log_prob * pos_mask.float()).sum(dim=1)
                safe_pos_counts = pos_counts.float().clamp(min=1.0)
                anchor_losses = -pos_log_prob_sum / safe_pos_counts

                loss = anchor_losses[valid_anchors].mean()
                assert torch.isfinite(loss), "SupervisedContrastiveLoss produced non-finite value!"
                return loss

else:

    class SupervisedContrastiveLoss:
        """
        NumPy dry-run fallback for SupervisedContrastiveLoss (tau = 0.07) with zero-positive-pair safeguard.
        """

        def __init__(self, temperature: float = 0.07) -> None:
            assert temperature > 0.0
            self.temperature = float(temperature)

        def __call__(self, z: np.ndarray, labels: np.ndarray) -> float:
            return self.forward(z, labels)

        def forward(self, z: np.ndarray, labels: np.ndarray) -> float:
            z_arr = np.asarray(z, dtype=np.float64)
            y_arr = np.asarray(labels).reshape(-1)
            assert z_arr.ndim == 2 and z_arr.shape[0] == y_arr.shape[0]
            b = z_arr.shape[0]
            if b <= 1:
                return 0.0

            norms = np.linalg.norm(z_arr, ord=2, axis=-1, keepdims=True)
            z_norm = z_arr / np.maximum(norms, 1e-8)
            sim_matrix = (z_norm @ z_norm.T) / self.temperature

            self_mask = np.eye(b, dtype=bool)
            non_self_mask = ~self_mask
            pos_mask = (y_arr[:, None] == y_arr[None, :]) & non_self_mask
            pos_counts = pos_mask.sum(axis=1)

            valid_anchors = pos_counts > 0
            if not np.any(valid_anchors):
                return 0.0

            masked_sim = np.where(self_mask, np.finfo(sim_matrix.dtype).min, sim_matrix)
            row_max = np.max(masked_sim, axis=1, keepdims=True)
            logits = sim_matrix - row_max

            exp_logits = np.exp(logits) * non_self_mask.astype(np.float64)
            log_denom = np.log(np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12))
            log_prob = logits - log_denom

            pos_sum = (log_prob * pos_mask.astype(np.float64)).sum(axis=1)
            anchor_losses = -pos_sum / np.maximum(pos_counts.astype(np.float64), 1.0)
            loss_val = float(np.mean(anchor_losses[valid_anchors]))
            assert np.isfinite(loss_val), "SupervisedContrastiveLoss produced non-finite value!"
            return loss_val


def run_dry_run() -> None:
    """
    Tests `SupervisedContrastiveLoss` on both a standard multi-generator batch and a degenerate
    batch with zero positive pairs to confirm it never produces NaN.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing losses/contrastive.py (L_con tau=0.07 + zero-positive-pair safeguard)...")

    con_fn = SupervisedContrastiveLoss(temperature=0.07)
    if HAS_TORCH:
        # 1. Normal balanced batch (4 Real, 4 Fake)
        z = F.normalize(torch.randn(8, 128, requires_grad=True), p=2, dim=-1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        loss_normal = con_fn(z, labels)
        loss_normal.backward()
        assert torch.isfinite(loss_normal) and float(loss_normal.item()) > 0.0

        # 2. Edge case: batch with zero positive pairs (1 Real, 1 Fake) -> must return 0.0, not NaN
        z_edge = F.normalize(torch.randn(2, 128, requires_grad=True), p=2, dim=-1)
        labels_edge = torch.tensor([0, 1], dtype=torch.long)
        loss_edge = con_fn(z_edge, labels_edge)
        assert torch.isfinite(loss_edge) and float(loss_edge.item()) == 0.0
        val_normal = float(loss_normal.item())
    else:
        z = np.random.randn(8, 128).astype(np.float32)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        val_normal = con_fn(z, labels)
        assert np.isfinite(val_normal) and val_normal > 0.0

        z_edge = np.random.randn(2, 128).astype(np.float32)
        labels_edge = np.array([0, 1], dtype=np.int64)
        loss_edge = con_fn(z_edge, labels_edge)
        assert np.isfinite(loss_edge) and loss_edge == 0.0

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] L_con={val_normal:.4f} (normal) and L_con=0.0000 (zero-positive-pairs) verified in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Generator Supervised Contrastive Loss (L_con).")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive temperature tau (default: 0.07).")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU contrastive loss verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] SupervisedContrastiveLoss ready. Import from losses.contrastive or run with --dry-run.")


if __name__ == "__main__":
    main()
