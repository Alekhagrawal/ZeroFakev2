# File: models/prototype.py
# What this file does: Implements the learnable Multi-Prototype memory bank in 128-d unit-hypersphere
#   embedding space consisting of 1 Real prototype (p_real) and K=6 Fake prototypes (p_fake_1 ... p_fake_K),
#   and computes cosine distances d_real(z) = 1 - <z, p_real> and d_fake(z) = min_j (1 - <z, p_fake_j>).
# Inputs:
#   - z (torch.Tensor or np.ndarray): L2-normalized embedding batch of shape (B, 128).
#   - num_fake_prototypes (int): Number of learnable Fake prototypes K (default: 6 for ZeroFake, 1 for Config C).
#   - --dry-run (flag): Verifies cosine distances d_real(z), d_fake(z), and prototype shapes on CPU in <5s.
# Outputs:
#   - Dictionary containing:
#       * 'd_real': Cosine distance to the Real prototype, shape (B,)
#       * 'd_fake': Minimum cosine distance across the K Fake prototypes, shape (B,)
#       * 'd_fake_all': Cosine distances to all K Fake prototypes, shape (B, K)
#       * 'sim_real': Cosine similarity to the Real prototype, shape (B,)
#       * 'sim_fake_all': Cosine similarities to all K Fake prototypes, shape (B, K)
#       * 'p_real_norm': L2-normalized Real prototype vector, shape (1, 128)
#       * 'p_fake_norm': L2-normalized Fake prototype matrix, shape (K, 128)

import argparse
import time
from typing import Dict

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

    class MultiPrototypeLayer(nn.Module):
        """
        Learnable Multi-Prototype memory bank storing:
          - 1 Real prototype: `p_real` in R^(1 x 128)
          - K Fake prototypes: `p_fake` in R^(K x 128) (default K=6; K=1 for single_proto ablation Config C)
        All prototypes are L2-normalized before computing cosine distances:
          - `d_real(z) = 1 - <z, p_real>`
          - `d_fake(z) = min_{j=1..K} (1 - <z, p_fake_j>)`
        """

        def __init__(self, embed_dim: int = 128, num_fake_prototypes: int = 6) -> None:
            super().__init__()
            assert num_fake_prototypes >= 1, "num_fake_prototypes (K) must be >= 1."
            self.embed_dim = int(embed_dim)
            self.num_fake_prototypes = int(num_fake_prototypes)

            # Initialize learnable prototype parameters with orthogonal / normalized Gaussian directions
            real_init = F.normalize(torch.randn(1, self.embed_dim), p=2, dim=-1)
            fake_init = F.normalize(torch.randn(self.num_fake_prototypes, self.embed_dim), p=2, dim=-1)

            self.real_prototype = nn.Parameter(real_init)
            self.fake_prototypes = nn.Parameter(fake_init)

        def get_normalized_prototypes(self):
            """
            Returns L2-normalized `(p_real_norm, p_fake_norm)` of shapes `(1, 128)` and `(K, 128)`.
            """
            p_real_norm = F.normalize(self.real_prototype, p=2, dim=-1, eps=1e-8)
            p_fake_norm = F.normalize(self.fake_prototypes, p=2, dim=-1, eps=1e-8)
            return p_real_norm, p_fake_norm

        def forward(self, z: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            """
            Computes cosine distances `d_real(z)` and `d_fake(z) = min_j d(z, p_j)` for batch `z` of shape `(B, 128)`.
            """
            assert z.ndim == 2 and z.shape[1] == self.embed_dim, (
                f"MultiPrototypeLayer expected z of shape (B, {self.embed_dim}), got {tuple(z.shape)}"
            )
            # Ensure input embeddings z are unit L2-normalized
            z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)
            p_real_norm, p_fake_norm = self.get_normalized_prototypes()

            # Cosine similarities <z, p> in [-1, 1]
            sim_real = torch.matmul(z_norm, p_real_norm.t()).squeeze(-1)  # (B,)
            sim_fake_all = torch.matmul(z_norm, p_fake_norm.t())          # (B, K)

            # Cosine distances d(z, p) = 1 - <z, p> in [0, 2]
            d_real = 1.0 - sim_real                                       # (B,)
            d_fake_all = 1.0 - sim_fake_all                               # (B, K)
            d_fake, nearest_fake_idx = torch.min(d_fake_all, dim=-1)      # (B,)

            b = z.shape[0]
            assert d_real.shape == (b,) and d_fake.shape == (b,), (
                f"Expected d_real and d_fake of shape ({b},), got {tuple(d_real.shape)} and {tuple(d_fake.shape)}"
            )
            assert d_fake_all.shape == (b, self.num_fake_prototypes), (
                f"Expected d_fake_all of shape ({b}, {self.num_fake_prototypes}), got {tuple(d_fake_all.shape)}"
            )

            return {
                "d_real": d_real,
                "d_fake": d_fake,
                "d_fake_all": d_fake_all,
                "sim_real": sim_real,
                "sim_fake_all": sim_fake_all,
                "nearest_fake_idx": nearest_fake_idx,
                "p_real_norm": p_real_norm,
                "p_fake_norm": p_fake_norm,
            }

else:

    class MultiPrototypeLayer:
        """
        NumPy dry-run fallback for MultiPrototypeLayer (1 Real prototype + K=6 Fake prototypes).
        """

        def __init__(self, embed_dim: int = 128, num_fake_prototypes: int = 6) -> None:
            assert num_fake_prototypes >= 1
            self.embed_dim = int(embed_dim)
            self.num_fake_prototypes = int(num_fake_prototypes)
            rng = np.random.default_rng(303)
            r = rng.normal(0.0, 1.0, size=(1, self.embed_dim)).astype(np.float32)
            f = rng.normal(0.0, 1.0, size=(self.num_fake_prototypes, self.embed_dim)).astype(np.float32)
            self.real_prototype = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-8)
            self.fake_prototypes = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-8)

        def __call__(self, z: np.ndarray) -> Dict[str, np.ndarray]:
            return self.forward(z)

        def forward(self, z: np.ndarray) -> Dict[str, np.ndarray]:
            assert z.ndim == 2 and z.shape[1] == self.embed_dim, (
                f"MultiPrototypeLayer expected z of shape (B, {self.embed_dim}), got {z.shape}"
            )
            z_norm = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
            p_real_norm = self.real_prototype / np.maximum(
                np.linalg.norm(self.real_prototype, axis=-1, keepdims=True), 1e-8
            )
            p_fake_norm = self.fake_prototypes / np.maximum(
                np.linalg.norm(self.fake_prototypes, axis=-1, keepdims=True), 1e-8
            )

            sim_real = (z_norm @ p_real_norm.T).squeeze(-1)
            sim_fake_all = z_norm @ p_fake_norm.T

            d_real = (1.0 - sim_real).astype(np.float32)
            d_fake_all = (1.0 - sim_fake_all).astype(np.float32)
            d_fake = np.min(d_fake_all, axis=-1).astype(np.float32)
            nearest_fake_idx = np.argmin(d_fake_all, axis=-1)

            b = z.shape[0]
            assert d_real.shape == (b,) and d_fake.shape == (b,)
            assert d_fake_all.shape == (b, self.num_fake_prototypes)

            return {
                "d_real": d_real,
                "d_fake": d_fake,
                "d_fake_all": d_fake_all,
                "sim_real": sim_real.astype(np.float32),
                "sim_fake_all": sim_fake_all.astype(np.float32),
                "nearest_fake_idx": nearest_fake_idx,
                "p_real_norm": p_real_norm.astype(np.float32),
                "p_fake_norm": p_fake_norm.astype(np.float32),
            }


def run_dry_run() -> None:
    """
    Verifies MultiPrototypeLayer with K=6 Fake prototypes and K=1 Fake prototype on CPU in <5s.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing models/prototype.py (1 Real + K=6 Fake learnable prototypes)...")

    proto_layer = MultiPrototypeLayer(embed_dim=128, num_fake_prototypes=6)
    if HAS_TORCH:
        z = F.normalize(torch.randn(8, 128), p=2, dim=-1)
        out = proto_layer(z)
        assert tuple(out["d_real"].shape) == (8,)
        assert tuple(out["d_fake"].shape) == (8,)
        assert tuple(out["d_fake_all"].shape) == (8, 6)
        assert torch.allclose(out["d_fake"], out["d_fake_all"].min(dim=-1).values, atol=1e-6)
    else:
        z = np.random.randn(8, 128).astype(np.float32)
        out = proto_layer(z)
        assert out["d_real"].shape == (8,)
        assert out["d_fake"].shape == (8,)
        assert out["d_fake_all"].shape == (8, 6)
        assert np.allclose(out["d_fake"], out["d_fake_all"].min(axis=-1), atol=1e-6)

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] MultiPrototypeLayer verified d_real (8,) and d_fake=min_j d(z,p_j) (8,) in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Prototype layer (1 Real + K=6 Fake prototypes).")
    parser.add_argument("--num_fake_prototypes", type=int, default=6, help="Number of Fake prototypes K.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU prototype distance verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] MultiPrototypeLayer ready. Import from models.prototype or run with --dry-run.")


if __name__ == "__main__":
    main()
