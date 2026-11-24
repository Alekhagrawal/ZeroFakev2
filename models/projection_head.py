# File: models/projection_head.py
# What this file does: Implements a 2-layer MLP projection head (768 -> 256 -> 128) with ReLU activation
#   and L2-normalization that maps 768-d backbone features h into unit-length 128-d contrastive embeddings z.
# Inputs:
#   - h (torch.Tensor or np.ndarray): Backbone feature tensor of shape (B, 768).
#   - --dry-run (flag): Passes dummy (8, 768) features on CPU and verifies output shape (8, 128) and unit L2 norm.
# Outputs:
#   - z (torch.Tensor or np.ndarray): L2-normalized embedding tensor of shape (B, 128) where ||z_i||_2 = 1.

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

    class ProjectionHead(nn.Module):
        """
        Two-layer MLP projection head (`768 -> 256 -> 128`) with ReLU activation and L2 normalization.
        Produces unit-norm embeddings `z = g(h) / ||g(h)||_2` on the 128-d hypersphere for
        Multi-Generator Supervised Contrastive Learning and Multi-Prototype distance calculation.
        """

        def __init__(
            self,
            in_dim: int = 768,
            hidden_dim: int = 256,
            embed_dim: int = 128,
        ) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.hidden_dim = int(hidden_dim)
            self.embed_dim = int(embed_dim)

            self.mlp = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.embed_dim),
            )

        def forward(self, h: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass:
              - Asserts `h` has shape `(B, 768)`.
              - Projects through `768 -> 256 -> 128` MLP and L2-normalizes along `dim=-1`.
              - Asserts output `z` has shape `(B, 128)`.
            """
            assert h.ndim == 2 and h.shape[1] == self.in_dim, (
                f"ProjectionHead expected input shape (B, {self.in_dim}), got {tuple(h.shape)}"
            )
            raw_z = self.mlp(h)
            z = F.normalize(raw_z, p=2, dim=-1, eps=1e-8)
            assert z.ndim == 2 and z.shape[1] == self.embed_dim, (
                f"ProjectionHead expected output shape (B, {self.embed_dim}), got {tuple(z.shape)}"
            )
            return z

else:

    class ProjectionHead:
        """
        NumPy dry-run fallback for ProjectionHead (768 -> 256 -> 128 with ReLU and L2 normalization).
        """

        def __init__(
            self,
            in_dim: int = 768,
            hidden_dim: int = 256,
            embed_dim: int = 128,
        ) -> None:
            self.in_dim = int(in_dim)
            self.hidden_dim = int(hidden_dim)
            self.embed_dim = int(embed_dim)
            rng = np.random.default_rng(101)
            self.w1 = rng.normal(0.0, 0.05, size=(self.in_dim, self.hidden_dim)).astype(np.float32)
            self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)
            self.w2 = rng.normal(0.0, 0.05, size=(self.hidden_dim, self.embed_dim)).astype(np.float32)
            self.b2 = np.zeros((self.embed_dim,), dtype=np.float32)

        def __call__(self, h: np.ndarray) -> np.ndarray:
            return self.forward(h)

        def forward(self, h: np.ndarray) -> np.ndarray:
            assert h.ndim == 2 and h.shape[1] == self.in_dim, (
                f"ProjectionHead expected input shape (B, {self.in_dim}), got {h.shape}"
            )
            hidden = np.maximum(0.0, h @ self.w1 + self.b1)
            raw_z = hidden @ self.w2 + self.b2
            norms = np.linalg.norm(raw_z, ord=2, axis=-1, keepdims=True)
            z = raw_z / np.maximum(norms, 1e-8)
            assert z.shape == (h.shape[0], self.embed_dim), (
                f"ProjectionHead expected output shape ({h.shape[0]}, {self.embed_dim}), got {z.shape}"
            )
            return z.astype(np.float32)


def run_dry_run() -> None:
    """
    Verifies that ProjectionHead maps (8, 768) features to unit-norm (8, 128) embeddings in <5s on CPU.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing models/projection_head.py (768 -> 256 -> 128 + L2 norm)...")

    proj = ProjectionHead(in_dim=768, hidden_dim=256, embed_dim=128)
    if HAS_TORCH:
        h = torch.randn(8, 768)
        z = proj(h)
        assert tuple(z.shape) == (8, 128), f"Expected shape (8, 128), got {tuple(z.shape)}"
        norms = torch.norm(z, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings z are not unit L2-normalized!"
    else:
        h = np.random.randn(8, 768).astype(np.float32)
        z = proj(h)
        assert z.shape == (8, 128), f"Expected shape (8, 128), got {z.shape}"
        norms = np.linalg.norm(z, ord=2, axis=-1)
        assert np.allclose(norms, np.ones_like(norms), atol=1e-5), "Embeddings z are not unit L2-normalized!"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] ProjectionHead mapped (8, 768) -> unit-norm (8, 128) in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2-layer MLP projection head (768 -> 256 -> 128).")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU shape and norm verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] ProjectionHead ready. Import from models.projection_head or run with --dry-run.")


if __name__ == "__main__":
    main()
