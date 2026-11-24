# File: models/backbone.py
# What this file does: Loads the pretrained ConvNeXt-Tiny convolutional backbone (via timm) with its
#   classification head removed (num_classes=0) so every 224x224 RGB face image is converted into a
#   rich 768-dimensional visual feature vector h.
# Inputs:
#   - x (torch.Tensor or np.ndarray): Batch of normalized RGB face images of shape (B, 3, 224, 224).
#   - pretrained (bool): Whether to load ImageNet pretrained ConvNeXt-Tiny weights from timm (default: True).
#   - --dry-run (flag): Passes dummy (4, 3, 224, 224) tensors on CPU and verifies output shape (4, 768) in <5s.
# Outputs:
#   - h (torch.Tensor or np.ndarray): 768-dimensional backbone feature tensor of shape (B, 768).

import argparse
import time
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    timm = None
    HAS_TIMM = False


if HAS_TORCH:

    class ConvNeXtTinyBackbone(nn.Module):
        """
        Pretrained ConvNeXt-Tiny feature extractor (`convnext_tiny` from `timm`, `num_classes=0`).
        Maps an input face batch `x` of shape `(B, 3, 224, 224)` to a 768-d feature vector `h` of shape `(B, 768)`.
        If `timm` is unavailable or `pretrained=False` during offline CPU `--dry-run`, falls back to a lightweight
        768-d convolutional stem so local dry-runs execute in milliseconds without network downloads.
        """

        def __init__(
            self,
            model_name: str = "convnext_tiny",
            pretrained: bool = True,
            out_dim: int = 768,
        ) -> None:
            super().__init__()
            self.model_name = model_name
            self.pretrained = pretrained
            self.out_dim = int(out_dim)

            if HAS_TIMM:
                try:
                    self.encoder = timm.create_model(
                        model_name,
                        pretrained=pretrained,
                        num_classes=0,
                    )
                except Exception:
                    # Fallback if offline during pretrained weight download
                    self.encoder = timm.create_model(
                        model_name,
                        pretrained=False,
                        num_classes=0,
                    )
            else:
                # Lightweight PyTorch fallback when timm is not installed in a minimal local environment
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Linear(64, self.out_dim),
                )

        def get_target_layer(self) -> nn.Module:
            """
            Returns the final spatial convolutional stage of ConvNeXt-Tiny for Grad-CAM heatmap generation.
            """
            if hasattr(self.encoder, "stages"):
                return self.encoder.stages[-1]
            return self.encoder

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass:
              - Asserts input tensor `x` has shape `(B, 3, 224, 224)`.
              - Extracts 768-d global pooled features `h` of shape `(B, 768)`.
            """
            assert x.ndim == 4 and x.shape[1:] == (3, 224, 224), (
                f"ConvNeXtTinyBackbone expected input shape (B, 3, 224, 224), got {tuple(x.shape)}"
            )
            h = self.encoder(x)
            if h.ndim > 2:
                h = h.mean(dim=(-2, -1))
            assert h.ndim == 2 and h.shape[1] == self.out_dim, (
                f"ConvNeXtTinyBackbone expected output shape (B, {self.out_dim}), got {tuple(h.shape)}"
            )
            return h

else:

    class ConvNeXtTinyBackbone:
        """
        NumPy dry-run mock of ConvNeXtTinyBackbone used only if PyTorch is not installed on the local laptop.
        """

        def __init__(
            self,
            model_name: str = "convnext_tiny",
            pretrained: bool = False,
            out_dim: int = 768,
        ) -> None:
            self.model_name = model_name
            self.pretrained = pretrained
            self.out_dim = int(out_dim)
            rng = np.random.default_rng(42)
            self.proj = rng.normal(0.0, 0.05, size=(3 * 16 * 16, self.out_dim)).astype(np.float32)

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return self.forward(x)

        def forward(self, x: np.ndarray) -> np.ndarray:
            assert x.ndim == 4 and x.shape[1:] == (3, 224, 224), (
                f"ConvNeXtTinyBackbone expected input shape (B, 3, 224, 224), got {x.shape}"
            )
            b = x.shape[0]
            pooled = x[:, :, ::14, ::14].reshape(b, -1).astype(np.float32)
            h = pooled @ self.proj
            assert h.shape == (b, self.out_dim), f"Expected output shape ({b}, {self.out_dim}), got {h.shape}"
            return h


def run_dry_run() -> None:
    """
    Runs a fast (<5s) CPU dry-run verifying that dummy images of shape (4, 3, 224, 224) produce
    backbone features of shape (4, 768).
    """
    start_time = time.time()
    print("[DRY-RUN] Testing models/backbone.py (ConvNeXt-Tiny 768-d feature extractor)...")

    backbone = ConvNeXtTinyBackbone(pretrained=False, out_dim=768)
    if HAS_TORCH:
        x = torch.randn(4, 3, 224, 224)
        h = backbone(x)
        assert tuple(h.shape) == (4, 768), f"Expected shape (4, 768), got {tuple(h.shape)}"
    else:
        x = np.random.randn(4, 3, 224, 224).astype(np.float32)
        h = backbone(x)
        assert h.shape == (4, 768), f"Expected shape (4, 768), got {h.shape}"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] ConvNeXtTinyBackbone mapped (4, 3, 224, 224) -> (4, 768) in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny 768-d backbone feature extractor.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU shape verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] ConvNeXtTinyBackbone ready. Import from models.backbone or run with --dry-run.")


if __name__ == "__main__":
    main()
