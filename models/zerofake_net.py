# File: models/zerofake_net.py
# What this file does: Assembles the complete ZeroFake neural network by combining the ConvNeXt-Tiny
#   backbone (768-d), the 2-layer MLP projection head (128-d L2-normalized embedding z), the binary
#   classifier head (fake logit & probability p), and the learnable multi-prototype memory bank
#   (1 Real + K Fake prototypes), supporting all ablation modes ('baseline', 'contrastive', 'single_proto', 'zerofake').
# Inputs:
#   - x (torch.Tensor or np.ndarray): Batch of normalized RGB face images of shape (B, 3, 224, 224).
#   - mode (str): Ablation mode in {'baseline', 'contrastive', 'single_proto', 'zerofake'} (default: 'zerofake').
#   - --dry-run (flag): Runs a forward check across all 4 ablation modes on CPU in under 5 seconds.
# Outputs:
#   - Dictionary containing:
#       * 'features': 768-d backbone representations h of shape (B, 768)
#       * 'embeddings': 128-d L2-normalized embeddings z of shape (B, 128)
#       * 'logits': Binary fake logits of shape (B, 1)
#       * 'probs': Sigmoid fake probabilities p of shape (B, 1)
#       * 'd_real', 'd_fake', 'd_fake_all', 'sim_real', 'sim_fake_all', 'p_real_norm', 'p_fake_norm'
#       * 'evidence_fake': Combined class evidence E_fake(x) in [0, 1] of shape (B,)
#       * 'mode': Active ablation mode string

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.backbone import ConvNeXtTinyBackbone
from models.classifier import BinaryClassifierHead
from models.projection_head import ProjectionHead
from models.prototype import MultiPrototypeLayer

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


VALID_MODES = ("baseline", "contrastive", "single_proto", "zerofake")


def get_mode_loss_weights_and_k(mode: str):
    """
    Returns `(lambda_con, lambda_proto, num_fake_prototypes)` for each ablation configuration
    defined in Section 4.4 of SPEC.md:
      - 'baseline'     (Config A): lambda_con=0.0, lambda_proto=0.0, K=6 (prototypes unused in loss)
      - 'contrastive'  (Config B): lambda_con=0.5, lambda_proto=0.0, K=6 (prototypes unused in loss)
      - 'single_proto' (Config C): lambda_con=0.5, lambda_proto=0.2, K=1 (1 Real + 1 Fake prototype)
      - 'zerofake'     (Config D/E): lambda_con=0.5, lambda_proto=0.2, K=6 (1 Real + 6 Fake prototypes)
    """
    mode_clean = mode.strip().lower()
    if mode_clean not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {VALID_MODES}.")

    if mode_clean == "baseline":
        return 0.0, 0.0, 6
    if mode_clean == "contrastive":
        return 0.5, 0.0, 6
    if mode_clean == "single_proto":
        return 0.5, 0.2, 1
    return 0.5, 0.2, 6


if HAS_TORCH:

    class ZeroFakeNet(nn.Module):
        """
        Unified ZeroFake Detector network combining:
          1. `backbone`: Pretrained ConvNeXt-Tiny (`(B, 3, 224, 224) -> (B, 768)`)
          2. `projection_head`: 2-layer MLP (`768 -> 256 -> 128` + L2 normalization)
          3. `classifier`: Linear binary classification head (`768 -> 1` logit & sigmoid prob `p`)
          4. `prototype_layer`: Learnable prototypes (`1` Real + `K` Fake in 128-d space, where `K=1` for
             `single_proto` and `K=6` for `zerofake`)
        """

        def __init__(
            self,
            mode: str = "zerofake",
            pretrained: bool = True,
            backbone_dim: int = 768,
            hidden_dim: int = 256,
            embed_dim: int = 128,
            num_fake_prototypes: int = 6,
            dropout_p: float = 0.2,
            proto_temp: float = 0.2,
        ) -> None:
            super().__init__()
            mode_clean = mode.strip().lower()
            assert mode_clean in VALID_MODES, f"Unsupported mode '{mode}'. Must be one of {VALID_MODES}."
            self.mode = mode_clean
            self.proto_temp = float(proto_temp)

            lambda_con, lambda_proto, default_k = get_mode_loss_weights_and_k(self.mode)
            self.lambda_con = lambda_con
            self.lambda_proto = lambda_proto
            self.num_fake_prototypes = 1 if self.mode == "single_proto" else int(num_fake_prototypes or default_k)

            self.backbone = ConvNeXtTinyBackbone(
                model_name="convnext_tiny",
                pretrained=pretrained,
                out_dim=backbone_dim,
            )
            self.projection_head = ProjectionHead(
                in_dim=backbone_dim,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
            )
            self.classifier = BinaryClassifierHead(
                in_dim=backbone_dim,
                dropout_p=dropout_p,
            )
            self.prototype_layer = MultiPrototypeLayer(
                embed_dim=embed_dim,
                num_fake_prototypes=self.num_fake_prototypes,
            )

        def forward(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            """
            Executes a complete forward pass and returns all intermediate and output tensors with strict shape checks.
            """
            assert x.ndim == 4 and x.shape[1:] == (3, 224, 224), (
                f"ZeroFakeNet expected input shape (B, 3, 224, 224), got {tuple(x.shape)}"
            )
            b = x.shape[0]

            # 1. Extract 768-d backbone features h
            h = self.backbone(x)
            assert h.shape == (b, 768), f"Expected features h of shape ({b}, 768), got {tuple(h.shape)}"

            # 2. Project to 128-d L2-normalized embedding z
            z = self.projection_head(h)
            assert z.shape == (b, 128), f"Expected embeddings z of shape ({b}, 128), got {tuple(z.shape)}"

            # 3. Compute binary classification logit and fake probability p
            logits, probs = self.classifier(h)
            assert logits.shape == (b, 1) and probs.shape == (b, 1)

            # 4. Compute multi-prototype cosine distances d_real(z) and d_fake(z) = min_j d(z, p_j)
            proto_out = self.prototype_layer(z)
            d_real = proto_out["d_real"]
            d_fake = proto_out["d_fake"]

            # 5. Compute combined fake evidence E_fake(x) per Section 3.3 of SPEC.md
            p_flat = probs.squeeze(-1)
            if self.mode in ("single_proto", "zerofake"):
                p_proto = torch.sigmoid((d_real - d_fake) / max(self.proto_temp, 1e-6))
                evidence_fake = 0.5 * p_flat + 0.5 * p_proto
            else:
                p_proto = p_flat
                evidence_fake = p_flat

            return {
                "features": h,
                "embeddings": z,
                "logits": logits,
                "probs": probs,
                "d_real": d_real,
                "d_fake": d_fake,
                "d_fake_all": proto_out["d_fake_all"],
                "sim_real": proto_out["sim_real"],
                "sim_fake_all": proto_out["sim_fake_all"],
                "nearest_fake_idx": proto_out["nearest_fake_idx"],
                "p_real_norm": proto_out["p_real_norm"],
                "p_fake_norm": proto_out["p_fake_norm"],
                "p_proto": p_proto,
                "evidence_fake": evidence_fake,
                "mode": self.mode,
            }

else:

    class ZeroFakeNet:
        """
        NumPy dry-run fallback for ZeroFakeNet supporting all 4 ablation modes on environments without PyTorch.
        """

        def __init__(
            self,
            mode: str = "zerofake",
            pretrained: bool = False,
            backbone_dim: int = 768,
            hidden_dim: int = 256,
            embed_dim: int = 128,
            num_fake_prototypes: int = 6,
            dropout_p: float = 0.0,
            proto_temp: float = 0.2,
        ) -> None:
            mode_clean = mode.strip().lower()
            assert mode_clean in VALID_MODES, f"Unsupported mode '{mode}'. Must be one of {VALID_MODES}."
            self.mode = mode_clean
            self.proto_temp = float(proto_temp)
            lambda_con, lambda_proto, default_k = get_mode_loss_weights_and_k(self.mode)
            self.lambda_con = lambda_con
            self.lambda_proto = lambda_proto
            self.num_fake_prototypes = 1 if self.mode == "single_proto" else int(num_fake_prototypes or default_k)

            self.backbone = ConvNeXtTinyBackbone(pretrained=False, out_dim=backbone_dim)
            self.projection_head = ProjectionHead(in_dim=backbone_dim, hidden_dim=hidden_dim, embed_dim=embed_dim)
            self.classifier = BinaryClassifierHead(in_dim=backbone_dim, dropout_p=dropout_p)
            self.prototype_layer = MultiPrototypeLayer(
                embed_dim=embed_dim,
                num_fake_prototypes=self.num_fake_prototypes,
            )

        def __call__(self, x: np.ndarray) -> Dict:
            return self.forward(x)

        def forward(self, x: np.ndarray) -> Dict:
            assert x.ndim == 4 and x.shape[1:] == (3, 224, 224), (
                f"ZeroFakeNet expected input shape (B, 3, 224, 224), got {x.shape}"
            )
            b = x.shape[0]
            h = self.backbone(x)
            z = self.projection_head(h)
            logits, probs = self.classifier(h)
            proto_out = self.prototype_layer(z)

            d_real = proto_out["d_real"]
            d_fake = proto_out["d_fake"]
            p_flat = probs.squeeze(-1)

            if self.mode in ("single_proto", "zerofake"):
                diff = np.clip((d_real - d_fake) / max(self.proto_temp, 1e-6), -30.0, 30.0)
                p_proto = (1.0 / (1.0 + np.exp(-diff))).astype(np.float32)
                evidence_fake = (0.5 * p_flat + 0.5 * p_proto).astype(np.float32)
            else:
                p_proto = p_flat
                evidence_fake = p_flat

            assert h.shape == (b, 768) and z.shape == (b, 128) and logits.shape == (b, 1)
            return {
                "features": h,
                "embeddings": z,
                "logits": logits,
                "probs": probs,
                "d_real": d_real,
                "d_fake": d_fake,
                "d_fake_all": proto_out["d_fake_all"],
                "sim_real": proto_out["sim_real"],
                "sim_fake_all": proto_out["sim_fake_all"],
                "nearest_fake_idx": proto_out["nearest_fake_idx"],
                "p_real_norm": proto_out["p_real_norm"],
                "p_fake_norm": proto_out["p_fake_norm"],
                "p_proto": p_proto,
                "evidence_fake": evidence_fake,
                "mode": self.mode,
            }


def run_dry_run(selected_mode: str = "zerofake") -> None:
    """
    Tests ZeroFakeNet across all four ablation modes ('baseline', 'contrastive', 'single_proto', 'zerofake')
    on synthetic dummy face images (4, 3, 224, 224) on CPU in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing models/zerofake_net.py across all 4 ablation modes...")

    modes_to_test = list(VALID_MODES) if selected_mode == "zerofake" else [selected_mode]
    for m in modes_to_test:
        net = ZeroFakeNet(mode=m, pretrained=False)
        expected_k = 1 if m == "single_proto" else 6

        if HAS_TORCH:
            x = torch.randn(4, 3, 224, 224)
            out = net(x)
            assert tuple(out["features"].shape) == (4, 768)
            assert tuple(out["embeddings"].shape) == (4, 128)
            assert tuple(out["logits"].shape) == (4, 1)
            assert tuple(out["probs"].shape) == (4, 1)
            assert tuple(out["d_real"].shape) == (4,)
            assert tuple(out["d_fake"].shape) == (4,)
            assert tuple(out["d_fake_all"].shape) == (4, expected_k)
        else:
            x = np.random.randn(4, 3, 224, 224).astype(np.float32)
            out = net(x)
            assert out["features"].shape == (4, 768)
            assert out["embeddings"].shape == (4, 128)
            assert out["logits"].shape == (4, 1)
            assert out["probs"].shape == (4, 1)
            assert out["d_real"].shape == (4,)
            assert out["d_fake"].shape == (4,)
            assert out["d_fake_all"].shape == (4, expected_k)

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Verified ZeroFakeNet modes {modes_to_test} "
        f"((4, 3, 224, 224) -> h:(4, 768), z:(4, 128), p:(4, 1), d_fake:(4,)) in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZeroFake neural network wrapper supporting all ablation modes.")
    parser.add_argument(
        "--mode",
        type=str,
        default="zerofake",
        choices=list(VALID_MODES),
        help="Ablation mode: baseline, contrastive, single_proto, or zerofake.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU verification across modes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(selected_mode=args.mode)
        return
    print(f"[INFO] ZeroFakeNet initialized in mode='{args.mode}'.")


if __name__ == "__main__":
    main()
