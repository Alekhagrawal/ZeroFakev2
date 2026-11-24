# File: losses/total_loss.py
# What this file does: Combines Binary Classification Loss (L_cls), Multi-Generator Supervised Contrastive
#   Loss (L_con), and Multi-Prototype Distance Loss (L_proto) into the unified training objective:
#     L_total = L_cls + lambda_1 * L_con + lambda_2 * L_proto
#   with default weights lambda_1=0.5 and lambda_2=0.2 (configurable via configs/default.yaml and ablation mode).
# Inputs:
#   - model_outputs (dict): Output dictionary from ZeroFakeNet containing 'logits', 'embeddings', 'd_real', 'd_fake', 'p_fake_norm'.
#   - labels (torch.Tensor or np.ndarray): Ground-truth binary labels y in {0, 1} of shape (B,).
#   - config_path (str, optional): Path to YAML config file (default: configs/default.yaml).
#   - --dry-run (flag): Runs an end-to-end forward/loss check across all 4 ablation modes on CPU in <5s.
# Outputs:
#   - Dictionary containing 'loss_total', 'loss_cls', 'loss_con', 'loss_proto', 'lambda_1', and 'lambda_2'.

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.classification import BinaryClassificationLoss
from losses.contrastive import SupervisedContrastiveLoss
from losses.prototype_loss import PrototypeDistanceLoss
from models.zerofake_net import VALID_MODES, ZeroFakeNet

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


def load_loss_config(config_path: str = "configs/default.yaml") -> Dict[str, float]:
    """
    Loads `lambda_1`, `lambda_2`, `temperature`, `proto_margin`, and `lambda_div` from `configs/default.yaml`.
    Falls back to lightweight regex parsing if PyYAML is not installed locally.
    """
    defaults = {
        "lambda_1": 0.5,
        "lambda_2": 0.2,
        "temperature": 0.07,
        "proto_margin": 0.5,
        "lambda_div": 0.1,
    }
    if not config_path or not os.path.exists(config_path):
        return defaults

    if HAS_YAML:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        loss_cfg = data.get("loss", {})
        for k in defaults:
            if k in loss_cfg:
                defaults[k] = float(loss_cfg[k])
        return defaults

    # Minimal parser when PyYAML is not installed in a local dry-run environment
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    for k in list(defaults.keys()):
        match = re.search(rf"^\s*{k}\s*:\s*([0-9.]+)", text, flags=re.MULTILINE)
        if match:
            defaults[k] = float(match.group(1))
    return defaults


def resolve_ablation_lambdas(mode: str, base_lambda_1: float = 0.5, base_lambda_2: float = 0.2):
    """
    Adjusts `(lambda_1, lambda_2)` according to the active ablation mode (Section 2.4 of SPEC.md):
      - 'baseline'     (Config A): lambda_1 = 0.0, lambda_2 = 0.0
      - 'contrastive'  (Config B): lambda_1 = base_lambda_1 (0.5), lambda_2 = 0.0
      - 'single_proto' (Config C): lambda_1 = base_lambda_1 (0.5), lambda_2 = base_lambda_2 (0.2)
      - 'zerofake'     (Config D/E): lambda_1 = base_lambda_1 (0.5), lambda_2 = base_lambda_2 (0.2)
    """
    m = mode.strip().lower()
    if m == "baseline":
        return 0.0, 0.0
    if m == "contrastive":
        return float(base_lambda_1), 0.0
    if m in ("single_proto", "zerofake"):
        return float(base_lambda_1), float(base_lambda_2)
    raise ValueError(f"Unsupported mode '{mode}'. Choose from {VALID_MODES}.")


if HAS_TORCH:

    class ZeroFakeTotalLoss(nn.Module):
        """
        Combined ZeroFake training criterion:
          `L_total = L_cls + lambda_1 * L_con + lambda_2 * L_proto`
        """

        def __init__(
            self,
            mode: str = "zerofake",
            lambda_1: Optional[float] = None,
            lambda_2: Optional[float] = None,
            temperature: Optional[float] = None,
            proto_margin: Optional[float] = None,
            lambda_div: Optional[float] = None,
            config_path: str = "configs/default.yaml",
        ) -> None:
            super().__init__()
            cfg = load_loss_config(config_path)
            b_l1 = cfg["lambda_1"] if lambda_1 is None else float(lambda_1)
            b_l2 = cfg["lambda_2"] if lambda_2 is None else float(lambda_2)
            tau = cfg["temperature"] if temperature is None else float(temperature)
            margin = cfg["proto_margin"] if proto_margin is None else float(proto_margin)
            l_div = cfg["lambda_div"] if lambda_div is None else float(lambda_div)

            self.mode = mode.strip().lower()
            self.lambda_1, self.lambda_2 = resolve_ablation_lambdas(self.mode, b_l1, b_l2)

            self.cls_loss_fn = BinaryClassificationLoss()
            self.con_loss_fn = SupervisedContrastiveLoss(temperature=tau)
            self.proto_loss_fn = PrototypeDistanceLoss(margin=margin, lambda_div=l_div)

        def forward(
            self,
            model_outputs: Dict[str, "torch.Tensor"],
            labels: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            logits_raw = model_outputs["logits"]
            device_type = "cuda" if logits_raw.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                logits = logits_raw.float()
                embeddings = model_outputs["embeddings"].float()
                d_real = model_outputs["d_real"].float()
                d_fake = model_outputs["d_fake"].float()
                p_fake_raw = model_outputs.get("p_fake_norm", None)
                p_fake_norm = p_fake_raw.float() if p_fake_raw is not None else None
                labels_f32 = labels.float()

                loss_cls = self.cls_loss_fn(logits, labels_f32)

                if self.lambda_1 > 0.0:
                    loss_con = self.con_loss_fn(embeddings, labels)
                else:
                    loss_con = embeddings.sum() * 0.0

                if self.lambda_2 > 0.0:
                    loss_proto = self.proto_loss_fn(d_real, d_fake, labels_f32, p_fake_norm=p_fake_norm)
                else:
                    loss_proto = d_real.sum() * 0.0

                loss_total = loss_cls + (self.lambda_1 * loss_con) + (self.lambda_2 * loss_proto)
                assert torch.isfinite(loss_total), "ZeroFakeTotalLoss produced non-finite L_total!"

                return {
                    "loss_total": loss_total,
                    "loss_cls": loss_cls,
                    "loss_con": loss_con,
                    "loss_proto": loss_proto,
                    "lambda_1": self.lambda_1,
                    "lambda_2": self.lambda_2,
                }

else:

    class ZeroFakeTotalLoss:
        """
        NumPy dry-run fallback for ZeroFakeTotalLoss (`L_total = L_cls + lambda_1 * L_con + lambda_2 * L_proto`).
        """

        def __init__(
            self,
            mode: str = "zerofake",
            lambda_1: Optional[float] = None,
            lambda_2: Optional[float] = None,
            temperature: Optional[float] = None,
            proto_margin: Optional[float] = None,
            lambda_div: Optional[float] = None,
            config_path: str = "configs/default.yaml",
        ) -> None:
            cfg = load_loss_config(config_path)
            b_l1 = cfg["lambda_1"] if lambda_1 is None else float(lambda_1)
            b_l2 = cfg["lambda_2"] if lambda_2 is None else float(lambda_2)
            tau = cfg["temperature"] if temperature is None else float(temperature)
            margin = cfg["proto_margin"] if proto_margin is None else float(proto_margin)
            l_div = cfg["lambda_div"] if lambda_div is None else float(lambda_div)

            self.mode = mode.strip().lower()
            self.lambda_1, self.lambda_2 = resolve_ablation_lambdas(self.mode, b_l1, b_l2)

            self.cls_loss_fn = BinaryClassificationLoss()
            self.con_loss_fn = SupervisedContrastiveLoss(temperature=tau)
            self.proto_loss_fn = PrototypeDistanceLoss(margin=margin, lambda_div=l_div)

        def __call__(self, model_outputs: Dict, labels: np.ndarray) -> Dict[str, float]:
            return self.forward(model_outputs, labels)

        def forward(self, model_outputs: Dict, labels: np.ndarray) -> Dict[str, float]:
            loss_cls = self.cls_loss_fn(model_outputs["logits"], labels)
            loss_con = (
                self.con_loss_fn(model_outputs["embeddings"], labels)
                if self.lambda_1 > 0.0
                else 0.0
            )
            loss_proto = (
                self.proto_loss_fn(
                    model_outputs["d_real"],
                    model_outputs["d_fake"],
                    labels,
                    p_fake_norm=model_outputs.get("p_fake_norm", None),
                )
                if self.lambda_2 > 0.0
                else 0.0
            )
            loss_total = float(loss_cls + self.lambda_1 * loss_con + self.lambda_2 * loss_proto)
            assert np.isfinite(loss_total), "ZeroFakeTotalLoss produced non-finite L_total!"
            return {
                "loss_total": loss_total,
                "loss_cls": float(loss_cls),
                "loss_con": float(loss_con),
                "loss_proto": float(loss_proto),
                "lambda_1": self.lambda_1,
                "lambda_2": self.lambda_2,
            }


def run_dry_run() -> None:
    """
    Runs a full forward pass through `ZeroFakeNet` and computes `L_total = L_cls + 0.5*L_con + 0.2*L_proto`
    across all 4 ablation modes on CPU in under 5 seconds.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing losses/total_loss.py across all 4 ablation modes...")

    for mode in VALID_MODES:
        net = ZeroFakeNet(mode=mode, pretrained=False)
        criterion = ZeroFakeTotalLoss(mode=mode, config_path=str(PROJECT_ROOT / "configs" / "default.yaml"))

        if HAS_TORCH:
            x = torch.randn(8, 3, 224, 224)
            labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
            outputs = net(x)
            loss_dict = criterion(outputs, labels)
            loss_dict["loss_total"].backward()

            # Verify gradients reach backbone, projection head, classifier, and prototypes as appropriate
            assert net.classifier.fc.weight.grad is not None, "Classifier weights received no gradient!"
            if mode in ("contrastive", "single_proto", "zerofake"):
                assert net.projection_head.mlp[0].weight.grad is not None, "Projection head received no gradient!"
            if mode in ("single_proto", "zerofake"):
                assert net.prototype_layer.real_prototype.grad is not None, "Real prototype received no gradient!"
                assert net.prototype_layer.fake_prototypes.grad is not None, "Fake prototypes received no gradient!"

            l_tot = float(loss_dict["loss_total"].item())
            l_cls = float(loss_dict["loss_cls"].item())
            l_con = float(loss_dict["loss_con"].item())
            l_pro = float(loss_dict["loss_proto"].item())
        else:
            x = np.random.randn(8, 3, 224, 224).astype(np.float32)
            labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
            outputs = net(x)
            loss_dict = criterion(outputs, labels)
            l_tot = loss_dict["loss_total"]
            l_cls = loss_dict["loss_cls"]
            l_con = loss_dict["loss_con"]
            l_pro = loss_dict["loss_proto"]

        expected_tot = l_cls + loss_dict["lambda_1"] * l_con + loss_dict["lambda_2"] * l_pro
        assert abs(l_tot - expected_tot) < 1e-4, f"Mismatch in L_total formula for mode={mode}"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Verified L_total = L_cls + 0.5*L_con + 0.2*L_proto across modes {list(VALID_MODES)} "
        f"(last zerofake L_total={l_tot:.4f}) in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined ZeroFake loss L_total = L_cls + 0.5*L_con + 0.2*L_proto.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU loss verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] ZeroFakeTotalLoss ready. Import from losses.total_loss or run with --dry-run.")


if __name__ == "__main__":
    main()
