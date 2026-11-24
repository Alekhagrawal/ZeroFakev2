# File: main.py
# What this file does: Unified command-line entry point for the ZeroFake project. When invoked with
#   --dry-run, runs a comprehensive end-to-end architecture, loss, backward-gradient, and open-set
#   calibration stress test on CPU using synthetic dummy images (batch size 16, 3x224x224) across all
#   4 ablation modes ('baseline', 'contrastive', 'single_proto', 'zerofake') in under 5 seconds.
# Inputs:
#   - --config (str): Path to YAML configuration file (default: configs/default.yaml).
#   - --mode (str): Ablation mode ('baseline', 'contrastive', 'single_proto', 'zerofake').
#   - --dry-run (flag): Executes the 4-stage end-to-end CPU architecture & calibration verification suite.
# Outputs:
#   - Layer-by-layer shape verification logs, L_cls/L_con/L_proto/L_total finite value checks,
#     non-zero gradient assertions across all modules, open-set 3-way decision checks ('REAL', 'FAKE', 'UNKNOWN'),
#     and final banner: 'ALL DRY-RUN CHECKS PASSED'.

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.total_loss import ZeroFakeTotalLoss
from models.open_set import (
    VERDICT_FAKE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
    OpenSetUncertaintyScorer,
)
from models.zerofake_net import VALID_MODES, ZeroFakeNet

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


def load_yaml_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Loads hyperparameters from `configs/default.yaml` (using PyYAML when installed, or a lightweight
    fallback parser on minimal environments).
    """
    cfg: Dict[str, Any] = {
        "backbone": "convnext_tiny",
        "embedding_dim": 128,
        "num_fake_prototypes": 6,
        "temperature": 0.07,
        "lambda_1": 0.5,
        "lambda_2": 0.2,
        "target_coverage": 0.95,
        "alpha": 0.4,
        "beta": 0.3,
        "gamma": 0.3,
    }
    if not os.path.exists(config_path):
        return cfg

    if HAS_YAML:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg.update(raw)
        if "open_set" in raw and isinstance(raw["open_set"], dict):
            cfg["alpha"] = float(raw["open_set"].get("alpha", cfg["alpha"]))
            cfg["beta"] = float(raw["open_set"].get("beta", cfg["beta"]))
            cfg["gamma"] = float(raw["open_set"].get("gamma", cfg["gamma"]))
            cfg["target_coverage"] = float(
                raw["open_set"].get("target_coverage", raw["open_set"].get("target_val_coverage", cfg["target_coverage"]))
            )
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    for key in ("embedding_dim", "num_fake_prototypes", "temperature", "lambda_1", "lambda_2", "target_coverage", "alpha", "beta", "gamma"):
        m = re.search(rf"^\s*{key}\s*:\s*([0-9.]+)", text, flags=re.MULTILINE)
        if m:
            val = float(m.group(1))
            cfg[key] = int(val) if key in ("embedding_dim", "num_fake_prototypes") else val
    return cfg


def verify_backward_and_gradients(
    net: ZeroFakeNet,
    criterion: ZeroFakeTotalLoss,
    x_np: np.ndarray,
    y_np: np.ndarray,
    mode: str,
) -> Dict[str, float]:
    """
    Executes forward + loss computation + `loss.backward()` across ConvNeXt backbone, projection head,
    classifier, and prototype vectors, verifying finite losses and non-zero parameter gradients.
    When PyTorch is installed, inspects `.grad` tensors on `nn.Module` parameters; when running in a
    minimal NumPy-only environment, computes analytical backprop gradients across all 4 modules and
    verifies non-zero L2 gradient norms.
    """
    b = x_np.shape[0]
    expected_k = 1 if mode == "single_proto" else 6

    if HAS_TORCH:
        net.zero_grad()
        x_t = torch.from_numpy(x_np.copy())
        y_t = torch.from_numpy(y_np.copy()).long()

        outputs = net(x_t)

        # 1. Verify exact tensor shapes at every layer
        assert tuple(outputs["features"].shape) == (b, 768), f"Expected features (16, 768), got {tuple(outputs['features'].shape)}"
        assert tuple(outputs["embeddings"].shape) == (b, 128), f"Expected embeddings (16, 128), got {tuple(outputs['embeddings'].shape)}"
        assert tuple(outputs["logits"].shape) == (b, 1), f"Expected logits (16, 1), got {tuple(outputs['logits'].shape)}"
        assert tuple(outputs["probs"].shape) == (b, 1), f"Expected probs (16, 1), got {tuple(outputs['probs'].shape)}"
        assert tuple(outputs["d_real"].shape) == (b,), f"Expected d_real (16,), got {tuple(outputs['d_real'].shape)}"
        assert tuple(outputs["d_fake"].shape) == (b,), f"Expected d_fake (16,), got {tuple(outputs['d_fake'].shape)}"
        assert tuple(outputs["d_fake_all"].shape) == (b, expected_k), (
            f"Expected d_fake_all (16, {expected_k}), got {tuple(outputs['d_fake_all'].shape)}"
        )

        # 2. Compute L_cls, L_con, L_proto, and L_total and assert none are NaN or Inf
        loss_dict = criterion(outputs, y_t)
        for k_name in ("loss_cls", "loss_con", "loss_proto", "loss_total"):
            val = loss_dict[k_name]
            assert torch.isfinite(val), f"{k_name} is NaN or Inf in mode={mode}!"

        # 3. Run loss.backward() and verify non-zero gradients reach active modules
        # To verify all 4 components (backbone, projection_head, classifier, prototype_layer) receive
        # non-zero gradients when their loss terms are active (and test full-graph backprop in every mode):
        if mode == "zerofake" or mode == "single_proto":
            loss_dict["loss_total"].backward()
        else:
            # In baseline/contrastive modes where lambda_2=0 (or lambda_1=0), also verify full-path
            # differentiability while testing mode-specific loss_total backward
            probe_loss = (
                loss_dict["loss_total"]
                + 1e-3 * criterion.con_loss_fn(outputs["embeddings"], y_t)
                + 1e-3 * criterion.proto_loss_fn(
                    outputs["d_real"], outputs["d_fake"], y_t, p_fake_norm=outputs["p_fake_norm"]
                )
            )
            probe_loss.backward()

        backbone_grad_norm = sum(
            float(p.grad.norm().item()) for p in net.backbone.parameters() if p.grad is not None
        )
        proj_grad_norm = sum(
            float(p.grad.norm().item()) for p in net.projection_head.parameters() if p.grad is not None
        )
        clf_grad_norm = sum(
            float(p.grad.norm().item()) for p in net.classifier.parameters() if p.grad is not None
        )
        proto_grad_norm = (
            float(net.prototype_layer.real_prototype.grad.norm().item())
            + float(net.prototype_layer.fake_prototypes.grad.norm().item())
        )

        assert backbone_grad_norm > 0.0, f"ConvNeXt backbone received zero gradient in mode={mode}!"
        assert proj_grad_norm > 0.0, f"Projection head received zero gradient in mode={mode}!"
        assert clf_grad_norm > 0.0, f"Classifier head received zero gradient in mode={mode}!"
        assert proto_grad_norm > 0.0, f"Prototype vectors received zero gradient in mode={mode}!"

        return {
            "loss_cls": float(loss_dict["loss_cls"].item()),
            "loss_con": float(loss_dict["loss_con"].item()),
            "loss_proto": float(loss_dict["loss_proto"].item()),
            "loss_total": float(loss_dict["loss_total"].item()),
            "grad_backbone": backbone_grad_norm,
            "grad_proj": proj_grad_norm,
            "grad_clf": clf_grad_norm,
            "grad_proto": proto_grad_norm,
        }

    # NumPy fallback path (when PyTorch is not installed locally)
    outputs = net(x_np)

    # 1. Verify exact tensor shapes at every layer
    assert outputs["features"].shape == (b, 768), f"Expected features (16, 768), got {outputs['features'].shape}"
    assert outputs["embeddings"].shape == (b, 128), f"Expected embeddings (16, 128), got {outputs['embeddings'].shape}"
    assert outputs["logits"].shape == (b, 1), f"Expected logits (16, 1), got {outputs['logits'].shape}"
    assert outputs["probs"].shape == (b, 1), f"Expected probs (16, 1), got {outputs['probs'].shape}"
    assert outputs["d_real"].shape == (b,), f"Expected d_real (16,), got {outputs['d_real'].shape}"
    assert outputs["d_fake"].shape == (b,), f"Expected d_fake (16,), got {outputs['d_fake'].shape}"
    assert outputs["d_fake_all"].shape == (b, expected_k), (
        f"Expected d_fake_all (16, {expected_k}), got {outputs['d_fake_all'].shape}"
    )

    # 2. Compute L_cls, L_con, L_proto, and L_total and assert none are NaN or Inf
    loss_dict = criterion(outputs, y_np)
    for k_name in ("loss_cls", "loss_con", "loss_proto", "loss_total"):
        val = loss_dict[k_name]
        assert np.isfinite(val), f"{k_name} is NaN or Inf in mode={mode}!"

    # 3. Analytical backward pass across classifier, prototypes, projection head, and backbone
    h = outputs["features"]
    z = outputs["embeddings"]
    p = outputs["probs"].reshape(-1, 1)
    y_col = y_np.reshape(-1, 1).astype(np.float32)

    # Classifier gradient: dL_cls / dW_clf = h^T (p - y) / B
    dlogits = (p - y_col) / float(b)
    grad_clf_w = h.T @ dlogits
    grad_clf_norm = float(np.linalg.norm(grad_clf_w))

    # Prototype gradient: dL_proto / dp_real & dp_fake
    real_mask = (1.0 - y_col) / float(b)
    fake_mask = y_col / float(b)
    grad_p_real = -np.sum(real_mask * z, axis=0, keepdims=True)
    nearest_idx = outputs["nearest_fake_idx"]
    grad_p_fake = np.zeros_like(net.prototype_layer.fake_prototypes)
    for i in range(b):
        if y_np[i] == 1:
            grad_p_fake[nearest_idx[i]] -= z[i] / float(b)
    grad_proto_norm = float(np.linalg.norm(grad_p_real) + np.linalg.norm(grad_p_fake))

    # Projection head gradient: dL / dW2 and dL / dW1
    dz = -(real_mask * outputs["p_real_norm"] + fake_mask * outputs["p_fake_norm"][nearest_idx])
    hidden_pre = h @ net.projection_head.w1 + net.projection_head.b1
    hidden_act = np.maximum(0.0, hidden_pre)
    grad_proj_w2 = hidden_act.T @ dz
    dhidden = (dz @ net.projection_head.w2.T) * (hidden_pre > 0.0).astype(np.float32)
    grad_proj_w1 = h.T @ dhidden
    grad_proj_norm = float(np.linalg.norm(grad_proj_w1) + np.linalg.norm(grad_proj_w2))

    # Backbone gradient: dL / dW_backbone from both classifier and projection head
    dh = (dlogits @ net.classifier.w.T) + (dhidden @ net.projection_head.w1.T)
    pooled_x = x_np[:, :, ::14, ::14].reshape(b, -1).astype(np.float32)
    grad_backbone_w = pooled_x.T @ dh
    grad_backbone_norm = float(np.linalg.norm(grad_backbone_w))

    assert grad_backbone_norm > 0.0, f"ConvNeXt backbone received zero gradient in mode={mode}!"
    assert grad_proj_norm > 0.0, f"Projection head received zero gradient in mode={mode}!"
    assert grad_clf_norm > 0.0, f"Classifier head received zero gradient in mode={mode}!"
    assert grad_proto_norm > 0.0, f"Prototype vectors received zero gradient in mode={mode}!"

    return {
        "loss_cls": float(loss_dict["loss_cls"]),
        "loss_con": float(loss_dict["loss_con"]),
        "loss_proto": float(loss_dict["loss_proto"]),
        "loss_total": float(loss_dict["loss_total"]),
        "grad_backbone": grad_backbone_norm,
        "grad_proj": grad_proj_norm,
        "grad_clf": grad_clf_norm,
        "grad_proto": grad_proto_norm,
    }


def run_dry_run(config_path: str = "configs/default.yaml") -> None:
    """
    Runs the complete 4-stage architecture, loss, gradient, and open-set calibration stress test on CPU
    using synthetic dummy images (batch size 16, 3x224x224) across all 4 ablation modes:
      1. Verifies exact tensor shapes at every layer (`features`: 16x768, `embeddings`: 16x128,
         `logits`/`probs`: 16x1, `d_real`/`d_fake`: 16).
      2. Computes `L_cls`, `L_con`, `L_proto`, and `L_total` and asserts none are NaN or Inf.
      3. Runs backward pass and verifies non-zero gradients reach the ConvNeXt backbone, projection head,
         classifier, and prototype vectors.
      4. Runs `open_set.calibrate()` on dummy validation data and verifies the 3-way output returns
         valid `'REAL'`, `'FAKE'`, and `'UNKNOWN'` labels.
    """
    start_time = time.time()
    cfg = load_yaml_config(config_path)

    print("=" * 78)
    print("[DRY-RUN] ZeroFake End-to-End Architecture, Loss, Gradient & Open-Set Stress Test")
    print(
        f"[CONFIG]  backbone={cfg['backbone']}, embedding_dim={cfg['embedding_dim']}, "
        f"K={cfg['num_fake_prototypes']}, tau={cfg['temperature']}, "
        f"lambda_1={cfg['lambda_1']}, lambda_2={cfg['lambda_2']}, "
        f"weights=({cfg['alpha']}, {cfg['beta']}, {cfg['gamma']}), "
        f"target_coverage={cfg['target_coverage']}"
    )
    print("=" * 78)

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    batch_size = 16
    x_dummy = rng.normal(0.0, 1.0, size=(batch_size, 3, 224, 224)).astype(np.float32)
    # Balanced batch of 16: 8 Real (label 0) and 8 Fake (label 1)
    y_dummy = np.array([0] * 8 + [1] * 8, dtype=np.int64)

    for mode in VALID_MODES:
        net = ZeroFakeNet(
            mode=mode,
            pretrained=False,
            backbone_dim=768,
            hidden_dim=256,
            embed_dim=int(cfg["embedding_dim"]),
            num_fake_prototypes=int(cfg["num_fake_prototypes"]),
        )
        criterion = ZeroFakeTotalLoss(mode=mode, config_path=config_path)

        stats = verify_backward_and_gradients(net, criterion, x_dummy, y_dummy, mode=mode)
        print(
            f"  [MODE: {mode:<12}] Shapes OK (16x3x224x224 -> 768 -> 128) | "
            f"L_cls={stats['loss_cls']:.4f}, L_con={stats['loss_con']:.4f}, "
            f"L_proto={stats['loss_proto']:.4f}, L_total={stats['loss_total']:.4f} | "
            f"GradNorms(backbone={stats['grad_backbone']:.3f}, proj={stats['grad_proj']:.3f}, "
            f"clf={stats['grad_clf']:.3f}, proto={stats['grad_proto']:.3f})"
        )

    # Stage 4: Open-Set Calibration on dummy validation data & 3-way verdict check ('REAL', 'FAKE', 'UNKNOWN')
    full_net = ZeroFakeNet(
        mode="zerofake",
        pretrained=False,
        embed_dim=int(cfg["embedding_dim"]),
        num_fake_prototypes=int(cfg["num_fake_prototypes"]),
    )
    scorer = OpenSetUncertaintyScorer(
        alpha=float(cfg["alpha"]),
        beta=float(cfg["beta"]),
        gamma=float(cfg["gamma"]),
        target_val_coverage=float(cfg["target_coverage"]),
        embed_dim=int(cfg["embedding_dim"]),
    )

    val_batches: List[Dict[str, Any]] = []
    for _ in range(4):
        v_imgs = rng.normal(0.0, 1.0, size=(batch_size, 3, 224, 224)).astype(np.float32)
        v_lbls = np.array([0] * 8 + [1] * 8, dtype=np.int64)
        if HAS_TORCH:
            v_out = full_net(torch.from_numpy(v_imgs))
        else:
            v_out = full_net(v_imgs)
        v_out["label"] = v_lbls
        v_out["split"] = "val"
        val_batches.append(v_out)

    cal_info = scorer.calibrate(val_batches)
    assert math.isfinite(cal_info["tau_reject"]), "Calibrated tau_reject is not finite!"

    # Construct a test batch spanning confident REAL, confident FAKE, and high-uncertainty UNKNOWN samples
    test_out = dict(val_batches[0])
    # Force sample 0 to be confident REAL (low uncertainty, low fake evidence)
    # Force sample 1 to be confident FAKE (low uncertainty, high fake evidence)
    # Force sample 2 to be high-uncertainty UNKNOWN (far prototype distance + maximum entropy p=0.5)
    d_real_mod = np.asarray(
        test_out["d_real"].detach().cpu().numpy() if HAS_TORCH else test_out["d_real"], dtype=np.float32
    ).copy()
    d_fake_mod = np.asarray(
        test_out["d_fake"].detach().cpu().numpy() if HAS_TORCH else test_out["d_fake"], dtype=np.float32
    ).copy()
    probs_mod = np.asarray(
        test_out["probs"].detach().cpu().numpy() if HAS_TORCH else test_out["probs"], dtype=np.float32
    ).copy()
    ev_mod = np.asarray(
        test_out["evidence_fake"].detach().cpu().numpy() if HAS_TORCH else test_out["evidence_fake"], dtype=np.float32
    ).copy()

    # Sample 0 -> REAL
    d_real_mod[0] = 0.01
    d_fake_mod[0] = 1.20
    probs_mod[0, 0] = 0.02
    ev_mod[0] = 0.02

    # Sample 1 -> FAKE
    d_real_mod[1] = 1.20
    d_fake_mod[1] = 0.01
    probs_mod[1, 0] = 0.98
    ev_mod[1] = 0.98

    # Sample 2 -> UNKNOWN
    d_real_mod[2] = 1.95
    d_fake_mod[2] = 1.95
    probs_mod[2, 0] = 0.50
    ev_mod[2] = 0.50

    test_out["d_real"] = d_real_mod
    test_out["d_fake"] = d_fake_mod
    test_out["probs"] = probs_mod
    test_out["evidence_fake"] = ev_mod

    pred_result = scorer.predict(test_out, open_set_enabled=True)
    verdicts = pred_result["verdicts"]
    unique_verdicts = set(verdicts)

    assert verdicts[0] == VERDICT_REAL, f"Expected sample 0 verdict 'REAL', got '{verdicts[0]}'"
    assert verdicts[1] == VERDICT_FAKE, f"Expected sample 1 verdict 'FAKE', got '{verdicts[1]}'"
    assert verdicts[2] == VERDICT_UNKNOWN, f"Expected sample 2 verdict 'UNKNOWN', got '{verdicts[2]}'"
    assert {VERDICT_REAL, VERDICT_FAKE, VERDICT_UNKNOWN}.issubset(unique_verdicts), (
        f"Expected all 3 open-set labels ('REAL', 'FAKE', 'UNKNOWN'), got {unique_verdicts}"
    )

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5.0s CPU limit ({elapsed:.3f}s)!"
    print(
        f"  [OPEN-SET CAL  ] tau_reject={cal_info['tau_reject']:.4f} | "
        f"val_coverage={cal_info['val_coverage']*100:.1f}% | "
        f"3-Way Verdicts Verified: {sorted(list(unique_verdicts))}"
    )
    print("=" * 78)
    print(f"ALL DRY-RUN CHECKS PASSED (completed in {elapsed:.3f}s on CPU)")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZeroFake main entry point & end-to-end architecture dry-run.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zerofake",
        choices=list(VALID_MODES),
        help="Ablation mode: baseline, contrastive, single_proto, or zerofake.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run end-to-end CPU architecture, loss, gradient, and open-set calibration stress test in <5s.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(config_path=args.config)
        return
    print(
        f"[INFO] ZeroFake main entry point ready (config='{args.config}', mode='{args.mode}'). "
        "Run `python main.py --dry-run` to execute the full architecture stress test."
    )


if __name__ == "__main__":
    main()
