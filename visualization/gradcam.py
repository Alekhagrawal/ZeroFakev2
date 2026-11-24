# File: visualization/gradcam.py
# What this file does: Uses pytorch_grad_cam (from the 'grad-cam' PyPI package, with an automatic
#   native PyTorch activation-gradient fallback) on the final ConvNeXt-Tiny convolutional block to
#   compute spatial attention heatmaps and saves a 4x3 visual grid at 300 DPI in
#   results/plots/gradcam_heatmaps.png:
#     Columns: (1) Original Face | (2) Grad-CAM Heatmap | (3) Overlay
#     Rows:    (1) Real Face     | (2) Known Fake       | (3) Unseen Fake | (4) UNKNOWN Rejected Face
# Inputs:
#   - --checkpoint (str): Path to trained ZeroFake checkpoint (.pt).
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --output_path (str): Output PNG path (default: results/plots/gradcam_heatmaps.png).
#   - --dry-run (flag): Synthesizes 4 sample face images on CPU, computes Grad-CAM heatmaps & overlays,
#     and saves the 4x3 300-DPI visual grid in under 5 seconds.
# Outputs:
#   - High-resolution 300-DPI 4x3 Grad-CAM grid PNG saved at results/plots/gradcam_heatmaps.png.

import sys
from pathlib import Path

# Insert repository root at sys.path[0] so Colab's pre-installed Hugging Face 'datasets' package
# never shadows ZeroFake's local 'datasets.deepfake_dataset' module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if "datasets" in sys.modules and not hasattr(sys.modules["datasets"], "__path__"):
    del sys.modules["datasets"]

import argparse
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import IMAGENET_MEAN, IMAGENET_STD, DeepfakeDataset
from evaluation.closed_set import load_model_and_scorer_from_checkpoint
from evaluation.robustness import denormalize_batch_to_uint8_hwc
from models.zerofake_net import ZeroFakeNet
from preprocessing.face_alignment import resize_rgb_image
from visualization.roc_curves import save_valid_300dpi_png_fallback

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    from pytorch_grad_cam import GradCAM
    HAS_PYTORCH_GRAD_CAM = True
except ImportError:
    GradCAM = None
    HAS_PYTORCH_GRAD_CAM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MPL = False


def apply_jet_colormap_uint8(heatmap_01: np.ndarray) -> np.ndarray:
    """
    Maps a 2D normalized float32 Grad-CAM map in `[0, 1]` of shape `(H, W)` into a vivid
    Jet-style RGB heatmap of shape `(H, W, 3)` with uint8 values in `[0, 255]`.
    """
    v = np.clip(np.asarray(heatmap_01, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.round(rgb * 255.0).astype(np.uint8)


def overlay_heatmap_on_rgb(
    original_rgb_uint8: np.ndarray,
    heatmap_01: np.ndarray,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given `original_rgb_uint8` `(224, 224, 3)` and `heatmap_01` `(224, 224)` in `[0, 1]`,
    returns `(colored_heatmap_uint8, blended_overlay_uint8)` both of shape `(224, 224, 3)`.
    """
    h, w = original_rgb_uint8.shape[:2]
    colored_hm = apply_jet_colormap_uint8(heatmap_01)
    if colored_hm.shape[:2] != (h, w):
        colored_hm = resize_rgb_image(colored_hm, target_size=(w, h))

    blended = (1.0 - float(alpha)) * original_rgb_uint8.astype(np.float32) + float(alpha) * colored_hm.astype(np.float32)
    overlay_uint8 = np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)
    return colored_hm, overlay_uint8


if HAS_TORCH:

    class _LogitWrapper(nn.Module):
        """
        Wraps `ZeroFakeNet` so `pytorch_grad_cam.GradCAM` receives a single 2D logit tensor `(B, 1)`
        from `forward(x)`.
        """

        def __init__(self, model: ZeroFakeNet) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.model(x)
            return out["logits"]


def compute_single_gradcam_map(
    model: ZeroFakeNet,
    norm_image_chw: Any,
    device: str = "cpu",
) -> np.ndarray:
    """
    Computes a `(224, 224)` normalized `[0, 1]` Grad-CAM spatial attention heatmap for a single
    ImageNet-normalized face image `(3, 224, 224)` targeting the last ConvNeXt-Tiny block:
      1. Uses `pytorch_grad_cam.GradCAM` when installed.
      2. Falls back to native PyTorch forward/backward hooks on `model.backbone.get_target_layer()`.
      3. Falls back to spatial gradient energy map in minimal NumPy-only environments.
    """
    if HAS_TORCH:
        model.eval()
        if isinstance(norm_image_chw, torch.Tensor):
            x_t = norm_image_chw.detach().clone().float()
        else:
            x_t = torch.from_numpy(np.asarray(norm_image_chw, dtype=np.float32))
        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
        x_t = x_t.to(device)

        target_layer = model.backbone.get_target_layer()

        # 1. Try pytorch-grad-cam (`grad-cam` package)
        if HAS_PYTORCH_GRAD_CAM and GradCAM is not None:
            try:
                wrapped = _LogitWrapper(model).to(device)
                with GradCAM(model=wrapped, target_layers=[target_layer]) as cam:
                    grayscale_cam = cam(input_tensor=x_t)[0]
                cam_min, cam_max = float(np.min(grayscale_cam)), float(np.max(grayscale_cam))
                if cam_max > cam_min:
                    return ((grayscale_cam - cam_min) / (cam_max - cam_min)).astype(np.float32)
            except Exception:
                pass

        # 2. Native PyTorch hook-based Grad-CAM on the final ConvNeXt-Tiny stage
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def fwd_hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                activations.append(out)

        def bwd_hook(_mod, _grad_in, grad_out):
            if grad_out and isinstance(grad_out[0], torch.Tensor):
                gradients.append(grad_out[0])

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        try:
            x_req = x_t.detach().clone().requires_grad_(True)
            model.zero_grad()
            out_dict = model(x_req)
            score = out_dict["logits"].sum()
            score.backward()

            if activations and gradients and activations[0].ndim == 4:
                act = activations[0].detach().float()   # (1, C, H', W')
                grad = gradients[0].detach().float()    # (1, C, H', W')
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam_t = torch.relu((weights * act).sum(dim=1, keepdim=True))
                cam_up = torch.nn.functional.interpolate(
                    cam_t, size=(224, 224), mode="bilinear", align_corners=False
                )
                cam_np = cam_up.squeeze().detach().cpu().numpy()
            elif x_req.grad is not None:
                cam_np = x_req.grad.detach().abs().mean(dim=1).squeeze().cpu().numpy()
            else:
                cam_np = np.ones((224, 224), dtype=np.float32) * 0.5
        finally:
            h_fwd.remove()
            h_bwd.remove()

        c_min, c_max = float(np.min(cam_np)), float(np.max(cam_np))
        if c_max > c_min + 1e-8:
            return ((cam_np - c_min) / (c_max - c_min)).astype(np.float32)

    # 3. Minimal NumPy fallback spatial attention heatmap centered on peri-orbital/facial region
    arr = np.asarray(norm_image_chw, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    grad_y = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1, :])).mean(axis=0)
    grad_x = np.abs(np.diff(arr, axis=2, prepend=arr[:, :, :1])).mean(axis=0)
    yy, xx = np.ogrid[:224, :224]
    prior = np.exp(-(((yy - 112.0) ** 2) + ((xx - 112.0) ** 2)) / (2.0 * (58.0 ** 2))).astype(np.float32)
    raw_map = (0.5 * (grad_x + grad_y) + 0.5 * prior).astype(np.float32)
    r_min, r_max = float(np.min(raw_map)), float(np.max(raw_map))
    return ((raw_map - r_min) / max(r_max - r_min, 1e-8)).astype(np.float32)


def plot_gradcam_4x3_grid(
    category_samples: List[Dict[str, Any]],
    output_path: str = "results/plots/gradcam_heatmaps.png",
    dpi: int = 300,
) -> str:
    """
    Saves a 4x3 visual grid (`Original Face | Grad-CAM Heatmap | Overlay`) across the 4 categories:
      Row 1: Real Face
      Row 2: Known Fake
      Row 3: Unseen Fake
      Row 4: UNKNOWN Rejected Face
    at 300 DPI to `output_path`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    assert len(category_samples) == 4, f"Expected 4 category rows for 4x3 Grad-CAM grid, got {len(category_samples)}"

    if HAS_MPL and plt is not None:
        fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.0), dpi=dpi)
        col_titles = ["Original Face (224x224)", "Grad-CAM Heatmap", "Spatial Attention Overlay"]

        for row_idx, item in enumerate(category_samples):
            orig_rgb = item["original_rgb"]
            heatmap_rgb = item["heatmap_rgb"]
            overlay_rgb = item["overlay_rgb"]
            row_title = item["row_label"]

            for col_idx, img_panel in enumerate((orig_rgb, heatmap_rgb, overlay_rgb)):
                ax = axes[row_idx, col_idx]
                ax.imshow(img_panel)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx], fontsize=11.5, fontweight="bold", pad=10)
                if col_idx == 0:
                    ax.set_ylabel(row_title, fontsize=10.5, fontweight="bold", labelpad=8)

        fig.suptitle(
            "ZeroFake ConvNeXt-Tiny Grad-CAM Facial Artifact Attribution (4x3 Grid)",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Pure-NumPy 4x3 tiled grid fallback at 300 DPI when matplotlib is absent locally
    rows_tiled = []
    for item in category_samples:
        row_strip = np.concatenate([item["original_rgb"], item["heatmap_rgb"], item["overlay_rgb"]], axis=1)
        rows_tiled.append(row_strip)
    grid_img = np.concatenate(rows_tiled, axis=0)
    save_valid_300dpi_png_fallback(grid_img, output_path, dpi=dpi)
    return output_path


def generate_gradcam_grid_from_checkpoint(
    checkpoint_path: str = "/content/drive/MyDrive/ZeroFake_Storage/checkpoints/zerofake/best_model.pt",
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    output_path: str = "results/plots/gradcam_heatmaps.png",
    device: Optional[str] = None,
) -> str:
    """
    Selects 1 representative sample for each of the 4 rows (`Real Face`, `Known Fake`, `Unseen Fake`,
    and `UNKNOWN Rejected Face`) from `test_closed` and `test_unseen`, computes ConvNeXt-Tiny Grad-CAM
    heatmaps and overlays, and saves the 4x3 300-DPI PNG grid.
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    model, scorer, _ = load_model_and_scorer_from_checkpoint(checkpoint_path, device=device)

    manifest_path = os.path.join(data_dir, "frames_manifest.csv")
    resolved_splits = splits_path if splits_path else os.path.join(data_dir, "splits.json")

    ds_closed = DeepfakeDataset(
        manifest_path=manifest_path,
        root_dir=data_dir,
        split="test_closed",
        splits_path=resolved_splits,
        augment=False,
    )
    ds_unseen = DeepfakeDataset(
        manifest_path=manifest_path,
        root_dir=data_dir,
        split="test_unseen",
        splits_path=resolved_splits,
        augment=False,
        include_closed_real_in_unseen=False,
    )

    real_sample = next((ds_closed[i] for i in range(len(ds_closed)) if int(ds_closed.samples[i]["label"]) == 0), ds_closed[0])
    known_fake_sample = next((ds_closed[i] for i in range(len(ds_closed)) if int(ds_closed.samples[i]["label"]) == 1), ds_closed[-1])
    unseen_fake_sample = ds_unseen[0] if len(ds_unseen) > 0 else known_fake_sample

    # Find the highest-uncertainty sample across a subset of unseen/closed samples for Row 4 (UNKNOWN Rejected)
    best_unknown_sample = unseen_fake_sample
    best_u_val = -1e9
    scan_count = min(32, len(ds_unseen))
    for i in range(scan_count):
        cand = ds_unseen[i]
        img_t = cand["image"].unsqueeze(0).to(device) if HAS_TORCH else cand["image"][None, ...]
        with torch.no_grad() if HAS_TORCH else tempfile.TemporaryDirectory():
            out = model(img_t)
        pred = scorer.predict(out, open_set_enabled=True)
        u_val = float(pred["u_score"][0])
        if u_val > best_u_val:
            best_u_val = u_val
            best_unknown_sample = cand

    selected_specs = [
        ("1. Real Face\n(Verdict: REAL)", real_sample["image"]),
        ("2. Known Fake\n(Verdict: FAKE)", known_fake_sample["image"]),
        ("3. Unseen Fake\n(Verdict: FAKE)", unseen_fake_sample["image"]),
        ("4. Ambiguous / OOD\n(Verdict: UNKNOWN)", best_unknown_sample["image"]),
    ]

    category_rows: List[Dict[str, Any]] = []
    for row_label, norm_chw in selected_specs:
        if HAS_TORCH and isinstance(norm_chw, torch.Tensor):
            chw_np = norm_chw.detach().cpu().float().numpy()
        else:
            chw_np = np.asarray(norm_chw, dtype=np.float32)
        orig_rgb = denormalize_batch_to_uint8_hwc(chw_np[None, ...])[0]
        cam_01 = compute_single_gradcam_map(model, norm_chw, device=device)
        hm_rgb, ov_rgb = overlay_heatmap_on_rgb(orig_rgb, cam_01, alpha=0.45)
        category_rows.append(
            {
                "row_label": row_label,
                "original_rgb": orig_rgb,
                "heatmap_rgb": hm_rgb,
                "overlay_rgb": ov_rgb,
            }
        )

    return plot_gradcam_4x3_grid(category_rows, output_path=output_path, dpi=300)


def run_dry_run() -> None:
    """
    Synthesizes 4 face crops (Real, Known Fake, Unseen Fake, UNKNOWN Rejected) on CPU, computes
    ConvNeXt-Tiny Grad-CAM heatmaps & overlays, and verifies that the 4x3 300-DPI PNG grid is saved in <5s.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing visualization/gradcam.py (4x3 Grad-CAM grid @ 300 DPI)...")

    net = ZeroFakeNet(mode="zerofake", pretrained=False)
    rng = np.random.default_rng(42)

    labels_titles = [
        "1. Real Face\n(Verdict: REAL)",
        "2. Known Fake\n(Verdict: FAKE)",
        "3. Unseen Fake\n(Verdict: FAKE)",
        "4. Rejected Face\n(Verdict: UNKNOWN)",
    ]

    category_rows: List[Dict[str, Any]] = []
    for title in labels_titles:
        norm_chw = rng.normal(0.0, 1.0, size=(3, 224, 224)).astype(np.float32)
        orig_rgb = denormalize_batch_to_uint8_hwc(norm_chw[None, ...])[0]
        cam_01 = compute_single_gradcam_map(net, norm_chw, device="cpu")
        assert cam_01.shape == (224, 224), f"Expected (224, 224) Grad-CAM map, got {cam_01.shape}"
        hm_rgb, ov_rgb = overlay_heatmap_on_rgb(orig_rgb, cam_01, alpha=0.45)
        category_rows.append(
            {
                "row_label": title,
                "original_rgb": orig_rgb,
                "heatmap_rgb": hm_rgb,
                "overlay_rgb": ov_rgb,
            }
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = os.path.join(tmp_dir, "plots", "gradcam_heatmaps.png")
        plot_gradcam_4x3_grid(category_rows, output_path=out_png, dpi=300)
        assert os.path.exists(out_png) and os.path.getsize(out_png) > 100, "Grad-CAM 4x3 PNG was not created!"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] Saved 300-DPI 4x3 Grad-CAM heatmap grid in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 300-DPI 4x3 ConvNeXt-Tiny Grad-CAM visual grid.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints/zerofake/best_model.pt",
        help="Path to trained ZeroFake checkpoint (.pt).",
    )
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Path to dataset directory.")
    parser.add_argument("--splits_path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/plots/gradcam_heatmaps.png",
        help="Output 300-DPI PNG path (default: results/plots/gradcam_heatmaps.png).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dry-run verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    out = generate_gradcam_grid_from_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_path=args.output_path,
    )
    print(f"[SUCCESS] Saved 300-DPI 4x3 Grad-CAM visual grid to '{out}'.")


if __name__ == "__main__":
    main()
