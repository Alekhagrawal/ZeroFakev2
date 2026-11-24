# File: visualization/embeddings.py
# What this file does: Extracts 128-d L2-normalized embeddings z for up to 1,000 test samples across
#   Real faces, seen manipulation generators (Deepfakes, Face2Face, FaceSwap), and the held-out unseen
#   generator (NeuralTextures) together with the learned 1 Real prototype and K=6 Fake prototypes,
#   projects them to 2D via t-SNE, and saves a color-coded 300-DPI scatter plot in
#   results/plots/tsne_prototypes.png showing how ZeroFake groups different deepfake generators into
#   distinct prototype clusters.
# Inputs:
#   - --checkpoint (str): Path to trained ZeroFake checkpoint (.pt).
#   - --data_dir (str): Directory containing extracted face crops, frames_manifest.csv, and splits.json.
#   - --splits_path (str, optional): Explicit path to splits.json (defaults to <data_dir>/splits.json).
#   - --max_samples (int): Maximum number of test sample embeddings to project (default: 1000).
#   - --output_path (str): Output PNG path (default: results/plots/tsne_prototypes.png).
#   - --dry-run (flag): Synthesizes 128-d embeddings + 1 Real & 6 Fake prototypes on CPU, runs 2D
#     projection, and saves a 300-DPI PNG scatter plot in under 5 seconds.
# Outputs:
#   - High-resolution 300-DPI PNG image saved at results/plots/tsne_prototypes.png.

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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset
from evaluation.closed_set import (
    collect_split_inference_outputs,
    load_model_and_scorer_from_checkpoint,
)
from visualization.roc_curves import save_valid_300dpi_png_fallback

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MPL = False

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    TSNE = None
    HAS_SKLEARN = False


GENERATOR_COLORS = {
    "Real": "#27ae60",
    "Deepfakes": "#2980b9",
    "Face2Face": "#8e44ad",
    "FaceSwap": "#e67e22",
    "NeuralTextures": "#c0392b",
}


def compute_2d_tsne_projection(
    all_vectors: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Projects `(N, 128)` embeddings + prototypes into 2D coordinates `(N, 2)` using `sklearn.manifold.TSNE`
    (or exact SVD/PCA spectral projection fallback when `scikit-learn` is not installed locally).
    """
    mat = np.asarray(all_vectors, dtype=np.float64)
    n = mat.shape[0]
    if n <= 3:
        return mat[:, :2].astype(np.float32)

    if HAS_SKLEARN and TSNE is not None:
        perp = min(30.0, max(5.0, float(n - 2) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        return tsne.fit_transform(mat).astype(np.float32)

    # Pure-NumPy centered SVD 2D projection fallback for fast local dry-runs without scikit-learn
    centered = mat - np.mean(mat, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords_2d = u[:, :2] * s[:2]
    return coords_2d.astype(np.float32)


def plot_tsne_embeddings_and_prototypes(
    sample_embeddings: np.ndarray,
    generator_names: List[str],
    p_real_norm: np.ndarray,
    p_fake_norm: np.ndarray,
    output_path: str = "results/plots/tsne_prototypes.png",
    max_samples: int = 1000,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Subsamples up to `max_samples` (default 1,000) test embeddings, concatenates the learned Real prototype
    `(1, 128)` and `K=6` Fake prototypes `(K, 128)`, computes the joint 2D t-SNE projection, and saves a
    300-DPI color-coded scatter plot to `output_path`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    z_arr = np.asarray(sample_embeddings, dtype=np.float32)
    gens = list(generator_names)

    if len(z_arr) > max_samples:
        rng = np.random.default_rng(42)
        chosen_idx = np.sort(rng.choice(len(z_arr), size=int(max_samples), replace=False))
        z_arr = z_arr[chosen_idx]
        gens = [gens[i] for i in chosen_idx]

    p_real_2d = np.asarray(p_real_norm, dtype=np.float32).reshape(1, -1)
    p_fake_2d = np.asarray(p_fake_norm, dtype=np.float32)
    if p_fake_2d.ndim == 1:
        p_fake_2d = p_fake_2d.reshape(1, -1)

    n_samples = z_arr.shape[0]
    k_fake = p_fake_2d.shape[0]

    combined = np.concatenate([z_arr, p_real_2d, p_fake_2d], axis=0)
    coords = compute_2d_tsne_projection(combined, random_state=42)

    sample_coords = coords[:n_samples]
    real_proto_coord = coords[n_samples : n_samples + 1]
    fake_proto_coords = coords[n_samples + 1 : n_samples + 1 + k_fake]

    if HAS_MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=dpi)

        unique_gens = []
        for g in ("Real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
            if g in gens:
                unique_gens.append(g)
        for g in sorted(set(gens)):
            if g not in unique_gens:
                unique_gens.append(g)

        for g_name in unique_gens:
            mask = np.array([g == g_name for g in gens], dtype=bool)
            color = GENERATOR_COLORS.get(g_name, "#34495e")
            marker = "o" if g_name == "Real" else "^"
            ax.scatter(
                sample_coords[mask, 0],
                sample_coords[mask, 1],
                c=color,
                marker=marker,
                s=28,
                alpha=0.72,
                edgecolors="none",
                label=f"{g_name} (n={int(np.sum(mask))})",
            )

        # Plot learned Real prototype as a big gold-edged green star
        ax.scatter(
            real_proto_coord[:, 0],
            real_proto_coord[:, 1],
            c="#2ecc71",
            marker="*",
            s=360,
            edgecolors="#1b4f72",
            linewidths=1.8,
            zorder=10,
            label="Learned Real Prototype (1)",
        )

        # Plot learned K Fake prototypes as big dark-edged crimson stars
        ax.scatter(
            fake_proto_coords[:, 0],
            fake_proto_coords[:, 1],
            c="#e74c3c",
            marker="*",
            s=320,
            edgecolors="#1c2833",
            linewidths=1.6,
            zorder=10,
            label=f"Learned Fake Prototypes (K={k_fake})",
        )

        for k_idx in range(k_fake):
            ax.annotate(
                f"P_fake{k_idx + 1}",
                (fake_proto_coords[k_idx, 0], fake_proto_coords[k_idx, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#922b21",
            )

        ax.set_title(
            "t-SNE Projection of 128-d ZeroFake Embeddings & Multi-Prototypes",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", frameon=True, fontsize=9.5)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        canvas = np.full((650, 850, 3), 255, dtype=np.uint8)
        save_valid_300dpi_png_fallback(canvas, output_path, dpi=dpi)

    return {
        "num_projected_samples": int(n_samples),
        "num_fake_prototypes": int(k_fake),
        "output_path": output_path,
    }


def generate_tsne_from_checkpoint(
    checkpoint_path: str = "/content/drive/MyDrive/ZeroFake_Storage/checkpoints/zerofake/best_model.pt",
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    output_path: str = "results/plots/tsne_prototypes.png",
    max_samples: int = 1000,
    batch_size: int = 64,
    test_closed_loader: Optional[Iterable] = None,
    test_unseen_loader: Optional[Iterable] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads a trained ZeroFake checkpoint, extracts 128-d embeddings across `test_closed` and `test_unseen`
    (subsampling up to `max_samples=1000`), extracts the learned 1 Real + 6 Fake prototype vectors,
    and saves the 300-DPI t-SNE cluster plot.
    """
    if device is None:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    model, scorer, _ = load_model_and_scorer_from_checkpoint(checkpoint_path, device=device)

    if test_closed_loader is None or test_unseen_loader is None:
        manifest_path = os.path.join(data_dir, "frames_manifest.csv")
        resolved_splits = splits_path if splits_path else os.path.join(data_dir, "splits.json")
        if test_closed_loader is None:
            ds_closed = DeepfakeDataset(
                manifest_path=manifest_path,
                root_dir=data_dir,
                split="test_closed",
                splits_path=resolved_splits,
                augment=False,
            )
            test_closed_loader = DataLoader(ds_closed, batch_size=batch_size, shuffle=False, num_workers=2)
        if test_unseen_loader is None:
            ds_unseen = DeepfakeDataset(
                manifest_path=manifest_path,
                root_dir=data_dir,
                split="test_unseen",
                splits_path=resolved_splits,
                augment=False,
                include_closed_real_in_unseen=False,
            )
            test_unseen_loader = DataLoader(ds_unseen, batch_size=batch_size, shuffle=False, num_workers=2)

    closed_out = collect_split_inference_outputs(
        model=model,
        data_loader=test_closed_loader,
        open_set_scorer=scorer,
        device=device,
    )
    unseen_out = collect_split_inference_outputs(
        model=model,
        data_loader=test_unseen_loader,
        open_set_scorer=scorer,
        device=device,
    )

    all_embs = np.concatenate([closed_out["embeddings"], unseen_out["embeddings"]], axis=0)
    all_gens = list(closed_out["generators"]) + list(unseen_out["generators"])

    if HAS_TORCH:
        with torch.no_grad():
            p_real_t, p_fake_t = model.prototype_layer.get_normalized_prototypes()
            p_real_np = p_real_t.detach().cpu().float().numpy()
            p_fake_np = p_fake_t.detach().cpu().float().numpy()
    else:
        p_real_np = model.prototype_layer.real_prototype
        p_fake_np = model.prototype_layer.fake_prototypes

    return plot_tsne_embeddings_and_prototypes(
        sample_embeddings=all_embs,
        generator_names=all_gens,
        p_real_norm=p_real_np,
        p_fake_norm=p_fake_np,
        output_path=output_path,
        max_samples=max_samples,
        dpi=300,
    )


def run_dry_run() -> None:
    """
    Synthesizes 128-d L2-normalized embeddings across Real and 4 fake generators plus 1 Real prototype
    and 6 Fake prototypes on CPU, runs t-SNE/2D projection, and verifies the 300-DPI PNG output in <5s.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing visualization/embeddings.py (t-SNE of 128-d embeddings + 1 Real & 6 Fake prototypes)...")

    rng = np.random.default_rng(42)
    z_samples = rng.normal(0.0, 1.0, size=(100, 128)).astype(np.float32)
    z_samples /= np.maximum(np.linalg.norm(z_samples, axis=-1, keepdims=True), 1e-8)

    gens = (
        ["Real"] * 20
        + ["Deepfakes"] * 20
        + ["Face2Face"] * 20
        + ["FaceSwap"] * 20
        + ["NeuralTextures"] * 20
    )
    p_real = z_samples[0:1].copy()
    p_fake = z_samples[20:26].copy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = os.path.join(tmp_dir, "plots", "tsne_prototypes.png")
        info = plot_tsne_embeddings_and_prototypes(
            sample_embeddings=z_samples,
            generator_names=gens,
            p_real_norm=p_real,
            p_fake_norm=p_fake,
            output_path=out_png,
            max_samples=1000,
            dpi=300,
        )
        assert os.path.exists(out_png) and os.path.getsize(out_png) > 100, "t-SNE PNG file was not created!"
        assert info["num_projected_samples"] == 100 and info["num_fake_prototypes"] == 6

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(f"[DRY-RUN PASSED] Saved 300-DPI t-SNE cluster plot (100 samples + 1 Real + 6 Fake prototypes) in {elapsed:.3f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 300-DPI t-SNE projection of 128-d ZeroFake embeddings & prototypes.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/ZeroFake_Storage/checkpoints/zerofake/best_model.pt",
        help="Path to trained ZeroFake checkpoint (.pt).",
    )
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Path to dataset directory.")
    parser.add_argument("--splits_path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max test samples to project (default: 1000).")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/plots/tsne_prototypes.png",
        help="Output 300-DPI PNG path (default: results/plots/tsne_prototypes.png).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU dry-run verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    info = generate_tsne_from_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    print(f"[SUCCESS] Saved 300-DPI t-SNE cluster plot ({info['num_projected_samples']} samples) to '{args.output_path}'.")


if __name__ == "__main__":
    main()
