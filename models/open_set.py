# File: models/open_set.py
# What this file does: Computes the calibrated Open-Set Uncertainty score U(x) from three signals
#   (nearest-prototype cosine distance D_prototype(x), predictive Shannon entropy H(p(y|x)), and
#   out-of-distribution centroid/Mahalanobis distance S_OOD(x)), calibrates Z-score scalers and the
#   95th-percentile rejection threshold tau_reject STRICTLY on the validation split ('val'), and
#   applies the 3-way selective decision rule ('REAL', 'FAKE', or 'UNKNOWN').
# Inputs:
#   - model_outputs (dict): Output dictionary from ZeroFakeNet containing 'embeddings' z, 'probs' p,
#     'd_real', 'd_fake', 'p_real_norm', 'p_fake_norm', and 'evidence_fake'.
#   - val_loader (iterable): DataLoader or batch iterator strictly from the 'val' split for calibration.
#   - --dry-run (flag): Fits training centroids + calibrates on synthetic val batches, verifies 95%
#     validation coverage, tests 3-way predictions, and checks checkpoint state_dict save/load in <5s.
# Outputs:
#   - Dictionary containing raw signals ('d_prototype', 'entropy', 's_ood'), Z-score normalized signals,
#     combined uncertainty 'u_score', threshold 'tau_reject', binary predictions, and 3-way string verdicts
#     ('REAL', 'FAKE', 'UNKNOWN'), plus serializable state_dict for model checkpoint dictionaries.

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.zerofake_net import ZeroFakeNet

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


VERDICT_REAL = "REAL"
VERDICT_FAKE = "FAKE"
VERDICT_UNKNOWN = "UNKNOWN"

LABEL_ID_REAL = 0
LABEL_ID_FAKE = 1
LABEL_ID_UNKNOWN = 2


def _to_numpy(x: Any) -> np.ndarray:
    """
    Converts a PyTorch Tensor, list, or NumPy array into a clean float64 NumPy array on CPU.
    """
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def compute_d_prototype(d_real: Any, d_fake: Any) -> np.ndarray:
    """
    Signal 1 — Nearest-Prototype Cosine Distance `D_prototype(x)`:
      `D_prototype(x) = min(d_real(z), d_fake(z))`
    where `d_fake(z) = min_{j=1..K} d(z, p_fake_j)`.
    Returns a 1D float64 array of shape `(B,)`.
    """
    dr = _to_numpy(d_real).reshape(-1)
    df = _to_numpy(d_fake).reshape(-1)
    assert dr.shape == df.shape, f"d_real shape {dr.shape} != d_fake shape {df.shape}"
    return np.minimum(dr, df)


def compute_predictive_entropy(probs: Any, eps: float = 1e-7) -> np.ndarray:
    """
    Signal 2 — Shannon Predictive Entropy `H(p(y|x))`:
      `H(p) = -(p * log2(p) + (1 - p) * log2(1 - p))` in `[0, 1]`.
    Returns a 1D float64 array of shape `(B,)`.
    """
    p = np.clip(_to_numpy(probs).reshape(-1), eps, 1.0 - eps)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
    return np.clip(entropy, 0.0, 1.0)


def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Row-wise L2 normalization for 2D arrays of shape `(N, D)`.
    """
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
    return arr / np.maximum(norms, eps)


class OpenSetUncertaintyScorer:
    """
    Calibrated Open-Set Uncertainty Scorer (Section 5.6 of SPEC.md & Rule 3 of AGENTS.md):
      1. Computes three raw uncertainty signals:
         - `D_prototype(x) = min(d_real(z), d_fake(z))`
         - `H(p(y|x))      = Shannon predictive entropy of p`
         - `S_OOD(x)       = Minimum cosine + diagonal-Mahalanobis distance from z to training class centroids`
      2. Combines Z-score normalized signals with weights `(alpha, beta, gamma)` (default `0.4, 0.3, 0.3`):
         `U(x) = alpha * Z(D_prototype(x)) + beta * Z(H(p(y|x))) + gamma * Z(S_OOD(x))`
      3. Calibrates Z-score means/stds and sets `tau_reject` at the 95th percentile of `U(x)`
         STRICTLY on the validation split (`val`).
      4. Applies the 3-way selective decision rule:
         - `'UNKNOWN'` if `U(x) > tau_reject`
         - `'FAKE'`    if `U(x) <= tau_reject` and `E_fake(x) >= 0.5`
         - `'REAL'`    if `U(x) <= tau_reject` and `E_fake(x) < 0.5`
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        target_val_coverage: float = 0.95,
        embed_dim: int = 128,
        eps: float = 1e-6,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.target_val_coverage = float(target_val_coverage)
        self.embed_dim = int(embed_dim)
        self.eps = float(eps)

        # Training class feature centroids and diagonal variances for S_OOD(x)
        self.real_centroid: np.ndarray = np.zeros((1, self.embed_dim), dtype=np.float64)
        self.fake_centroid: np.ndarray = np.zeros((1, self.embed_dim), dtype=np.float64)
        self.real_var: np.ndarray = np.ones((1, self.embed_dim), dtype=np.float64)
        self.fake_var: np.ndarray = np.ones((1, self.embed_dim), dtype=np.float64)
        self.centroids_fitted: bool = False

        # Validation-only Z-score normalization parameters (mean and std for each of the 3 signals)
        self.mean_d_proto: float = 0.0
        self.std_d_proto: float = 1.0
        self.mean_entropy: float = 0.0
        self.std_entropy: float = 1.0
        self.mean_s_ood: float = 0.0
        self.std_s_ood: float = 1.0

        # Calibrated open-set rejection threshold tau_reject (95th percentile of validation U(x))
        self.tau_reject: float = float("inf")
        self.val_coverage_achieved: float = 1.0
        self.is_calibrated: bool = False

    def fit_training_centroids(self, embeddings: Any, labels: Any) -> None:
        """
        Fits the Real (`y=0`) and Fake (`y=1`) class feature centroids and shrinkage-regularized diagonal
        variances from training embeddings `z` (shape `(N, 128)`).
        """
        z = l2_normalize_rows(_to_numpy(embeddings))
        y = _to_numpy(labels).reshape(-1).astype(np.int64)
        assert z.shape[0] == y.shape[0], "Embeddings and labels count mismatch in fit_training_centroids!"

        real_mask = y == 0
        fake_mask = y == 1

        if np.any(real_mask):
            z_real = z[real_mask]
            self.real_centroid = l2_normalize_rows(np.mean(z_real, axis=0, keepdims=True))
            self.real_var = np.var(z_real, axis=0, keepdims=True) + 0.05
        if np.any(fake_mask):
            z_fake = z[fake_mask]
            self.fake_centroid = l2_normalize_rows(np.mean(z_fake, axis=0, keepdims=True))
            self.fake_var = np.var(z_fake, axis=0, keepdims=True) + 0.05

        self.centroids_fitted = True

    def compute_s_ood(self, embeddings: Any, model_outputs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Signal 3 — Out-of-Distribution Feature Distance `S_OOD(x)`:
        Computes the minimum hybrid cosine + diagonal-Mahalanobis distance from embedding `z` to the
        training class feature centroids `{mu_real, mu_fake}`.
        If `fit_training_centroids()` has not been explicitly called prior to calibration, uses the
        learned prototype centroids (`p_real_norm` and mean of `p_fake_norm`) as fallback centroids.
        """
        z = l2_normalize_rows(_to_numpy(embeddings))

        if self.centroids_fitted:
            mu_real = self.real_centroid
            mu_fake = self.fake_centroid
            var_real = self.real_var
            var_fake = self.fake_var
        elif model_outputs is not None and "p_real_norm" in model_outputs and "p_fake_norm" in model_outputs:
            mu_real = l2_normalize_rows(_to_numpy(model_outputs["p_real_norm"]))
            mu_fake = l2_normalize_rows(np.mean(_to_numpy(model_outputs["p_fake_norm"]), axis=0, keepdims=True))
            var_real = np.ones((1, z.shape[1]), dtype=np.float64)
            var_fake = np.ones((1, z.shape[1]), dtype=np.float64)
        else:
            mu_real = self.real_centroid
            mu_fake = self.fake_centroid
            var_real = self.real_var
            var_fake = self.fake_var

        # Cosine distance to Real and Fake class centroids
        cos_dist_real = 1.0 - (z @ mu_real.T).squeeze(-1)
        cos_dist_fake = 1.0 - (z @ mu_fake.T).squeeze(-1)

        # Diagonal Mahalanobis distance normalized by sqrt(D)
        dim_scale = math.sqrt(float(z.shape[1]))
        maha_real = np.sqrt(np.sum(((z - mu_real) ** 2) / np.maximum(var_real, 1e-4), axis=-1)) / dim_scale
        maha_fake = np.sqrt(np.sum(((z - mu_fake) ** 2) / np.maximum(var_fake, 1e-4), axis=-1)) / dim_scale

        dist_real = 0.5 * cos_dist_real + 0.5 * maha_real
        dist_fake = 0.5 * cos_dist_fake + 0.5 * maha_fake
        return np.minimum(dist_real, dist_fake)

    def compute_raw_signals(self, model_outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the three raw uncertainty signal vectors `(d_proto, entropy, s_ood)` each of shape `(B,)`
        from a `ZeroFakeNet` forward output dictionary.
        """
        d_proto = compute_d_prototype(model_outputs["d_real"], model_outputs["d_fake"])
        entropy = compute_predictive_entropy(model_outputs["probs"])
        s_ood = self.compute_s_ood(model_outputs["embeddings"], model_outputs=model_outputs)
        return d_proto, entropy, s_ood

    def compute_uncertainty(self, model_outputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Computes Z-score normalized signals and the combined uncertainty score:
          `U(x) = alpha * Z(D_prototype(x)) + beta * Z(H(p(y|x))) + gamma * Z(S_OOD(x))`
        """
        d_proto, entropy, s_ood = self.compute_raw_signals(model_outputs)

        z_d_proto = (d_proto - self.mean_d_proto) / max(self.std_d_proto, self.eps)
        z_entropy = (entropy - self.mean_entropy) / max(self.std_entropy, self.eps)
        z_s_ood = (s_ood - self.mean_s_ood) / max(self.std_s_ood, self.eps)

        u_score = (
            self.alpha * z_d_proto
            + self.beta * z_entropy
            + self.gamma * z_s_ood
        )

        return {
            "d_prototype": d_proto.astype(np.float32),
            "entropy": entropy.astype(np.float32),
            "s_ood": s_ood.astype(np.float32),
            "z_d_prototype": z_d_proto.astype(np.float32),
            "z_entropy": z_entropy.astype(np.float32),
            "z_s_ood": z_s_ood.astype(np.float32),
            "u_score": u_score.astype(np.float32),
        }

    def calibrate(
        self,
        val_loader: Iterable,
        model: Optional[Any] = None,
        device: Optional[str] = None,
        train_loader: Optional[Iterable] = None,
    ) -> Dict[str, float]:
        """
        Calibrates open-set Z-score normalization statistics and the 95th-percentile rejection threshold
        `tau_reject` STRICTLY on the validation split (`val_loader`).
        Enforces Rule 3 of AGENTS.md: raises an AssertionError if any batch or dataset attached to
        `val_loader` belongs to `'test_unseen'`, `'test_closed'`, or `'celeb_df'`.
        """
        # Rule 3 Guardrail: Verify val_loader is not a test split DataLoader
        dataset_obj = getattr(val_loader, "dataset", None)
        if dataset_obj is not None and hasattr(dataset_obj, "split"):
            split_attr = str(getattr(dataset_obj, "split")).lower()
            assert split_attr == "val", (
                f"Rule 3 Violation: OpenSetUncertaintyScorer.calibrate() must ONLY be called on 'val' split, "
                f"but received split='{split_attr}'!"
            )

        if model is not None and HAS_TORCH and hasattr(model, "eval"):
            model.eval()

        # Optional: fit training class centroids from train_loader if provided
        if train_loader is not None:
            train_embs: List[np.ndarray] = []
            train_lbls: List[np.ndarray] = []
            for batch in train_loader:
                out, lbl = self._extract_outputs_and_labels(batch, model=model, device=device)
                train_embs.append(_to_numpy(out["embeddings"]))
                if lbl is not None:
                    train_lbls.append(_to_numpy(lbl).reshape(-1))
            if train_embs and train_lbls:
                self.fit_training_centroids(np.concatenate(train_embs, axis=0), np.concatenate(train_lbls, axis=0))

        all_d_proto: List[np.ndarray] = []
        all_entropy: List[np.ndarray] = []
        all_embs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        last_outputs: Optional[Dict[str, Any]] = None

        for batch in val_loader:
            if isinstance(batch, dict) and "split" in batch:
                splits_in_batch = batch["split"]
                if isinstance(splits_in_batch, (list, tuple)):
                    for s in splits_in_batch:
                        assert str(s).lower() == "val", f"Rule 3 Violation: non-val sample '{s}' in calibrate()!"
                elif isinstance(splits_in_batch, str):
                    assert splits_in_batch.lower() == "val", (
                        f"Rule 3 Violation: non-val split '{splits_in_batch}' in calibrate()!"
                    )

            outputs, labels = self._extract_outputs_and_labels(batch, model=model, device=device)
            last_outputs = outputs
            all_d_proto.append(compute_d_prototype(outputs["d_real"], outputs["d_fake"]))
            all_entropy.append(compute_predictive_entropy(outputs["probs"]))
            all_embs.append(_to_numpy(outputs["embeddings"]))
            if labels is not None:
                all_labels.append(_to_numpy(labels).reshape(-1))

        if not all_d_proto:
            raise ValueError("val_loader yielded zero validation samples during open-set calibration!")

        val_embs = np.concatenate(all_embs, axis=0)
        if not self.centroids_fitted and all_labels:
            val_lbls = np.concatenate(all_labels, axis=0)
            self.fit_training_centroids(val_embs, val_lbls)

        val_d_proto = np.concatenate(all_d_proto, axis=0)
        val_entropy = np.concatenate(all_entropy, axis=0)
        val_s_ood = self.compute_s_ood(val_embs, model_outputs=last_outputs)

        # 1. Fit validation Z-score means and standard deviations
        self.mean_d_proto = float(np.mean(val_d_proto))
        self.std_d_proto = float(max(np.std(val_d_proto), self.eps))

        self.mean_entropy = float(np.mean(val_entropy))
        self.std_entropy = float(max(np.std(val_entropy), self.eps))

        self.mean_s_ood = float(np.mean(val_s_ood))
        self.std_s_ood = float(max(np.std(val_s_ood), self.eps))

        # 2. Compute Z-score normalized validation uncertainty U(x)
        z_d = (val_d_proto - self.mean_d_proto) / self.std_d_proto
        z_h = (val_entropy - self.mean_entropy) / self.std_entropy
        z_s = (val_s_ood - self.mean_s_ood) / self.std_s_ood

        val_u = self.alpha * z_d + self.beta * z_h + self.gamma * z_s

        # 3. Set tau_reject at the 95th percentile of validation U(x) (retaining 95% of validation samples)
        percentile_q = float(np.clip(self.target_val_coverage * 100.0, 1.0, 99.9))
        self.tau_reject = float(np.percentile(val_u, percentile_q))
        self.val_coverage_achieved = float(np.mean(val_u <= self.tau_reject))
        self.is_calibrated = True

        return {
            "tau_reject": self.tau_reject,
            "val_coverage": self.val_coverage_achieved,
            "mean_d_proto": self.mean_d_proto,
            "std_d_proto": self.std_d_proto,
            "mean_entropy": self.mean_entropy,
            "std_entropy": self.std_entropy,
            "mean_s_ood": self.mean_s_ood,
            "std_s_ood": self.std_s_ood,
        }

    def predict(
        self,
        model_outputs: Dict[str, Any],
        open_set_enabled: bool = True,
        fake_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Applies the 3-way selective decision rule:
          - If `open_set_enabled=True` and `U(x) > tau_reject`: verdict is `'UNKNOWN'` (class ID 2)
          - Otherwise, if `evidence_fake >= fake_threshold`: verdict is `'FAKE'` (class ID 1)
          - Otherwise: verdict is `'REAL'` (class ID 0)
        """
        unc_dict = self.compute_uncertainty(model_outputs)
        u_score = unc_dict["u_score"]

        if "evidence_fake" in model_outputs:
            evidence_fake = _to_numpy(model_outputs["evidence_fake"]).reshape(-1).astype(np.float32)
        else:
            evidence_fake = _to_numpy(model_outputs["probs"]).reshape(-1).astype(np.float32)

        binary_preds = (evidence_fake >= fake_threshold).astype(np.int64)
        is_rejected = (u_score > self.tau_reject) if (open_set_enabled and self.is_calibrated) else np.zeros_like(binary_preds, dtype=bool)

        verdicts: List[str] = []
        three_way_ids = np.zeros_like(binary_preds, dtype=np.int64)

        for i in range(len(binary_preds)):
            if bool(is_rejected[i]):
                verdicts.append(VERDICT_UNKNOWN)
                three_way_ids[i] = LABEL_ID_UNKNOWN
            elif binary_preds[i] == 1:
                verdicts.append(VERDICT_FAKE)
                three_way_ids[i] = LABEL_ID_FAKE
            else:
                verdicts.append(VERDICT_REAL)
                three_way_ids[i] = LABEL_ID_REAL

        return {
            **unc_dict,
            "tau_reject": float(self.tau_reject),
            "evidence_fake": evidence_fake,
            "binary_preds": binary_preds,
            "is_rejected": is_rejected,
            "three_way_ids": three_way_ids,
            "verdicts": verdicts,
        }

    def state_dict(self) -> Dict[str, Any]:
        """
        Exports all calibration parameters into a plain Python dictionary ready to be stored
        inside a model checkpoint dictionary (`checkpoint['open_set_state'] = scorer.state_dict()`).
        """
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "target_val_coverage": self.target_val_coverage,
            "embed_dim": self.embed_dim,
            "real_centroid": self.real_centroid.tolist(),
            "fake_centroid": self.fake_centroid.tolist(),
            "real_var": self.real_var.tolist(),
            "fake_var": self.fake_var.tolist(),
            "centroids_fitted": self.centroids_fitted,
            "mean_d_proto": self.mean_d_proto,
            "std_d_proto": self.std_d_proto,
            "mean_entropy": self.mean_entropy,
            "std_entropy": self.std_entropy,
            "mean_s_ood": self.mean_s_ood,
            "std_s_ood": self.std_s_ood,
            "tau_reject": self.tau_reject,
            "val_coverage_achieved": self.val_coverage_achieved,
            "is_calibrated": self.is_calibrated,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restores calibration parameters from a saved checkpoint dictionary.
        """
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta = float(state.get("beta", self.beta))
        self.gamma = float(state.get("gamma", self.gamma))
        self.target_val_coverage = float(state.get("target_val_coverage", self.target_val_coverage))
        self.embed_dim = int(state.get("embed_dim", self.embed_dim))

        if "real_centroid" in state:
            self.real_centroid = np.asarray(state["real_centroid"], dtype=np.float64).reshape(1, self.embed_dim)
        if "fake_centroid" in state:
            self.fake_centroid = np.asarray(state["fake_centroid"], dtype=np.float64).reshape(1, self.embed_dim)
        if "real_var" in state:
            self.real_var = np.asarray(state["real_var"], dtype=np.float64).reshape(1, self.embed_dim)
        if "fake_var" in state:
            self.fake_var = np.asarray(state["fake_var"], dtype=np.float64).reshape(1, self.embed_dim)

        self.centroids_fitted = bool(state.get("centroids_fitted", False))
        self.mean_d_proto = float(state.get("mean_d_proto", 0.0))
        self.std_d_proto = float(state.get("std_d_proto", 1.0))
        self.mean_entropy = float(state.get("mean_entropy", 0.0))
        self.std_entropy = float(state.get("std_entropy", 1.0))
        self.mean_s_ood = float(state.get("mean_s_ood", 0.0))
        self.std_s_ood = float(state.get("std_s_ood", 1.0))
        self.tau_reject = float(state.get("tau_reject", float("inf")))
        self.val_coverage_achieved = float(state.get("val_coverage_achieved", 1.0))
        self.is_calibrated = bool(state.get("is_calibrated", False))

    def _extract_outputs_and_labels(
        self,
        batch: Any,
        model: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """
        Helper supporting either precomputed `ZeroFakeNet` output dicts or raw image batches `{'image': ..., 'label': ...}`
        passed together with `model`.
        """
        if isinstance(batch, dict) and "embeddings" in batch and "probs" in batch and "d_real" in batch:
            return batch, batch.get("label", batch.get("labels", None))

        if model is None:
            raise ValueError("When val_loader yields image batches rather than model_outputs, pass `model` to calibrate().")

        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch.get("label", batch.get("labels", None))
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch, None

        if HAS_TORCH and isinstance(images, torch.Tensor):
            if device is not None:
                images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
        else:
            outputs = model(images)

        return outputs, labels


def run_dry_run() -> None:
    """
    Runs a fast (<5s) CPU dry-run that:
      1. Passes synthetic validation batches through `ZeroFakeNet` and `OpenSetUncertaintyScorer.calibrate(val_loader)`.
      2. Asserts validation coverage is 95% (`tau_reject` at 95th percentile).
      3. Verifies 3-way selective decisions (`REAL`, `FAKE`, `UNKNOWN`) and checkpoint state_dict save/load.
      4. Confirms Rule 3 guardrail blocks calibration if a non-`val` split is passed.
    """
    start_time = time.time()
    print("[DRY-RUN] Testing models/open_set.py (U(x) = 0.4*D_proto + 0.3*H + 0.3*S_OOD & 95% val calibration)...")

    net = ZeroFakeNet(mode="zerofake", pretrained=False)
    scorer = OpenSetUncertaintyScorer(alpha=0.4, beta=0.3, gamma=0.3, target_val_coverage=0.95)

    # Synthesize 5 validation batches of 20 samples each = 100 validation samples
    rng = np.random.default_rng(42)
    val_batches: List[Dict[str, Any]] = []
    for _ in range(5):
        if HAS_TORCH:
            imgs = torch.from_numpy(rng.normal(0.0, 1.0, size=(20, 3, 224, 224)).astype(np.float32))
            lbls = torch.from_numpy(rng.integers(0, 2, size=(20,)).astype(np.int64))
        else:
            imgs = rng.normal(0.0, 1.0, size=(20, 3, 224, 224)).astype(np.float32)
            lbls = rng.integers(0, 2, size=(20,)).astype(np.int64)
        out = net(imgs)
        out["label"] = lbls
        out["split"] = "val"
        val_batches.append(out)

    cal_stats = scorer.calibrate(val_batches)
    assert scorer.is_calibrated, "Scorer should be marked calibrated after calibrate()!"
    assert abs(cal_stats["val_coverage"] - 0.95) <= 0.02, (
        f"Expected ~95% validation coverage, got {cal_stats['val_coverage'] * 100:.1f}%"
    )

    # Evaluate 3-way predictions across all 100 validation samples
    total_unknown = 0
    total_real_or_fake = 0
    for b in val_batches:
        pred_out = scorer.predict(b, open_set_enabled=True)
        for v in pred_out["verdicts"]:
            assert v in (VERDICT_REAL, VERDICT_FAKE, VERDICT_UNKNOWN)
            if v == VERDICT_UNKNOWN:
                total_unknown += 1
            else:
                total_real_or_fake += 1

    assert total_unknown == 5 and total_real_or_fake == 95, (
        f"Expected 95 accepted and 5 UNKNOWN on 100 val samples, got {total_real_or_fake} accepted and {total_unknown} UNKNOWN"
    )

    # Test checkpoint dictionary serialization and restoration
    ckpt_payload = {"open_set_state": scorer.state_dict()}
    restored_scorer = OpenSetUncertaintyScorer()
    restored_scorer.load_state_dict(ckpt_payload["open_set_state"])
    assert restored_scorer.is_calibrated
    assert abs(restored_scorer.tau_reject - scorer.tau_reject) < 1e-8

    # Verify Rule 3 guardrail blocks test_unseen split during calibration
    blocked_unseen = False
    try:
        bad_batch = dict(val_batches[0])
        bad_batch["split"] = "test_unseen"
        scorer.calibrate([bad_batch])
    except AssertionError:
        blocked_unseen = True
    assert blocked_unseen, "Rule 3 Guardrail failed to block 'test_unseen' inside calibrate()!"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Calibrated tau_reject={scorer.tau_reject:.4f} on 'val' split "
        f"(coverage={cal_stats['val_coverage']*100:.1f}%, 95 accepted / 5 UNKNOWN, Rule 3 isolation enforced) "
        f"in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrated Open-Set Uncertainty Scorer U(x) and 3-way decision rule.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for nearest-prototype distance D_prototype(x).")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for predictive entropy H(p(y|x)).")
    parser.add_argument("--gamma", type=float, default=0.3, help="Weight for OOD centroid distance S_OOD(x).")
    parser.add_argument("--dry-run", action="store_true", help="Run fast (<5s) CPU open-set calibration verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return
    print("[INFO] OpenSetUncertaintyScorer ready. Import from models.open_set or run with --dry-run.")


if __name__ == "__main__":
    main()
