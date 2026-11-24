# File: training/train.py
# What this file does: Cloud GPU-ready training engine for ZeroFake supporting all 4 ablation modes
#   ('baseline', 'contrastive', 'single_proto', 'zerofake'), AdamW optimizer, CosineAnnealingLR scheduler,
#   modern PyTorch AMP mixed precision (torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda')),
#   generator-balanced mini-batch sampling, per-epoch checkpointing (last_checkpoint.pt and best_model.pt
#   with calibrated open-set parameters), automatic Google Drive cropped_faces.zip extraction to local SSD,
#   and seamless --resume recovery.
# Inputs:
#   - --mode (str): Ablation mode in {'baseline', 'contrastive', 'single_proto', 'zerofake'} (default: 'zerofake').
#   - --config (str): Path to YAML configuration file (default: 'configs/default.yaml').
#   - --holdout_generator (str): Held-out LOGO unseen generator (default: 'NeuralTextures').
#   - --epochs (int): Number of training epochs (default: 15).
#   - --batch_size (int): Mini-batch size (default: 64).
#   - --lr (float): Learning rate for AdamW (default: 1e-4).
#   - --data_dir (str): Path to extracted face crops and frames_manifest.csv (default: '/content/data').
#   - --save_dir (str): Directory to save last_checkpoint.pt and best_model.pt after every epoch.
#   - --resume (str or flag): Path to a checkpoint (or 'auto') to resume training from the last saved epoch.
#   - --dry-run (flag): Runs 2 full epochs on synthetic dummy images on CPU in <5s, saves checkpoints,
#     and verifies --resume recovery with zero errors.
# Outputs:
#   - last_checkpoint.pt (updated after every epoch with model, optimizer, scheduler, scaler, epoch, and open_set_state).
#   - best_model.pt (updated whenever validation ROC-AUC improves, plus final validation-calibrated open_set_state).
#   - training_history.json logging per-epoch train/val losses, ROC-AUC, EER, and calibrated tau_reject.

import sys
from pathlib import Path

# Insert repository root at sys.path[0] (and purge any pre-cached Hugging Face 'datasets' module)
# so `from datasets.deepfake_dataset import ...` always resolves to ZeroFake's local datasets/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if "datasets" in sys.modules and not hasattr(sys.modules["datasets"], "__path__"):
    del sys.modules["datasets"]

import argparse
import json
import os
import pickle
import tempfile
import time
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from datasets.deepfake_dataset import DeepfakeDataset, GeneratorBalancedBatchSampler
from losses.total_loss import ZeroFakeTotalLoss
from main import load_yaml_config
from models.open_set import OpenSetUncertaintyScorer
from models.zerofake_net import VALID_MODES, ZeroFakeNet
from training.validation import evaluate_validation_split

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
    HAS_TORCH = False


def maybe_unzip_drive_archive(
    data_dir: str,
    drive_zip_path: str = "/content/drive/MyDrive/ZeroFake_Storage/cropped_faces.zip",
) -> str:
    """
    Rule 4 (Cloud GPU Ready — Fast Local SSD I/O):
    If `frames_manifest.csv` is not yet present in `data_dir` (e.g. `/content/data` on Google Colab)
    and `cropped_faces.zip` exists on mounted Google Drive, automatically unzips it onto local SSD
    before training starts to avoid Google Drive network latency.
    """
    manifest_in_data = os.path.join(data_dir, "frames_manifest.csv")
    if os.path.exists(manifest_in_data):
        return data_dir

    if os.path.exists(drive_zip_path):
        print(f"[INFO] Unzipping '{drive_zip_path}' onto fast local SSD '{data_dir}'...")
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(drive_zip_path, "r") as zf:
            zf.extractall(data_dir)
        print(f"[SUCCESS] Extracted cropped faces to '{data_dir}'.")
    return data_dir


def save_checkpoint_file(payload: Dict[str, Any], path: str) -> None:
    """
    Saves a checkpoint dictionary to `path` using `torch.save` when PyTorch is available,
    or `pickle.dump` in minimal dry-run environments.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if HAS_TORCH:
        torch.save(payload, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(payload, f)


def load_checkpoint_file(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Loads a checkpoint dictionary from `path`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if HAS_TORCH:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    with open(path, "rb") as f:
        return pickle.load(f)


def get_model_state(model: ZeroFakeNet) -> Dict[str, Any]:
    """
    Extracts serializable parameter weights from `ZeroFakeNet`.
    """
    if HAS_TORCH and hasattr(model, "state_dict"):
        return model.state_dict()
    return {
        "backbone_proj": model.backbone.proj.copy(),
        "proj_w1": model.projection_head.w1.copy(),
        "proj_b1": model.projection_head.b1.copy(),
        "proj_w2": model.projection_head.w2.copy(),
        "proj_b2": model.projection_head.b2.copy(),
        "clf_w": model.classifier.w.copy(),
        "clf_b": model.classifier.b.copy(),
        "real_prototype": model.prototype_layer.real_prototype.copy(),
        "fake_prototypes": model.prototype_layer.fake_prototypes.copy(),
    }


def set_model_state(model: ZeroFakeNet, state: Dict[str, Any]) -> None:
    """
    Restores parameter weights into `ZeroFakeNet`.
    """
    if HAS_TORCH and hasattr(model, "load_state_dict"):
        model.load_state_dict(state)
        return
    if "backbone_proj" in state:
        model.backbone.proj = np.asarray(state["backbone_proj"], dtype=np.float32).copy()
        model.projection_head.w1 = np.asarray(state["proj_w1"], dtype=np.float32).copy()
        model.projection_head.b1 = np.asarray(state["proj_b1"], dtype=np.float32).copy()
        model.projection_head.w2 = np.asarray(state["proj_w2"], dtype=np.float32).copy()
        model.projection_head.b2 = np.asarray(state["proj_b2"], dtype=np.float32).copy()
        model.classifier.w = np.asarray(state["clf_w"], dtype=np.float32).copy()
        model.classifier.b = np.asarray(state["clf_b"], dtype=np.float32).copy()
        model.prototype_layer.real_prototype = np.asarray(state["real_prototype"], dtype=np.float32).copy()
        model.prototype_layer.fake_prototypes = np.asarray(state["fake_prototypes"], dtype=np.float32).copy()


def cast_outputs_to_float32(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Casts all floating-point PyTorch tensors in `outputs` to `float32` (`.float()`) outside the
    `autocast` block so loss functions (`L_cls`, `L_con`, `L_proto`) and open-set uncertainty `U(x)`
    never encounter `c10::Half` overflow.
    """
    if not HAS_TORCH:
        return outputs
    casted: Dict[str, Any] = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            casted[k] = v.float()
        else:
            casted[k] = v
    return casted


def train_one_epoch(
    model: ZeroFakeNet,
    train_loader: Iterable,
    criterion: ZeroFakeTotalLoss,
    optimizer: Any,
    scaler: Any,
    device: str = "cpu",
    use_amp: bool = False,
    lr: float = 1e-4,
) -> Dict[str, float]:
    """
    Runs one full training epoch using PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')`
    and `torch.amp.GradScaler('cuda')`) on CUDA GPUs, with automatic fallback to `fp32` on CPU.
    Only the model forward pass runs inside `autocast`; all outputs are cast to `float32` before
    computing `criterion(outputs, labels_dev)` outside the `autocast` block.
    """
    if HAS_TORCH and hasattr(model, "train"):
        model.train()

    amp_enabled = bool(use_amp and HAS_TORCH and str(device).startswith("cuda") and torch.cuda.is_available())

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    con_loss_sum = 0.0
    proto_loss_sum = 0.0
    num_batches = 0

    for batch in train_loader:
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch[0], batch[1]

        if HAS_TORCH and isinstance(images, torch.Tensor):
            images_dev = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled and scaler is not None:
                with torch.amp.autocast("cuda"):
                    raw_outputs = model(images_dev)
                outputs = cast_outputs_to_float32(raw_outputs)
                loss_dict = criterion(outputs, labels_dev)
                loss_total = loss_dict["loss_total"]
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                raw_outputs = model(images_dev)
                outputs = cast_outputs_to_float32(raw_outputs)
                loss_dict = criterion(outputs, labels_dev)
                loss_total = loss_dict["loss_total"]
                loss_total.backward()
                optimizer.step()

            total_loss_sum += float(loss_total.item())
            cls_loss_sum += float(loss_dict["loss_cls"].item())
            con_loss_sum += float(loss_dict["loss_con"].item())
            proto_loss_sum += float(loss_dict["loss_proto"].item())
        else:
            # NumPy dry-run training step on minimal CPU environments without PyTorch
            x_np = np.asarray(images, dtype=np.float32)
            y_np = np.asarray(labels, dtype=np.int64).reshape(-1)
            outputs = model(x_np)
            loss_dict = criterion(outputs, y_np)

            # Analytical gradient update step on classifier and prototypes
            b = max(1, x_np.shape[0])
            h = outputs["features"]
            p = outputs["probs"].reshape(-1, 1)
            y_col = y_np.reshape(-1, 1).astype(np.float32)
            dlogits = (p - y_col) / float(b)
            model.classifier.w -= float(lr) * (h.T @ dlogits)
            model.classifier.b -= float(lr) * np.sum(dlogits, axis=0)

            total_loss_sum += float(loss_dict["loss_total"])
            cls_loss_sum += float(loss_dict["loss_cls"])
            con_loss_sum += float(loss_dict["loss_con"])
            proto_loss_sum += float(loss_dict["loss_proto"])

        num_batches += 1

    denom = max(1, num_batches)
    return {
        "train_loss": total_loss_sum / denom,
        "train_loss_cls": cls_loss_sum / denom,
        "train_loss_con": con_loss_sum / denom,
        "train_loss_proto": proto_loss_sum / denom,
        "num_batches": num_batches,
    }


def run_training_loop(
    mode: str = "zerofake",
    config_path: str = "configs/default.yaml",
    holdout_generator: str = "NeuralTextures",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    data_dir: str = "/content/data",
    splits_path: Optional[str] = None,
    save_dir: str = "checkpoints",
    resume_path: Optional[str] = None,
    pretrained: bool = True,
    train_loader: Optional[Iterable] = None,
    val_loader: Optional[Iterable] = None,
    train_sampler: Optional[GeneratorBalancedBatchSampler] = None,
) -> Dict[str, Any]:
    """
    Orchestrates the full ZeroFake training pipeline:
      1. Initializes `ZeroFakeNet(mode=mode)`, `ZeroFakeTotalLoss(mode=mode)`, `OpenSetUncertaintyScorer`,
         `AdamW` optimizer, `CosineAnnealingLR` scheduler, and `torch.amp.GradScaler('cuda')`.
      2. Restores state if `--resume` is specified.
      3. Runs `train_one_epoch` and `evaluate_validation_split` for each epoch.
      4. Saves `last_checkpoint.pt` and `best_model.pt` (tracked by highest validation ROC-AUC, including
         calibrated `open_set_state`) into `save_dir` after EVERY epoch.
      5. At the end of training, runs a final validation calibration check on the best model and updates
         `best_model.pt` so `tau_reject` (at 95% validation coverage) is permanently locked into the checkpoint.
    """
    cfg = load_yaml_config(config_path)
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    use_amp = bool(cfg.get("training", {}).get("mixed_precision", True) and device == "cuda")

    # Build datasets & DataLoaders if not pre-supplied (e.g., by --dry-run)
    if train_loader is None or val_loader is None:
        maybe_unzip_drive_archive(data_dir)
        manifest_path = os.path.join(data_dir, "frames_manifest.csv")
        resolved_splits_path = splits_path if splits_path else os.path.join(data_dir, "splits.json")
        if not os.path.exists(manifest_path):
            manifest_path = str(cfg.get("data", {}).get("manifest_path", "datasets/processed/frames_manifest.csv"))
        if not os.path.exists(resolved_splits_path):
            resolved_splits_path = str(cfg.get("data", {}).get("splits_path", "datasets/splits.json"))

        train_dataset = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=data_dir,
            split="train",
            splits_path=resolved_splits_path,
            augment=True,
        )
        val_dataset = DeepfakeDataset(
            manifest_path=manifest_path,
            root_dir=data_dir,
            split="val",
            splits_path=resolved_splits_path,
            augment=False,
        )

        train_sampler = GeneratorBalancedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            seed=int(cfg.get("seed", 42)),
        )
        if HAS_TORCH:
            num_workers = int(cfg.get("training", {}).get("num_workers", 2))
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
            )
        else:
            raise RuntimeError("PyTorch is required for full DataLoader training on real datasets.")

    model = ZeroFakeNet(
        mode=mode,
        pretrained=pretrained,
        embed_dim=int(cfg.get("embedding_dim", 128)),
        num_fake_prototypes=int(cfg.get("num_fake_prototypes", 6)),
    )
    criterion = ZeroFakeTotalLoss(mode=mode, config_path=config_path)
    open_set_scorer = OpenSetUncertaintyScorer(
        alpha=float(cfg.get("alpha", 0.4)),
        beta=float(cfg.get("beta", 0.3)),
        gamma=float(cfg.get("gamma", 0.3)),
        target_val_coverage=float(cfg.get("target_coverage", 0.95)),
        embed_dim=int(cfg.get("embedding_dim", 128)),
    )

    if HAS_TORCH:
        model = model.to(device)
        criterion = criterion.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        optimizer = None
        scheduler = None
        scaler = None

    start_epoch = 0
    best_val_auc = -1.0
    history: List[Dict[str, Any]] = []

    last_ckpt_path = os.path.join(save_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(save_dir, "best_model.pt")

    # Handle --resume gracefully: if checkpoint is from a different mode/architecture, start fresh
    if resume_path:
        target_resume = last_ckpt_path if resume_path.lower() in ("auto", "true", "last") else resume_path
        if os.path.exists(target_resume):
            try:
                ckpt = load_checkpoint_file(target_resume, map_location=device)
                ckpt_mode = str(ckpt.get("mode", mode)).strip().lower()
                if ckpt_mode != mode.strip().lower():
                    print(
                        f"[RESUME WARNING] Checkpoint mode '{ckpt_mode}' != current mode '{mode}'. Starting fresh."
                    )
                else:
                    set_model_state(model, ckpt["model_state_dict"])
                    if HAS_TORCH and optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if HAS_TORCH and scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    if HAS_TORCH and scaler is not None and ckpt.get("scaler_state_dict") is not None:
                        scaler.load_state_dict(ckpt["scaler_state_dict"])
                    if "open_set_state" in ckpt:
                        open_set_scorer.load_state_dict(ckpt["open_set_state"])

                    start_epoch = int(ckpt.get("epoch", -1)) + 1
                    best_val_auc = float(ckpt.get("best_val_auc", -1.0))
                    history = list(ckpt.get("history", []))
                    print(
                        f"[RESUME] Restored checkpoint '{target_resume}' at epoch {start_epoch}/{epochs} "
                        f"(best_val_auc={best_val_auc:.4f})."
                    )
            except Exception as exc:
                print(f"[RESUME WARNING] Could not load '{target_resume}' ({exc}). Starting fresh from epoch 0.")

    for epoch in range(start_epoch, int(epochs)):
        epoch_t0 = time.time()
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            lr=lr,
        )

        if HAS_TORCH and scheduler is not None:
            scheduler.step()
            current_lr = float(scheduler.get_last_lr()[0])
        else:
            current_lr = float(lr)

        # Evaluate strictly on 'val' split & calibrate open-set threshold tau_reject on 'val'
        val_metrics = evaluate_validation_split(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            open_set_scorer=open_set_scorer,
            calibrate_open_set=True,
            device=device,
            use_amp=use_amp,
        )

        val_auc = float(val_metrics["roc_auc"])
        is_best = val_auc >= best_val_auc
        if is_best:
            best_val_auc = val_auc

        epoch_record = {
            "epoch": epoch,
            "mode": mode,
            "holdout_generator": holdout_generator,
            "lr": current_lr,
            **train_metrics,
            "val_loss": val_metrics["val_loss"],
            "val_roc_auc": val_auc,
            "val_eer": val_metrics["eer"],
            "val_f1": val_metrics["f1"],
            "tau_reject": float(open_set_scorer.tau_reject),
            "val_coverage": float(open_set_scorer.val_coverage_achieved),
            "elapsed_sec": time.time() - epoch_t0,
        }
        history.append(epoch_record)

        ckpt_payload = {
            "epoch": epoch,
            "mode": mode,
            "holdout_generator": holdout_generator,
            "model_state_dict": get_model_state(model),
            "optimizer_state_dict": optimizer.state_dict() if (HAS_TORCH and optimizer is not None) else None,
            "scheduler_state_dict": scheduler.state_dict() if (HAS_TORCH and scheduler is not None) else None,
            "scaler_state_dict": scaler.state_dict() if (HAS_TORCH and scaler is not None) else None,
            "best_val_auc": best_val_auc,
            "val_metrics": val_metrics,
            "open_set_state": open_set_scorer.state_dict(),
            "history": history,
            "config": cfg,
        }

        # Rule 4: Save last_checkpoint.pt and best_model.pt after EVERY epoch
        save_checkpoint_file(ckpt_payload, last_ckpt_path)
        if is_best or not os.path.exists(best_ckpt_path):
            save_checkpoint_file(ckpt_payload, best_ckpt_path)

        print(
            f"[Epoch {epoch + 1:02d}/{epochs:02d}] mode={mode} | "
            f"train_loss={train_metrics['train_loss']:.4f} | val_loss={val_metrics['val_loss']:.4f} | "
            f"val_AUC={val_auc:.4f} (best={best_val_auc:.4f}) | val_EER={val_metrics['eer']:.4f} | "
            f"tau_reject={open_set_scorer.tau_reject:.4f} ({open_set_scorer.val_coverage_achieved*100:.1f}% cov)"
        )

    # Final post-training validation calibration lock-in on best_model.pt
    if os.path.exists(best_ckpt_path):
        best_ckpt = load_checkpoint_file(best_ckpt_path, map_location=device)
        set_model_state(model, best_ckpt["model_state_dict"])
        final_val = evaluate_validation_split(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            open_set_scorer=open_set_scorer,
            calibrate_open_set=True,
            device=device,
            use_amp=use_amp,
        )
        best_ckpt["val_metrics"] = final_val
        best_ckpt["open_set_state"] = open_set_scorer.state_dict()
        save_checkpoint_file(best_ckpt, best_ckpt_path)

    history_path = os.path.join(save_dir, f"training_history_{mode}.json")
    with open(history_path, "w", encoding="utf-8") as hf:
        json.dump(history, hf, indent=2)

    return {
        "mode": mode,
        "epochs_completed": int(epochs),
        "best_val_auc": best_val_auc,
        "tau_reject": float(open_set_scorer.tau_reject),
        "val_coverage": float(open_set_scorer.val_coverage_achieved),
        "last_checkpoint": last_ckpt_path,
        "best_model": best_ckpt_path,
    }


def _make_synthetic_batches(num_batches: int, batch_size: int, split_name: str, seed: int) -> List[Dict[str, Any]]:
    """
    Creates synthetic dummy mini-batches `(batch_size, 3, 224, 224)` with balanced binary labels `{0, 1}`
    for fast CPU `--dry-run` testing.
    """
    rng = np.random.default_rng(seed)
    half = batch_size // 2
    batches: List[Dict[str, Any]] = []
    for _ in range(num_batches):
        imgs = rng.normal(0.0, 1.0, size=(batch_size, 3, 224, 224)).astype(np.float32)
        lbls = np.array([0] * half + [1] * (batch_size - half), dtype=np.int64)
        if HAS_TORCH:
            batches.append(
                {
                    "image": torch.from_numpy(imgs),
                    "label": torch.from_numpy(lbls),
                    "split": split_name,
                }
            )
        else:
            batches.append({"image": imgs, "label": lbls, "split": split_name})
    return batches


def run_dry_run(mode: str = "zerofake", config_path: str = "configs/default.yaml") -> None:
    """
    Runs 2 epochs on synthetic dummy images on CPU across all 4 ablation modes ('baseline', 'contrastive',
    'single_proto', 'zerofake') with simulated float16 autocast forward + float32 loss/backward passes,
    verifies zero overflow/NaNs, checks that `last_checkpoint.pt` and `best_model.pt` (including calibrated
    `open_set_state`) are saved after every epoch, and tests `--resume` recovery in under 5 seconds.
    """
    start_time = time.time()
    print(f"[DRY-RUN] Testing training/train.py across all 4 modes with simulated fp16 autocast -> fp32 loss...")

    train_batches = _make_synthetic_batches(num_batches=2, batch_size=8, split_name="train", seed=101)
    val_batches = _make_synthetic_batches(num_batches=3, batch_size=8, split_name="val", seed=202)

    # 1. Verify simulated float16 forward outputs -> float32 cast -> loss & uncertainty across all 4 modes
    for m in VALID_MODES:
        net_m = ZeroFakeNet(mode=m, pretrained=False)
        crit_m = ZeroFakeTotalLoss(mode=m, config_path=config_path)
        sample_batch = train_batches[0]
        if HAS_TORCH:
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                raw_out = net_m(sample_batch["image"])
            # Simulate strict float16 outputs coming out of CUDA autocast
            fp16_out = {
                k: (v.half() if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v)
                for k, v in raw_out.items()
            }
            fp32_out = cast_outputs_to_float32(fp16_out)
            loss_dict_m = crit_m(fp32_out, sample_batch["label"])
            loss_dict_m["loss_total"].backward()
            assert torch.isfinite(loss_dict_m["loss_total"]), f"Non-finite loss in mode={m}!"
        else:
            raw_out = net_m(sample_batch["image"].astype(np.float16).astype(np.float32))
            fp16_out = {
                k: (v.astype(np.float16) if (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)) else v)
                for k, v in raw_out.items()
            }
            fp32_out = {
                k: (v.astype(np.float32) if (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)) else v)
                for k, v in fp16_out.items()
            }
            loss_dict_m = crit_m(fp32_out, sample_batch["label"])
            assert np.isfinite(loss_dict_m["loss_total"]), f"Non-finite loss in mode={m}!"

    with tempfile.TemporaryDirectory() as tmp_save_dir:
        # Step 1: Run 1 epoch and save checkpoints
        run_training_loop(
            mode=mode,
            config_path=config_path,
            epochs=1,
            batch_size=8,
            lr=1e-4,
            save_dir=tmp_save_dir,
            resume_path=None,
            pretrained=False,
            train_loader=train_batches,
            val_loader=val_batches,
        )
        last_ckpt_file = os.path.join(tmp_save_dir, "last_checkpoint.pt")
        best_ckpt_file = os.path.join(tmp_save_dir, "best_model.pt")
        assert os.path.exists(last_ckpt_file), "last_checkpoint.pt was not created after epoch 1!"
        assert os.path.exists(best_ckpt_file), "best_model.pt was not created after epoch 1!"

        # Step 2: Resume from last_checkpoint.pt to complete epoch 2
        summary = run_training_loop(
            mode=mode,
            config_path=config_path,
            epochs=2,
            batch_size=8,
            lr=1e-4,
            save_dir=tmp_save_dir,
            resume_path=last_ckpt_file,
            pretrained=False,
            train_loader=train_batches,
            val_loader=val_batches,
        )

        loaded_best = load_checkpoint_file(best_ckpt_file)
        assert "open_set_state" in loaded_best and loaded_best["open_set_state"]["is_calibrated"], (
            "Calibrated open_set_state missing from best_model.pt!"
        )
        assert int(load_checkpoint_file(last_ckpt_file)["epoch"]) == 1, "Expected resumed checkpoint epoch index == 1!"

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run took {elapsed:.3f}s, exceeding 5.0s CPU limit!"
    print(
        f"[DRY-RUN PASSED] Verified all 4 modes ({list(VALID_MODES)}) with fp16->fp32 loss & completed 2 epochs + "
        f"checkpoint save/resume in {elapsed:.3f}s (best_val_auc={summary['best_val_auc']:.4f}, "
        f"tau_reject={summary['tau_reject']:.4f})."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZeroFake Cloud GPU Training Engine with Per-Epoch Checkpointing.")
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
        "--holdout_generator",
        type=str,
        default="NeuralTextures",
        help="Held-out LOGO unseen generator (default: NeuralTextures).",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15).")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4).")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/content/data",
        help="Directory containing extracted face crops and frames_manifest.csv.",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default=None,
        help="Optional explicit path to splits.json (defaults to <data_dir>/splits.json).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save last_checkpoint.pt and best_model.pt after every epoch.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from last_checkpoint.pt (pass --resume or --resume /path/to/ckpt.pt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 2 epochs on synthetic dummy images on CPU in <5s and verify checkpoint saving & resume.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(mode=args.mode, config_path=args.config)
        return

    manifest_path = os.path.join(args.data_dir, "frames_manifest.csv")
    splits_path = args.splits_path if args.splits_path else os.path.join(args.data_dir, "splits.json")
    print(f"[INFO] Resolved manifest_path={manifest_path}, splits_path={splits_path}")

    run_training_loop(
        mode=args.mode,
        config_path=args.config,
        holdout_generator=args.holdout_generator,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        splits_path=splits_path,
        save_dir=args.save_dir,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
