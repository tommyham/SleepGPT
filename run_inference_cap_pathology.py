"""Inference-only evaluation for SleepGPT on CAP pathology data.

Loads a trained model checkpoint and runs evaluation on all train/val/test splits
without any training. Outputs:
  - Per-window sequence CSVs: {output_dir}/fold_{k}/sequences/{split}/{subject_id}.csv
  - Per-fold metrics JSON (VisualizeEvaluationResult format):
    {output_dir}/fold_{k}/label_sequence_jmetrics.json

Usage example (single checkpoint for all folds):
    python run_inference_cap_pathology.py \\
        --checkpoint /path/to/model.ckpt \\
        --cap-root /data/capslpdb/process \\
        --output-dir ./inference_results

Usage example (one checkpoint per fold):
    python run_inference_cap_pathology.py \\
        --checkpoint /path/fold0.ckpt /path/fold1.ckpt /path/fold2.ckpt /path/fold3.ckpt \\
        --cap-root /data/capslpdb/process \\
        --kfold 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from main.datamodules.Multi_datamodule import MultiDataModule
from main.modules import Model

# CAP pathology class order matches CAP_datasets named config in config.py
CAP_PATHOLOGIES = ["n", "ins", "narco", "nfle", "plm", "rbd", "sdb"]
NUM_CLASSES = len(CAP_PATHOLOGIES)  # 7


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> dict:
    """Build _config dict equivalent to finetune_CAP + CAP_datasets + run_ft overrides.

    All values are taken from:
      - main/config.py @ex.config defaults
      - finetune_CAP named config
      - CAP_datasets named config
      - run_ft_cap_pathology_kfold.py build_config_updates()
    """
    cap_root = args.cap_root
    use_cuda = torch.cuda.is_available() and not args.cpu

    return {
        # ── Identity / experiment ────────────────────────────────────────────
        "extra_name": "Finetune_cap_all",
        "exp_name": "sleep",
        "seed": 3407,
        "random_seed": [3407],
        "precision": "16-mixed",
        "mode": "Finetune_cap_all",
        "kfold": 4,  # matches run_ft_cap_pathology_kfold.py (overridden per fold in run_fold())

        # ── Batch / training schedule (not used for inference) ───────────────
        "batch_size": args.batch_size,
        "max_epoch": 30,
        "max_steps": -1,
        "accum_iter": 2,
        "start_epoch": 0,

        # ── Data ────────────────────────────────────────────────────────────
        "datasets": [f"CAP_{p}" for p in CAP_PATHOLOGIES],
        "data_dir": [os.path.join(cap_root, p) for p in CAP_PATHOLOGIES],
        "data_setting": {"CAP": None},

        # ── Loss / task config (Pathology=1 → current_tasks=['Pathology']) ──
        "dropout": 0.0,
        "loss_names": {
            "Spindle": 0,
            "CrossEntropy": 0,
            "mtm": 0,
            "itc": 0,
            "itm": 0,
            "Apnea": 0,
            "Pathology": 1,
        },
        "transform_keys": {"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]},
        "num_workers": args.num_workers,
        "drop_path_rate": 0.5,
        "patch_size": 200,
        "lr_mult": 20,
        "blr": 1.5e-5,
        "end_lr": 0,
        "warmup_steps": 0.1,
        "smoothing": 0.1,
        "mixup": 0,

        # ── Directories ─────────────────────────────────────────────────────
        "output_dir": "./checkpoint/2201210064/experiments",
        "log_dir": "./checkpoint_log/2201210064/experiments",
        "load_path": None,   # overridden in load_model()
        "kfold_load_path": "",
        "resume_ckpt_path": "",

        # ── Optimizer (read by Model.__init__ / configure_optimizers) ───────
        "lr_policy": "cosine",
        "optim": "adamw",
        "clip_grad": False,
        "weight_decay": 0.05,
        "lr": 5e-4,
        "min_lr": 0,
        "warmup_lr": 0,
        "layer_decay": 0.75,
        "get_param_method": "layer_decay",
        "Lambda": 1.0,
        "poly": False,
        "gradient_clip_val": 1.0,

        # ── Device ──────────────────────────────────────────────────────────
        "device": "cuda" if use_cuda else "cpu",
        "deepspeed": False,
        "dist_on_itp": False,
        "num_gpus": 1,
        "num_nodes": 1,

        # ── Evaluation flags ────────────────────────────────────────────────
        "dist_eval": False,
        "eval": True,
        "get_recall_metric": False,
        "limit_val_batches": 1.0,
        "limit_train_batches": 1.0,
        "val_check_interval": 0.5,
        "check_val_every_n_epoch": 1,
        "fast_dev_run": 7,

        # ── Architecture ────────────────────────────────────────────────────
        "model_arch": "backbone_large_patch200",
        "epoch_duration": 30,
        "fs": 100,
        "mask_ratio": None,
        "max_time_len": 1,
        "random_choose_channels": 8,
        "actual_channels": None,
        "time_only": False,
        "fft_only": False,
        "loss_function": "l1",
        "resume_during_training": 0,
        "use_triton": False,
        "use_relative_pos_emb": False,
        "use_global_fft": True,
        "use_multiway": False,
        "use_g_mid": False,
        "local_pooling": False,
        "multi_y": ["tf"],
        "num_encoder_layers": 4,
        "use_cb": True,

        # ── Time-series / sliding window ────────────────────────────────────
        "all_time": True,
        "time_size": 100,
        "split_len": 100,
        "use_all_label": "all",

        # ── Decoder / pooler ────────────────────────────────────────────────
        "use_pooling": "longnet",
        "pool": None,
        "decoder_features": 1024,
        "decoder_depth": 4,
        "decoder_heads": 32,
        "decoder_selected_layers": "2-3",
        "longnet_dr": [1, 2, 4, 8, 16],
        "longnet_sl": [32, 64, 128, 512, 1024],
        "longnet_pool": False,

        # ── Swin / FPN (not used for longnet, but Model reads them) ─────────
        "Swin_window_size": 60,
        "Use_FPN": None,
        "FPN_resnet": False,
        "num_queries": 400,
        "Event_decoder_depth": 4,
        "Event_enc_dim": 384,

        # ── Spindle / Apnea (inactive for CAP pathology) ────────────────────
        "mass_aug_times": 0,
        "expert": None,
        "IOU_th": 0.2,
        "sp_prob": 0.55,
        "patch_time": 30,
        "use_fpfn": None,
        "CE_Weight": 10,

        # ── Misc ────────────────────────────────────────────────────────────
        "aug_test": None,
        "EDF_Mode": None,
        "subset": None,
        "aug_dir": None,
        "aug_prob": 0.0,
        "kfold_test": None,
        "grad_name": "partial_10",
        "save_top_k": 2,
        "show_transform_param": False,
        "mask_strategies": None,

        # ── Visualization (disabled) ─────────────────────────────────────────
        "visual": False,
        "visual_setting": {"mask_same": False, "mode": None, "save_extra_name": None},
        "persub": None,
        "return_alpha": False,

        # ── Classification ───────────────────────────────────────────────────
        "num_classes": NUM_CLASSES,
        "stage1_epoch": (5,),
        "stage2_epoch": 10,
        "freeze_stage": False,

        # ── SpO2 / ODS (inactive) ────────────────────────────────────────────
        "spo2_ods_settings": {
            "inj": False,
            "d_spo2": 128,
            "xattn_layers": [12],
            "hybrid_loss": True,
            "ods_pos_w": 10,
            "model_type": "lstm",
            "use_seq": False,
            "concat": False,
        },
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: dict, checkpoint_path: str, device: str) -> Model:
    """Instantiate Model and load weights via the built-in load_pretrained_weight()."""
    cfg = dict(config)
    cfg["load_path"] = checkpoint_path
    model = Model(cfg)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

def _move_list_to_device(lst: list, device: str) -> list:
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in lst]


def _move_batch_to_device(batch: dict, device: str) -> dict:
    result: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, list):
            result[k] = _move_list_to_device(v, device)
        else:
            result[k] = v
    return result


def _preprocess_batch(batch: dict, model: Model) -> dict:
    """Replicate the preprocessing done inside Model.forward() before calling infer().

    Model.forward() stacks label lists, computes FFT, and builds the full attention
    mask before passing the batch to infer(). We replicate that here so we can call
    infer() directly and retrieve logits without computing loss.
    """
    # Stack label lists into proper tensors (mirrors forward() lines 1167-1177)
    if "Stage_label" in batch:
        batch["Stage_label"] = torch.stack(batch["Stage_label"], dim=0).squeeze(-1)
    if "Pathology_label" in batch:
        batch["Pathology_label"] = (
            torch.stack(batch["Pathology_label"], dim=0).squeeze(1).squeeze(-1)
        )
    if "Spindle_label" in batch:
        batch["Spindle_label"] = (
            torch.stack(batch["Spindle_label"], dim=0).squeeze(1).squeeze(-1)
        )
    if "Apnea_label" in batch:
        batch["Apnea_label"] = (
            torch.stack(batch["Apnea_label"], dim=0).squeeze(1).squeeze(-1)
        )

    # Compute FFT branch and build full attention mask (mirrors forward() lines 1180-1185)
    epochs_fft, attn_mask_fft = model.transformer.get_fft(
        batch["epochs"][0], batch["mask"][0], aug=False
    )
    batch["epochs"] = (batch["epochs"][0], epochs_fft)
    attention_mask = model.get_attention_mask(
        batch["mask"][0], attn_mask_fft, manual=False
    )
    batch["mask"] = attention_mask

    return batch


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_on_split(
    model: Model,
    dataloader,
    device: str,
) -> List[dict]:
    """Run model inference on one data split.

    Returns a list of per-window records with keys:
      subject_id, window_idx, true_label, pred_label
    """
    records: List[dict] = []
    subject_window_counter: Dict[int, int] = defaultdict(int)

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        batch = _preprocess_batch(batch, model)

        # Forward through transformer; cls_feats: [B, num_classes]
        output = model.infer(batch, time_mask=False, stage="test")
        logits = output["cls_feats"]
        pred_labels = torch.argmax(logits, dim=-1).cpu()  # [B]

        # True labels: same pathology for all epochs in a window → take first
        if model.time_size != 1:
            true_labels = batch["Pathology_label"][:, 0].cpu()  # [B]
        else:
            true_labels = batch["Pathology_label"].cpu()  # [B]

        # Subject IDs are stored as a list of scalar tensors by the collate fn
        names = batch["name"]

        for i in range(len(names)):
            raw_name = names[i]
            subject_id = int(raw_name.item()) if isinstance(raw_name, torch.Tensor) else int(raw_name)
            records.append(
                {
                    "subject_id": subject_id,
                    "window_idx": subject_window_counter[subject_id],
                    "true_label": int(true_labels[i].item()),
                    "pred_label": int(pred_labels[i].item()),
                }
            )
            subject_window_counter[subject_id] += 1

    return records


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_sequence_csvs(records: List[dict], split_dir: Path) -> None:
    """Save one CSV per subject under split_dir/{subject_id}.csv.

    Columns: subject_id, window_idx, true_label, pred_label
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    by_subject: Dict[int, List[dict]] = defaultdict(list)
    for rec in records:
        by_subject[rec["subject_id"]].append(rec)

    for subject_id, rows in sorted(by_subject.items()):
        csv_path = split_dir / f"{subject_id}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["subject_id", "window_idx", "true_label", "pred_label"]
            )
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> dict:
    """Compute classification metrics in VisualizeEvaluationResult format.

    All values are in [0, 1] (not percentages).
    """
    labels = list(range(num_classes))
    arr_true = np.array(y_true, dtype=int)
    arr_pred = np.array(y_pred, dtype=int)
    return {
        "num_samples": int(len(arr_true)),
        "accuracy": float(accuracy_score(arr_true, arr_pred)),
        "precision": float(
            precision_score(arr_true, arr_pred, average="macro", labels=labels, zero_division=0)
        ),
        "recall": float(
            recall_score(arr_true, arr_pred, average="macro", labels=labels, zero_division=0)
        ),
        "f1_score": float(
            f1_score(arr_true, arr_pred, average="macro", labels=labels, zero_division=0)
        ),
        "confusion_matrix": sk_confusion_matrix(arr_true, arr_pred, labels=labels).tolist(),
        "precision_per_class": precision_score(
            arr_true, arr_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
        "recall_per_class": recall_score(
            arr_true, arr_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
        "f1_score_per_class": f1_score(
            arr_true, arr_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
    }


def build_split_entry(records: List[dict], num_classes: int) -> dict:
    """Build the {'overall': ..., 'subjects': ...} entry for one split."""
    if not records:
        return {"overall": {}, "subjects": {}}

    y_true_all = [r["true_label"] for r in records]
    y_pred_all = [r["pred_label"] for r in records]
    overall = compute_metrics(y_true_all, y_pred_all, num_classes)

    # Per-subject: aggregate all windows for that subject
    by_subject: Dict[int, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for rec in records:
        sid = rec["subject_id"]
        by_subject[sid][0].append(rec["true_label"])
        by_subject[sid][1].append(rec["pred_label"])

    subjects = {
        str(sid): compute_metrics(trues, preds, num_classes)
        for sid, (trues, preds) in sorted(by_subject.items())
    }

    return {"overall": overall, "subjects": subjects}


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_metrics_json(
    fold_results: Dict[str, List[dict]],
    output_path: Path,
    model_name: str,
    fold: int,
    num_classes: int,
) -> None:
    """Save VisualizeEvaluationResult-compatible label_sequence_jmetrics.json.

    Structure:
        {
          "model_name": ...,
          "fold": ...,
          "num_classes": ...,
          "splits": {
            "train": {"overall": {...}, "subjects": {...}},
            "val":   {"overall": {...}, "subjects": {...}},
            "test":  {"overall": {...}, "subjects": {...}}
          }
        }
    """
    splits = {
        split_name: build_split_entry(records, num_classes)
        for split_name, records in fold_results.items()
        if records
    }
    payload = {
        "model_name": model_name,
        "fold": fold,
        "num_classes": num_classes,
        "splits": splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[fold {fold}] Metrics JSON saved → {output_path}")


# ---------------------------------------------------------------------------
# Per-fold orchestration
# ---------------------------------------------------------------------------

def run_fold(
    fold_idx: int,
    config: dict,
    checkpoint_path: str,
    output_dir: Path,
    device: str,
) -> None:
    """Load model and run inference on train/val/test for one fold."""
    print(f"\n{'='*60}")
    print(f" Fold {fold_idx}  |  checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    fold_dir = output_dir / f"fold_{fold_idx}"

    # Build per-fold config (kfold controls which split indices are used)
    cfg = dict(config)
    cfg["kfold"] = fold_idx

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading model from {checkpoint_path} …")
    model = load_model(cfg, checkpoint_path, device)
    model.set_task()  # populates model.current_tasks from loss_names

    # ── Build dataloaders ─────────────────────────────────────────────────────
    # Two separate MultiDataModule instances are required because the underlying
    # CAPDataModule uses a setup_flag that prevents calling setup() twice.
    print("  Setting up dataloaders …")
    dm_fit = MultiDataModule(cfg, kfold=fold_idx)
    dm_fit.setup("fit")
    train_loader = dm_fit.train_dataloader()
    val_loader = dm_fit.val_dataloader()

    dm_test = MultiDataModule(cfg, kfold=fold_idx)
    dm_test.setup("test")
    test_loader = dm_test.test_dataloader()

    # ── Run inference on all splits ───────────────────────────────────────────
    split_records: Dict[str, List[dict]] = {}
    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        print(f"  → {split_name} split …", flush=True)
        records = run_inference_on_split(model, loader, device)
        split_records[split_name] = records

        # Save per-subject CSVs
        seq_dir = fold_dir / "sequences" / split_name
        save_sequence_csvs(records, seq_dir)

        n_windows = len(records)
        n_subjects = len({r["subject_id"] for r in records})
        print(f"     {n_windows} windows, {n_subjects} subjects  →  {seq_dir}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    json_path = fold_dir / "label_sequence_jmetrics.json"
    model_name = cfg.get("model_arch", "backbone_large_patch200")
    save_metrics_json(split_records, json_path, model_name, fold_idx, NUM_CLASSES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_checkpoints(args: argparse.Namespace, num_folds: int) -> List[str]:
    """Return one checkpoint path per fold."""
    if len(args.checkpoint) == 1:
        return [args.checkpoint[0]] * num_folds
    if len(args.checkpoint) == num_folds:
        return args.checkpoint
    raise ValueError(
        f"--checkpoint: expected 1 or {num_folds} paths, got {len(args.checkpoint)}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only evaluation of a trained SleepGPT model on CAP pathology data.\n"
            "Saves per-window sequence CSVs and a VisualizeEvaluationResult-compatible "
            "label_sequence_jmetrics.json for each fold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "Checkpoint path(s). Supply one path to reuse it for every fold, or supply "
            "one path per fold (space-separated, ordered fold 0, 1, …)."
        ),
    )
    parser.add_argument(
        "--cap-root",
        default="/mnt/e/DataSet/Local/OpenData/capslpdb/process",
        help=(
            "Root directory that contains the CAP pathology subfolders "
            "(n, ins, narco, nfle, plm, rbd, sdb). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./inference_results",
        help="Directory to write outputs (CSVs and JSON). Default: %(default)s",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=4,
        help="Number of folds (must match the training kfold setting). Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference. Default: %(default)s",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers. Default: %(default)s",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when a GPU is available.",
    )

    args = parser.parse_args()

    num_folds = args.kfold
    checkpoints = resolve_checkpoints(args, num_folds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    device = config["device"]

    print(f"SleepGPT CAP Pathology Inference")
    print(f"  Device     : {device}")
    print(f"  Folds      : {num_folds}")
    print(f"  CAP root   : {args.cap_root}")
    print(f"  Output dir : {output_dir}")
    print(f"  Checkpoints: {checkpoints}")

    for fold_idx, ckpt_path in enumerate(checkpoints):
        run_fold(fold_idx, config, ckpt_path, output_dir, device)

    print("\n[done] All folds complete.")


if __name__ == "__main__":
    main()
