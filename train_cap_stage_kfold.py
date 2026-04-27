"""Fine-tune SleepGPT for sleep stage classification on the CAP dataset.

No Sacred (ex). Pure PyTorch training loop with timm CosineLRScheduler.
Loads a pretrained backbone checkpoint and fine-tunes the CrossEntropy
(5-class sleep stage) head using all seven CAP pathology subgroups combined.

Tensor shape flow (time_size=100, batch_size=B, num_patches P=15):
  Dataset item  : Stage_label (100, 1), epochs (100, 8, 3000)
  After collate : batch['epochs'][0] (B*100, 8, 3000)
                  batch['Stage_label'] list of B tensors (100, 1)
  After preprocess: batch['Stage_label'] (B, 100) via stack+squeeze
  Training target : batch['Stage_label'].reshape(-1) -> (B*100,)
  Model output    : logits (B*100, 5)  [per-epoch predictions via LongNet sleep_stage=True]
  Loss            : F.cross_entropy(logits, target, ignore_index=-100)

Usage:
    python train_cap_stage_kfold.py \\
        --cap-root /data/capslpdb/process \\
        --pretrain-ckpt /path/to/pretrained.ckpt \\
        --output-dir ./stage_checkpoints
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import ConcatDataset, DataLoader

from main.datasets.CAP_Pathology_dataset import (
    CAPDataset_ins,
    CAPDataset_n,
    CAPDataset_narco,
    CAPDataset_nfle,
    CAPDataset_plm,
    CAPDataset_rbd,
    CAPDataset_sdb,
)
from main.modules import Model
from main.modules.get_optm import param_groups_lrd, param_groups_no_layer_decay

CAP_DATASET_CLASSES = {
    "n":     CAPDataset_n,
    "ins":   CAPDataset_ins,
    "narco": CAPDataset_narco,
    "nfle":  CAPDataset_nfle,
    "plm":   CAPDataset_plm,
    "rbd":   CAPDataset_rbd,
    "sdb":   CAPDataset_sdb,
}

# ---------------------------------------------------------------------------
# Architecture-specific default parameters
# ---------------------------------------------------------------------------
# These values are automatically applied when --model-arch is specified.
# Any parameter explicitly passed via CLI overrides the arch default.
#
# decoder_heads is passed as dim_head to LongNetTransformer, so
#   num_attention_heads = decoder_features / decoder_heads
# must be an integer.  Current choices give 32 heads for both archs.
#
# longnet_dr and longnet_sl must have equal length (zipped in DilatedAttention).
# base uses 4 dilation levels to reduce decoder compute; large uses 5.
#
# grad_name controls partial backbone unfreezing in freeze_model():
#   "partial_N" unfreezes blocks N … depth-1.
#   base (depth=8):  partial_6 → last 2 blocks (25 %)
#   large (depth=12): partial_10 → last 2 blocks (17 %)
_ARCH_DEFAULTS: dict = {
    "backbone_base_patch200": {
        "decoder_features": 512,
        "decoder_heads":    16,
        "decoder_depth":    4,
        "grad_name":        "partial_6",
        "longnet_dr":       [1, 2, 4, 8],
        "longnet_sl":       [32, 64, 128, 512],
    },
    "backbone_large_patch200": {
        "decoder_features": 1024,
        "decoder_heads":    32,
        "decoder_depth":    4,
        "grad_name":        "partial_10",
        "longnet_dr":       [1, 2, 4, 8, 16],
        "longnet_sl":       [32, 64, 128, 512, 1024],
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> dict:
    """Build full config dict for CrossEntropy sleep staging on CAP data.

    Based on run_inference_cap_pathology.py::build_config() with modifications
    for fine-tuning (training) instead of inference.

    Architecture-dependent parameters (decoder size, grad_name, longnet_dr/sl)
    are resolved from _ARCH_DEFAULTS[args.model_arch] unless explicitly
    overridden by the corresponding CLI flag.
    """
    use_cuda = torch.cuda.is_available() and not args.cpu
    ad = _ARCH_DEFAULTS[args.model_arch]

    # Resolve arch-dependent params: explicit CLI value wins over arch default.
    decoder_features = args.decoder_features if args.decoder_features is not None else ad["decoder_features"]
    decoder_heads    = args.decoder_heads    if args.decoder_heads    is not None else ad["decoder_heads"]
    decoder_depth    = args.decoder_depth    if args.decoder_depth    is not None else ad["decoder_depth"]
    grad_name        = args.grad_name        if args.grad_name        is not None else ad["grad_name"]
    longnet_dr       = ad["longnet_dr"]
    longnet_sl       = ad["longnet_sl"]

    return {
        # ── Identity / experiment ─────────────────────────────────────────
        "extra_name": "Finetune_cap_stage",
        "exp_name": "sleep",
        "seed": 3407,
        "random_seed": [3407],
        "precision": "16-mixed",
        "mode": "Finetune_cap_all",   # Required for CAPDataModule compatibility
        "kfold": None,                 # Set per-fold at runtime

        # ── Batch / training schedule ──────────────────────────────────────
        "batch_size": args.batch_size,
        "max_epoch": args.max_epoch,
        "max_steps": -1,
        "accum_iter": 1,
        "start_epoch": 0,

        # ── Data ──────────────────────────────────────────────────────────
        "datasets": [f"CAP_{p}" for p in CAP_DATASET_CLASSES],
        "data_dir": [os.path.join(args.cap_root, p) for p in CAP_DATASET_CLASSES],
        "data_setting": {"CAP": None},

        # ── Loss / task (CrossEntropy=1 → sleep stage classification) ─────
        "dropout": 0.0,
        "loss_names": {
            "Spindle":      0,
            "CrossEntropy": 1,
            "mtm":          0,
            "itc":          0,
            "itm":          0,
            "Apnea":        0,
            "Pathology":    0,
        },
        "transform_keys": {"keys": [[0, 1, 2, 3, 4, 6]], "mode": ["shuffle"]},
        "num_workers": args.num_workers,
        "drop_path_rate": 0.1,
        "patch_size": 200,
        "lr_mult": 20,
        "blr": 1.5e-5,
        "end_lr": 0,
        "warmup_steps": args.warmup_steps,
        "smoothing": 0.1,
        "mixup": 0,

        # ── Directories ────────────────────────────────────────────────────
        "output_dir": str(args.output_dir),
        "log_dir":    str(args.output_dir),
        "load_path":  args.pretrain_ckpt,   # Loaded by load_pretrained_weight()
        "kfold_load_path": "",
        "resume_ckpt_path": "",

        # ── Optimizer ──────────────────────────────────────────────────────
        "lr_policy":        "cosine",
        "optim":            "adamw",
        "clip_grad":        False,
        "weight_decay":     args.weight_decay,
        "lr":               args.lr,
        "min_lr":           args.min_lr,
        "warmup_lr":        args.warmup_lr,
        "layer_decay":      args.layer_decay,
        "get_param_method": "layer_decay",
        "Lambda":           1.0,
        "poly":             False,
        "gradient_clip_val": 1.0,

        # ── Device ─────────────────────────────────────────────────────────
        "device":      "cuda" if use_cuda else "cpu",
        "deepspeed":   False,
        "dist_on_itp": False,
        "num_gpus":    -1,
        "num_nodes":   -1,

        # ── Evaluation flags ───────────────────────────────────────────────
        "dist_eval":             False,
        "eval":                  False,
        "get_recall_metric":     False,
        "limit_val_batches":     1.0,
        "limit_train_batches":   1.0,
        "val_check_interval":    1000,
        "check_val_every_n_epoch": None,
        "fast_dev_run":          7,

        # ── Architecture ───────────────────────────────────────────────────
        "model_arch":          args.model_arch,
        "epoch_duration":      30,
        "fs":                  100,
        "mask_ratio":          None,
        "max_time_len":        1,
        "random_choose_channels": 8,
        "actual_channels":     None,
        "time_only":           False,
        "fft_only":            False,
        "loss_function":       "l1",
        "resume_during_training": None,
        "use_triton":          False,
        "use_relative_pos_emb": False,
        "use_global_fft":      True,
        "use_multiway":        False,
        "use_g_mid":           False,
        "local_pooling":       False,
        "multi_y":             ["tf"],
        "num_encoder_layers":  4,
        "use_cb":              True,

        # ── Time-series / sliding window ────────────────────────────────────
        "all_time":      True,
        "time_size":     100,
        "split_len":     100,
        "use_all_label": "all",

        # ── Decoder / pooler ────────────────────────────────────────────────
        "use_pooling":            "longnet",
        "pool":                   None,
        "decoder_features":       decoder_features,
        "decoder_depth":          decoder_depth,
        "decoder_heads":          decoder_heads,
        "decoder_selected_layers": "2-3",
        "longnet_dr":             longnet_dr,
        "longnet_sl":             longnet_sl,
        "longnet_pool":           False,

        # ── Swin / FPN (unused for longnet, required by Model.__init__) ────
        "Swin_window_size":   60,
        "Use_FPN":            None,
        "FPN_resnet":         False,
        "num_queries":        400,
        "Event_decoder_depth": 4,
        "Event_enc_dim":      384,

        # ── Spindle / Apnea (inactive) ──────────────────────────────────────
        "mass_aug_times": 0,
        "expert":         None,
        "IOU_th":         0.2,
        "sp_prob":        0.55,
        "patch_time":     30,
        "use_fpfn":       None,
        "CE_Weight":      10,

        # ── Misc ────────────────────────────────────────────────────────────
        "aug_test":           None,
        "EDF_Mode":           None,
        "subset":             None,
        "aug_dir":            None,
        "aug_prob":           0.0,
        "kfold_test":         None,
        "grad_name":          grad_name,
        "save_top_k":         2,
        "show_transform_param": False,
        "mask_strategies":    None,

        # ── Visualization (disabled) ──────────────────────────────────────
        "visual": False,
        "visual_setting": {"mask_same": False, "mode": None, "save_extra_name": None},
        "persub":         None,
        "return_alpha":   False,

        # ── Classification ────────────────────────────────────────────────
        "num_classes":   5,
        "stage1_epoch":  (5,),
        "stage2_epoch":  10,
        "freeze_stage":  False,

        # ── SpO2 / ODS (inactive) ─────────────────────────────────────────
        "spo2_ods_settings": {
            "inj":       False,
            "d_spo2":    128,
            "xattn_layers": [12],
            "hybrid_loss": True,
            "ods_pos_w": 10,
            "model_type": "lstm",
            "use_seq":   False,
            "concat":    False,
        },
    }


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

def make_loaders(cap_root: str, kfold: int, config: dict):
    """Create train and validation DataLoaders from all CAP pathology groups."""
    dataset_kwargs = dict(
        transform_keys=config["transform_keys"],
        column_names=["signal", "stage", "good_channels", "pathology"],
        stage=True,
        pathology=False,
        spindle=False,
        random_choose_channels=config["random_choose_channels"],
        mask_ratio=config["mask_ratio"],
        all_time=config["all_time"],
        time_size=config["time_size"],
        pool_all=True,        # Non-overlapping windows; set directly
        split_len=config["split_len"],
        patch_size=config["patch_size"],
        settings=None,
        kfold=kfold,
    )

    train_ds_list, val_ds_list = [], []
    for name, cls in CAP_DATASET_CLASSES.items():
        data_dir = os.path.join(cap_root, name)
        train_ds_list.append(cls(split="train", data_dir=data_dir, **dataset_kwargs))
        val_ds_list.append(  cls(split="val",   data_dir=data_dir, **dataset_kwargs))

    # All datasets share an identical collate implementation
    collate_fn = train_ds_list[0].collate

    train_dl = DataLoader(
        ConcatDataset(train_ds_list),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_dl = DataLoader(
        ConcatDataset(val_ds_list),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

def _move_batch_to_device(batch: dict, device: str) -> dict:
    result: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, list):
            result[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        else:
            result[k] = v
    return result


_STAGE_IGNORE_INDEX = -100
_NUM_STAGE_CLASSES  = 5  # W, N1, N2, N3, REM


def _sanitize_stage_labels(labels: torch.Tensor) -> torch.Tensor:
    """Remap out-of-range stage labels (e.g. MT=7) to the cross-entropy ignore_index.

    The CAP dataset stores Movement Time epochs as label 7.  PyTorch's
    cross_entropy CUDA kernel raises a device-side assert when any target value
    is outside [0, num_classes) and is not the ignore_index.  This function
    remaps such values to _STAGE_IGNORE_INDEX (-100) so they are skipped during
    loss computation, matching the semantics of already-masked (-100) entries.
    """
    valid   = (labels >= 0) & (labels < _NUM_STAGE_CLASSES)
    ignored = labels == _STAGE_IGNORE_INDEX
    return labels.masked_fill(~(valid | ignored), _STAGE_IGNORE_INDEX)


def preprocess_batch(batch: dict, model: Model) -> dict:
    """Replicate Model.forward() preprocessing so infer() can be called directly.

    Mirrors run_inference_cap_pathology.py::_preprocess_batch().
    Stack label lists → tensors, compute FFT, build attention mask.
    """
    if "Stage_label" in batch:
        # list of B tensors (T, 1) → (B, T, 1) → (B, T)
        batch["Stage_label"] = _sanitize_stage_labels(
            torch.stack(batch["Stage_label"], dim=0).squeeze(-1)
        )
    if "Pathology_label" in batch:
        batch["Pathology_label"] = (
            torch.stack(batch["Pathology_label"], dim=0).squeeze(1).squeeze(-1)
        )

    # Compute FFT branch and build full attention mask
    epochs_fft, attn_mask_fft = model.transformer.get_fft(
        batch["epochs"][0], batch["mask"][0], aug=False
    )
    batch["epochs"] = (batch["epochs"][0], epochs_fft)
    batch["mask"] = model.get_attention_mask(batch["mask"][0], attn_mask_fft, manual=False)
    return batch


# ---------------------------------------------------------------------------
# Parameter freezing
# ---------------------------------------------------------------------------

def freeze_model(model: Model, grad_name_prefix: str = "partial_10") -> None:
    """Freeze all parameters, then unfreeze downstream heads and top transformer blocks.

    Mirrors main_kfold.py lines 193-210, but adds 'head' (LongnetClassificationHead)
    which must be trainable for sleep stage classification.

    The upper bound for partial unfreezing uses the actual block count from the
    loaded transformer so that backbone_base_patch200 (8 blocks) and
    backbone_large_patch200 (12 blocks) both work correctly.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable_keys = [
        "fc_norm",
        "transformer.norm",
        "pooler",
        "decoder_transformer_block",
        "stage_pred",
        "head",   # LongnetClassificationHead
    ]
    if grad_name_prefix.startswith("partial"):
        start_block = int(grad_name_prefix.split("_")[-1])
        num_blocks = len(model.transformer.blocks)  # actual depth: 8 for base, 12 for large
        for blk_idx in range(start_block, num_blocks):
            trainable_keys.append(f"transformer.blocks.{blk_idx}")

    for name, param in model.named_parameters():
        if any(key in name for key in trainable_keys) and "pe" not in name:
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model: Model, val_dl: DataLoader, device: str) -> tuple:
    """Run validation and return (accuracy, macro_f1)."""
    model.eval()
    model.current_tasks = ["CrossEntropy"]
    all_preds, all_labels = [], []

    for batch in val_dl:
        batch = _move_batch_to_device(batch, device)
        batch = preprocess_batch(batch, model)

        target = batch["Stage_label"].reshape(-1).long()   # [B*T]
        out    = model.infer(batch, time_mask=False, stage="val")
        logits = out["cls_feats"]["tf"]                    # [B*T, num_classes]
        preds  = logits.argmax(dim=-1)                     # [B*T]

        valid = target != -100
        all_preds.extend(preds[valid].cpu().numpy())
        all_labels.extend(target[valid].cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1


# ---------------------------------------------------------------------------
# Training loop (one fold)
# ---------------------------------------------------------------------------

def train_fold(config: dict, cap_root: str, fold: int, output_dir: Path) -> None:
    device  = config["device"]
    use_amp = device == "cuda"

    # ── Model ──────────────────────────────────────────────────────────────
    # load_pretrained_weight() is called inside __init__ using config["load_path"]
    model = Model(config, fold_now=fold, num_classes=config["num_classes"])
    model.current_tasks = ["CrossEntropy"]
    model = model.to(device)

    # ── Freeze / unfreeze ──────────────────────────────────────────────────
    freeze_model(model, grad_name_prefix=config["grad_name"])

    # ── Optimizer with layer-wise LR decay ─────────────────────────────────
    if config["get_param_method"] == "layer_decay":
        param_groups = param_groups_lrd(
            model,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            layer_decay=config["layer_decay"],
        )
    else:
        param_groups = param_groups_no_layer_decay(
            model, lr=config["lr"], weight_decay=config["weight_decay"]
        )
    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"], eps=1e-8, betas=(0.9, 0.999))

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_dl, val_dl = make_loaders(cap_root, fold, config)

    # ── Cosine LR Scheduler (step-based, with linear warmup) ────────────────
    max_steps = len(train_dl) * config["max_epoch"]
    warmup_cfg = config["warmup_steps"]
    warmup_steps = int(max_steps * warmup_cfg) if isinstance(warmup_cfg, float) else int(warmup_cfg)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max_steps,
        lr_min=config["min_lr"],
        warmup_lr_init=config["warmup_lr"],
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=True,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = output_dir / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    step = 0

    for epoch in range(config["max_epoch"]):
        # ── Training epoch ──────────────────────────────────────────────────
        model.train()
        model.current_tasks = ["CrossEntropy"]
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_dl:
            batch = _move_batch_to_device(batch, device)
            batch = preprocess_batch(batch, model)

            # Labels: list of B (T, 1) → (B, T) → (B*T,)
            target = batch["Stage_label"].reshape(-1).long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out    = model.infer(batch, time_mask=False, stage="train")
                logits = out["cls_feats"]["tf"]          # [B*T, num_classes]
                loss   = F.cross_entropy(logits, target, ignore_index=-100)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(step)
            step      += 1
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Validation epoch ────────────────────────────────────────────────
        val_acc, val_macro_f1 = run_validation(model, val_dl, device)
        print(
            f"Fold {fold}  Epoch {epoch:02d}  "
            f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  macro_f1={val_macro_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict":   model.state_dict(),
                    "epoch":        epoch,
                    "fold":         fold,
                    "val_acc":      val_acc,
                    "val_macro_f1": val_macro_f1,
                },
                ckpt_dir / "best.ckpt",
            )
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch, "fold": fold},
            ckpt_dir / "last.ckpt",
        )

    print(f"Fold {fold} complete. Best val_acc={best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SleepGPT for sleep stage classification on CAP dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cap-root", required=True,
        help="Root dir containing CAP subfolders (n, ins, narco, nfle, plm, rbd, sdb).",
    )
    parser.add_argument(
        "--pretrain-ckpt", required=True,
        help="Pretrained backbone checkpoint (loaded via load_pretrained_weight).",
    )
    parser.add_argument(
        "--output-dir", default="./stage_checkpoints",
        help="Directory to write fold checkpoints. Default: %(default)s",
    )
    parser.add_argument("--kfold",        type=int,   default=4,    help="Number of k-folds. Default: %(default)s")
    parser.add_argument("--batch-size",   type=int,   default=8,    help="Training batch size. Default: %(default)s")
    parser.add_argument("--max-epoch",    type=int,   default=30,   help="Training epochs per fold. Default: %(default)s")
    parser.add_argument("--lr",           type=float, default=5e-4, help="Base learning rate. Default: %(default)s")
    parser.add_argument("--min-lr",       type=float, default=0.0,  help="Minimum LR for cosine decay. Default: %(default)s")
    parser.add_argument("--warmup-lr",    type=float, default=0.0,  help="Warmup start LR. Default: %(default)s")
    parser.add_argument("--warmup-steps", type=float, default=0.1,  help="Warmup fraction of total steps. Default: %(default)s")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay. Default: %(default)s")
    parser.add_argument("--layer-decay",  type=float, default=0.75, help="Layer-wise LR decay factor. Default: %(default)s")
    parser.add_argument("--num-workers",  type=int,   default=0,    help="DataLoader num_workers. Default: %(default)s")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when GPU is available.")

    # ── Architecture / decoder size (tune for memory) ─────────────────────────
    parser.add_argument(
        "--model-arch",
        default="backbone_large_patch200",
        choices=["backbone_base_patch200", "backbone_large_patch200"],
        help=(
            "Backbone architecture. "
            "'backbone_base_patch200' (embed_dim=384, depth=8) uses less memory but "
            "pre-trained weights trained on the large model will not transfer. "
            "'backbone_large_patch200' (embed_dim=768, depth=12) matches published "
            "pre-trained checkpoints but requires more GPU memory. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--decoder-features", type=int, default=None,
        help=(
            "LongNet decoder hidden dim. "
            "Default: 512 for backbone_base_patch200, 1024 for backbone_large_patch200."
        ),
    )
    parser.add_argument(
        "--decoder-depth", type=int, default=None,
        help="LongNet decoder transformer depth. Default: 4 for both archs.",
    )
    parser.add_argument(
        "--decoder-heads", type=int, default=None,
        help=(
            "LongNet decoder attention dim_head "
            "(num_heads = decoder_features / decoder_heads must be integer). "
            "Default: 16 for backbone_base_patch200, 32 for backbone_large_patch200."
        ),
    )
    parser.add_argument(
        "--grad-name", default=None,
        help=(
            "Partial backbone unfreeze spec, e.g. 'partial_6' unfreezes blocks 6+ . "
            "Default: 'partial_6' for backbone_base_patch200, 'partial_10' for backbone_large_patch200."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)

    print("SleepGPT Sleep Stage Fine-Tuning (CAP dataset)")
    print(f"  Device          : {config['device']}")
    print(f"  Model arch      : {config['model_arch']}")
    print(f"  Decoder         : features={config['decoder_features']}  depth={config['decoder_depth']}  dim_head={config['decoder_heads']}")
    print(f"  LongNet dr/sl   : {config['longnet_dr']} / {config['longnet_sl']}")
    print(f"  Grad name       : {config['grad_name']}")
    print(f"  Folds           : {args.kfold}")
    print(f"  CAP root        : {args.cap_root}")
    print(f"  Checkpoint      : {args.pretrain_ckpt}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Max epochs      : {args.max_epoch}")
    print(f"  LR              : {args.lr}")

    for fold in range(args.kfold):
        print(f"\n{'='*60}")
        print(f" Fold {fold}")
        print(f"{'='*60}")
        train_fold(config, args.cap_root, fold, output_dir)

    print("\n[done] All folds complete.")


if __name__ == "__main__":
    main()
