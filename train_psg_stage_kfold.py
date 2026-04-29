"""Fine-tune SleepGPT for sleep stage classification on custom PSG HDF5 data.

Expected preprocessing output:
  processed_root/
    manifest.json
    psg_split.npy
    <subject_id>/
      data.h5        # signal: [N, 8, 3000], stage: [N], good_channels: [8]
      meta.json

Each fold uses a rotating split dictionary with keys train_k / val_k / test_k.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader

from main.datasets.PSG_dataset import PSGDataset
from main.modules import Model
from main.modules.get_optm import param_groups_lrd, param_groups_no_layer_decay

_ARCH_DEFAULTS: dict = {
    "backbone_base_patch200": {
        "decoder_features": 512,
        "decoder_heads": 16,
        "decoder_depth": 4,
        "grad_name": "partial_6",
        "longnet_dr": [1, 2, 4, 8],
        "longnet_sl": [32, 64, 128, 512],
    },
    "backbone_large_patch200": {
        "decoder_features": 256,
        "decoder_heads":    32,
        "decoder_depth":    4,
        "grad_name":        "partial_10",
        "longnet_dr":       [1, 2, 4],
        "longnet_sl":       [32, 64, 128],
    },
}


def build_model(config: dict, fold: int, load_path: str | None = None) -> Model:
    cfg = dict(config)
    if load_path is not None:
        cfg["load_path"] = load_path
    model = Model(cfg, fold_now=fold, num_classes=cfg["num_classes"])
    model.current_tasks = ["CrossEntropy"]
    return model.to(cfg["device"])


def build_config(args: argparse.Namespace) -> dict:
    use_cuda = torch.cuda.is_available() and not args.cpu
    ad = _ARCH_DEFAULTS[args.model_arch]

    decoder_features = (
        args.decoder_features
        if args.decoder_features is not None
        else ad["decoder_features"]
    )
    decoder_heads = (
        args.decoder_heads if args.decoder_heads is not None else ad["decoder_heads"]
    )
    decoder_depth = (
        args.decoder_depth if args.decoder_depth is not None else ad["decoder_depth"]
    )
    grad_name = args.grad_name if args.grad_name is not None else ad["grad_name"]
    longnet_dr = ad["longnet_dr"]
    longnet_sl = ad["longnet_sl"]

    return {
        "extra_name": "Finetune_psg_stage",
        "exp_name": "sleep",
        "seed": args.seed,
        "random_seed": [args.seed],
        "precision": "16-mixed",
        "mode": "Finetune_psg_stage",
        "kfold": None,
        "batch_size": args.batch_size,
        "max_epoch": args.max_epoch,
        "max_steps": -1,
        "accum_iter": 1,
        "start_epoch": 0,
        "datasets": ["PSG"],
        "data_dir": [args.processed_root],
        "data_setting": {"PSG": None},
        "dropout": 0.0,
        "loss_names": {
            "Spindle": 0,
            "CrossEntropy": 1,
            "mtm": 0,
            "itc": 0,
            "itm": 0,
            "Apnea": 0,
            "Pathology": 0,
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
        "output_dir": str(args.output_dir),
        "log_dir": str(args.output_dir),
        "load_path": args.pretrain_ckpt,
        "kfold_load_path": "",
        "resume_ckpt_path": "",
        "lr_policy": "cosine",
        "optim": "adamw",
        "clip_grad": False,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_lr": args.warmup_lr,
        "layer_decay": args.layer_decay,
        "get_param_method": "layer_decay",
        "Lambda": 1.0,
        "poly": False,
        "gradient_clip_val": 1.0,
        "device": "cuda" if use_cuda else "cpu",
        "deepspeed": False,
        "dist_on_itp": False,
        "num_gpus": -1,
        "num_nodes": -1,
        "dist_eval": False,
        "eval": False,
        "get_recall_metric": False,
        "limit_val_batches": 1.0,
        "limit_train_batches": 1.0,
        "val_check_interval": 1000,
        "check_val_every_n_epoch": None,
        "fast_dev_run": 7,
        "model_arch": args.model_arch,
        "epoch_duration": 30,
        "fs": 100,
        "mask_ratio": None,
        "max_time_len": 1,
        "random_choose_channels": 8,
        "actual_channels": None,
        "time_only": False,
        "fft_only": False,
        "loss_function": "l1",
        "resume_during_training": None,
        "use_triton": False,
        "use_relative_pos_emb": False,
        "use_global_fft": True,
        "use_multiway": False,
        "use_g_mid": False,
        "local_pooling": False,
        "multi_y": ["tf"],
        "num_encoder_layers": 4,
        "use_cb": True,
        "all_time": True,
        "time_size": 30,
        "split_len": 30,
        "use_all_label": "all",
        "use_pooling": "longnet",
        "pool": None,
        "decoder_features": decoder_features,
        "decoder_depth": decoder_depth,
        "decoder_heads": decoder_heads,
        "decoder_selected_layers": "2-3",
        "longnet_dr": longnet_dr,
        "longnet_sl": longnet_sl,
        "longnet_pool": False,
        "Swin_window_size": 60,
        "Use_FPN": None,
        "FPN_resnet": False,
        "num_queries": 400,
        "Event_decoder_depth": 4,
        "Event_enc_dim": 384,
        "mass_aug_times": 0,
        "expert": None,
        "IOU_th": 0.2,
        "sp_prob": 0.55,
        "patch_time": 30,
        "use_fpfn": None,
        "CE_Weight": 10,
        "aug_test": None,
        "EDF_Mode": None,
        "subset": None,
        "aug_dir": None,
        "aug_prob": 0.0,
        "kfold_test": None,
        "grad_name": grad_name,
        "save_top_k": 2,
        "show_transform_param": False,
        "mask_strategies": None,
        "visual": False,
        "visual_setting": {"mask_same": False, "mode": None, "save_extra_name": None},
        "persub": None,
        "return_alpha": False,
        "num_classes": 5,
        "stage1_epoch": (5,),
        "stage2_epoch": 10,
        "freeze_stage": False,
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


def make_loaders(processed_root: str, split_file: str, kfold: int, config: dict):
    dataset_kwargs = dict(
        transform_keys=config["transform_keys"],
        column_names=["signal", "stage", "good_channels"],
        stage=True,
        pathology=False,
        spindle=False,
        random_choose_channels=config["random_choose_channels"],
        mask_ratio=config["mask_ratio"],
        all_time=config["all_time"],
        time_size=config["time_size"],
        pool_all=True,
        split_len=config["split_len"],
        patch_size=config["patch_size"],
        settings=None,
        kfold=kfold,
        file_name=split_file,
    )

    train_ds = PSGDataset(split="train", data_dir=processed_root, **dataset_kwargs)
    val_ds = PSGDataset(split="val", data_dir=processed_root, **dataset_kwargs)
    test_ds = PSGDataset(split="test", data_dir=processed_root, **dataset_kwargs)
    collate_fn = train_ds.collate

    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )
    return train_dl, val_dl, test_dl


def _move_batch_to_device(batch: dict, device: str) -> dict:
    result: dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, list):
            result[key] = [
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def preprocess_batch(batch: dict, model: Model) -> dict:
    if "Stage_label" in batch:
        batch["Stage_label"] = torch.stack(batch["Stage_label"], dim=0).squeeze(-1)

    epochs_fft, attn_mask_fft = model.transformer.get_fft(
        batch["epochs"][0], batch["mask"][0], aug=False
    )
    batch["epochs"] = (batch["epochs"][0], epochs_fft)
    batch["mask"] = model.get_attention_mask(
        batch["mask"][0], attn_mask_fft, manual=False
    )
    return batch


def freeze_model(model: Model, grad_name_prefix: str = "partial_10") -> None:
    for param in model.parameters():
        param.requires_grad = False

    trainable_keys = [
        "fc_norm",
        "transformer.norm",
        "pooler",
        "decoder_transformer_block",
        "stage_pred",
        "head",
    ]
    if grad_name_prefix.startswith("partial"):
        start_block = int(grad_name_prefix.split("_")[-1])
        num_blocks = len(model.transformer.blocks)
        for blk_idx in range(start_block, num_blocks):
            trainable_keys.append(f"transformer.blocks.{blk_idx}")

    for name, param in model.named_parameters():
        if any(key in name for key in trainable_keys) and "pe" not in name:
            param.requires_grad = True


@torch.no_grad()
def evaluate_model(
    model: Model, data_loader: DataLoader, device: str
) -> dict[str, float | int]:
    model.eval()
    model.current_tasks = ["CrossEntropy"]
    all_preds, all_labels = [], []

    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)
        batch = preprocess_batch(batch, model)

        target = batch["Stage_label"].reshape(-1).long()
        out = model.infer(batch, time_mask=False, stage="eval")
        logits = out["cls_feats"]["tf"]
        preds = logits.argmax(dim=-1)

        valid = target != -100
        all_preds.extend(preds[valid].cpu().numpy())
        all_labels.extend(target[valid].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "num_samples": int(len(all_labels)),
    }


def train_fold(
    config: dict,
    processed_root: str,
    split_file: str,
    fold: int,
    output_dir: Path,
) -> dict[str, float | int]:
    device = config["device"]
    use_amp = device == "cuda"
    model = build_model(config, fold)
    freeze_model(model, grad_name_prefix=config["grad_name"])

    if config["get_param_method"] == "layer_decay":
        param_groups = param_groups_lrd(
            model,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            layer_decay=config["layer_decay"],
        )
    else:
        param_groups = param_groups_no_layer_decay(
            model,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    optimizer = torch.optim.AdamW(
        param_groups, lr=config["lr"], eps=1e-8, betas=(0.9, 0.999)
    )
    train_dl, val_dl, test_dl = make_loaders(processed_root, split_file, fold, config)
    if len(train_dl) == 0:
        raise ValueError(
            "Training DataLoader is empty. Increase subject count or reduce --time-size / --batch-size."
        )

    max_steps = len(train_dl) * config["max_epoch"]
    warmup_cfg = config["warmup_steps"]
    warmup_steps = (
        int(max_steps * warmup_cfg)
        if isinstance(warmup_cfg, float)
        else int(warmup_cfg)
    )
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
    best_val_acc = -1.0
    best_metrics: dict[str, float | int] | None = None
    step = 0
    # logger_path = os.path.join(config["log_dir"], "finetune")
    # os.makedirs(logger_path, exist_ok=True)
    # logger = pl.loggers.TensorBoardLogger(
    #     logger_path,
    #     name=f'fold_{fold}',
    # )
    
    # trainer = pl.Trainer(
    #     profiler="simple",
    #     devices=config["num_gpus"],
    #     precision=config["precision"],
    #     accelerator=config["device"],
    #     strategy="auto",
    #     deterministic=True,
    #     max_epochs=config["max_epoch"],
    #     max_steps=max_steps,
    #     # callbacks=callbacks,
    #     logger=logger,
    #     accumulate_grad_batches= config['accum_iter'],
    #     log_every_n_steps=1,
    #     val_check_interval=config["val_check_interval"],
    #     limit_val_batches=config['limit_val_batches'],
    #     gradient_clip_val=config['gradient_clip_val']
    # )
    
    # flag = True
    # if config['resume_during_training'] is not None and fold == int(config['resume_during_training']):
    #     if config["resume_ckpt_path"] != "":
    #         flag = False
    #         trainer.fit(model, datamodule=dm,
    #                     ckpt_path=config['resume_ckpt_path'])
    # if flag is True:
    #     trainer.fit(model, datamodule=dm,)

    for epoch in range(config["max_epoch"]):
        model.train()
        model.current_tasks = ["CrossEntropy"]
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            batch = _move_batch_to_device(batch, device)
            batch = preprocess_batch(batch, model)
            target = batch["Stage_label"].reshape(-1).long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model.infer(batch, time_mask=False, stage="train")
                logits = out["cls_feats"]["tf"]
                loss = F.cross_entropy(logits, target, ignore_index=-100)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(step)
            step += 1
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_model(model, val_dl, device)
        print(
            f"Fold {fold}  Epoch {epoch:02d}  loss={avg_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if float(val_metrics["accuracy"]) > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "fold": fold,
                    "val_metrics": val_metrics,
                },
                ckpt_dir / "best.ckpt",
            )
            best_metrics = val_metrics

        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch, "fold": fold},
            ckpt_dir / "last.ckpt",
        )

    checkpoint = torch.load(ckpt_dir / "best.ckpt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    test_metrics = evaluate_model(model, test_dl, device)

    fold_metrics = {
        "fold": fold,
        "best_val_accuracy": float(
            best_metrics["accuracy"] if best_metrics is not None else 0.0
        ),
        "best_val_macro_f1": float(
            best_metrics["macro_f1"] if best_metrics is not None else 0.0
        ),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_kappa": float(test_metrics["kappa"]),
        "test_num_samples": int(test_metrics["num_samples"]),
    }
    (ckpt_dir / "metrics.json").write_text(
        json.dumps(fold_metrics, indent=2), encoding="utf-8"
    )
    print(
        f"Fold {fold} complete. test_acc={fold_metrics['test_accuracy']:.4f}  "
        f"test_macro_f1={fold_metrics['test_macro_f1']:.4f}"
    )
    return fold_metrics


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[str]:
    if args.checkpoint:
        checkpoint_paths = []
        for fold in range(args.kfold):
            checkpoint_path = Path(args.checkpoint[0]) / f"fold_{fold}" / "best.ckpt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint path for fold {fold} does not exist: {checkpoint_path}"
                )
            checkpoint_paths.append(checkpoint_path)
        return checkpoint_paths
            
        # if len(args.checkpoint) == 1:
        #     return [args.checkpoint[0]] * args.kfold
        # if len(args.checkpoint) == args.kfold:
        #     return args.checkpoint
        # raise ValueError(
        #     f"--checkpoint expects 1 or {args.kfold} paths, got {len(args.checkpoint)}"
        # )

    resolved = []
    output_dir = Path(args.output_dir)
    for fold in range(args.kfold):
        checkpoint_path = output_dir / f"fold_{fold}" / "best.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Could not infer checkpoint for fold {fold}: {checkpoint_path}. "
                "Pass --checkpoint explicitly or run training first."
            )
        resolved.append(str(checkpoint_path))
    return resolved


def evaluate_fold_checkpoint(
    config: dict,
    processed_root: str,
    split_file: str,
    fold: int,
    checkpoint_path: str,
) -> dict[str, object]:
    device = config["device"]
    model = build_model(config, fold, load_path=checkpoint_path)
    _, val_dl, test_dl = make_loaders(processed_root, split_file, fold, config)

    val_metrics = evaluate_model(model, val_dl, device)
    test_metrics = evaluate_model(model, test_dl, device)
    return {
        "fold": fold,
        "checkpoint": checkpoint_path,
        "val": val_metrics,
        "test": test_metrics,
    }


def summarize_eval_only_metrics(
    fold_metrics: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {"folds": fold_metrics}
    for split_name in ("val", "test"):
        acc = np.array(
            [metric[split_name]["accuracy"] for metric in fold_metrics],
            dtype=np.float64,
        )
        macro_f1 = np.array(
            [metric[split_name]["macro_f1"] for metric in fold_metrics],
            dtype=np.float64,
        )
        kappa = np.array(
            [metric[split_name]["kappa"] for metric in fold_metrics], dtype=np.float64
        )
        summary[split_name] = {
            "mean_accuracy": float(acc.mean()),
            "std_accuracy": float(acc.std(ddof=0)),
            "mean_macro_f1": float(macro_f1.mean()),
            "std_macro_f1": float(macro_f1.std(ddof=0)),
            "mean_kappa": float(kappa.mean()),
            "std_kappa": float(kappa.std(ddof=0)),
        }
    return summary


def summarize_metrics(
    fold_metrics: list[dict[str, float | int]],
) -> dict[str, float | list[dict[str, float | int]]]:
    test_acc = np.array(
        [metric["test_accuracy"] for metric in fold_metrics], dtype=np.float64
    )
    test_macro_f1 = np.array(
        [metric["test_macro_f1"] for metric in fold_metrics], dtype=np.float64
    )
    test_kappa = np.array(
        [metric["test_kappa"] for metric in fold_metrics], dtype=np.float64
    )
    return {
        "folds": fold_metrics,
        "mean_test_accuracy": float(test_acc.mean()),
        "std_test_accuracy": float(test_acc.std(ddof=0)),
        "mean_test_macro_f1": float(test_macro_f1.mean()),
        "std_test_macro_f1": float(test_macro_f1.std(ddof=0)),
        "mean_test_kappa": float(test_kappa.mean()),
        "std_test_kappa": float(test_kappa.std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SleepGPT for 5-class sleep staging on custom PSG HDF5 data.",
    )
    parser.add_argument(
        "--processed-root",
        required=True,
        help="Root directory containing processed subject folders.",
    )
    parser.add_argument(
        "--split-file",
        default="psg_split.npy",
        help="Split .npy file produced by sleepgpt_preprocessing.py. Default: %(default)s",
    )
    parser.add_argument(
        "--pretrain-ckpt",
        default="",
        help="Pretrained backbone checkpoint loaded by Model.load_pretrained_weight().",
    )
    parser.add_argument("--output-dir", default="./psg_stage_checkpoints")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument("--time-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-lr", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--model-arch",
        default="backbone_large_patch200",
        choices=["backbone_base_patch200", "backbone_large_patch200"],
    )
    parser.add_argument("--decoder-features", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--grad-name", default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate existing fold checkpoints on val/test splits.",
    )
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=None,
        help=(
            "Checkpoint path(s) for --eval-only. Supply one path to reuse it for all folds, "
            "or one path per fold. If omitted, {output_dir}/fold_k/best.ckpt is used."
        ),
    )
    args = parser.parse_args()

    if not args.eval_only and args.pretrain_ckpt == "":
        parser.error("--pretrain-ckpt is required unless --eval-only is set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)

    print("SleepGPT Sleep Stage Fine-Tuning (custom PSG dataset)")
    print(f"  Device          : {config['device']}")
    print(f"  Processed root  : {args.processed_root}")
    print(f"  Split file      : {args.split_file}")
    print(
        f"  Init checkpoint : {args.pretrain_ckpt if args.pretrain_ckpt else '(not used)'}"
    )
    print(f"  Model arch      : {config['model_arch']}")
    print(f"  Time size       : {config['time_size']}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Max epochs      : {args.max_epoch}")

    if args.eval_only:
        checkpoint_paths = resolve_checkpoint_paths(args)
        eval_metrics: list[dict[str, object]] = []
        for fold, checkpoint_path in enumerate(checkpoint_paths):
            print(f"\n{'=' * 60}")
            print(f" Evaluate Fold {fold}")
            print(f"{'=' * 60}")
            metrics = evaluate_fold_checkpoint(
                config,
                args.processed_root,
                args.split_file,
                fold,
                checkpoint_path,
            )
            eval_metrics.append(metrics)
            fold_dir = output_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "evaluation_metrics.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
            print(
                f"Fold {fold} eval complete. "
                f"val_acc={metrics['val']['accuracy']:.4f}  test_acc={metrics['test']['accuracy']:.4f}"
            )

        summary = summarize_eval_only_metrics(eval_metrics)
        summary_path = output_dir / "cross_validation_eval_metrics.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nEvaluation complete.")
        print(f"  mean_val_accuracy  : {summary['val']['mean_accuracy']:.4f}")
        print(f"  mean_test_accuracy : {summary['test']['mean_accuracy']:.4f}")
        print(f"  mean_test_macro_f1 : {summary['test']['mean_macro_f1']:.4f}")
        print(f"  metrics saved to   : {summary_path}")
        return

    fold_metrics: list[dict[str, float | int]] = []
    for fold in range(args.kfold):
        print(f"\n{'=' * 60}")
        print(f" Fold {fold}")
        print(f"{'=' * 60}")
        fold_metrics.append(
            train_fold(config, args.processed_root, args.split_file, fold, output_dir)
        )

    summary = summarize_metrics(fold_metrics)
    summary_path = output_dir / "cross_validation_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nCross-validation complete.")
    print(f"  mean_test_accuracy : {summary['mean_test_accuracy']:.4f}")
    print(f"  mean_test_macro_f1 : {summary['mean_test_macro_f1']:.4f}")
    print(f"  mean_test_kappa    : {summary['mean_test_kappa']:.4f}")
    print(f"  metrics saved to   : {summary_path}")


if __name__ == "__main__":
    main()
