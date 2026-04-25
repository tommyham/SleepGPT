#!/usr/bin/env python3
"""
Run CAP fine-tuning or per-subject evaluation without srun/slurm.

This script reproduces the fold loop tasks that were previously submitted by srun,
by launching the same Sacred entrypoints with normal Python subprocesses.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import Iterable, List


PATHOLOGIES = ["n", "ins", "narco", "nfle", "plm", "rbd", "sdb"]


def parse_folds(folds_text: str) -> List[int]:
    folds_text = folds_text.strip()
    if "-" in folds_text:
        start, end = folds_text.split("-", 1)
        start_i = int(start)
        end_i = int(end)
        if start_i > end_i:
            raise ValueError(f"Invalid fold range: {folds_text}")
        return list(range(start_i, end_i + 1))

    folds = [int(x.strip()) for x in folds_text.split(",") if x.strip()]
    if not folds:
        raise ValueError("No folds were provided.")
    return folds


def cap_data_dirs(cap_root: str) -> List[str]:
    return [os.path.join(cap_root, p) for p in PATHOLOGIES]


def pick_checkpoint_for_fold(checkpoint_root: str, fold_idx_zero_based: int, prefer: str) -> str:
    pattern = os.path.join(checkpoint_root, "**", f"fold_{fold_idx_zero_based}", "**", "*.ckpt")
    candidates = glob.glob(pattern, recursive=True)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for fold_{fold_idx_zero_based} under: {checkpoint_root}"
        )

    # Prefer model checkpoints over last.ckpt when available.
    model_ckpts = [c for c in candidates if os.path.basename(c).startswith("ModelCheckpoint")]
    last_ckpts = [c for c in candidates if os.path.basename(c) == "last.ckpt"]

    if prefer == "best" and model_ckpts:
        pool = model_ckpts
    elif prefer == "last" and last_ckpts:
        pool = last_ckpts
    elif model_ckpts:
        pool = model_ckpts
    else:
        pool = candidates

    pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pool[0]


def run_command(cmd: List[str], dry_run: bool) -> int:
    printable = " ".join(cmd)
    print(f"[run] {printable}")
    if dry_run:
        return 0

    completed = subprocess.run(cmd)
    return int(completed.returncode)


def extend_with_extra(args: List[str], extra: Iterable[str]) -> List[str]:
    out = list(args)
    for item in extra:
        if item:
            out.append(item)
    return out


def build_test_cmd(ns: argparse.Namespace, fold_one_based: int, load_path: str, data_dirs: List[str]) -> List[str]:
    base = [
        ns.python,
        "main_test_kfold_persub.py",
        "with",
        "finetune_CAP",
        "CAP_datasets",
        f"num_gpus={ns.num_gpus}",
        "num_nodes=1",
        f"num_workers={ns.num_workers}",
        f"batch_size={ns.test_batch_size}",
        "model_arch=backbone_large_patch200",
        "lr_mult=20",
        "warmup_lr=0",
        "val_check_interval=0.5",
        "check_val_every_n_epoch=1",
        "limit_train_batches=1.0",
        "max_steps=-1",
        "all_time=True",
        "time_size=1",
        "decoder_features=768",
        "pool=None",
        "lr=1e-3",
        "min_lr=0",
        "random_choose_channels=8",
        "max_epoch=50",
        "lr_policy=cosine",
        "loss_function='l1'",
        "drop_path_rate=0.5",
        "warmup_steps=0.1",
        "split_len=1",
        "use_pooling='swin'",
        "use_relative_pos_emb=False",
        "mixup=0",
        "smoothing=0.1",
        "decoder_heads=16",
        "use_all_label='all'",
        "use_multiway='multiway'",
        "use_g_mid=False",
        "get_param_method='layer_decay'",
        "local_pooling=False",
        "optim='adamw'",
        "poly=False",
        "weight_decay=0.05",
        "layer_decay=0.75",
        "Lambda=1.0",
        "patch_size=200",
        "use_cb=True",
        f"kfold={fold_one_based}",
        "grad_name='all'",
        "resume_during_training=0",
        "resume_ckpt_path=''",
        "save_top_k=1",
        "eval=True",
        "dist_on_itp=False",
        f"load_path={repr(load_path)}",
        f"data_dir={repr(data_dirs)}",
    ]
    return extend_with_extra(base, ns.extra_arg)


def build_train_cmd(ns: argparse.Namespace, fold_one_based: int, pretrained_ckpt: str, data_dirs: List[str]) -> List[str]:
    # main_kfold.py loops range(start_idx, end_idx).
    # To train one fold per process, set:
    #   start_idx = fold_idx (resume_during_training)
    #   end_idx = fold_one_based (kfold)
    fold_idx_zero_based = fold_one_based - 1

    base = [
        ns.python,
        "main_kfold.py",
        "with",
        "finetune_CAP",
        "CAP_datasets",
        f"num_gpus={ns.num_gpus}",
        "num_nodes=1",
        f"num_workers={ns.num_workers}",
        f"batch_size={ns.train_batch_size}",
        "model_arch=backbone_large_patch200",
        "lr_mult=20",
        "warmup_lr=0",
        "val_check_interval=0.5",
        "check_val_every_n_epoch=1",
        "limit_train_batches=1.0",
        "max_steps=-1",
        "all_time=True",
        "time_size=100",
        "pool=None",
        "lr=5e-4",
        "min_lr=0",
        "random_choose_channels=8",
        "max_epoch=30",
        "lr_policy=cosine",
        "loss_function='l1'",
        "drop_path_rate=0.5",
        "warmup_steps=0.1",
        "split_len=100",
        f"load_path={repr(pretrained_ckpt)}",
        "mixup=0",
        "smoothing=0.1",
        "use_global_fft=True",
        "use_all_label='all'",
        "get_param_method='layer_decay'",
        "local_pooling=False",
        "optim='adamw'",
        "poly=False",
        "weight_decay=0.05",
        "layer_decay=0.75",
        "Lambda=1.0",
        "patch_size=200",
        "use_cb=True",
        "grad_name='partial_10'",
        f"resume_during_training={fold_idx_zero_based}",
        "resume_ckpt_path=''",
        f"kfold={fold_one_based}",
        "use_triton=False",
        "decoder_features=1024",
        "decoder_depth=4",
        "decoder_selected_layers='2-3'",
        "decoder_heads=32",
        "dist_on_itp=False",
        f"data_dir={repr(data_dirs)}",
    ]
    return extend_with_extra(base, ns.extra_arg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CAP loops without srun")
    parser.add_argument("mode", choices=["train", "test"], help="train: fine-tune loop, test: evaluation loop")
    parser.add_argument("--python", default="python3", help="Python executable")
    parser.add_argument("--folds", default="1-4", help="Fold list, e.g. 1-4 or 1,3,4")
    parser.add_argument("--cap-root", default="/mnt/e/DataSet/Local/OpenData/capslpdb/process",
                        help="Root directory that contains pathology folders: n, ins, ...")
    parser.add_argument("--num-gpus", type=int, default=1, help="num_gpus passed to Sacred")
    parser.add_argument("--num-workers", type=int, default=4, help="num_workers passed to Sacred")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Additional Sacred override, can be specified multiple times")

    parser.add_argument("--pretrained-ckpt", default="",
                        help="Required for train mode. Pretrained checkpoint path")
    parser.add_argument("--checkpoint-root", default="",
                        help="Required for test mode if --load-path-template is not set")
    parser.add_argument("--load-path-template", default="",
                        help="Template for test ckpt path. Supports {fold} and {fold_idx}")
    parser.add_argument("--prefer", choices=["best", "last"], default="best",
                        help="When auto-discovering checkpoints, choose best or last")

    ns = parser.parse_args()
    folds = parse_folds(ns.folds)
    data_dirs = cap_data_dirs(ns.cap_root)

    if ns.mode == "train" and not ns.pretrained_ckpt:
        parser.error("--pretrained-ckpt is required in train mode")

    if ns.mode == "test" and not ns.load_path_template and not ns.checkpoint_root:
        parser.error("In test mode, set --load-path-template or --checkpoint-root")

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"[warn] missing data_dir: {data_dir}")

    for fold in folds:
        if fold <= 0:
            raise ValueError(f"Fold must be >= 1: {fold}")

        if ns.mode == "train":
            cmd = build_train_cmd(ns, fold, ns.pretrained_ckpt, data_dirs)
        else:
            fold_idx = fold - 1
            if ns.load_path_template:
                load_path = ns.load_path_template.format(fold=fold, fold_idx=fold_idx)
            else:
                try:
                    load_path = pick_checkpoint_for_fold(ns.checkpoint_root, fold_idx, ns.prefer)
                except FileNotFoundError as exc:
                    if ns.dry_run:
                        print(f"[warn] {exc}")
                        load_path = f"/tmp/MISSING_CHECKPOINT_fold_{fold_idx}.ckpt"
                    else:
                        raise
            cmd = build_test_cmd(ns, fold, load_path, data_dirs)

        code = run_command(cmd, ns.dry_run)
        if code != 0:
            print(f"[error] fold {fold} failed with exit code {code}")
            return code

    print("[done] all folds completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
