"""Evaluate SleepGPT on custom PSG data: per-subject prediction CSVs + metrics.

Usage:
    python eval_psg_stage_kfold.py \\
        --processed-root /path/to/data \\
        --output-dir ./psg_stage_checkpoints \\
        --splits test val \\
        --kfold 5

Output structure::

    output_dir/
      fold_0/
        eval/
          test/
            subject-001.csv   # columns: subject_id, true_label_id, pred_label_id, is_correct
            subject-002.csv
          val/
            ...
          label_sequence_metrics.json
      fold_1/
        ...
      cross_fold_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from main.datasets.PSG_dataset import PSGDataset
from train_psg_stage_kfold import (
    _move_batch_to_device,
    build_config,
    build_model,
    preprocess_batch,
    resolve_checkpoint_paths,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_eval_loader(
    processed_root: str,
    split_file: str,
    fold: int,
    split: str,
    config: dict,
) -> tuple[DataLoader, PSGDataset]:
    """Create a DataLoader and Dataset for the given split/fold.

    Uses split_len=time_size to produce non-overlapping epoch windows.
    Overlapping epochs at subject boundaries are deduplicated in
    infer_per_subject() using batch['index'].
    """
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
        split_len=config["time_size"],
        patch_size=config["patch_size"],
        settings=None,
        kfold=fold,
        file_name=split_file,
    )
    dataset = PSGDataset(split=split, data_dir=processed_root, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=dataset.collate,
    )
    return loader, dataset


def build_item_to_subject_list(dataset: PSGDataset) -> list[str]:
    """Map each dataset item index to its subject directory basename.

    With pool_all=True and shuffle=False, dataset items are ordered by subject.
    nums_2_idx[subj] gives the first item index belonging to that subject.
    """
    item_to_subject: list[str] = []
    n_subjects = len(dataset.idx_2_name)
    for subj_idx in range(n_subjects):
        start = dataset.nums_2_idx[subj_idx]
        end = dataset.nums_2_idx[subj_idx + 1]
        path = dataset.idx_2_name[subj_idx]
        basename = os.path.basename(path.rstrip("/\\"))
        item_to_subject.extend([basename] * (end - start))
    return item_to_subject


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_per_subject(
    model,
    data_loader: DataLoader,
    device: str,
    item_to_subject: list[str],
) -> dict[str, tuple[list[int], list[int]]]:
    """Run inference and return per-subject (true_labels, pred_labels).

    Uses batch['index'] (absolute epoch index within each subject) to
    deduplicate epochs that appear in multiple overlapping windows. The last
    window's prediction is kept for each epoch. Padding epochs
    (label == -100 or epoch_idx < 0) are excluded.
    """
    model.eval()
    model.current_tasks = ["CrossEntropy"]
    # subject_id -> {epoch_abs_idx: (true_label, pred_label)}
    subject_epoch_data: dict[str, dict[int, tuple[int, int]]] = {}
    item_counter = 0

    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)
        batch = preprocess_batch(batch, model)

        # target_2d: (B, T) after preprocess_batch stacks and squeezes Stage_label
        target_2d = batch["Stage_label"]
        batch_size, time_size = target_2d.shape

        out = model.infer(batch, time_mask=False, stage="eval")
        logits = out["cls_feats"]["tf"]           # (B*T, num_classes)
        preds_flat = logits.argmax(dim=-1)         # (B*T,)
        preds_2d = preds_flat.reshape(batch_size, time_size)  # (B, T)

        # batch['index'] is (B*T, 1) after collate (all_time=True path)
        has_index = "index" in batch and batch["index"] is not None
        if has_index:
            indices_2d = batch["index"].reshape(batch_size, time_size)  # (B, T)

        for i in range(batch_size):
            global_item_idx = item_counter + i
            subject_id = (
                item_to_subject[global_item_idx]
                if global_item_idx < len(item_to_subject)
                else f"unknown_{global_item_idx}"
            )

            if subject_id not in subject_epoch_data:
                subject_epoch_data[subject_id] = {}

            labels_i = target_2d[i]  # (T,)
            preds_i = preds_2d[i]    # (T,)

            for t in range(time_size):
                label_t = int(labels_i[t].item())
                pred_t = int(preds_i[t].item())
                if label_t == -100:
                    continue
                if has_index:
                    epoch_idx = int(indices_2d[i, t].item())
                    if epoch_idx < 0:
                        continue
                    # Later windows overwrite earlier for overlapping epochs
                    subject_epoch_data[subject_id][epoch_idx] = (label_t, pred_t)
                else:
                    next_idx = len(subject_epoch_data[subject_id])
                    subject_epoch_data[subject_id][next_idx] = (label_t, pred_t)

        item_counter += batch_size

    result: dict[str, tuple[list[int], list[int]]] = {}
    for subject_id, epoch_dict in subject_epoch_data.items():
        sorted_items = sorted(epoch_dict.items())
        result[subject_id] = (
            [v[0] for _, v in sorted_items],
            [v[1] for _, v in sorted_items],
        )
    return result


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_subject_csvs(
    subject_data: dict[str, tuple[list[int], list[int]]],
    output_dir: Path,
    fold: int,
    split: str,
) -> list[str]:
    """Save per-subject prediction CSVs under output_dir/fold_{fold}/eval/{split}/."""
    split_dir = output_dir / f"fold_{fold}" / "eval" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[str] = []

    for subject_id, (true_labels, pred_labels) in subject_data.items():
        is_correct = [int(t == p) for t, p in zip(true_labels, pred_labels)]
        df = pd.DataFrame(
            {
                "subject_id": [subject_id] * len(true_labels),
                "true_label_id": true_labels,
                "pred_label_id": pred_labels,
                "is_correct": is_correct,
            }
        )
        csv_path = split_dir / f"{subject_id}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        output_paths.append(str(csv_path))

    return output_paths


# ---------------------------------------------------------------------------
# Metrics (ported from SleepStage label_sequence_metrics.py)
# ---------------------------------------------------------------------------

def _compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    if y_true.size == 0:
        return matrix
    np.add.at(
        matrix,
        (np.clip(y_true, 0, num_classes - 1), np.clip(y_pred, 0, num_classes - 1)),
        1,
    )
    return matrix


def compute_metrics_from_labels(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict:
    """Compute accuracy, per-class precision/recall/F1, macro averages,
    Cohen's kappa, and confusion matrix from label arrays."""
    conf_matrix = _compute_confusion_matrix(y_true, y_pred, num_classes)
    total = float(np.sum(conf_matrix))
    accuracy = float(np.trace(conf_matrix) / total) if total > 0 else 0.0

    m = conf_matrix.astype(np.float64)
    tp = np.diag(m)
    pred_sum = np.sum(m, axis=0)
    true_sum = np.sum(m, axis=1)

    precision_per_class = np.divide(
        tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0
    )
    recall_per_class = np.divide(
        tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0
    )
    denom = precision_per_class + recall_per_class
    f1_per_class = np.divide(
        2.0 * precision_per_class * recall_per_class,
        denom,
        out=np.zeros_like(tp),
        where=denom > 0,
    )

    kappa = 0.0
    if total > 0 and len(np.unique(y_true)) > 1:
        try:
            kappa = float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            kappa = 0.0

    return {
        "num_samples": int(total),
        "accuracy": accuracy,
        "kappa": kappa,
        "precision": float(np.mean(precision_per_class)),
        "recall": float(np.mean(recall_per_class)),
        "f1_score": float(np.mean(f1_per_class)),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_score_per_class": f1_per_class.tolist(),
        "confusion_matrix": conf_matrix.tolist(),
    }


def collect_fold_metrics_from_eval_dir(
    fold_eval_dir: Path, splits: list[str], num_classes: int
) -> dict:
    """Read per-subject CSVs and compute per-subject + overall metrics per split."""
    metrics_by_split: dict = {}

    for split in splits:
        split_dir = fold_eval_dir / split
        subject_metrics: dict = {}
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []

        if split_dir.exists():
            for csv_path in sorted(split_dir.glob("*.csv")):
                df = pd.read_csv(csv_path)
                if df.empty or not {"true_label_id", "pred_label_id"}.issubset(df.columns):
                    continue
                subject_id = csv_path.stem
                y_true = df["true_label_id"].to_numpy(dtype=np.int64)
                y_pred = df["pred_label_id"].to_numpy(dtype=np.int64)
                subject_metrics[subject_id] = compute_metrics_from_labels(
                    y_true, y_pred, num_classes
                )
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

        if y_true_all:
            y_true_np = np.concatenate(y_true_all)
            y_pred_np = np.concatenate(y_pred_all)
        else:
            y_true_np = np.asarray([], dtype=np.int64)
            y_pred_np = np.asarray([], dtype=np.int64)

        metrics_by_split[split] = {
            "overall": compute_metrics_from_labels(y_true_np, y_pred_np, num_classes),
            "subjects": subject_metrics,
        }

    return {"splits": metrics_by_split}


def write_fold_metrics_json(
    fold_eval_dir: Path, fold: int, num_classes: int, splits: list[str]
) -> dict:
    """Write label_sequence_metrics.json and return the payload dict."""
    payload = {
        "fold": fold,
        "num_classes": num_classes,
        **collect_fold_metrics_from_eval_dir(fold_eval_dir, splits, num_classes),
    }
    output_path = fold_eval_dir / "label_sequence_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def summarize_cross_fold(fold_results: list[dict], splits: list[str]) -> dict:
    """Compute mean/std of overall metrics across folds for each split."""
    summary: dict = {
        "num_folds": len(fold_results),
        "num_classes": fold_results[0].get("num_classes") if fold_results else None,
        "splits": {},
    }

    for split in splits:
        split_metrics = [
            res["splits"][split]["overall"]
            for res in fold_results
            if split in res.get("splits", {})
        ]
        if not split_metrics:
            continue

        acc = np.array([m["accuracy"] for m in split_metrics])
        f1 = np.array([m["f1_score"] for m in split_metrics])
        kappa = np.array([m["kappa"] for m in split_metrics])

        summary["splits"][split] = {
            "mean_accuracy": float(acc.mean()),
            "std_accuracy": float(acc.std(ddof=0)),
            "mean_f1_score": float(f1.mean()),
            "std_f1_score": float(f1.std(ddof=0)),
            "mean_kappa": float(kappa.mean()),
            "std_kappa": float(kappa.std(ddof=0)),
            "per_fold": [
                {"fold": res.get("fold", i), **res["splits"][split]["overall"]}
                for i, res in enumerate(fold_results)
                if split in res.get("splits", {})
            ],
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SleepGPT per subject on custom PSG HDF5 data.",
    )
    parser.add_argument(
        "--processed-root",
        required=True,
        help="Root directory containing processed subject folders.",
    )
    parser.add_argument(
        "--split-file",
        default="psg_split.npy",
        help="Split .npy or .json file produced by preprocessing. Default: %(default)s",
    )
    parser.add_argument("--output-dir", default="./psg_stage_checkpoints")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=None,
        help=(
            "Checkpoint path(s). One path (reused for all folds) or one per fold. "
            "If omitted, {output_dir}/fold_k/best.ckpt is used."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
        help="Which splits to evaluate. Default: test",
    )
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model-arch",
        default="backbone_large_patch200",
        choices=["backbone_base_patch200", "backbone_large_patch200"],
    )
    parser.add_argument("--decoder-features", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--grad-name", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cpu", action="store_true")
    # Dummy training-only args required by build_config
    parser.add_argument("--pretrain-ckpt", default="")
    parser.add_argument("--max-epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-lr", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    config["num_classes"] = args.num_classes
    checkpoint_paths = resolve_checkpoint_paths(args)

    print("SleepGPT Per-Subject Evaluation")
    print(f"  Device         : {config['device']}")
    print(f"  Processed root : {args.processed_root}")
    print(f"  Split file     : {args.split_file}")
    print(f"  Splits         : {args.splits}")
    print(f"  Num classes    : {args.num_classes}")
    print(f"  Model arch     : {config['model_arch']}")
    print(f"  Output dir     : {output_dir}")

    fold_results: list[dict] = []
    for fold, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n{'=' * 60}")
        print(f" Fold {fold}  |  checkpoint: {ckpt_path}")
        print(f"{'=' * 60}")

        model = build_model(config, fold, load_path=ckpt_path)
        model.eval()

        for split in args.splits:
            print(f"\n[Fold {fold}] Split: {split}")
            dl, dataset = make_eval_loader(
                args.processed_root, args.split_file, fold, split, config
            )
            if len(dl) == 0:
                print(f"  Empty DataLoader — skipping.")
                continue

            item_to_subject = build_item_to_subject_list(dataset)
            subject_data = infer_per_subject(model, dl, config["device"], item_to_subject)
            csv_paths = save_subject_csvs(subject_data, output_dir, fold, split)
            print(
                f"  Saved {len(csv_paths)} subject CSV(s) in "
                f"{output_dir}/fold_{fold}/eval/{split}/"
            )

        fold_eval_dir = output_dir / f"fold_{fold}" / "eval"
        fold_metrics = write_fold_metrics_json(
            fold_eval_dir, fold, args.num_classes, args.splits
        )
        fold_results.append(fold_metrics)

        for split in args.splits:
            if split in fold_metrics.get("splits", {}):
                ov = fold_metrics["splits"][split]["overall"]
                print(
                    f"  [{split}] accuracy={ov['accuracy']:.4f}  "
                    f"f1={ov['f1_score']:.4f}  kappa={ov['kappa']:.4f}  "
                    f"n={ov['num_samples']}"
                )

    if fold_results:
        summary = summarize_cross_fold(fold_results, args.splits)
        summary_path = output_dir / "cross_fold_eval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"\n{'=' * 60}")
        print("Cross-fold summary")
        print(f"{'=' * 60}")
        for split in args.splits:
            if split in summary.get("splits", {}):
                s = summary["splits"][split]
                print(
                    f"  [{split}] "
                    f"mean_accuracy={s['mean_accuracy']:.4f}±{s['std_accuracy']:.4f}  "
                    f"mean_f1={s['mean_f1_score']:.4f}±{s['std_f1_score']:.4f}  "
                    f"mean_kappa={s['mean_kappa']:.4f}±{s['std_kappa']:.4f}"
                )
        print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
