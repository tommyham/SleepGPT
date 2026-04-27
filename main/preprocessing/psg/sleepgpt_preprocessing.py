from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import h5py
import numpy as np
from scipy.signal import resample_poly


CANONICAL_CHANNEL_ORDER = ("C3", "C4", "EMG", "EOG", "F3", "Fpz", "O1", "Pz")
CANONICAL_CHANNEL_IDS = np.array([4, 5, 16, 18, 22, 36, 38, 52], dtype=np.int64)
VALID_STAGE_LABELS = {0, 1, 2, 3, 4}
DEFAULT_DISCARD_LABELS = {-1, 5, 6, 7}

DEFAULT_CHANNEL_ALIASES: Dict[str, tuple[str, ...]] = {
    "C3": (
        "c3",
        "c3a2",
        "c3m2",
        "eegc3",
        "eegc3a2",
        "eegc3m2",
    ),
    "C4": (
        "c4",
        "c4a1",
        "c4m1",
        "eegc4",
        "eegc4a1",
        "eegc4m1",
    ),
    "EMG": (
        "emg",
        "emg1",
        "emgchin",
        "chin",
        "chin1",
        "chin2",
        "submental",
        "submentalemg",
    ),
    "EOG": (
        "eog",
        "eog1",
        "eog2",
        "eogleft",
        "eogright",
        "lefteog",
        "righteog",
        "loc",
        "roc",
        "eogloc",
        "eogroc",
        "loca2",
        "locm2",
        "roca1",
        "rocm1",
        "eogloca2",
        "eoglocm2",
        "eogroca1",
        "eogrocm1",
    ),
    "F3": (
        "f3",
        "f3a2",
        "f3m2",
        "eegf3",
        "eegf3a2",
        "eegf3m2",
    ),
    "Fpz": (
        "fpz",
        "fpza1",
        "fpzm1",
        "eegfpz",
        "eegfpza1",
        "eegfpzm1",
    ),
    "O1": (
        "o1",
        "o1a2",
        "o1m2",
        "eego1",
        "eego1a2",
        "eego1m2",
    ),
    "Pz": (
        "pz",
        "pza1",
        "pzm1",
        "eegpz",
        "eegpza1",
        "eegpzm1",
    ),
}

EOG_LEFT_ALIASES = (
    "loc",
    "loca2",
    "locm2",
    "eogloc",
    "eogloca2",
    "eoglocm2",
    "lefteog",
    "eogleft",
)

EOG_RIGHT_ALIASES = (
    "roc",
    "roca1",
    "rocm1",
    "eogroc",
    "eogroca1",
    "eogrocm1",
    "righteog",
    "eogright",
)


@dataclass
class SleepGPTPSGPreprocessConfig:
    source_frequency: int = 1000
    target_frequency: int = 100
    epoch_seconds: int = 30
    trim_edge_seconds: int = 3600
    min_epochs_per_subject: int = 100
    discard_labels: set[int] = field(
        default_factory=lambda: set(DEFAULT_DISCARD_LABELS)
    )
    label_mapping: dict[int, int] | None = None
    channel_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_ALIASES)
    )

    @property
    def samples_per_epoch(self) -> int:
        return self.target_frequency * self.epoch_seconds

    @property
    def trim_edge_epochs(self) -> int:
        return self.trim_edge_seconds // self.epoch_seconds


def normalize_channel_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def parse_label_mapping(raw: str | None) -> dict[int, int] | None:
    if raw is None or raw.strip() == "":
        return None
    mapping: dict[int, int] = {}
    for item in raw.split(","):
        src, dst = item.split(":", maxsplit=1)
        mapping[int(src.strip())] = int(dst.strip())
    return mapping


def _iter_filtered_subject_dirs(input_root: str | Path) -> Iterable[Path]:
    root = Path(input_root)
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "Signals").is_dir() and (candidate / "label.npy").exists():
            yield candidate


def _load_signal_arrays(subject_dir: Path) -> dict[str, np.ndarray]:
    signal_dir = subject_dir / "Signals"
    loaded: dict[str, np.ndarray] = {}
    for npy_path in sorted(signal_dir.glob("*.npy")):
        signal = np.load(npy_path)
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D: {npy_path}")
        loaded[npy_path.stem] = signal.astype(np.float32, copy=False)
    if not loaded:
        raise FileNotFoundError(f"No signal npy files found under {signal_dir}")
    return loaded


def _resample_signal(
    signal: np.ndarray, source_frequency: int, target_frequency: int
) -> np.ndarray:
    if source_frequency == target_frequency:
        return signal.astype(np.float32, copy=False)
    gcd = math.gcd(source_frequency, target_frequency)
    up = target_frequency // gcd
    down = source_frequency // gcd
    return resample_poly(signal, up, down).astype(np.float32, copy=False)


def _build_normalized_lookup(
    signals: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    for name, signal in signals.items():
        normalized = normalize_channel_name(name)
        lookup.setdefault(normalized, signal)
    return lookup


def _pick_first_signal(
    lookup: Mapping[str, np.ndarray], aliases: Sequence[str]
) -> np.ndarray | None:
    for alias in aliases:
        signal = lookup.get(normalize_channel_name(alias))
        if signal is not None:
            return signal
    return None


def _build_eog_signal(lookup: Mapping[str, np.ndarray]) -> np.ndarray | None:
    left = _pick_first_signal(lookup, EOG_LEFT_ALIASES)
    right = _pick_first_signal(lookup, EOG_RIGHT_ALIASES)
    if left is not None and right is not None:
        length = min(len(left), len(right))
        return (right[:length] - left[:length]).astype(np.float32, copy=False)
    if left is not None:
        return left
    if right is not None:
        return right
    return _pick_first_signal(lookup, DEFAULT_CHANNEL_ALIASES["EOG"])


def _build_canonical_signals(
    signals: Mapping[str, np.ndarray],
    config: SleepGPTPSGPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    lookup = _build_normalized_lookup(signals)
    canonical_signals: list[np.ndarray] = []
    good_channels = np.zeros(len(CANONICAL_CHANNEL_ORDER), dtype=np.int8)
    source_names: dict[str, str] = {}

    candidate_signals: dict[str, np.ndarray | None] = {}
    for channel_name in CANONICAL_CHANNEL_ORDER:
        if channel_name == "EOG":
            candidate_signals[channel_name] = _build_eog_signal(lookup)
        else:
            candidate_signals[channel_name] = _pick_first_signal(
                lookup,
                config.channel_aliases[channel_name],
            )

    lengths = [
        len(signal) for signal in candidate_signals.values() if signal is not None
    ]
    if not lengths:
        raise ValueError("No canonical channels were matched from filtered signals.")
    common_length = min(lengths)

    for idx, channel_name in enumerate(CANONICAL_CHANNEL_ORDER):
        signal = candidate_signals[channel_name]
        if signal is None:
            canonical_signals.append(np.zeros(common_length, dtype=np.float32))
            continue
        canonical_signals.append(signal[:common_length].astype(np.float32, copy=False))
        good_channels[idx] = 1
        source_names[channel_name] = channel_name

    stacked = np.stack(canonical_signals, axis=0)
    return stacked, good_channels, source_names


def _normalize_labels(
    labels: np.ndarray, config: SleepGPTPSGPreprocessConfig
) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    normalized = labels.astype(np.int64, copy=True)

    if config.label_mapping is not None:
        remapped = np.full_like(normalized, fill_value=-100)
        for source_label, target_label in config.label_mapping.items():
            remapped[normalized == source_label] = target_label
        missing = sorted(set(normalized.tolist()) - set(config.label_mapping.keys()))
        if missing:
            raise ValueError(f"Found labels without mapping: {missing}")
        normalized = remapped

    discard_mask = np.isin(normalized, list(config.discard_labels))
    normalized[discard_mask] = -100

    valid_labels = set(normalized[normalized >= 0].tolist())
    unexpected = sorted(valid_labels - VALID_STAGE_LABELS)
    if unexpected:
        raise ValueError(
            "Normalized labels must be in {0,1,2,3,4}; "
            f"found unexpected labels: {unexpected}"
        )
    return normalized


def convert_filtered_subject_to_h5(
    subject_dir: str | Path,
    output_root: str | Path,
    config: SleepGPTPSGPreprocessConfig | None = None,
) -> dict[str, int | str | list[str]]:
    config = config or SleepGPTPSGPreprocessConfig()
    subject_dir = Path(subject_dir)
    output_root = Path(output_root)
    output_dir = output_root / subject_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_signals = _load_signal_arrays(subject_dir)
    labels = np.load(subject_dir / "label.npy")

    resampled_signals = {
        name: _resample_signal(signal, config.source_frequency, config.target_frequency)
        for name, signal in raw_signals.items()
    }

    canonical_signals, good_channels, source_names = _build_canonical_signals(
        resampled_signals, config
    )
    labels = _normalize_labels(labels, config)

    trim_samples = config.trim_edge_seconds * config.target_frequency
    trim_epochs = config.trim_edge_epochs
    if trim_samples > 0 and canonical_signals.shape[1] > 2 * trim_samples:
        canonical_signals = canonical_signals[:, trim_samples:-trim_samples]
    if trim_epochs > 0 and labels.shape[0] > 2 * trim_epochs:
        labels = labels[trim_epochs:-trim_epochs]

    max_epochs_from_signal = canonical_signals.shape[1] // config.samples_per_epoch
    max_epochs = min(max_epochs_from_signal, labels.shape[0])
    if max_epochs <= 0:
        raise ValueError(f"No complete epochs available for {subject_dir}")

    canonical_signals = canonical_signals[:, : max_epochs * config.samples_per_epoch]
    labels = labels[:max_epochs]

    epoch_signal = canonical_signals.reshape(
        len(CANONICAL_CHANNEL_ORDER),
        max_epochs,
        config.samples_per_epoch,
    ).transpose(1, 0, 2)

    valid_mask = labels >= 0
    epoch_signal = epoch_signal[valid_mask]
    labels = labels[valid_mask]
    if labels.shape[0] == 0:
        raise ValueError(
            f"Subject {subject_dir.name} has no valid epochs after label filtering."
        )

    h5_path = output_dir / "data.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset(
            "signal", data=epoch_signal.astype(np.float32), compression="gzip"
        )
        handle.create_dataset("stage", data=labels.astype(np.int64), compression="gzip")
        handle.create_dataset("good_channels", data=good_channels.astype(np.int8))

    metadata = {
        "subject_id": subject_dir.name,
        "channel_order": list(CANONICAL_CHANNEL_ORDER),
        "channel_ids": CANONICAL_CHANNEL_IDS.tolist(),
        "good_channels": good_channels.astype(int).tolist(),
        "source_frequency": config.source_frequency,
        "target_frequency": config.target_frequency,
        "epoch_seconds": config.epoch_seconds,
        "trim_edge_seconds": config.trim_edge_seconds,
        "valid_epochs": int(labels.shape[0]),
        "available_canonical_channels": [
            channel_name
            for channel_name, flag in zip(CANONICAL_CHANNEL_ORDER, good_channels)
            if flag
        ],
        "resolved_channels": source_names,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def preprocess_filtered_psg_dataset(
    input_root: str | Path,
    output_root: str | Path,
    config: SleepGPTPSGPreprocessConfig | None = None,
) -> list[dict[str, int | str | list[str]]]:
    config = config or SleepGPTPSGPreprocessConfig()
    summaries: list[dict[str, int | str | list[str]]] = []
    for subject_dir in _iter_filtered_subject_dirs(input_root):
        summaries.append(
            convert_filtered_subject_to_h5(subject_dir, output_root, config=config)
        )
    if not summaries:
        raise FileNotFoundError(f"No filtered PSG subjects found under {input_root}")
    manifest = {
        "config": {
            **asdict(config),
            "discard_labels": sorted(config.discard_labels),
            "channel_aliases": {k: list(v) for k, v in config.channel_aliases.items()},
        },
        "subjects": summaries,
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return summaries


def build_rotating_kfold_split(
    processed_root: str | Path,
    output_path: str | Path,
    n_splits: int = 5,
    min_epochs_per_subject: int = 100,
    seed: int = 3407,
) -> dict[str, dict[str, list[str] | list[int]]]:
    if n_splits < 3:
        raise ValueError(
            "n_splits must be at least 3 so train/val/test can be separated."
        )

    processed_root = Path(processed_root)
    subject_dirs = sorted(
        path
        for path in processed_root.iterdir()
        if path.is_dir() and (path / "data.h5").exists()
    )
    if len(subject_dirs) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} processed subjects, found {len(subject_dirs)}"
        )

    names: list[str] = []
    nums: list[int] = []
    for subject_dir in subject_dirs:
        with h5py.File(subject_dir / "data.h5", "r") as handle:
            num_epochs = int(handle["signal"].shape[0])
        if num_epochs < min_epochs_per_subject:
            continue
        names.append(str(subject_dir / "data.h5"))
        nums.append(num_epochs)

    if len(names) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} subjects with >= {min_epochs_per_subject} epochs; found {len(names)}"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    shuffled_names = [names[idx] for idx in order]
    shuffled_nums = [nums[idx] for idx in order]
    fold_indices = [
        np.array(chunk, dtype=np.int64)
        for chunk in np.array_split(np.arange(len(shuffled_names)), n_splits)
    ]

    split_dict: dict[str, dict[str, list[str] | list[int]]] = {}
    for fold in range(n_splits):
        test_idx = fold_indices[fold]
        val_idx = fold_indices[(fold + 1) % n_splits]
        train_idx = np.concatenate(
            [
                fold_indices[idx]
                for idx in range(n_splits)
                if idx not in {fold, (fold + 1) % n_splits}
            ],
            axis=0,
        )
        for split_name, split_idx in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            split_dict[f"{split_name}_{fold}"] = {
                "names": [shuffled_names[idx] for idx in split_idx.tolist()],
                "nums": [shuffled_nums[idx] for idx in split_idx.tolist()],
            }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, split_dict, allow_pickle=True)
    return split_dict


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert filtered PSG npy signals into SleepGPT-compatible HDF5 files.",
    )
    parser.add_argument(
        "--input-root", required=True, help="Root directory containing subject folders."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory to write SleepGPT HDF5 subjects.",
    )
    parser.add_argument(
        "--split-output",
        default="",
        help="Optional .npy path for rotating k-fold split metadata.",
    )
    parser.add_argument("--source-frequency", type=int, default=1000)
    parser.add_argument("--target-frequency", type=int, default=100)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--trim-edge-seconds", type=int, default=3600)
    parser.add_argument("--min-epochs-per-subject", type=int, default=100)
    parser.add_argument(
        "--label-map",
        default="",
        help="Optional label remapping, e.g. '0:0,1:1,2:2,3:3,4:4,5:-1'.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = SleepGPTPSGPreprocessConfig(
        source_frequency=args.source_frequency,
        target_frequency=args.target_frequency,
        epoch_seconds=args.epoch_seconds,
        trim_edge_seconds=args.trim_edge_seconds,
        min_epochs_per_subject=args.min_epochs_per_subject,
        label_mapping=parse_label_mapping(args.label_map),
    )
    summaries = preprocess_filtered_psg_dataset(
        args.input_root, args.output_root, config=config
    )
    print(f"Converted {len(summaries)} subjects into SleepGPT HDF5 format.")
    if args.split_output:
        split_dict = build_rotating_kfold_split(
            args.output_root,
            args.split_output,
            n_splits=args.n_splits,
            min_epochs_per_subject=args.min_epochs_per_subject,
            seed=args.seed,
        )
        print(
            f"Saved rotating k-fold split with {args.n_splits} folds to {args.split_output}."
        )
        print(f"Available split keys: {sorted(split_dict.keys())[:3]} ...")


if __name__ == "__main__":
    main()
