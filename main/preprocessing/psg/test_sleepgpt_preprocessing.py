from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

MODULE_PATH = Path(__file__).resolve().parent / "sleepgpt_preprocessing.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "sleepgpt_preprocessing", MODULE_PATH
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Could not load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

CANONICAL_CHANNEL_ORDER = MODULE.CANONICAL_CHANNEL_ORDER
SleepGPTPSGPreprocessConfig = MODULE.SleepGPTPSGPreprocessConfig
build_rotating_kfold_split = MODULE.build_rotating_kfold_split
convert_filtered_subject_to_h5 = MODULE.convert_filtered_subject_to_h5


class SleepGPTPSGPreprocessingTest(unittest.TestCase):
    def test_convert_filtered_subject_to_h5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            subject_dir = tmp_path / "input" / "subject_01"
            signals_dir = subject_dir / "Signals"
            signals_dir.mkdir(parents=True)

            source_frequency = 200
            target_frequency = 100
            epoch_seconds = 30
            total_epochs = 4
            total_samples = total_epochs * epoch_seconds * source_frequency
            time = np.arange(total_samples, dtype=np.float32) / source_frequency

            np.save(
                signals_dir / "EEG C3-A2.npy",
                np.sin(2 * np.pi * 1.0 * time).astype(np.float32),
            )
            np.save(
                signals_dir / "EEG C4-A1.npy",
                np.cos(2 * np.pi * 1.5 * time).astype(np.float32),
            )
            np.save(
                signals_dir / "EMG Chin.npy",
                (0.1 * np.sin(2 * np.pi * 10.0 * time)).astype(np.float32),
            )
            np.save(
                signals_dir / "EOG LOC-A2.npy",
                np.sin(2 * np.pi * 0.4 * time).astype(np.float32),
            )
            np.save(
                signals_dir / "EOG ROC-A1.npy",
                np.cos(2 * np.pi * 0.4 * time).astype(np.float32),
            )
            np.save(
                signals_dir / "EEG F3-A2.npy",
                np.sin(2 * np.pi * 0.8 * time).astype(np.float32),
            )
            np.save(subject_dir / "label.npy", np.array([0, 1, -1, 4], dtype=np.int64))

            output_root = tmp_path / "output"
            config = SleepGPTPSGPreprocessConfig(
                source_frequency=source_frequency,
                target_frequency=target_frequency,
                epoch_seconds=epoch_seconds,
                trim_edge_seconds=0,
                min_epochs_per_subject=2,
            )
            metadata = convert_filtered_subject_to_h5(
                subject_dir, output_root, config=config
            )

            self.assertEqual(metadata["valid_epochs"], 3)
            h5_path = output_root / "subject_01" / "data.h5"
            with h5py.File(h5_path, "r") as handle:
                signal = handle["signal"][:]
                stage = handle["stage"][:]
                good_channels = handle["good_channels"][:]

            self.assertEqual(signal.shape, (3, len(CANONICAL_CHANNEL_ORDER), 3000))
            np.testing.assert_array_equal(stage, np.array([0, 1, 4], dtype=np.int64))
            np.testing.assert_array_equal(
                good_channels,
                np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int8),
            )
            self.assertTrue(np.any(signal[:, 3, :] != 0.0))

    def test_build_rotating_kfold_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            processed_root = tmp_path / "processed"
            processed_root.mkdir()

            for subject_idx in range(5):
                subject_dir = processed_root / f"subject_{subject_idx:02d}"
                subject_dir.mkdir()
                with h5py.File(subject_dir / "data.h5", "w") as handle:
                    handle.create_dataset(
                        "signal", data=np.zeros((120, 8, 3000), dtype=np.float32)
                    )
                    handle.create_dataset(
                        "stage", data=np.zeros((120,), dtype=np.int64)
                    )
                    handle.create_dataset(
                        "good_channels", data=np.ones((8,), dtype=np.int8)
                    )

            split_path = processed_root / "psg_split.npy"
            split_dict = build_rotating_kfold_split(
                processed_root,
                split_path,
                n_splits=5,
                min_epochs_per_subject=100,
                seed=7,
            )

            self.assertTrue(split_path.exists())
            self.assertEqual(sorted(split_dict.keys())[0], "test_0")
            held_out_subjects = set()
            for fold in range(5):
                test_names = split_dict[f"test_{fold}"]["names"]
                self.assertEqual(len(test_names), 1)
                held_out_subjects.update(test_names)
                self.assertEqual(len(split_dict[f"val_{fold}"]["names"]), 1)
                self.assertEqual(len(split_dict[f"train_{fold}"]["names"]), 3)

            self.assertEqual(len(held_out_subjects), 5)


if __name__ == "__main__":
    unittest.main()
