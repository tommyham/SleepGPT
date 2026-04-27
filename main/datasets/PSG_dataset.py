import os
import json
from pathlib import Path

import h5py
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from .new_base_dataset import Aug_BaseDataset


class PSGDataset(Aug_BaseDataset):

    split = "train"
    transform_keys = ["full"]
    data_dir = ["./"]
    column_names = ["signal", "stage", "good_channels"]
    fs = 100
    epoch_duration = 30
    stage = True
    spindle = False
    pathology = False

    @staticmethod
    def _resolve_json_fold(folds, k):
        k_int = int(k)
        for item in folds:
            if int(item.get("fold", -1)) == k_int:
                return item
        # Allow 0-based fold indexing from training loops.
        for item in folds:
            if int(item.get("fold", -1)) == (k_int + 1):
                return item
        raise KeyError(f"Cannot find fold={k} (or fold={k_int + 1}) in folds.json")

    @staticmethod
    def _build_subject_dir_map(data_dir: Path):
        subject_map = {}
        for candidate in sorted(data_dir.iterdir()):
            if candidate.is_dir() and (candidate / "data.h5").exists():
                subject_map[candidate.name] = candidate
        return subject_map

    @staticmethod
    def _match_subject_dir(subject_id: str, subject_map):
        if subject_id in subject_map:
            return subject_map[subject_id]
        for key, path in subject_map.items():
            if key.endswith(subject_id):
                return path
        return None

    @staticmethod
    def _collect_names_nums_from_subject_ids(data_dir: Path, subject_ids):
        subject_map = PSGDataset._build_subject_dir_map(data_dir)
        names = []
        nums = []
        missing_subjects = []

        for subject_id in subject_ids:
            subject_dir = PSGDataset._match_subject_dir(subject_id, subject_map)
            if subject_dir is None:
                missing_subjects.append(subject_id)
                continue
            h5_path = subject_dir / "data.h5"
            with h5py.File(h5_path, "r") as handle:
                num_epochs = int(handle["signal"].shape[0])
            names.append(str(subject_dir))
            nums.append(num_epochs)

        if missing_subjects:
            preview = sorted(subject_map.keys())[:10]
            raise FileNotFoundError(
                f"Subjects from folds.json not found under {data_dir}: {missing_subjects}. "
                f"Available subject dirs (first 10): {preview}"
            )

        return names, nums

    def __init__(self, split="", *args, **kwargs):
        assert split in ["train", "val", "test"]
        k = kwargs.get("kfold")
        if k is None:
            raise NotImplementedError

        file_name = kwargs.pop("file_name", "psg_split.npy")
        split_path = Path(file_name)
        if not split_path.is_absolute():
            data_dir_split_path = Path(kwargs["data_dir"]) / split_path
            dataset_dir_split_path = Path(__file__).resolve().parent / split_path
            split_path = (
                data_dir_split_path
                if data_dir_split_path.exists()
                else dataset_dir_split_path
            )

        rank_zero_info(f"psg datasets items file name: {split_path}, kfold is {k}")
        if split_path.suffix.lower() == ".json":
            fold_payload = json.loads(split_path.read_text(encoding="utf-8"))
            fold_spec = self._resolve_json_fold(fold_payload["folds"], k)
            split_subject_ids = fold_spec[f"{split}_subjects"]
            names, nums = self._collect_names_nums_from_subject_ids(
                Path(kwargs["data_dir"]), split_subject_ids
            )
        else:
            items = np.load(split_path, allow_pickle=True)
            if items.dtype == np.dtype("O"):
                data = items.item()[f"{split}_{k}"]
                names = data["names"]
                nums = data.get("nums")
            else:
                names = items["names"]
                nums = items["nums"] if "nums" in items.keys() else None

        kwargs.pop("kfold")
        rank_zero_info(f"data_dir: {kwargs['data_dir']}")
        super().__init__(
            names=names, concatenate=False, nums=nums, split=split, *args, **kwargs
        )

    def __getitem__(self, index):
        return self.get_suite(index)

    @property
    def channels(self):
        return np.array([4, 5, 16, 18, 22, 36, 38, 52])

    def get_name(self, index):
        idx = np.where(self.idx_2_nums <= index)[0][-1]
        start_idx = index - self.nums_2_idx[idx]
        if self.pool_all:
            start_idx *= self.split_len
        try:
            return int(
                os.path.basename(os.path.dirname(self.idx_2_name[idx])).split("-")[-1]
            )
        except Exception:
            return os.path.basename(os.path.dirname(self.idx_2_name[idx]))
