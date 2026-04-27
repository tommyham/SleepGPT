import os
from pathlib import Path

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

    def __init__(self, split="", *args, **kwargs):
        assert split in ["train", "val", "test"]
        k = kwargs["kfold"]
        if k is None:
            raise NotImplementedError

        file_name = kwargs.pop("file_name", "psg_split.npy")
        split_path = Path(file_name)
        if not split_path.is_absolute():
            split_path = Path(kwargs["data_dir"]) / split_path

        rank_zero_info(f"psg datasets items file name: {split_path}, kfold is {k}")
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
