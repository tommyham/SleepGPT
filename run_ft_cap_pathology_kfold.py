"""Run main_kfold with CAP pathology fine-tuning settings without srun.

This script mirrors start_ft_cap_pathology.sh by invoking Sacred directly:
  - named configs: finetune_CAP, CAP_datasets
  - config overrides: same values passed in the shell script
"""

from main.config import ex
import main_kfold  # noqa: F401  # Registers @ex.automain on import.


def build_config_updates() -> dict:
    load_path = (
        "/mnt/e/DataSet/Local/OpenData/capslpdb/checkpoint/"
        ""
        "ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt"
    )

    _config = {
        "num_gpus": 1,
        "num_nodes": 1,
        "num_workers": 0,
        "batch_size": 8,
        "model_arch": "backbone_large_patch200",
        "lr_mult": 20,
        "warmup_lr": 0,
        "val_check_interval": 0.5,
        "check_val_every_n_epoch": 1,
        "limit_train_batches": 1.0,
        "max_steps": -1,
        "all_time": True,
        "dist_on_itp": False,
        "time_size": 100,
        "pool": None,
        "lr": 5e-4,
        "min_lr": 0,
        "random_choose_channels": 8,
        "max_epoch": 30,
        "lr_policy": "cosine",
        "loss_function": "l1",
        "drop_path_rate": 0.5,
        "warmup_steps": 0.1,
        "split_len": 100,
        "load_path": load_path,
        "mixup": 0,
        "smoothing": 0.1,
        "use_global_fft": True,
        "use_all_label": "all",
        "get_param_method": "layer_decay",
        "use_pooling": "longnet",
        "local_pooling": False,
        "optim": "adamw",
        "poly": False,
        "weight_decay": 0.05,
        "layer_decay": 0.75,
        "Lambda": 1.0,
        "patch_size": 200,
        "use_cb": True,
        "grad_name": "partial_10",
        "resume_during_training": 0,
        "resume_ckpt_path": "",
        "kfold": 4,
        "use_triton": False,
        "decoder_features": 1024,
        "decoder_depth": 4,
        "decoder_selected_layers": "2-3",
        "decoder_heads": 32,
    }

    return _config


def main() -> None:
    _config = build_config_updates()
    ex.run(
        named_configs=["finetune_CAP", "CAP_datasets"],
        config_updates=_config,
    )


if __name__ == "__main__":
    main()
