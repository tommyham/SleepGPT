"""Run main_test_kfold_persub with CAP settings without srun.

This script mirrors Test_cap.sh by invoking Sacred directly:
  - named configs: finetune_CAP, CAP_datasets
  - config overrides: same values passed in the shell script
"""

from main.config import ex
import main_test_kfold_persub  # noqa: F401  # Registers @ex.automain on import.


def build_config_updates() -> dict:
    kfold_load_path = "/mnt/e/DataSet/Local/OpenData/capslpdb/checkpoint"
    return {
        "num_gpus": 1,
        "num_nodes": 1,
        "num_workers": 1,
        "batch_size": 64,
        "model_arch": "backbone_large_patch200",
        "lr_mult": 20,
        "warmup_lr": 0,
        "val_check_interval": 0.5,
        "check_val_every_n_epoch": 1,
        "limit_train_batches": 1.0,
        "max_steps": -1,
        "all_time": True,
        "time_size": 1,
        "decoder_features": 768,
        "pool": None,
        "lr": 1e-3,
        "min_lr": 0,
        "random_choose_channels": 8,
        "max_epoch": 50,
        "lr_policy": "cosine",
        "loss_function": "l1",
        "drop_path_rate": 0.5,
        "warmup_steps": 0.1,
        "split_len": 1,
        "kfold_load_path": kfold_load_path,
        "use_pooling": "swin",
        "use_relative_pos_emb": False,
        "mixup": 0,
        "smoothing": 0.1,
        "decoder_heads": 16,
        "use_all_label": "all",
        "use_multiway": "multiway",
        "use_g_mid": False,
        "get_param_method": "layer_decay",
        "local_pooling": False,
        "optim": "adamw",
        "poly": False,
        "weight_decay": 0.05,
        "layer_decay": 0.75,
        "Lambda": 1.0,
        "patch_size": 200,
        "use_cb": True,
        "kfold": 4,
        "grad_name": "all",
        "resume_during_training": 0,
        "resume_ckpt_path": "",
        "save_top_k": 1,
        "eval": True,
        "dist_on_itp": False,
    }


def main() -> None:
    ex.run(
        named_configs=["finetune_CAP", "CAP_datasets"],
        config_updates=build_config_updates(),
    )


if __name__ == "__main__":
    main()