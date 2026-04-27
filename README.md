
# SleepGPT: A Unified Time-Frequency Foundation Model for Sleep Decoding

SleepGPT is a foundation model designed for comprehensive sleep decoding. Built upon PyTorch Lightning and generative pretraining, SleepGPT generalizes across multiple sleep-related tasks and heterogeneous polysomnography (PSG) datasets. The model integrates time- and frequency-domain information using a unified transformer framework, and adapts dynamically to varying EEG channel configurations.

---

## 🚀 Key Features

- Pretrained on over **86,000 hours** of PSG recordings from **8,377 subjects**
- Supports multiple tasks: **sleep staging**, **spindle detection**, **apnea classification**, and **signal generation**
- Unified **time-frequency transformer architecture**
- Channel-adaptive fusion mechanism for diverse PSG configurations
- Compatible with over 10+ public PSG datasets

---

## 📦 Repository Structure

### 🧠 Model Components
| File                        | Description                                                       |
|-----------------------------|-------------------------------------------------------------------|
| `backbone.py`              | Main model with time-frequency fusion and attention handling      |
| `multiway_transformer.py`  | Core domain-aware transformer encoder                             |
| `Swin_transformer.py`      | Global-context encoder based on Swin Transformer                  |
| `backbone_pretrain.py`     | Self-supervised pretraining variant                               |
| `heads.py`                 | Pooling and projection heads for classification tasks             |
| `objectives.py`            | Implements contrastive, classification, and reconstruction losses |
| `get_optm.py`              | Optimizer and learning rate scheduler setup                       |

### 📚 Dataset and DataModule
| File/Folder           | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `*_datamodule.py`     | Lightning DataModules for datasets like MASS, SHHS, EDF, ISRUC, Apnea, etc. |
| `*_dataset.py`        | Dataset implementations with task-specific processing                       |
| `BaseDataModule.py`   | Base class for all DataModules                                              |
| `new_base_dataset.py` | Base class for all Dataset                                                  |

### 🧰 Utilities
| File              | Description |
|-------------------|-------------|
| `my_metrics.py`   | Custom metrics: Accuracy, Scalar, etc. |
| `transform.py`    | Frequency-based and dual-stream data augmentation |
| `others.py`       | Loss functions: Focal, Dice, Weighted BCE |

### 📊 Visualization
Scripts under `Visualization/` include:
- `visual_mask.py`: attention mask heatmaps
- `visual_fft.py`: frequency domain plots
- `visual_umap.py`: embedding space visualization
- `visual_spindles.py`: spindle detection overlays
- `visual_portion.py`: per-epoch prediction visualization

### 🧹 Preprocessing
| File/Folder            | Description                                                             |
|------------------------|-------------------------------------------------------------------------|
| `preprocessing.py`     | Preprocess raw PSG into h5 format or pyarrow format, normalize channels |
| `generate_list.py`     | Generate index and dataset split metadata                               |
| `cap/`, `edf/`, ...    | Subdirectories for dataset-specific preprocessing scripts               |

### 🧪 Training & Evaluation
| File                         | Purpose |
|------------------------------|---------|
| `main.py`                    | Launch main training procedure |
| `main_kfold.py`              | K-fold training |
| `main_test_kfold.py`         | K-fold evaluation |
| `main_test_kfold_persub.py` | Per-subject evaluation mode |
| `.sh` files                  | Slurm / shell job scripts |

---

## ⚙️ Getting Started

### 1. Install Dependencies

```bash
conda create -n sleepgpt python=3.8
conda activate sleepgpt
pip install -r requirements.txt
```

### 2. Preprocess Your Dataset

```bash
python preprocessing/dataset/preprocessing.py
```

### 3. Launch Training


To run experiments, use the provided SLURM scripts. All configurations are managed using [Sacred](https://sacred.readthedocs.io/), allowing you to define experiments by name.

---

### 🔧 Pretraining

Pretraining runs use `main.py`. You can launch it with SLURM like this:

```bash
sbatch Pt_unify_slurm.sh
```

Internally, it uses:

```bash
srun python3 main.py with pretrain_shhs_stage2 SHHS1_WM_datasets
```

- `pretrain_shhs_stage2`: pretraining mode configuration
- `SHHS1_WM_datasets`: dataset loader setup
- Additional arguments (e.g. `mask_ratio`, `loss_function`, `model_arch`) are passed via CLI.

### 💾 Pretrained Checkpoint

We provide a pretrained checkpoint that can be used for downstream tasks such as sleep staging and spindle detection.

- **Download link**: [Google Drive](https://drive.google.com/file/d/1aSU60xUDtXhOAaCrkx6lrIxHSO1dVMQc/view?usp=drive_link)

To use the checkpoint, specify the `load_path` in your training or fine-tuning script:

```bash
load_path=/your/path/to/ModelCheckpoint.ckpt
```

---

### 🧪 Fine-tuning (K-Fold)

Fine-tuning runs use `main_kfold.py`, usually with k-fold evaluation and resume support.

Launch with (e.g. MASS SS2 dataset):

```bash
sbatch Start_ft_mass_stage_p.sh
```

Internally:

```bash
srun python3 main_kfold.py with finetune_mass_stage MASS2_datasets
```

- `finetune_mass_stage`: fine-tuning mode configuration (e.g. lr schedule, decoder head)
- `MASS2_datasets`: MASS dataset loader with augmentation & label mapping

All configurations are defined in [`config.py`](./config.py), so you don’t need to modify code—just pass the names.

---
## 📂 Supported Tasks

- 💤 Sleep staging
- ⚡ Sleep signal generation
- 🫁 Sleep-related pathology classification
- 🌙 Sleep spindle detection
---

## 🔍 Demo: Visualizing Masked Reconstruction

See [`masked_reconstruction_demo.md`](docs/masked_reconstruction_demo.md) for a full explanation and how to run the visualization script.

## 🧾 Custom PSG Workflow

For adapting a filtered custom PSG dataset into SleepGPT-native HDF5, building rotating k-fold splits, fine-tuning the sleep staging head, and re-running evaluation from saved checkpoints, see [`custom_psg_sleepgpt_pipeline.md`](docs/custom_psg_sleepgpt_pipeline.md).


## 📝 Citation

If you use SleepGPT in your research, please cite:

```bibtex
@article{huang2026unified,
  title={A unified time-frequency foundation model for sleep decoding},
  author={Huang, Weixuan and Wang, Yan and Cheng, Hanrong and Xu, Wei and Li, Tingyue and Wu, Xiuwen and Xu, Hui and Liao, Pan and Cui, Zaixu and Zou, Qihong and others},
  journal={Nature Communications},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

---

## 📬 Contact

- Maintainer: [Weixuan Huang](mailto:weixuan.huang@pku.edu.cn)
- Institution: Peking University

---

