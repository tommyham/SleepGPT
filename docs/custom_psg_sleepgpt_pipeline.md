# Custom PSG SleepGPT Pipeline

This workflow converts filtered PSG `.npy` signals into the native SleepGPT HDF5 layout, builds rotating cross-validation splits, fine-tunes the sleep staging head, and re-evaluates saved checkpoints later without retraining.

## 1. Input Assumptions

The converter expects one subject per folder with this structure:

```text
FilteredRoot/
  subject_001/
    Signals/
      EEG C3-A2.npy
      EEG C4-A1.npy
      EOG LOC-A2.npy
      EOG ROC-A1.npy
      EMG Chin.npy
      ...
    label.npy
```

- Each signal file must be one-dimensional and continuous over the whole recording.
- `label.npy` must contain one label per 30-second epoch before trimming.
- Missing canonical channels are allowed. They are zero-filled and tracked in `good_channels`.

## 2. Canonical SleepGPT Layout

The converter writes one `data.h5` per subject with:

- `signal`: `[n_epochs, 8, 3000]`
- `stage`: `[n_epochs]`
- `good_channels`: `[8]`

The canonical channel order is:

```text
C3, C4, EMG, EOG, F3, Fpz, O1, Pz
```

These map to the SleepGPT large-mode channel IDs:

```text
[4, 5, 16, 18, 22, 36, 38, 52]
```

## 3. Preprocess Filtered PSG into HDF5

Run the converter from the repository root:

```bash
python main/preprocessing/psg/sleepgpt_preprocessing.py \
  --input-root /path/to/FilteredRoot \
  --output-root /path/to/ProcessedPSG \
  --split-output /path/to/ProcessedPSG/psg_split.npy \
  --source-frequency 1000 \
  --target-frequency 100 \
  --trim-edge-seconds 3600 \
  --min-epochs-per-subject 100
```

If your labels are not already in `0..4`, remap them explicitly:

```bash
python main/preprocessing/psg/sleepgpt_preprocessing.py \
  --input-root /path/to/FilteredRoot \
  --output-root /path/to/ProcessedPSG \
  --split-output /path/to/ProcessedPSG/psg_split.npy \
  --label-map "0:0,1:1,2:2,3:3,4:4,5:-1,6:-1,7:-1"
```

Outputs:

```text
ProcessedPSG/
  manifest.json
  psg_split.npy
  subject_001/
    data.h5
    meta.json
```

## 4. Fine-Tune with Cross-Validation

Use a pretrained SleepGPT backbone checkpoint as initialization:

```bash
python train_psg_stage_kfold.py \
  --processed-root /path/to/ProcessedPSG \
  --split-file psg_split.npy \
  --pretrain-ckpt /path/to/pretrained_sleepgpt.ckpt \
  --output-dir /path/to/psg_stage_runs \
  --kfold 5 \
  --time-size 100 \
  --batch-size 8 \
  --max-epoch 30
```

Per fold, the script writes:

```text
psg_stage_runs/
  fold_0/
    best.ckpt
    last.ckpt
    metrics.json
  ...
  cross_validation_metrics.json
```

## 5. Evaluate Saved Checkpoints Only

After training, you can recompute validation and test metrics without retraining.

If the checkpoints are already under `output-dir/fold_k/best.ckpt`:

```bash
python train_psg_stage_kfold.py \
  --processed-root /path/to/ProcessedPSG \
  --split-file psg_split.npy \
  --output-dir /path/to/psg_stage_runs \
  --kfold 5 \
  --eval-only
```

If you want to pass checkpoints explicitly:

```bash
python train_psg_stage_kfold.py \
  --processed-root /path/to/ProcessedPSG \
  --split-file psg_split.npy \
  --output-dir /path/to/psg_stage_eval \
  --kfold 5 \
  --eval-only \
  --checkpoint \
    /path/to/fold0.ckpt \
    /path/to/fold1.ckpt \
    /path/to/fold2.ckpt \
    /path/to/fold3.ckpt \
    /path/to/fold4.ckpt
```

Outputs:

- `fold_k/evaluation_metrics.json`
- `cross_validation_eval_metrics.json`

## 6. Operational Notes

- `time_size=100` means each subject should have at least 100 valid epochs after trimming and label filtering.
- Subjects shorter than `--min-epochs-per-subject` are excluded when `psg_split.npy` is built.
- Missing canonical channels do not break training; they are masked by `good_channels`.
- The converter derives `EOG` from right-minus-left when both ROC and LOC are available.