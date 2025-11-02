# SlowFast Age Rating Classifier

This project provides a training pipeline for a SlowFast video classification model that predicts film age-rating categories from short video clips. The implementation relies on PyTorch and PyTorchVideo.

## Project layout

```
src/
  config.py        # Training hyper-parameters and file paths
  dataset.py       # Dataset + preprocessing utilities
  model.py         # SlowFast backbone with custom head
  train.py         # Training script and evaluation loop
requirements.txt   # Python dependencies
```

## Dataset format

Prepare a single CSV metadata file (default `data/train_metadata.csv`) with the columns:

- `path` – relative or absolute path to each video file
- `label` – categorical (e.g. `G`, `PG-13`) or numeric label

Example CSV:

```
path,label
trailers/clip_0001.mp4,G
trailers/clip_0002.mp4,PG
```

Place all videos under `data/videos` by default or provide a different root with `--video-root`.
The trainer automatically splits the metadata into train/validation subsets (80/20 by default).

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training

PowerShell (Windows):

```powershell
python -m src.train `
  --train-csv data/train_metadata.csv `
  --video-root data/videos `
  --output-dir runs/slowfast `
  --batch-size 4 `
  --max-epochs 30
```

Bash (Linux/macOS):

```bash
python -m src.train \
  --train-csv data/train_metadata.csv \
  --video-root data/videos \
  --output-dir runs/slowfast \
  --batch-size 4 \
  --max-epochs 30
```

run predict
python -m src.predict

Key CLI flags:

- `--alpha` and `--frames` adjust the SlowFast temporal sampling (default `alpha=4`, `frames=32`).
- `--device` can be `cuda` or `cpu`.
- `--val-split` controls the validation proportion (default `0.2`).
- `--checkpoint` loads a previously saved state before continuing training.

Models are saved to `output_dir` with `latest.pt` (last epoch) and `best.pt` (highest validation accuracy).

## Notes

- Mixed precision is enabled automatically on CUDA.
- If the effective validation split ends up empty (e.g., too few samples), the trainer still runs and only serialises `latest.pt`.
- Update `TrainConfig.class_names` in `src/config.py` to match your rating schema.
