# Efficient HAR: Lightweight Hybrid CNN + Transformer for Human Action Recognition

A compact PyTorch codebase for video-based human action recognition that measures
**accuracy and computational cost together**. An ImageNet-pretrained 2D CNN encodes each
sampled frame; a temporal module (Transformer, LSTM or averaging) models how the action
unfolds over time.

Designed to run on **free Kaggle / Colab GPUs** and to benchmark inference on a plain CPU.

```
video clip (T frames) ──► 2D CNN per frame ──► projection ──► temporal module ──► action class
                         MobileNetV3 / EfficientNet           Transformer [CLS] / LSTM / mean
                         / ResNet-18 (ImageNet)
```

## Features

- **Backbones:** MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0, ResNet-18 (torchvision, ImageNet weights)
- **Temporal modules:** pre-norm Transformer encoder with a `[CLS]` token and sinusoidal positions, LSTM, or TSN-style averaging
- **Data:** UCF101 with official split files, any folder-per-class dataset (e.g. HMDB51) with a leakage-free group split, and a synthetic motion set for tests
- **Sampling:** TSN segment sampling (random within each segment for training, centre for evaluation)
- **Training:** AdamW, warm-up + cosine schedule, lower learning rate for the pretrained CNN, label smoothing, mixed precision on GPU
- **Evaluation:** top-1, top-5, macro F1, per-class recall, confusion matrix plot
- **Cost benchmark:** parameters, GFLOPs per clip, median CPU latency and clips/s for every backbone × temporal module × frame count

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                       # 5 fast tests, CPU only
python benchmark.py                       # cost table, no data needed
python train.py --dataset synthetic --frames 8 --img-size 64 --epochs 3 --no-pretrained
```

### UCF101

```bash
python train.py --dataset ucf101 --data-root /path/UCF-101 \
    --split-dir /path/ucfTrainTestlist --max-classes 10 --epochs 15 --out runs/ucf10_transformer
python evaluate.py --ckpt runs/ucf10_transformer/best.pt --dataset ucf101 \
    --data-root /path/UCF-101 --split-dir /path/ucfTrainTestlist --max-classes 10
```

Drop `--max-classes` for all 101 classes. For HMDB51 or your own videos, use
`--dataset folder --data-root /path/<class folders>`.

### Kaggle

Open [`kaggle_ucf101.ipynb`](kaggle_ucf101.ipynb) on Kaggle, enable a T4 GPU and internet,
add a UCF101 dataset, and run all cells. The notebook locates the data, runs the cost
benchmark, trains the Transformer and averaging models, evaluates them, and builds a
combined accuracy + cost table.

## Results

### Sanity check: synthetic motion (CPU, trained from scratch)

Each clip shows a square moving left, right, up or down. Every frame looks alike, so only
a model that uses **frame order** can solve it. 8 frames, 64 px, 3 epochs, 800/200 clips.

| Model | Params | Val top-1 |
|---|---|---|
| MobileNetV3-S + Transformer | 2.13 M | **100.0%** |
| MobileNetV3-S + average (TSN-style) | 1.08 M | 44.0% |

Averaging separates horizontal from vertical motion but cannot tell left from right
(about 50% at best), which shows why a temporal model matters.

### Computational cost (112 × 112 input, 101 classes)

`python benchmark.py --runs 5 --threads 4` on a laptop CPU. GFLOPs are multiply-accumulates
per clip, the convention used by TSM, X3D and MoViNet. Latency is indicative only: use more
`--runs` and a quiet machine for published numbers.

| Backbone | Temporal | Frames | Params (M) | GFLOPs | CPU ms/clip |
|---|---|---|---|---|---|
| MobileNetV3-S | avg | 8 | 1.10 | 0.12 | 57 |
| MobileNetV3-S | avg | 16 | 1.10 | 0.25 | 75 |
| MobileNetV3-S | LSTM | 16 | 2.15 | 0.25 | 112 |
| MobileNetV3-S | Transformer | 8 | 2.16 | 0.12 | 85 |
| MobileNetV3-S | Transformer | 16 | 2.16 | 0.25 | 97 |
| EfficientNet-B0 | Transformer | 8 | 5.42 | 0.85 | 321 |
| EfficientNet-B0 | Transformer | 16 | 5.42 | 1.71 | 889 |

The CNN backbone dominates the cost and grows linearly with the number of frames; the
Transformer adds about 1 M parameters and very few FLOPs. Reducing how many frames reach the
backbone is therefore the main lever for efficiency.

### UCF101

Run `kaggle_ucf101.ipynb` to reproduce; results are written to `runs/<name>/metrics.json`.

## Project layout

```
har/model.py      backbones, temporal modules, HybridHAR
har/data.py       video decoding, TSN sampling, UCF101 / folder / synthetic datasets
har/metrics.py    accuracy, F1, confusion matrix, params, GFLOPs, CPU latency
train.py          training loop
evaluate.py       metrics + confusion matrix for a checkpoint
benchmark.py      cost comparison table
tests/            fast CPU tests
```

## Implementation note

torchvision's MobileNetV3 and EfficientNet set BatchNorm momentum to 0.01, tuned for long
ImageNet training. In short fine-tuning runs the running statistics then lag behind the
data, and eval-mode accuracy can collapse to chance even while training loss falls. This
codebase resets BatchNorm momentum to 0.1 (`har/model.py`).

## References

- Simonyan & Zisserman, Two-Stream Convolutional Networks for Action Recognition, NeurIPS 2014
- Wang et al., Temporal Segment Networks, ECCV 2016
- Lin et al., TSM: Temporal Shift Module for Efficient Video Understanding, ICCV 2019
- Howard et al., Searching for MobileNetV3, ICCV 2019
- Bertasius et al., Is Space-Time Attention All You Need for Video Understanding? (TimeSformer), ICML 2021
- Soomro et al., UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild, 2012
