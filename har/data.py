"""Video datasets: UCF101 / HMDB51-style folders, and a synthetic motion set for testing."""
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}


def sample_indices(n_total: int, num_frames: int, train: bool, rng: random.Random) -> list[int]:
    """TSN-style segment sampling: one frame per equal segment.

    Training picks a random frame inside each segment; evaluation takes the centre.
    """
    n_total = max(n_total, 1)
    seg = n_total / num_frames
    idx = []
    for i in range(num_frames):
        start = seg * i
        offset = rng.uniform(0, seg) if train else seg / 2
        idx.append(min(int(start + offset), n_total - 1))
    return idx


def read_video_frames(path: Path, num_frames: int, train: bool, rng: random.Random) -> np.ndarray:
    """Decode `num_frames` RGB frames from a video file -> uint8 array (T, H, W, 3)."""
    import cv2  # imported lazily so the rest of the package works without OpenCV

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:  # some containers do not report a frame count: decode everything
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        picked = [frames[i] for i in sample_indices(len(frames), num_frames, train, rng)]
    else:
        wanted = sample_indices(total, num_frames, train, rng)
        needed = set(wanted)
        decoded = {}
        last = None
        for i in range(max(wanted) + 1):  # sequential read is faster than seeking for AVI
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
            if i in needed:
                decoded[i] = frame
        cap.release()
        if last is None:
            raise RuntimeError(f"no frames decoded from {path}")
        # frame counts in headers can overshoot; fall back to the last decoded frame
        picked = [decoded.get(i, last) for i in wanted]
    return np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in picked])


def to_clip_tensor(frames: np.ndarray, img_size: int, train: bool, rng: random.Random) -> torch.Tensor:
    """uint8 (T, H, W, 3) -> normalised float (T, 3, img_size, img_size)."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
    h, w = x.shape[-2:]
    short = int(round(img_size * 1.15))
    scale = short / min(h, w)
    x = F.interpolate(x, size=(max(short, round(h * scale)), max(short, round(w * scale))),
                      mode="bilinear", align_corners=False, antialias=True)
    h, w = x.shape[-2:]
    if train:
        top, left = rng.randint(0, h - img_size), rng.randint(0, w - img_size)
    else:
        top, left = (h - img_size) // 2, (w - img_size) // 2
    x = x[..., top:top + img_size, left:left + img_size]
    if train and rng.random() < 0.5:
        x = x.flip(-1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class VideoClipDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], classes: list[str],
                 num_frames: int = 16, img_size: int = 112, train: bool = False):
        self.samples, self.classes = samples, classes
        self.num_frames, self.img_size, self.train = num_frames, img_size, train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        rng = random.Random() if self.train else random.Random(i)
        frames = read_video_frames(path, self.num_frames, self.train, rng)
        return to_clip_tensor(frames, self.img_size, self.train, rng), label


def _keep_classes(classes: list[str], max_classes: int | None) -> list[str]:
    return classes[:max_classes] if max_classes else classes


def ucf101_samples(root: str, split_dir: str, split: int = 1, max_classes: int | None = None):
    """Read the official UCF101 split files (classInd.txt, trainlistNN.txt, testlistNN.txt)."""
    root, split_dir = Path(root), Path(split_dir)
    classes = []
    for line in (split_dir / "classInd.txt").read_text().split("\n"):
        if line.strip():
            classes.append(line.split()[1])
    classes = _keep_classes(classes, max_classes)
    index = {c: i for i, c in enumerate(classes)}

    def parse(name: str):
        out = []
        for line in (split_dir / name).read_text().split("\n"):
            if not line.strip():
                continue
            rel = line.split()[0]
            cls = rel.split("/")[0]
            if cls not in index:
                continue
            path = root / rel
            if not path.exists():  # some mirrors store videos without class sub-folders
                path = root / Path(rel).name
            out.append((path, index[cls]))
        return out

    return parse(f"trainlist{split:02d}.txt"), parse(f"testlist{split:02d}.txt"), classes


_GROUP = re.compile(r"_g(\d+)_c\d+", re.IGNORECASE)


def folder_samples(root: str, val_ratio: float = 0.2, max_classes: int | None = None, seed: int = 0):
    """Folder-per-class layout (root/<class>/<video>) with a reproducible train/val split.

    UCF101-style names (v_Action_g08_c01.avi) are split by group, so clips of the
    same actor/scene never land in both train and val.
    """
    root = Path(root)
    classes = _keep_classes(sorted(p.name for p in root.iterdir() if p.is_dir()), max_classes)
    rng = random.Random(seed)
    train, val = [], []
    for label, cls in enumerate(classes):
        vids = sorted(p for p in (root / cls).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
        groups: dict[str, list[Path]] = {}
        for v in vids:
            m = _GROUP.search(v.stem)
            groups.setdefault(m.group(1) if m else v.stem, []).append(v)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_ratio)) if len(keys) > 1 else 0
        for k_i, k in enumerate(keys):
            (val if k_i < n_val else train).extend((v, label) for v in groups[k])
    return train, val, classes


class SyntheticMotionDataset(Dataset):
    """A square moving left/right/up/down over noise; the class is the direction.

    Every frame looks alike, so only a model that uses frame ORDER can solve it.
    It needs no download and is used for smoke tests.
    """

    classes = ["left", "right", "up", "down"]

    def __init__(self, size: int = 400, num_frames: int = 8, img_size: int = 64, seed: int = 0):
        self.size, self.num_frames, self.img_size, self.seed = size, num_frames, img_size, seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        label = i % 4
        s, t, sq = self.img_size, self.num_frames, self.img_size // 4
        clip = torch.rand(t, 3, s, s, generator=g) * 0.3
        color = torch.rand(3, 1, 1, generator=g) * 0.6 + 0.4
        cross = int(torch.randint(0, s - sq, (1,), generator=g))
        for f in range(t):
            p = round(f * (s - sq) / (t - 1))
            p = {0: s - sq - p, 1: p, 2: s - sq - p, 3: p}[label]
            y, x = (cross, p) if label < 2 else (p, cross)
            clip[f, :, y:y + sq, x:x + sq] = color
        return (clip - IMAGENET_MEAN) / IMAGENET_STD, label


def build_datasets(dataset: str, data_root: str | None = None, split_dir: str | None = None,
                   split: int = 1, max_classes: int | None = None, num_frames: int = 16,
                   img_size: int = 112, val_ratio: float = 0.2, seed: int = 0):
    """Return (train_set, val_set, class_names)."""
    if dataset == "synthetic":
        tr = SyntheticMotionDataset(800, num_frames, img_size, seed=1)
        va = SyntheticMotionDataset(200, num_frames, img_size, seed=2)
        return tr, va, SyntheticMotionDataset.classes
    if not data_root:
        raise ValueError("--data-root is required for real datasets")
    if dataset == "ucf101":
        if not split_dir:
            raise ValueError("--split-dir (folder with classInd.txt) is required for ucf101")
        tr, va, classes = ucf101_samples(data_root, split_dir, split, max_classes)
    elif dataset == "folder":
        tr, va, classes = folder_samples(data_root, val_ratio, max_classes, seed)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if not tr or not va:
        raise ValueError(f"empty split: {len(tr)} train / {len(va)} val videos found under {data_root}")
    return (VideoClipDataset(tr, classes, num_frames, img_size, train=True),
            VideoClipDataset(va, classes, num_frames, img_size, train=False), classes)
