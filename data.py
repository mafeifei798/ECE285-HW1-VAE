"""Load the official splits, with duplicate removal within the training set."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, root, split, image_size=64, cache_dir=None):
        self.split = split
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        cache = Path(cache_dir) if cache_dir else None
        matches = sorted(cache.glob(f"{split}-{image_size}*.npy")) if cache else []
        if len(matches) > 1:
            raise ValueError(f"Multiple {split} caches found in {cache}")
        if matches:
            self.images = np.load(matches[0], mmap_mode="r", allow_pickle=False)
            if self.images.dtype != np.uint8 or self.images.shape[1:] != (image_size, image_size):
                raise ValueError(f"Unexpected cache shape or dtype: {matches[0]}")
            return

        root = Path(root).expanduser()
        if not (root / "train").is_dir():
            root = root / "chest_xray"
        files = [path for label in ("NORMAL", "PNEUMONIA")
                 for path in sorted((root / split / label).glob("*"))
                 if not path.name.startswith(".") and path.suffix.lower() in (".jpeg", ".jpg", ".png")]
        if not files:
            raise FileNotFoundError(f"No images in {root / split}")
        images, seen = [], set()
        for path in files:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("L")
                if split == "train":
                    # Compare original grayscale pixels, before resizing.
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                images.append(np.array(image.resize(
                    (image_size, image_size), Image.Resampling.BILINEAR)))
        self.images = np.stack(images)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / f"{split}-{image_size}.npy", self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.images[index], copy=True)).unsqueeze(0).float().div_(255)
