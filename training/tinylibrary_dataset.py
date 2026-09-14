"""TinyLibrary dataset adapters for nanoVLM.

TinyLibraryRaw yields FineVision-schema items ({'images': [PIL], 'texts':
[{'user','assistant'}]}) that nanoVLM's VQADataset consumes unchanged.
OrderedView materializes a curriculum order over it.
"""
import json

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # scraped book scans; tolerate short files


def load_manifest(path):
    return [json.loads(l) for l in open(path)]


def split_train_val(records, val_fraction=0.02, seed=42):
    """Deterministic held-out split, identical across all curriculum conditions."""
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(records))
    n_val = int(len(records) * val_fraction)
    val_idx = set(perm[:n_val].tolist())
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


class TinyLibraryRaw(Dataset):
    def __init__(self, records, image_root):
        self.records = records
        self.image_root = image_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(f"{self.image_root}/{rec['image']}")
        return {"images": [image], "texts": rec["texts"]}


class TinyLibraryTextRaw(Dataset):
    """Text-only control: identical text/order to TinyLibraryRaw but yields NO
    images, so nanoVLM trains the decoder on the same words with no image tokens
    (no disk/PIL cost). Used to assess whether visual inputs shift the linguistic
    results under the tested setup."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {"images": [], "texts": self.records[idx]["texts"]}


class OrderedView(Dataset):
    """Reindex a dataset by a precomputed (multi-epoch) curriculum sequence."""

    def __init__(self, base, order_seq):
        self.base = base
        self.order_seq = order_seq

    def __len__(self):
        return len(self.order_seq)

    def __getitem__(self, idx):
        return self.base[int(self.order_seq[idx])]
