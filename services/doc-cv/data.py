"""Generates synthetic 'document-like' images for doc-type classification training.
Each class gets a distinct layout/texture pattern so a small CNN can learn to
distinguish them. Deterministic seed for reproducibility. No real scanned
documents are used - this is a structural proxy, swap in real labeled scans
for production use."""
import numpy as np

CLASSES = ["id_card", "pay_stub", "bank_statement", "other"]
IMG_SIZE = 64


def _base(rng, bg, n):
    return np.tile(np.array(bg, dtype=np.float32), (n, IMG_SIZE, IMG_SIZE, 1)) / 255.0


def generate(n_per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    imgs, labels = [], []

    for idx, cls in enumerate(CLASSES):
        bg = _base(rng, [200, 200, 210], n_per_class) if cls == "id_card" else \
             _base(rng, [255, 255, 255], n_per_class) if cls == "pay_stub" else \
             _base(rng, [235, 245, 255], n_per_class) if cls == "bank_statement" else \
             _base(rng, [180, 180, 180], n_per_class)

        # add class-specific horizontal "text line" bands + noise so patterns differ
        n_bands = {"id_card": 3, "pay_stub": 6, "bank_statement": 10, "other": 1}[cls]
        for i in range(n_per_class):
            for b in range(n_bands):
                y = rng.integers(4, IMG_SIZE - 4)
                bg[i, y:y + 2, 4:IMG_SIZE - 4, :] -= 0.3
            bg[i] += rng.normal(0, 0.03, bg[i].shape)

        imgs.append(bg.clip(0, 1))
        labels.extend([idx] * n_per_class)

    X = np.concatenate(imgs, axis=0).astype(np.float32).transpose(0, 3, 1, 2)  # NCHW
    y = np.array(labels, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]