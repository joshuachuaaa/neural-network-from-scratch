import gzip
import struct
from pathlib import Path

import numpy as np

from neural_network_from_scratch import settings

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


def load_mnist_images(filename):
    """Read IDX image data from a gzip file and normalize pixels to [0, 1]."""
    with gzip.open(Path(filename), "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != IMAGE_MAGIC:
            raise ValueError(f"Invalid MNIST image magic number: {magic}")
        if rows != settings.MNIST_IMAGE_ROWS or cols != settings.MNIST_IMAGE_COLS:
            raise ValueError(f"Expected {settings.MNIST_IMAGE_ROWS}x{settings.MNIST_IMAGE_COLS} images, got {rows}x{cols}")

        pixels = np.frombuffer(handle.read(), dtype=np.uint8)
        expected_values = count * settings.IN_DIMS
        if pixels.size != expected_values:
            raise ValueError(f"Expected {expected_values} image values, got {pixels.size}")

    return pixels.reshape(count, settings.IN_DIMS).astype(np.float64) / 255.0


def load_mnist_labels(filename):
    """Read IDX label data from a gzip file."""
    with gzip.open(Path(filename), "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != LABEL_MAGIC:
            raise ValueError(f"Invalid MNIST label magic number: {magic}")

        labels = np.frombuffer(handle.read(), dtype=np.uint8)
        if labels.size != count:
            raise ValueError(f"Expected {count} labels, got {labels.size}")

    return labels


def one_hot_encode(labels, num_classes=settings.OUT_DIM):
    """Convert integer class labels to a one-hot matrix."""
    labels = np.asarray(labels)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"Labels must be in range [0, {num_classes})")
    return np.eye(num_classes)[labels]

