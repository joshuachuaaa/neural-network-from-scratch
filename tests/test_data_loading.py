import gzip
import struct

import numpy as np

from neural_network_from_scratch.train import load_mnist_images, load_mnist_labels


def test_load_mnist_images_reads_and_normalizes_idx_gzip(tmp_path):
    path = tmp_path / "images.gz"
    pixels = np.arange(2 * 28 * 28, dtype=np.uint8)

    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, 2, 28, 28))
        handle.write(pixels.tobytes())

    images = load_mnist_images(path)

    assert images.shape == (2, 784)
    np.testing.assert_allclose(images[0, 1], 1 / 255.0)


def test_load_mnist_labels_reads_idx_gzip(tmp_path):
    path = tmp_path / "labels.gz"

    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, 3))
        handle.write(bytes([7, 2, 9]))

    labels = load_mnist_labels(path)

    np.testing.assert_array_equal(labels, np.array([7, 2, 9], dtype=np.uint8))

