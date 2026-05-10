import gzip
import struct

import numpy as np

from neural_network_from_scratch.data import load_mnist_images, load_mnist_labels, one_hot_encode


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


def test_load_mnist_images_rejects_wrong_dimensions(tmp_path):
    path = tmp_path / "images.gz"

    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, 1, 14, 14))
        handle.write(bytes(14 * 14))

    try:
        load_mnist_images(path)
    except ValueError as error:
        assert "Expected 28x28 images" in str(error)
    else:
        raise AssertionError("Expected invalid dimensions to raise ValueError")


def test_one_hot_encode_rejects_out_of_range_labels():
    try:
        one_hot_encode(np.array([0, 10]))
    except ValueError as error:
        assert "Labels must be in range" in str(error)
    else:
        raise AssertionError("Expected out-of-range label to raise ValueError")
