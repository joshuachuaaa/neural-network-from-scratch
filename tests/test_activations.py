import numpy as np

from neural_network_from_scratch.activations import ReLU, Softmax


def test_relu_activation_and_mask():
    values = np.array([[-2.0, 0.0, 3.0]])

    np.testing.assert_array_equal(ReLU.activate(values), np.array([[0.0, 0.0, 3.0]]))
    np.testing.assert_array_equal(ReLU.getActiveNeurons(values), np.array([[0, 0, 1]]))


def test_softmax_is_row_wise_and_stable():
    values = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])

    probabilities = Softmax.activate(values)

    assert probabilities.shape == values.shape
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(probabilities[0], probabilities[1], rtol=1e-12)

