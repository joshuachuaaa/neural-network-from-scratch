import numpy as np
import pytest

from neural_network_from_scratch import settings
from neural_network_from_scratch.layers import LayerType
from neural_network_from_scratch.network import NeuralNetwork


def test_predict_returns_batch_by_output_dimension():
    network = NeuralNetwork()

    predictions = network.predict(np.zeros((3, settings.IN_DIMS)))

    assert predictions.shape == (3, settings.OUT_DIM)
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(3))


@pytest.mark.xfail(reason="Current constructor creates HIDDEN_LAYERS - 1 hidden layers.")
def test_hidden_layer_count_matches_settings():
    network = NeuralNetwork()

    hidden_layers = [layer for layer in network.layer_array if layer.layerType is LayerType.HIDDEN]

    assert len(hidden_layers) == settings.HIDDEN_LAYERS


@pytest.mark.xfail(reason="Current backpropagation has incompatible batch matrix shapes.")
def test_backprop_populates_gradients_matching_weight_shapes():
    network = NeuralNetwork()
    batch = np.zeros((2, settings.IN_DIMS))
    labels = np.eye(settings.OUT_DIM)[[0, 1]]

    network.predict(batch)
    network.backProp(labels)

    for layer in network.layer_array:
        if layer.layerType is LayerType.INPUT:
            continue
        assert layer.gradientMatrix.shape == layer.weights.shape
        assert layer.errorVector.shape == layer.biases.shape

