import numpy as np

from neural_network_from_scratch import settings
from neural_network_from_scratch.layers import LayerType
from neural_network_from_scratch.network import NeuralNetwork


def test_predict_returns_batch_by_output_dimension():
    network = NeuralNetwork()

    predictions = network.predict(np.zeros((3, settings.IN_DIMS)))

    assert predictions.shape == (3, settings.OUT_DIM)
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(3))


def test_hidden_layer_count_matches_settings():
    network = NeuralNetwork()

    hidden_layers = [layer for layer in network.layer_array if layer.layerType is LayerType.HIDDEN]

    assert len(hidden_layers) == settings.HIDDEN_LAYERS


def test_network_accepts_per_layer_hidden_dimensions():
    network = NeuralNetwork(hidden_layer_dims=[8, 4])

    hidden_layers = [layer for layer in network.layer_array if layer.layerType is LayerType.HIDDEN]

    assert [layer.neuronDim for layer in hidden_layers] == [8, 4]
    assert network.predict(np.zeros((2, settings.IN_DIMS))).shape == (2, settings.OUT_DIM)


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


def test_one_training_step_updates_non_input_weights():
    network = NeuralNetwork()
    batch = np.random.default_rng(0).normal(size=(2, settings.IN_DIMS))
    labels = np.eye(settings.OUT_DIM)[[0, 1]]
    network.predict(batch)

    before = [
        layer.weights.copy()
        for layer in network.layer_array
        if layer.layerType is not LayerType.INPUT
    ]

    network.backProp(labels)
    for layer in network.layer_array:
        layer.updateValues(settings.LEARNING_RATE)

    after = [
        layer.weights
        for layer in network.layer_array
        if layer.layerType is not LayerType.INPUT
    ]

    assert any(not np.array_equal(old, new) for old, new in zip(before, after))
