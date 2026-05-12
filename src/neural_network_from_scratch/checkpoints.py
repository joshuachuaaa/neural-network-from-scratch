from __future__ import annotations

from pathlib import Path

import numpy as np

from neural_network_from_scratch.layers import LayerType
from neural_network_from_scratch.network import NeuralNetwork


def save_checkpoint(network: NeuralNetwork, path) -> Path:
    """Save network architecture and parameters as a NumPy checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "hidden_layer_dims": np.array(network.hidden_layer_dims, dtype=np.int64),
    }
    trainable_layers = [layer for layer in network.layer_array if layer.layerType is not LayerType.INPUT]
    for index, layer in enumerate(trainable_layers):
        payload[f"weights_{index}"] = layer.weights
        payload[f"biases_{index}"] = layer.biases

    np.savez(checkpoint_path, **payload)
    return checkpoint_path


def load_checkpoint(path) -> NeuralNetwork:
    """Load a network checkpoint saved by `save_checkpoint`."""
    checkpoint_path = Path(path)
    with np.load(checkpoint_path, allow_pickle=False) as payload:
        hidden_layer_dims = payload["hidden_layer_dims"].astype(int).tolist()
        network = NeuralNetwork(hidden_layer_dims=hidden_layer_dims)
        trainable_layers = [layer for layer in network.layer_array if layer.layerType is not LayerType.INPUT]

        for index, layer in enumerate(trainable_layers):
            layer.weights = payload[f"weights_{index}"].copy()
            layer.biases = payload[f"biases_{index}"].copy()

    return network

