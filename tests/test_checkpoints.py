import numpy as np

from neural_network_from_scratch.checkpoints import load_checkpoint, save_checkpoint
from neural_network_from_scratch.network import NeuralNetwork


def test_checkpoint_round_trip_preserves_architecture_and_predictions(tmp_path):
    network = NeuralNetwork(hidden_layer_dims=[8, 4])
    batch = np.random.default_rng(0).normal(size=(3, 784))
    before = network.predict(batch)

    checkpoint_path = save_checkpoint(network, tmp_path / "model.npz")
    loaded = load_checkpoint(checkpoint_path)
    after = loaded.predict(batch)

    assert loaded.hidden_layer_dims == [8, 4]
    np.testing.assert_allclose(after, before)
