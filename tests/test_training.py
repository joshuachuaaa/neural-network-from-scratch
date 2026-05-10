import numpy as np

from neural_network_from_scratch.train import evaluate_accuracy, iter_batches


class ConstantNetwork:
    def predict(self, features):
        predictions = np.zeros((len(features), 3))
        predictions[:, 1] = 1.0
        return predictions


def test_iter_batches_includes_remainder_without_shuffle():
    features = np.arange(10).reshape(10, 1)
    targets = np.arange(10)

    batches = list(iter_batches(features, targets, batch_size=4, shuffle=False))

    assert [len(batch_features) for batch_features, _ in batches] == [4, 4, 2]
    np.testing.assert_array_equal(batches[-1][0].ravel(), np.array([8, 9]))


def test_iter_batches_can_shuffle_with_seeded_rng():
    features = np.arange(6).reshape(6, 1)
    targets = np.arange(6)
    rng = np.random.default_rng(0)

    batches = list(iter_batches(features, targets, batch_size=3, rng=rng))
    observed = np.concatenate([batch_targets for _, batch_targets in batches])

    assert sorted(observed.tolist()) == list(range(6))
    assert observed.tolist() != list(range(6))


def test_evaluate_accuracy_uses_batches():
    network = ConstantNetwork()
    features = np.zeros((5, 2))
    labels = np.array([1, 0, 1, 2, 1])

    assert evaluate_accuracy(network, features, labels, batch_size=2) == 3 / 5

