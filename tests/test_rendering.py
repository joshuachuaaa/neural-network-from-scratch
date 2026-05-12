import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from neural_network_from_scratch.network import NeuralNetwork
from neural_network_from_scratch.rendering import VisualMetrics, dashboard, layer_table, probability_table, render_digit, sparkline
from neural_network_from_scratch.visual_train import layer_update_norms


def test_render_digit_returns_14_compact_lines():
    rendered = render_digit(np.zeros((28, 28)))

    assert isinstance(rendered, Text)
    assert len(rendered.plain.splitlines()) == 14


def test_layer_and_probability_tables_are_rich_tables():
    network = NeuralNetwork(hidden_layers=1, hidden_layer_dim=4)
    network.predict(np.zeros((1, 784)))

    assert isinstance(layer_table(network), Table)
    assert isinstance(probability_table(np.ones(10) / 10, label=3), Table)


def test_sparkline_limits_width():
    assert len(sparkline(range(100), width=10)) == 10


def test_sparkline_shows_constant_positive_values():
    assert sparkline([2.3]) == "█"


def test_dashboard_is_rich_group():
    network = NeuralNetwork(hidden_layers=1, hidden_layer_dim=4)
    probabilities = network.predict(np.zeros((1, 784)))[0]
    metrics = VisualMetrics(
        epoch=1,
        epochs=1,
        batch=1,
        batches=1,
        loss=2.3,
        accuracy=0.1,
        sample_label=0,
        sample_prediction=int(np.argmax(probabilities)),
    )

    rendered = dashboard(network, np.zeros((28, 28)), 0, probabilities, metrics)

    assert isinstance(rendered, Group)


def test_layer_update_norms_reports_non_input_layers():
    network = NeuralNetwork(hidden_layers=1, hidden_layer_dim=4)
    batch = np.zeros((2, 784))
    labels = np.eye(10)[[0, 1]]

    network.predict(batch)
    network.backProp(labels)
    norms = layer_update_norms(network, learning_rate=0.1)

    assert set(norms) == {1, 2}
    assert all(value >= 0 for value in norms.values())
