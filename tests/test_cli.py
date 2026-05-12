from rich.panel import Panel
from rich.table import Table

from neural_network_from_scratch import settings
from neural_network_from_scratch.cli import (
    AppState,
    TrainingPoint,
    architecture_summary,
    architecture_table,
    architecture_visual,
    build_network,
    parameter_count,
    training_history_panel,
)
from neural_network_from_scratch.layers import LayerType


def test_architecture_summary_lists_layers():
    assert architecture_summary([32, 16]) == f"Input({settings.IN_DIMS}) -> Hidden(32) -> Hidden(16) -> Output({settings.OUT_DIM})"


def test_build_network_uses_variable_layer_sizes():
    network = build_network([12, 6])

    hidden_layers = [layer for layer in network.layer_array if layer.layerType is LayerType.HIDDEN]

    assert [layer.neuronDim for layer in hidden_layers] == [12, 6]


def test_parameter_count_includes_weights_and_biases():
    assert parameter_count([4]) == (settings.IN_DIMS * 4 + 4) + (4 * settings.OUT_DIM + settings.OUT_DIM)


def test_main_screen_parts_are_rich_renderables():
    state = AppState(hidden_layers=[24])

    assert isinstance(architecture_table(state), Table)
    assert isinstance(architecture_visual(state), Panel)
    assert isinstance(training_history_panel(state), Panel)


def test_training_history_panel_accepts_recent_metrics():
    state = AppState(training_history=[TrainingPoint(epoch=1, loss=2.0, accuracy=0.25)])

    assert isinstance(training_history_panel(state), Panel)
