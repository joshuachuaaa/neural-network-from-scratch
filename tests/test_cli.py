from rich.console import Group
from rich.table import Table

from neural_network_from_scratch import settings
from neural_network_from_scratch.cli import AppState, architecture_summary, architecture_table, build_network, main_screen
from neural_network_from_scratch.layers import LayerType


def test_architecture_summary_lists_layers():
    assert architecture_summary([32, 16]) == f"Input({settings.IN_DIMS}) -> Hidden(32) -> Hidden(16) -> Output({settings.OUT_DIM})"


def test_build_network_uses_variable_layer_sizes():
    network = build_network([12, 6])

    hidden_layers = [layer for layer in network.layer_array if layer.layerType is LayerType.HIDDEN]

    assert [layer.neuronDim for layer in hidden_layers] == [12, 6]


def test_main_screen_and_architecture_table_are_rich_renderables():
    state = AppState(hidden_layers=[24])

    assert isinstance(architecture_table(state), Table)
    assert isinstance(main_screen(state), Group)
