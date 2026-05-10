from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich import box
from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from neural_network_from_scratch.layers import LayerType

BLOCKS = " ▁▂▃▄▅▆▇█"
DIGIT_PIXELS = " .:-=+*#%@"


@dataclass
class VisualMetrics:
    epoch: int
    epochs: int
    batch: int
    batches: int
    loss: float
    accuracy: float
    sample_label: int
    sample_prediction: int


def render_digit(image) -> Text:
    pixels = np.asarray(image).reshape(28, 28)
    pixels = pixels.reshape(14, 2, 14, 2).mean(axis=(1, 3))
    text = Text()
    for row in pixels:
        for value in row:
            text.append(DIGIT_PIXELS[min(int(value * (len(DIGIT_PIXELS) - 1)), len(DIGIT_PIXELS) - 1)])
        text.append("\n")
    return text


def _bar(value: float, width: int = 18) -> str:
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "█" * filled + " " * (width - filled)


def _sparkline(values, width: int = 24) -> str:
    array = np.asarray(values).ravel()
    if array.size == 0:
        return ""
    if array.size > width:
        sample_indices = np.linspace(0, array.size - 1, width).astype(int)
        array = array[sample_indices]
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value == min_value:
        scaled = np.zeros_like(array, dtype=int)
    else:
        scaled = np.round((array - min_value) / (max_value - min_value) * (len(BLOCKS) - 1)).astype(int)
    return "".join(BLOCKS[index] for index in scaled)


def layer_table(network, update_norms=None) -> Table:
    table = Table(title="Layers, Activations, Weights", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right")
    table.add_column("Type")
    table.add_column("Neurons", justify="right")
    table.add_column("Activation sample")
    table.add_column("Act μ/max", justify="right")
    table.add_column("Weight μ/σ", justify="right")
    table.add_column("Δ update", justify="right")

    update_norms = update_norms or {}
    for index, layer in enumerate(network.layer_array):
        activations = np.asarray(layer.activatedNeurons)
        activation_mean = float(np.mean(activations)) if activations.size else 0.0
        activation_max = float(np.max(activations)) if activations.size else 0.0

        if layer.layerType is LayerType.INPUT:
            weight_summary = "-"
            update_summary = "-"
        else:
            weight_summary = f"{np.mean(layer.weights):+.4f}/{np.std(layer.weights):.4f}"
            update_summary = f"{update_norms.get(index, 0.0):.2e}"

        table.add_row(
            str(index),
            layer.layerType.value,
            str(layer.neuronDim),
            _sparkline(activations[0] if activations.ndim > 1 else activations),
            f"{activation_mean:.3f}/{activation_max:.3f}",
            weight_summary,
            update_summary,
        )

    return table


def probability_table(probabilities, label: int) -> Table:
    probabilities = np.asarray(probabilities).ravel()
    prediction = int(np.argmax(probabilities))

    table = Table(title=f"Output Probabilities | label={label} prediction={prediction}", box=box.SIMPLE)
    table.add_column("Digit", justify="right")
    table.add_column("Probability")
    table.add_column("Value", justify="right")

    for digit, probability in enumerate(probabilities):
        style = "bold green" if digit == prediction else ""
        if digit == label:
            style = "bold cyan" if digit != prediction else "bold green"
        table.add_row(str(digit), _bar(float(probability)), f"{probability:.3f}", style=style)

    return table


def metrics_panel(metrics: VisualMetrics) -> Panel:
    text = Text()
    text.append(f"Epoch {metrics.epoch}/{metrics.epochs}  ", style="bold")
    text.append(f"Batch {metrics.batch}/{metrics.batches}\n", style="bold")
    text.append(f"Loss: {metrics.loss:.4f}\n", style="yellow")
    text.append(f"Accuracy: {metrics.accuracy * 100:.2f}%\n", style="green")
    text.append(f"Sample label: {metrics.sample_label}  Prediction: {metrics.sample_prediction}")
    return Panel(text, title="Training", border_style="blue")


def dashboard(network, sample_image, sample_label: int, probabilities, metrics: VisualMetrics, update_norms=None) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=7),
    )
    layout["main"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=1))
    layout["right"].split_column(Layout(name="digit"), Layout(name="probabilities"))

    layout["header"].update(Panel(Align.center("[bold]Neural Network Terminal Visualizer[/bold]"), border_style="magenta"))
    layout["left"].update(Panel(layer_table(network, update_norms), title="Network", border_style="green"))
    layout["digit"].update(Panel(render_digit(sample_image), title="MNIST Sample", border_style="cyan"))
    layout["probabilities"].update(Panel(probability_table(probabilities, sample_label), title="Inference", border_style="yellow"))
    layout["footer"].update(metrics_panel(metrics))
    return layout
