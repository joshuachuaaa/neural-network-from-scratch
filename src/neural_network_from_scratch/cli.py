from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from neural_network_from_scratch import settings
from neural_network_from_scratch.checkpoints import load_checkpoint, save_checkpoint
from neural_network_from_scratch.data import load_mnist_images, load_mnist_labels, one_hot_encode
from neural_network_from_scratch.network import NeuralNetwork
from neural_network_from_scratch.rendering import VisualMetrics, dashboard, probability_table, render_digit, sparkline
from neural_network_from_scratch.train import DEFAULT_DATA_DIR, calculate_loss, evaluate_accuracy, iter_batches
from neural_network_from_scratch.visual_train import layer_update_norms


@dataclass
class TrainingPoint:
    epoch: int
    loss: float
    accuracy: float


@dataclass
class AppState:
    hidden_layers: list[int] = field(default_factory=lambda: [settings.HIDDEN_LAYER_DIM] * settings.HIDDEN_LAYERS)
    network: NeuralNetwork | None = None
    trained_epochs: int = 0
    last_loss: float | None = None
    last_accuracy: float | None = None
    last_eval_accuracy: float | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_path: Path | None = None
    training_history: list[TrainingPoint] = field(default_factory=list)


def architecture_summary(hidden_layers: list[int]) -> str:
    parts = [f"Input({settings.IN_DIMS})"]
    parts.extend(f"Hidden({neurons})" for neurons in hidden_layers)
    parts.append(f"Output({settings.OUT_DIM})")
    return " -> ".join(parts)


def layer_dimensions(hidden_layers: list[int]) -> list[int]:
    return [settings.IN_DIMS, *hidden_layers, settings.OUT_DIM]


def layer_parameter_count(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


def parameter_count(hidden_layers: list[int]) -> int:
    dims = layer_dimensions(hidden_layers)
    return sum(layer_parameter_count(previous, current) for previous, current in zip(dims, dims[1:]))


def build_network(hidden_layers: list[int]) -> NeuralNetwork:
    return NeuralNetwork(hidden_layer_dims=hidden_layers)


def architecture_table(state: AppState) -> Table:
    table = Table(title="Current Architecture", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right")
    table.add_column("Layer")
    table.add_column("Neurons", justify="right")
    table.add_column("Params in", justify="right")

    table.add_row("0", "Input", str(settings.IN_DIMS), "-")
    previous_dim = settings.IN_DIMS
    for index, neurons in enumerate(state.hidden_layers, start=1):
        table.add_row(str(index), "Hidden", str(neurons), f"{layer_parameter_count(previous_dim, neurons):,}")
        previous_dim = neurons
    table.add_row(
        str(len(state.hidden_layers) + 1),
        "Output",
        str(settings.OUT_DIM),
        f"{layer_parameter_count(previous_dim, settings.OUT_DIM):,}",
    )
    return table


def status_table(state: AppState) -> Table:
    table = Table(title="Session State", box=box.SIMPLE)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Model", "[green]built[/green]" if state.network is not None else "[yellow]not built[/yellow]")
    table.add_row("Parameters", f"{parameter_count(state.hidden_layers):,}")
    table.add_row("Trained epochs", str(state.trained_epochs))
    table.add_row("Last train loss", "-" if state.last_loss is None else f"{state.last_loss:.4f}")
    table.add_row("Last train accuracy", "-" if state.last_accuracy is None else f"{state.last_accuracy * 100:.2f}%")
    table.add_row("Last eval accuracy", "-" if state.last_eval_accuracy is None else f"{state.last_eval_accuracy * 100:.2f}%")
    table.add_row("Checkpoint", "-" if state.checkpoint_path is None else str(state.checkpoint_path))
    return table


def menu_table() -> Table:
    table = Table(title="Main Menu", box=box.ROUNDED)
    table.add_column("Key", justify="right", style="bold cyan")
    table.add_column("Area", style="dim")
    table.add_column("Action")
    table.add_row("1", "Design", "Add hidden layer")
    table.add_row("2", "Design", "Change hidden layer neurons")
    table.add_row("3", "Design", "Remove hidden layer")
    table.add_row("4", "Model", "Build/reset model")
    table.add_row("5", "Run", "Train current model")
    table.add_row("6", "Run", "Run inference on MNIST sample")
    table.add_row("7", "Run", "Evaluate current model")
    table.add_row("8", "Artifact", "Save checkpoint")
    table.add_row("9", "Artifact", "Load checkpoint")
    table.add_row("10", "View", "Refresh current model screen")
    table.add_row("0", "System", "Exit")
    return table


def compact_neuron_row(neurons: int, width: int = 12) -> str:
    visible = min(neurons, width)
    dots = " ".join("●" for _ in range(visible))
    if neurons > visible:
        dots = f"{dots}  +{neurons - visible}"
    return dots


def architecture_visual(state: AppState) -> Panel:
    layer_names = ["Input", *[f"Hidden {index}" for index in range(1, len(state.hidden_layers) + 1)], "Output"]
    layer_sizes = layer_dimensions(state.hidden_layers)
    layer_styles = ["cyan", *["green" for _ in state.hidden_layers], "magenta"]
    text = Text()

    for index, (name, neurons, style) in enumerate(zip(layer_names, layer_sizes, layer_styles)):
        text.append(f"[{index}] ", style="dim")
        text.append(f"{name:<9}", style=f"bold {style}")
        text.append(f"{neurons:>5}  ", style="bold")
        text.append(compact_neuron_row(neurons), style=style)
        if index > 0:
            params = layer_parameter_count(layer_sizes[index - 1], neurons)
            text.append(f"  {params:,} params", style="dim")
        if index < len(layer_sizes) - 1:
            text.append("\n      │\n      ▼\n", style="dim")

    return Panel(text, title="Network Map", border_style="green")


def training_history_panel(state: AppState) -> Panel:
    if not state.training_history:
        empty = Text("No training run yet.\nUse action 5 to train and watch the live dashboard.", style="dim")
        return Panel(empty, title="Training History", border_style="blue")

    recent = state.training_history[-20:]
    losses = [point.loss for point in recent]
    accuracies = [point.accuracy for point in recent]
    latest = recent[-1]

    text = Text()
    text.append("Loss      ", style="bold yellow")
    text.append(sparkline(losses, width=28), style="yellow")
    text.append(f"  {latest.loss:.4f}\n", style="yellow")
    text.append("Accuracy  ", style="bold green")
    text.append(sparkline(accuracies, width=28), style="green")
    text.append(f"  {latest.accuracy * 100:.2f}%\n", style="green")
    text.append(f"Latest epoch: {latest.epoch}", style="dim")
    return Panel(text, title="Training History", border_style="blue")


def header_panel(state: AppState) -> Panel:
    title = Text()
    title.append("nnfs", style="bold magenta")
    title.append("  Neural Network From Scratch", style="bold white")
    title.append("\n")
    title.append(architecture_summary(state.hidden_layers), style="dim")
    return Panel(Align.left(title), subtitle="interactive Rich terminal lab", border_style="magenta")


def main_screen(state: AppState) -> Group:
    return Group(
        header_panel(state),
        Columns(
            [
                architecture_visual(state),
                Group(status_table(state), training_history_panel(state)),
            ],
            equal=False,
            expand=True,
        ),
        architecture_table(state),
        menu_table(),
    )


def reset_run_state(state: AppState, clear_checkpoint: bool = True):
    state.trained_epochs = 0
    state.last_loss = None
    state.last_accuracy = None
    state.last_eval_accuracy = None
    state.training_history.clear()
    if clear_checkpoint:
        state.checkpoint_path = None


def mark_architecture_changed(state: AppState):
    state.network = None
    reset_run_state(state, clear_checkpoint=True)


def add_hidden_layer(state: AppState):
    neurons = IntPrompt.ask("Neurons in new hidden layer", default=settings.HIDDEN_LAYER_DIM)
    if neurons < 1:
        raise ValueError("Hidden layer must have at least one neuron")
    state.hidden_layers.append(neurons)
    mark_architecture_changed(state)


def change_hidden_layer(state: AppState):
    if not state.hidden_layers:
        raise ValueError("There are no hidden layers to change")
    layer_number = IntPrompt.ask("Hidden layer number", default=1)
    if layer_number < 1 or layer_number > len(state.hidden_layers):
        raise ValueError("Hidden layer number is out of range")
    neurons = IntPrompt.ask("New neuron count", default=state.hidden_layers[layer_number - 1])
    if neurons < 1:
        raise ValueError("Hidden layer must have at least one neuron")
    state.hidden_layers[layer_number - 1] = neurons
    mark_architecture_changed(state)


def remove_hidden_layer(state: AppState):
    if not state.hidden_layers:
        raise ValueError("There are no hidden layers to remove")
    layer_number = IntPrompt.ask("Hidden layer number to remove", default=len(state.hidden_layers))
    if layer_number < 1 or layer_number > len(state.hidden_layers):
        raise ValueError("Hidden layer number is out of range")
    del state.hidden_layers[layer_number - 1]
    mark_architecture_changed(state)


def ensure_network(state: AppState):
    if state.network is None:
        state.network = build_network(state.hidden_layers)


def load_data(state: AppState, limit_train: int | None = None, limit_test: int | None = None):
    X_train = load_mnist_images(state.data_dir / "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(state.data_dir / "train-labels-idx1-ubyte.gz")
    X_test = load_mnist_images(state.data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(state.data_dir / "t10k-labels-idx1-ubyte.gz")

    if limit_train is not None:
        X_train = X_train[:limit_train]
        y_train = y_train[:limit_train]
    if limit_test is not None:
        X_test = X_test[:limit_test]
        y_test = y_test[:limit_test]
    return X_train, y_train, X_test, y_test


def train_current_model(state: AppState, console: Console):
    ensure_network(state)
    epochs = IntPrompt.ask("Epochs", default=1)
    batch_size = IntPrompt.ask("Batch size", default=settings.BATCH_SIZE)
    learning_rate = FloatPrompt.ask("Learning rate", default=settings.LEARNING_RATE)
    limit_train = IntPrompt.ask("Training samples", default=512)
    limit_test = IntPrompt.ask("Test samples", default=128)
    sample_index = IntPrompt.ask("Sample index to watch", default=0)

    if epochs < 1 or batch_size < 1 or limit_train < 1 or limit_test < 1:
        raise ValueError("Epochs, batch size, and sample limits must be positive")

    X_train, y_train, X_test, y_test = load_data(state, limit_train=limit_train, limit_test=limit_test)
    y_train_encoded = one_hot_encode(y_train)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    batches = int(np.ceil(len(X_train) / batch_size))
    starting_epoch = state.trained_epochs
    sample_index = max(0, min(sample_index, len(X_test) - 1))
    sample_image = X_test[sample_index]
    sample_label = int(y_test[sample_index])
    probabilities = state.network.predict(sample_image.reshape(1, -1))[0]
    metrics = VisualMetrics(0, epochs, 0, batches, 0.0, 0.0, sample_label, int(np.argmax(probabilities)))

    with Live(dashboard(state.network, sample_image, sample_label, probabilities, metrics), console=console, refresh_per_second=6) as live:
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            correct = 0
            seen = 0
            for batch_number, (X_batch, y_batch) in enumerate(
                iter_batches(X_train, y_train_encoded, batch_size, rng=rng),
                start=1,
            ):
                predictions = state.network.predict(X_batch)
                state.network.backProp(y_batch)
                update_norms = layer_update_norms(state.network, learning_rate)
                for layer in state.network.layer_array:
                    layer.updateValues(learning_rate)

                batch_count = len(X_batch)
                running_loss += calculate_loss(predictions, y_batch) * batch_count
                correct += int(np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1)))
                seen += batch_count

                probabilities = state.network.predict(sample_image.reshape(1, -1))[0]
                metrics = VisualMetrics(
                    epoch,
                    epochs,
                    batch_number,
                    batches,
                    running_loss / seen,
                    correct / seen,
                    sample_label,
                    int(np.argmax(probabilities)),
                )
                live.update(dashboard(state.network, sample_image, sample_label, probabilities, metrics, update_norms))

            state.training_history.append(TrainingPoint(starting_epoch + epoch, metrics.loss, metrics.accuracy))

    state.trained_epochs += epochs
    state.last_loss = metrics.loss
    state.last_accuracy = metrics.accuracy


def run_inference(state: AppState, console: Console):
    ensure_network(state)
    _, _, X_test, y_test = load_data(state)
    sample_index = IntPrompt.ask("Test sample index", default=0)
    sample_index = max(0, min(sample_index, len(X_test) - 1))
    sample_image = X_test[sample_index]
    sample_label = int(y_test[sample_index])
    probabilities = state.network.predict(sample_image.reshape(1, -1))[0]

    console.print(Panel(render_digit(sample_image), title=f"MNIST sample #{sample_index} | label={sample_label}", border_style="cyan"))
    console.print(probability_table(probabilities, sample_label))


def evaluate_current_model(state: AppState, console: Console):
    ensure_network(state)
    limit_test = IntPrompt.ask("Test samples", default=512)
    _, _, X_test, y_test = load_data(state, limit_test=limit_test)
    state.last_eval_accuracy = evaluate_accuracy(state.network, X_test, y_test, settings.BATCH_SIZE)
    console.print(f"[green]Evaluation accuracy:[/green] {state.last_eval_accuracy * 100:.2f}%")


def save_current_checkpoint(state: AppState, console: Console):
    ensure_network(state)
    default_path = Path("artifacts") / "model.npz"
    checkpoint_path = Path(Prompt.ask("Checkpoint path", default=str(default_path)))
    state.checkpoint_path = save_checkpoint(state.network, checkpoint_path)
    console.print(f"[green]Saved checkpoint:[/green] {state.checkpoint_path}")


def load_model_checkpoint(state: AppState, console: Console):
    default_path = state.checkpoint_path or Path("artifacts") / "model.npz"
    checkpoint_path = Path(Prompt.ask("Checkpoint path", default=str(default_path)))
    state.network = load_checkpoint(checkpoint_path)
    state.hidden_layers = list(state.network.hidden_layer_dims)
    reset_run_state(state, clear_checkpoint=False)
    state.checkpoint_path = checkpoint_path
    console.print(f"[green]Loaded checkpoint:[/green] {checkpoint_path}")


def wait_for_user():
    Prompt.ask("\nPress Enter to continue", default="")


def run_app(console: Console | None = None):
    console = console or Console()
    state = AppState()

    while True:
        console.clear()
        console.print(main_screen(state))
        choice = Prompt.ask("Choose an action", choices=[str(number) for number in range(11)], default="0")

        try:
            if choice == "0":
                break
            if choice == "1":
                add_hidden_layer(state)
            elif choice == "2":
                change_hidden_layer(state)
            elif choice == "3":
                remove_hidden_layer(state)
            elif choice == "4":
                state.network = build_network(state.hidden_layers)
                reset_run_state(state, clear_checkpoint=True)
                console.print("[green]Model built.[/green]")
                wait_for_user()
            elif choice == "5":
                train_current_model(state, console)
                wait_for_user()
            elif choice == "6":
                run_inference(state, console)
                wait_for_user()
            elif choice == "7":
                evaluate_current_model(state, console)
                wait_for_user()
            elif choice == "8":
                save_current_checkpoint(state, console)
                wait_for_user()
            elif choice == "9":
                load_model_checkpoint(state, console)
                wait_for_user()
            elif choice == "10":
                wait_for_user()
        except Exception as error:
            console.print(f"[bold red]Error:[/bold red] {error}")
            wait_for_user()


def main():
    run_app()


if __name__ == "__main__":
    main()
