from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.prompt import Confirm, IntPrompt

from neural_network_from_scratch import settings
from neural_network_from_scratch.data import load_mnist_images, load_mnist_labels, one_hot_encode
from neural_network_from_scratch.layers import LayerType
from neural_network_from_scratch.network import NeuralNetwork
from neural_network_from_scratch.rendering import VisualMetrics, dashboard
from neural_network_from_scratch.train import DEFAULT_DATA_DIR, calculate_loss, iter_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Rich terminal visualization of neural-network training.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=settings.LEARNING_RATE)
    parser.add_argument("--hidden-layers", type=int, default=settings.HIDDEN_LAYERS)
    parser.add_argument("--hidden-neurons", type=int, default=settings.HIDDEN_LAYER_DIM)
    parser.add_argument("--limit-train", type=int, default=512)
    parser.add_argument("--limit-test", type=int, default=128)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--refresh-every", type=int, default=1)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.hidden_layers < 0:
        raise SystemExit("--hidden-layers must be 0 or greater")
    if args.hidden_neurons < 1:
        raise SystemExit("--hidden-neurons must be at least 1")
    if args.limit_train < 1:
        raise SystemExit("--limit-train must be at least 1")
    if args.limit_test < 1:
        raise SystemExit("--limit-test must be at least 1")
    if args.refresh_every < 1:
        raise SystemExit("--refresh-every must be at least 1")
    return args


def apply_interactive_prompts(args):
    console = Console()
    console.print("[bold]Configure the visual training run[/bold]")
    args.hidden_layers = IntPrompt.ask("Hidden layers", default=args.hidden_layers)
    args.hidden_neurons = IntPrompt.ask("Neurons per hidden layer", default=args.hidden_neurons)
    args.epochs = IntPrompt.ask("Epochs", default=args.epochs)
    args.batch_size = IntPrompt.ask("Batch size", default=args.batch_size)
    args.limit_train = IntPrompt.ask("Training samples to use", default=args.limit_train)
    args.limit_test = IntPrompt.ask("Test samples to use", default=args.limit_test)
    args.sample_index = IntPrompt.ask("Sample index to visualize", default=args.sample_index)
    if not Confirm.ask("Start visual training?", default=True):
        raise SystemExit(0)
    return args


def layer_update_norms(network, learning_rate):
    norms = {}
    for index, layer in enumerate(network.layer_array):
        if layer.layerType is LayerType.INPUT:
            continue
        weight_update = learning_rate * layer.gradientMatrix
        bias_update = learning_rate * layer.biasGradient
        norms[index] = float(np.linalg.norm(weight_update) + np.linalg.norm(bias_update))
    return norms


def main():
    args = validate_args(parse_args())
    if args.interactive:
        args = validate_args(apply_interactive_prompts(args))

    data_dir = Path(args.data_dir)
    X_train = load_mnist_images(data_dir / "train-images-idx3-ubyte.gz")[:args.limit_train]
    y_train = load_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")[:args.limit_train]
    X_test = load_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")[:args.limit_test]
    y_test = load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")[:args.limit_test]
    y_train_encoded = one_hot_encode(y_train)

    network = NeuralNetwork(hidden_layers=args.hidden_layers, hidden_layer_dim=args.hidden_neurons)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    sample_index = max(0, min(args.sample_index, len(X_test) - 1))
    sample_image = X_test[sample_index]
    sample_label = int(y_test[sample_index])
    batches = int(np.ceil(len(X_train) / args.batch_size))
    probabilities = network.predict(sample_image.reshape(1, -1))[0]

    metrics = VisualMetrics(
        epoch=0,
        epochs=args.epochs,
        batch=0,
        batches=batches,
        loss=0.0,
        accuracy=0.0,
        sample_label=sample_label,
        sample_prediction=int(np.argmax(probabilities)),
    )

    with Live(dashboard(network, sample_image, sample_label, probabilities, metrics), refresh_per_second=6, screen=False) as live:
        for epoch in range(1, args.epochs + 1):
            running_loss = 0.0
            correct = 0
            seen = 0
            for batch_number, (X_batch, y_batch) in enumerate(
                iter_batches(X_train, y_train_encoded, args.batch_size, rng=rng),
                start=1,
            ):
                predictions = network.predict(X_batch)
                network.backProp(y_batch)
                update_norms = layer_update_norms(network, args.learning_rate)
                for layer in network.layer_array:
                    layer.updateValues(args.learning_rate)

                batch_count = len(X_batch)
                running_loss += calculate_loss(predictions, y_batch) * batch_count
                correct += int(np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1)))
                seen += batch_count

                probabilities = network.predict(sample_image.reshape(1, -1))[0]
                metrics = VisualMetrics(
                    epoch=epoch,
                    epochs=args.epochs,
                    batch=batch_number,
                    batches=batches,
                    loss=running_loss / seen,
                    accuracy=correct / seen,
                    sample_label=sample_label,
                    sample_prediction=int(np.argmax(probabilities)),
                )
                if batch_number % args.refresh_every == 0 or batch_number == batches:
                    live.update(dashboard(network, sample_image, sample_label, probabilities, metrics, update_norms))


if __name__ == "__main__":
    main()
