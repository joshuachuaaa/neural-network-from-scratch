# Neural Network from Scratch

This repository implements a small fully connected neural network for MNIST digit classification using NumPy. The code is organized as an educational project: activation functions, layer state, network orchestration, data loading, and training are separated into focused modules.

## Project Structure

```text
.
├── data/mnist/                         # MNIST gzip files
├── src/neural_network_from_scratch/
│   ├── activations.py                  # ReLU and Softmax
│   ├── data.py                         # MNIST IDX gzip loading and validation
│   ├── layers.py                       # Layer state, forward pass, parameter updates
│   ├── network.py                      # Network topology, predict, backpropagation
│   ├── settings.py                     # Model and training constants
│   └── train.py                        # Training/evaluation orchestration
├── tests/                              # Unit tests and smoke coverage
├── main.py                             # Backwards-compatible script entrypoint
├── pyproject.toml
└── requirements.txt
```

## Runtime Flow

1. `main.py` calls `neural_network_from_scratch.train.main()`.
2. `data.py` loads and validates MNIST gzip files from `data/mnist/`.
3. `Network` builds an input layer, configured hidden layers, and an output layer.
4. `predict()` performs the forward pass.
5. `backProp()` computes batch gradients for all non-input layers.
6. `Layer.updateValues()` applies gradient descent updates.
7. Training prints epoch loss/accuracy, then batched test accuracy.

## Model

- Input dimension: `28 * 28 = 784`
- Output classes: `10`
- Hidden layers: configured in `settings.py`
- Hidden activation: ReLU
- Output activation: Softmax
- Loss: cross-entropy
- Optimization: mini-batch gradient descent

## Setup

```bash
python -m pip install -r requirements.txt
```

For editable package usage:

```bash
python -m pip install -e ".[dev]"
```

## Data

The repository expects MNIST gzip files under `data/mnist/`:

```text
data/mnist/train-images-idx3-ubyte.gz
data/mnist/train-labels-idx1-ubyte.gz
data/mnist/t10k-images-idx3-ubyte.gz
data/mnist/t10k-labels-idx1-ubyte.gz
```

The loader validates IDX magic numbers, image dimensions, byte counts, and label ranges before training.

To verify the tracked dataset files against the checksum manifest:

```bash
python scripts/verify_mnist.py
```

## Run

From the repository root:

```bash
python main.py
```

After editable install:

```bash
nnfs-train
```

## Terminal Visualization

Launch the interactive terminal lab:

```bash
PYTHONPATH=src python -m neural_network_from_scratch.cli
```

After editable install:

```bash
nnfs
```

The main menu lets you add/remove hidden layers, change neuron counts, build or reset the model, train with a live dashboard, evaluate, run inference on MNIST samples, and save/load checkpoints. Editing the architecture resets the current model so training and inference always match the architecture shown on the main screen.

Run a Rich-powered terminal dashboard that shows:

- the configured network layers
- sampled layer activations
- weight mean/standard deviation per layer
- per-batch update magnitude
- a rendered MNIST sample
- live output probabilities during training

Quick visual smoke run:

```bash
PYTHONPATH=src python -m neural_network_from_scratch.visual_train \
  --epochs 1 \
  --limit-train 128 \
  --limit-test 32 \
  --batch-size 32 \
  --hidden-layers 2 \
  --hidden-neurons 32
```

Interactive prompt mode:

```bash
PYTHONPATH=src python -m neural_network_from_scratch.visual_train --interactive
```

After editable install, use:

```bash
nnfs-visual --interactive
```

## Test

```bash
pytest
```

The test suite covers activations, MNIST loading validation, network topology, forward output shape, backpropagation gradient shapes, one training update, batching, and evaluation helpers.
