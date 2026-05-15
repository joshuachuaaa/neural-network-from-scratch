# Reader Report: Neural Network Core

## Scope

`src/neural_network_from_scratch/activations.py`, `src/neural_network_from_scratch/layers.py`, `src/neural_network_from_scratch/network.py`.

## Component Purpose

Implements the from-scratch fully connected network: activations, layer state, forward pass, gradient storage, parameter updates, and backpropagation.

## Files Reviewed

| File | Status | Notes |
|---|---|---|
| `activations.py` | reviewed | `Softmax`, `ReLU`. |
| `layers.py` | reviewed | `LayerType`, `Layer`. |
| `network.py` | reviewed | `NeuralNetwork`. |

## File-by-File Review

### `activations.py`

- Purpose: Activation functions.
- Key symbols: `Softmax.activate`, `ReLU.activate`, `ReLU.getActiveNeurons`.
- Inputs/outputs: NumPy arrays in, NumPy arrays out.
- Error handling: None explicit.
- Tests: `tests/test_activations.py`.
- Important details: Softmax subtracts per-row max before exponentiation.
- Risks/unknowns: Naming `getActiveNeurons` is Java-style and returns mask from activated values.
- Confidence: Confirmed.

### `layers.py`

- Purpose: Represents layer parameters and activation state.
- Key symbols: `LayerType`, `Layer.__init__`, `Layer.forward`, `Layer._activate`, `Layer.getActiveNeurons`, `Layer.updateValues`.
- Inputs: Previous layer dimension, neuron dimension, layer type, input arrays, learning rate.
- Outputs: Activated neurons; mutates layer state and parameters.
- Internal dependencies: `ReLU`, `Softmax`.
- Side effects: Initializes random weights with `np.random.randn(inputDim, neuronDim) * 0.01`; updates `weights` and `biases`.
- Error handling: None for shape mismatch or invalid layer type beyond implicit NumPy errors.
- Tests: indirectly through `tests/test_network.py`.
- Security-sensitive areas: none.
- Performance-sensitive areas: matrix multiplication and full state retention per layer.
- Risks/unknowns: Weight initialization is not seed-injected and not ideal for deep ReLU networks; global random state harms reproducibility.
- Confidence: Confirmed.

### `network.py`

- Purpose: Wires layers and implements prediction/backpropagation.
- Key symbols: `NeuralNetwork.__init__`, `predict`, `backProp`, `_calcFinalError`, `_calcGradientMatrix`, `calcErrorTerm`.
- Inputs: hidden layer config, feature batches, one-hot labels.
- Outputs: class probability matrix; mutates layer gradients.
- Internal dependencies: `settings`, `Layer`, `LayerType`.
- Side effects: Builds random layer parameters.
- Error handling: None explicit for input/label shapes.
- Tests: `tests/test_network.py`.
- Important details: Backprop uses softmax + cross-entropy simplification via `activatedNeurons - y_batch`; divides delta by batch size.
- Risks/unknowns: Method names and attributes are inconsistent with Python style (`backProp`, `layer_array`, `neuronDim`); no explicit config/model object; no seed/RNG injection.
- Confidence: Confirmed.

## Key Flows

`NeuralNetwork.__init__` -> creates input layer -> hidden layers -> output layer. `predict()` passes data through each `Layer.forward()`. `backProp()` traverses layers backwards, calculating gradients for trainable layers.

## Data Flow

Feature matrix -> input layer state -> hidden ReLU activations -> output softmax probabilities -> gradients -> `Layer.updateValues()`.

## Component API / Boundaries

- Public API: `NeuralNetwork.predict`, `NeuralNetwork.backProp`, `Layer.updateValues`.
- Boundary issue: Callers must remember to call `predict()` before `backProp()` because gradients depend on stored activations.

## Duplication Observed

- None inside core beyond naming/style inconsistencies.

## Risks and Unknowns

- High: Model initialization is not reproducible even though training order uses `RANDOM_SEED`.
- Medium: Fixed small normal init can slow or weaken deeper ReLU training.
- Medium: No explicit shape validation for training data or labels.
- Medium: Stateful forward/backprop API is easy to misuse; `backProp()` relies on previous `predict()`.

## Summary for Orchestrator

The core is compact and educational. Highest-value improvements are seedable/He initialization, shape validation, Pythonic naming aliases or migration, and a single `train_batch()` abstraction to hide call-order hazards.
