# Flow Trace: Important Runtime Flows

## Summary

Important flows are interactive design/build/train/infer/evaluate/checkpoint workflows and standard training.

## Entry Points

- `nnfs`
- `nnfs-train`
- `nnfs-visual`

## Step-by-Step Flow

| Step | File | Function/Class | What Happens | Inputs | Outputs | Side Effects |
|---|---|---|---|---|---|---|
| 1 | `cli.py` | `add_hidden_layer()` / `change_hidden_layer()` / `remove_hidden_layer()` | Mutates hidden layer list and resets model state. | Prompted layer counts. | Updated `AppState`. | Clears network/training metrics. |
| 2 | `cli.py` | `build_network()` | Builds `NeuralNetwork(hidden_layer_dims=...)`. | Hidden layer list. | Network. | Random parameter initialization. |
| 3 | `cli.py` | `train_current_model()` | Prompts hyperparameters, loads data, trains with live dashboard. | Epochs, batch size, learning rate, sample limits. | Updated model and history. | Reads data, mutates network, terminal rendering. |
| 4 | `cli.py` | `run_inference()` | Loads test data and predicts a sample. | Sample index. | Digit render and probability table. | Reads data. |
| 5 | `cli.py` | `evaluate_current_model()` | Loads test data and computes accuracy. | Test sample count. | Evaluation accuracy. | Reads data. |
| 6 | `cli.py` / `checkpoints.py` | save/load checkpoint | Writes or reads `.npz` checkpoint. | Path. | Persisted or loaded network. | File IO. |
| 7 | `train.py` | `main()` | Full standard training from settings. | Data files/settings. | Epoch and test accuracy prints. | Reads data, mutates network. |
| 8 | `visual_train.py` | `main()` | Visual one-shot training with args/prompts. | Args/prompts. | Live dashboard. | Reads data, mutates network. |

## Error Handling

- CLI catches broad errors and returns to menu.
- Visual command validates numeric args but not file existence before loaders.
- Standard train lets exceptions propagate.

## Tests

- Unit tests plus scan-time smoke commands.
- No full prompt automation tests in repository.

## Risks

- High: Three training flow implementations can diverge.
- Medium: Fixed default watched sample can make users misunderstand training behavior.
- Medium: Data reload on every CLI action is inefficient.

## Unknowns

- Unknown: Whether the interactive CLI should become the only supported primary training flow.

## Confidence

Confirmed.
