# Flow Trace: Data Flow

## Summary

Confirmed: Data flows from tracked MNIST gzip files through IDX loaders into NumPy arrays, then through network forward/backprop and Rich renderers.

## Entry Points

- `train.main()`
- `visual_train.main()`
- `cli.train_current_model()`
- `cli.run_inference()`
- `cli.evaluate_current_model()`

## Step-by-Step Flow

| Step | File | Function/Class | What Happens | Inputs | Outputs | Side Effects |
|---|---|---|---|---|---|---|
| 1 | `data.py` | `load_mnist_images()` | Opens gzip, validates magic/dimensions/count, normalizes pixels. | IDX image gzip. | `float64` array `(n, 784)`. | File read. |
| 2 | `data.py` | `load_mnist_labels()` | Opens gzip, validates magic/count. | IDX label gzip. | `uint8` labels. | File read. |
| 3 | `data.py` | `one_hot_encode()` | Validates label range and builds one-hot matrix. | Label array. | One-hot labels. | None. |
| 4 | `train.py` / `cli.py` / `visual_train.py` | training loops | Batches features/targets and predicts. | Feature/target arrays. | Predictions. | Network layer state changes. |
| 5 | `network.py` | `backProp()` | Computes gradients from labels and stored activations. | One-hot labels. | Gradient state. | Mutates layer gradients. |
| 6 | `layers.py` | `updateValues()` | Applies gradient descent. | Learning rate. | Updated weights/biases. | Mutates parameters. |
| 7 | `rendering.py` | `dashboard()` | Displays activations/probabilities/sample. | Network state, sample image, metrics. | Rich renderables. | Terminal rendering by caller. |

## Error Handling

- Data validation raises `ValueError`.
- Shape mismatches generally produce NumPy exceptions.

## Tests

- `tests/test_data_loading.py`
- `tests/test_network.py`
- `tests/test_training.py`
- `tests/test_rendering.py`

## Risks

- Data is reloaded repeatedly in CLI rather than cached.
- `float64` image arrays cost more memory than likely needed for MNIST demo.
- Training labels are range-validated by one-hot encoding; eval labels are not range-validated.

## Unknowns

- Unknown: Whether future datasets beyond MNIST are planned.

## Confidence

Confirmed.
