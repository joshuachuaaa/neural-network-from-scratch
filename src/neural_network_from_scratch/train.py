import numpy as np
from pathlib import Path

from neural_network_from_scratch.data import load_mnist_images, load_mnist_labels, one_hot_encode
from neural_network_from_scratch import settings
from neural_network_from_scratch.network import NeuralNetwork

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mnist"


def iter_batches(features, targets, batch_size, rng=None, shuffle=True):
    indices = np.arange(len(features))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield features[batch_indices], targets[batch_indices]

def calculate_loss(predictions, targets):
    """Calculate the cross-entropy loss."""
    return -np.sum(targets * np.log(predictions + 1e-9)) / targets.shape[0]


def train_epoch(network, X_train, y_train_encoded, batch_size, learning_rate, rng):
    epoch_loss = 0.0
    correct_predictions = 0
    samples_seen = 0

    for X_batch, y_batch in iter_batches(X_train, y_train_encoded, batch_size, rng=rng):
        predictions = network.predict(X_batch)
        network.backProp(y_batch)
        for layer in network.layer_array:
            layer.updateValues(learning_rate)

        batch_size_actual = len(X_batch)
        epoch_loss += calculate_loss(predictions, y_batch) * batch_size_actual
        correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))
        samples_seen += batch_size_actual

    return epoch_loss / samples_seen, correct_predictions / samples_seen


def evaluate_accuracy(network, X_test, y_test, batch_size):
    correct_predictions = 0

    for X_batch, y_batch in iter_batches(X_test, y_test, batch_size, shuffle=False):
        predictions = network.predict(X_batch)
        correct_predictions += np.sum(np.argmax(predictions, axis=1) == y_batch)

    return correct_predictions / len(y_test)

def main(data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)

    # Load and preprocess the data
    X_train = load_mnist_images(data_dir / 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(data_dir / 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(data_dir / 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(data_dir / 't10k-labels-idx1-ubyte.gz')
    
    y_train_encoded = one_hot_encode(y_train)

    # Initialize the neural network
    nn = NeuralNetwork()
    rng = np.random.default_rng(settings.RANDOM_SEED)
    
    for epoch in range(settings.EPOCHS):
        epoch_loss, epoch_accuracy = train_epoch(
            nn,
            X_train,
            y_train_encoded,
            settings.BATCH_SIZE,
            settings.LEARNING_RATE,
            rng,
        )
        print(f"Epoch {epoch + 1} complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")
    
    # Evaluate the network
    accuracy = evaluate_accuracy(nn, X_test, y_test, settings.BATCH_SIZE)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main() 
