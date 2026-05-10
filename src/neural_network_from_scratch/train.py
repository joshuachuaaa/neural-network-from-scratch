import numpy as np
import struct
import gzip
from pathlib import Path

from neural_network_from_scratch.network import NeuralNetwork
from neural_network_from_scratch import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mnist"

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 784)
        return images / 255.0  # Normalize to [0, 1]

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

def calculate_loss(predictions, targets):
    """Calculate the cross-entropy loss."""
    return -np.sum(targets * np.log(predictions + 1e-9)) / targets.shape[0]

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
    
    # Train the network using mini-batch gradient descent
    num_batches = len(X_train) // settings.BATCH_SIZE
    
    for epoch in range(settings.EPOCHS):
        epoch_loss = 0
        correct_predictions = 0
        
        for batch in range(num_batches):
            start = batch * settings.BATCH_SIZE
            end = start + settings.BATCH_SIZE
            X_batch = X_train[start:end]
            y_batch = y_train_encoded[start:end]
            
            # Forward and backward pass for each mini-batch
            predictions = nn.predict(X_batch)
            nn.backProp(y_batch)
            for layer in nn.layer_array:
                layer.updateValues(settings.LEARNING_RATE)
            
            # Calculate loss
            batch_loss = calculate_loss(predictions, y_batch)
            epoch_loss += batch_loss
            
            # Calculate accuracy
            correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))
        
        # Print epoch progress
        epoch_accuracy = correct_predictions / len(X_train)
        print(epoch_loss)
        print(f"Epoch {epoch + 1} complete - Loss: {epoch_loss / num_batches:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")
    
    # Evaluate the network
    correct_predictions = 0
    for i in range(len(X_test)):
        prediction = nn.predict(X_test[i:i+1])
        if np.argmax(prediction) == y_test[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main() 
