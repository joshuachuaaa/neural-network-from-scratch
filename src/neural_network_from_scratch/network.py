import numpy as np
from typing import List

from neural_network_from_scratch import settings
from neural_network_from_scratch.layers import Layer, LayerType

class NeuralNetwork:
    """
    Neural Network class -> Serves as a wrapper for all the layers
    Create input layer -> hidden layers -> output layer
    """

    def __init__(self, hidden_layers=None, hidden_layer_dim=None, hidden_layer_dims=None):

        # Declare number of neurons in the input & output layers
        self.input_dims = settings.IN_DIMS
        self.output_dims = settings.OUT_DIM

        if hidden_layer_dims is None:
            hidden_layers = settings.HIDDEN_LAYERS if hidden_layers is None else hidden_layers
            hidden_layer_dim = settings.HIDDEN_LAYER_DIM if hidden_layer_dim is None else hidden_layer_dim
            hidden_layer_dims = [hidden_layer_dim] * hidden_layers

        self.hidden_layer_dims = list(hidden_layer_dims)
        self.hidden_layers = len(self.hidden_layer_dims)
        self.hidden_layers_dim = self.hidden_layer_dims[0] if self.hidden_layer_dims else 0

        # To Store all the layers within a single array
        self.layer_array : List[Layer] = []

        # Create input and add input layer to array
        self.input_layer: Layer = Layer(None, self.input_dims, LayerType.INPUT)
        self.layer_array.append(self.input_layer)

        # Create Hidden Layers
        for layer_dim in self.hidden_layer_dims:
            previousLayerDim = self.layer_array[-1].neuronDim
            new_hidden_layer = Layer(previousLayerDim, layer_dim, LayerType.HIDDEN)
            self.layer_array.append(new_hidden_layer)

        # Creating Output Layer
        previousLayerDim = self.layer_array[-1].neuronDim
        self.output_layer : Layer = Layer(previousLayerDim, self.output_dims, LayerType.OUTPUT)

        # Add it to Array
        self.layer_array.append(self.output_layer)

    def predict(self, X):
        """To feed data forward and get the predicted result"""
        out = X
        for layer in self.layer_array:
            out = layer.forward(out)
        return out


    def backProp(self, y_batch):
        """Back propagate the error for training and weight/bias adjustment"""
        batch_size = y_batch.shape[0]
        delta = self._calcFinalError(y_batch) / batch_size

        for idx in range(len(self.layer_array) - 1, 0, -1):
            layer = self.layer_array[idx]
            previousLayer = self.layer_array[idx - 1]

            layer.errorVector = np.sum(delta, axis=0, keepdims=True)
            layer.biasGradient = layer.errorVector
            layer.gradientMatrix = self._calcGradientMatrix(delta, previousLayer)

            if previousLayer.layerType is not LayerType.INPUT:
                delta = self.calcErrorTerm(delta, previousLayer, layer)

        return
        
    def _calcFinalError(self, rightVals):
        """Find the error term in the output"""
        return self.layer_array[-1].activatedNeurons - rightVals
    
    def _calcGradientMatrix(self, delta, prevLayer):
        """Returns the gradient matrix"""
        return np.transpose(prevLayer.activatedNeurons) @ delta
    
    def calcErrorTerm(self, delta, layer: Layer, nextLayer: Layer):
        """Find Error term for layer"""
        return (delta @ np.transpose(nextLayer.weights)) * layer.boolActiveNeurons
