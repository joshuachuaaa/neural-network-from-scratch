import numpy as np
from enum import Enum

from neural_network_from_scratch.activations import ReLU, Softmax

class LayerType(Enum):
    INPUT = "Input"
    OUTPUT = "Output"
    HIDDEN = "Hidden"

class Layer : 

    def __init__( self, inputDim:int, neuronDim:int, layerType:LayerType):
        """
        Initialization of Layer Class
        Type of Layer determined by activation type
        """
        self.layerType = layerType
        self.neuronDim = neuronDim

        if self.layerType is not LayerType.INPUT:

            self.weights = np.random.randn(inputDim, neuronDim) * 0.01
            self.biases = np.zeros((1, neuronDim))

            self.errorVector = np.zeros((1, neuronDim))
            self.gradientMatrix = np.zeros((inputDim, neuronDim))

        # For sake of clarity,
        self.input = None
        self.activatedNeurons = np.zeros((1, neuronDim))
        self.boolActiveNeurons = np.zeros((1, neuronDim))
        

    def forward(self, X):
        """
        X shape: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        # If Input Layer, Do Nothing
        if self.layerType == LayerType.INPUT:
            return X
        
        # Calculate Neuron Value if Hidden or Output Layer
        self.input = X
        preActivatedNeurons = (X @ self.weights) + self.biases
        activatedNeurons = self._activate(preActivatedNeurons)

        # ReLU Mask
        self.boolActiveNeurons = self.getActiveNeurons()

        # Debugging: Print summary statistics of activations
        #print(f"Layer {self.layerType} - Activated Neurons Mean: {np.mean(activatedNeurons):.4f}, Std: {np.std(activatedNeurons):.4f}")
        
        return activatedNeurons


    def _activate(self, preActivatedNeurons):
        """Activate Neurons"""
        
        if self.layerType is LayerType.HIDDEN:
            self.activatedNeurons = ReLU.activate(preActivatedNeurons)

        elif self.layerType is LayerType.OUTPUT:
            self.activatedNeurons = Softmax.activate(preActivatedNeurons)

        return self.activatedNeurons
    
    def getActiveNeurons(self):
        """Return 1 for active neurons and 0 for inactive Neurons"""
        return ReLU.getActiveNeurons(self.activatedNeurons)
        


    def updateValues(self, learning_rate: int):
        """Update the weights and bias values based on the learning rate"""
        
        # Skip updating if it's an input layer
        if self.layerType == LayerType.INPUT:
            return

        # Debugging: Print summary statistics of weights and biases before update
        #print(f"Before update - Weights Mean: {np.mean(self.weights):.4f}, Std: {np.std(self.weights):.4f}")
        #print(f"Before update - Biases Mean: {np.mean(self.biases):.4f}, Std: {np.std(self.biases):.4f}")

        # Update the biases
        self.biases -= learning_rate * self.errorVector

        # Update the weights
        self.weights -= learning_rate * self.gradientMatrix

        # Debugging: Print summary statistics of weights and biases after update
        #print(f"After update - Weights Mean: {np.mean(self.weights):.4f}, Std: {np.std(self.weights):.4f}")
        #print(f"After update - Biases Mean: {np.mean(self.biases):.4f}, Std: {np.std(self.biases):.4f}")













        

        



        
