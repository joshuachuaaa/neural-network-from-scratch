from enum import Enum
import numpy as np

class Softmax:
    """
    This class is meant to be used for the final neuron values so that the final activation 
    values of the final layer of the output layer all equates to 1, essentially
    turning the output values into probabilities.
    """
    @staticmethod
    def activate(neuron_array):
 
        #This is to make sure that the values aren't too big when doing the exponentitaion
        adjusted_neuron_values = np.exp(neuron_array - np.max(neuron_array, axis=1, keepdims=True))  # Stability improvement
        
        #Turning the array of neuron values into probabilities
        return adjusted_neuron_values / np.sum(adjusted_neuron_values, axis=1, keepdims=True)

class ReLU:

    @staticmethod
    def activate(neuron_vector):
        """Applies ReLU activation using np.maximium"""
        return np.maximum(0, neuron_vector)
    
    @staticmethod
    def getActiveNeurons(neuron_vector):
        """Check if Neuron is active, assign 1 or 0"""
        return np.where(neuron_vector > 0, 1, 0)
        


