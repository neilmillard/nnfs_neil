import numpy as np


class Dense:
    """
    A Dense Neural Network Layer.
    :arg n_inputs: Number of inputs accepted by this layer
    :arg n_neurons: Number of neurons in this layer
    """
    def __init__(self, n_inputs=2, n_neurons=4):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = []

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
