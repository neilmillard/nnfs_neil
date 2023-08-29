import numpy as np

from utils import one_hot


class Dense:
    """
    A Dense Neural Network Layer.
    :arg n_inputs: Number of inputs accepted by this layer
    :arg n_neurons: Number of neurons in this layer
    """
    def __init__(self, n_inputs=2, n_neurons=4, activation_function=None):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = []
        self.activation_output = []
        self.activation_function = activation_function

    def forward(self, X_inputs):
        self.output = np.dot(X_inputs, self.weights) + self.biases
        self.activation_output = self.activation_function.forward(self.output)

    def backward(self, Y_answers):
        m = Y_answers.size
        one_hot_Y = one_hot(Y_answers)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
