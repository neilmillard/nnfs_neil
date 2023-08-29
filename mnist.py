import numpy as np

import activation
import layer
from utils import one_hot


def init_network():
    layer1 = layer.Dense(784, 10, activation.ReLU())
    layer2 = layer.Dense(10, 10, activation.Softmax())

    return layer1, layer2


def forward_prop(layer1, layer2, _X):
    layer1.forward(_X)
    # activation1 = activation.ReLU()
    # activation1.forward(layer1.output)
    Z1 = layer1.output
    A1 = layer1.activation_output
    layer2.forward(A1)
    # activation2 = activation.Softmax()
    # activation2.forward(layer2.output)
    Z2 = layer2.output
    A2 = layer2.activation_output

    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    This calculates the derivative of each Weight and bias in the network
    derivative of Weights and biases of both layers
    :param Z1 Output of first layer neuron
    :param A1 Output of first layer after activation function
    :param Z2 Output of second layer neuron
    :param A2 Output of second layer after activation function
    :param W1 Current weights of the first layer
    :param W2 Current weights of the second layer
    :param X Inputs
    :param Y Answers
    :return dW1, db1, dW2, db2
    """
    m = Y.size  # number of samples
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Return the updated weights and bias for a 2 layer network
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def back_prop(Z1, A1, Z2, A2, W2, Y):


    print(activation2.output[:5])

    loss_function = CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)

    print("Loss:", loss)


def main():
    # Get some data
    # we want the data as a column for inputs
    y_dev = []  # Y for testing Y = answers
    X_dev = []  # X for testing X = inputs

    y_train = []
    X_train = []

    dense1, dense2 = init_network()


main()