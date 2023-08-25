import nnfs
from nnfs.datasets import spiral_data

import Activation
import Layer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def spiral():
    # Example of a classifier
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer.Dense(2, 3)
    activation1 = Activation.ReLU()

    dense2 = Layer.Dense(3, 3)
    activation2 = Activation.Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    spiral()