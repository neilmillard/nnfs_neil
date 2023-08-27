import matplotlib.pyplot as plt
import numpy as np


def f(_x):
    return 2*_x**2


def approximate_tangent_line(_x, approx_derivative, _b):
    return approx_derivative*_x + _b


x = np.arange(0, 50, 0.001)
y = f(x)

plt.plot(x, y)

colours = ['k', 'g', 'r', 'b', 'c']

for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1+p2_delta

    y1 = f(x1)
    y2 = f(x2)
    print((x1, y1), (x2, y2))

    approximate_derivative = (y2-y1)/(x2-x1)
    # b is the y = mx + b: formula for a line
    b = y2 - approximate_derivative*x2

    to_plot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1, y1, c=colours[i])
    plt.plot(to_plot,
             [approximate_tangent_line(point, approximate_derivative, b)
                for point in to_plot],
             c=colours[i])

    print('Approx derivative for f(x)',
          f'where x = {x1} is {approximate_derivative}')

plt.show()
