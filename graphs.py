import matplotlib.pyplot as plt
import numpy as np


def f(_x):
    # return _x * _x - 3 * _x + 4
    return 0.2 * _x ** 4 + 0.1 * _x ** 3 - _x ** 2 + 2


def approximate_tangent_line(_x, _approx_derivative, _b):
    return _approx_derivative * _x + _b


x = np.arange(-3, 3, 0.001)
y = f(x)

plt.plot(x, y)

colours = ['k', 'g', 'r', 'b', 'c']


def approx_derivative(_i):
    p2_delta = 0.0001
    x1 = _i
    x2 = x1 + p2_delta
    y1 = f(x1)
    y2 = f(x2)
    print((x1, y1), (x2, y2))
    slope = (y2 - y1) / (x2 - x1)
    # b is the y = mx + b: formula for a line
    b = y2 - slope * x2
    draw_line(_i, slope, b, x1, y1)
    print('Approx derivative for f(x)',
          f'where x = {x1} is {slope}')


def draw_line(colour_index, _slope, b, x1, y1):
    to_plot = [x1 - 0.9, x1 + 0.9]
    plt.scatter(x1, y1, c=colours[colour_index])  # Plot the point
    plt.plot(to_plot,
             [approximate_tangent_line(point, _slope, b)
              for point in to_plot],
             c=colours[colour_index])


for i in range(5):
    approx_derivative(i)

plt.show()
