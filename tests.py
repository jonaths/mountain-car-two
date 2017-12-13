import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# def test():
#     items = 5
#     budget = 100
#
#     x1 = np.linspace(-0.06, 0.06, num=items)
#     x2 = np.linspace(-1.2, 0.6, num=items)
#
#     print x1
#     print x2
#
#     x1x, y1y = np.meshgrid(x1, x2)
#     print x1x
#     print y1y
#
#     def func(b, x1, x2):
#         return x1 * x2 + x1 * b
#
#     step = 10
#     # enumera del 10 a budget incluyedolo
#     bs = range(10, budget + step, step)
#
#     # el numero de filas que contiene el arreglo (una para cada combinacion b x1 x2)
#     length = items ** 2 * len(bs)
#
#     # el numero de columnas del arreglo (una para cada parametro y la ultima para y)
#     columns = 4
#
#     v = np.zeros((length, columns))
#
#     index = 0
#     for b in bs:
#         v[index * items ** 2: (index + 1) * items ** 2, 0] = np.full((1, items ** 2), b)
#         v[index * items ** 2: (index + 1) * items ** 2, 1] = x1x.ravel()
#         v[index * items ** 2: (index + 1) * items ** 2, 2] = y1y.ravel()
#         index += 1
#
#     print v.shape
#     print v
#
#     for r in v:
#         r[-1] = func(r[0], r[1], r[2])
#
#     new = False
#     print v.shape
#     print v
#
#     filename = 'values/' + 'filename' + '.npy'
#     try:
#         arr = np.load(filename)
#         print "arr found", arr.shape
#         pass
#     except IOError:
#         arr = np.zeros((1, length, 4))
#         arr[0] = v
#         new = True
#         print "arr created", arr.shape
#         pass
#
#     print arr.shape
#     print arr
#
#
#
#     if new:
#         print "if"
#         pass
#     else:
#         print "else"
#         print arr.shape
#         print v.shape
#         arr = np.append(arr, [v], axis=0)
#
#     print "arr:"
#     print arr.shape
#     np.save(filename, arr)
#
#
#
#
# for i in range(3):
#     test()

filename = 'values/' + 'b-0040_ep-0000' + '.npy'
arr = np.load(filename)

print "loaded file shape"
print arr.shape
# print arr

print "average shape"
arr = np.average(arr, axis=0)
print arr.shape
print arr

budgets_to_print = [40, 20]

for b in budgets_to_print:
    filter = arr[:, 0] == b
    print filter.shape

    v = arr[filter, :]
    print v.shape

    fig = plt.figure()
    ax = Axes3D(fig)

    print v[:, 0]
    print v[:, 1]
    print v[:, 2]
    print v[:, 3]

    ax.scatter(v[:, 1], v[:, 2], v[:, 3])
    plt.show()
