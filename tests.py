import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# def test():
#     items = 5
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
#     v = np.zeros((items ** 2, 4))
#
#     v[:, 0] = np.full((1, items ** 2), 100)
#     v[:, 1] = x1x.ravel()
#     v[:, 2] = y1y.ravel()
#
#     print v.shape
#     print v
#
#
#     for r in v:
#         r[-1] = func(r[0], r[1], r[2])
#
#     new = False
#     print v.shape
#     print v
#
#
#
#     filename = 'values/' + 'filename' + '.npy'
#     try:
#         arr = np.load(filename)
#         print "arr found", arr.shape
#         pass
#     except IOError:
#         arr = np.zeros((1, items ** 2, 4))
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

filename = 'values/' + 'b-0020_ep-0000' + '.npy'

print "XXX"
arr = np.load(filename)

print arr.shape
print arr

fig = plt.figure()
ax = Axes3D(fig)

v = arr[0]
print v.shape
print v

print v[:, 0]
print v[:, 1]
print v[:, 2]
print v[:, 3]

ax.scatter(v[:, 1], v[:, 2], v[:, 3])
plt.show()
