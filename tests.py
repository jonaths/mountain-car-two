import numpy as np
import sys
from matplotlib import pyplot as plt


# x = np.arange(-5, 5, 0.1)
# y = np.arange(-5, 5, 0.1)
# xx, yy = np.meshgrid(x, y, sparse=True)
# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
# h = plt.contourf(x,y,z)
# plt.show()

def test():
    items = 5

    b = np.full((1, items), 100)
    x1 = np.linspace(-0.06, 0.06, num=items)
    x2 = np.linspace(-1.2, 0.6, num=items)

    def func(b, x1, x2):
        return x1 * x2 + b

    v = np.zeros((items, 4))

    v[:, 0] = b
    v[:, 1] = x1
    v[:, 2] = x2

    print v[0][0:3]

    for r in v:
        r[-1] = func(r[0], r[1], r[2])

    new =False
    print v.shape
    filename = 'values/' + 'filename' + '.npy'
    try:
        arr = np.load(filename)
        pass
    except IOError:
        arr = np.zeros((1, items, 4))
        arr[0] = v
        new = True
        pass

    print len(arr)

    if new:
        print "if"
        pass
    else:
        print "else"
        arr = np.append(arr, [v], axis=0)

    np.save(filename, arr)


for i in range(3):
    test()

filename = 'values/' + 'filename' + '.npy'
print "XXX"
print np.load(filename)
