from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_value(b, ep):
    b_str = '{:04}'.format(b)
    ep_str = '{:04}'.format(ep)

    filename = 'b-' + b_str + '_ep-' + ep_str + '.npy'
    print filename
    arr = np.load('values/' + filename)

    print "loaded file shape"
    print arr.shape

    print "average shape"
    arr = np.average(arr, axis=0)
    print arr.shape

    # imprime la funcion de valor desde 20 hasta b (el +1 es para incluirlo)
    # de 20 en 20
    budgets_to_print = range(20, b + 1, 20)

    print budgets_to_print

    unfold_index = 20

    for b in budgets_to_print:
        filter = arr[:, 0] == b
        print filter.shape

        v = arr[filter, :]
        print "v shape"
        print v.shape

        # cp = plt.contour(
        #     v[:, 1].reshape((unfold_index, unfold_index)),
        #     v[:, 2].reshape((unfold_index, unfold_index)),
        #     v[:, 3].reshape((unfold_index, unfold_index))
        # )
        # plt.clabel(cp, inline=True,
        #            fontsize=10)
        # plt.title('Contour Plot')
        # plt.show()

        fig = plt.figure()

        # Plot the surface.
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(
            v[:, 1].reshape((unfold_index, unfold_index)),
            v[:, 2].reshape((unfold_index, unfold_index)),
            v[:, 3].reshape((unfold_index, unfold_index)),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True
        )

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        title = 'Ep-' + ep_str + ', B-' + b_str + '@ ' + str(b)

        plt.title(title)

        # plt.show()
        plt.savefig('value_results/' + filename + '_b-' + str(b) + '.png')
