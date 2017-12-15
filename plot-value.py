from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def plot_value(b, ep):

    filename = 'values/' + 'b-0100_ep-0950' + '.npy'
    arr = np.load(filename)

    print "loaded file shape"
    print arr.shape
    # print arr

    print "average shape"
    arr = np.average(arr, axis=0)
    print arr.shape
    print arr

    budgets_to_print = [40]
    unfold_index = 20

    for b in budgets_to_print:
        filter = arr[:, 0] == b
        print filter.shape

        v = arr[filter, :]
        print "v shape"
        print v.shape

        cp = plt.contour(
            v[:, 1].reshape((unfold_index, unfold_index)),
            v[:, 2].reshape((unfold_index, unfold_index)),
            v[:, 3].reshape((unfold_index, unfold_index))
        )
        plt.clabel(cp, inline=True,
                   fontsize=10)
        plt.title('Contour Plot')
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(
            v[:, 1].reshape((unfold_index, unfold_index)),
            v[:, 2].reshape((unfold_index, unfold_index)),
            v[:, 3].reshape((unfold_index, unfold_index)),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False
        )

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

