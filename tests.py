from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

global_b = '0100'
ep = '0950'

filename = 'values/' + 'b-' + global_b + '_ep-' + ep + '.npy'
arr = np.load(filename)

print "loaded file shape"
print arr.shape
# print arr

print "average shape"
arr = np.average(arr, axis=0)
print arr.shape
print arr

budgets_to_print = [40]
unfold_index = 10
features_num = 4

for b in budgets_to_print:
    filter = arr[:, 0] == b
    print filter.shape

    v = arr[filter, :]
    print "v shape"
    print v.shape

    fig1 = plt.figure()

    cp = plt.contour(
        v[:, 1].reshape((unfold_index, unfold_index)),
        v[:, 2].reshape((unfold_index, unfold_index)),
        v[:, 3].reshape((unfold_index, unfold_index))
    )
    plt.clabel(cp, inline=True,
               fontsize=10)
    plt.title('Contour Plot')
    fig1.savefig('value_results/countour-' + global_b + '-' + ep + '-' + str(b))

    fig2 = plt.figure()

    # Plot the surface.
    ax = fig2.gca(projection='3d')

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
    fig2.colorbar(surf, shrink=0.5, aspect=5)

    fig2.savefig('value_results/surf-' + global_b + '-' + ep + '-' + str(b))
