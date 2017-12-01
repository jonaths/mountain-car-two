from lib import plotting
import numpy as np
from matplotlib import pyplot as plt

def load_stats(file_name='test.npy'):
    """
    Recupera un archivo de resultados para graficarlo
    :param file_name:
    :return:
    """
    loaded = np.load(file_name + '.npy')

    stats_array = plotting.EpisodeStats(
        episode_lengths=loaded[0],
        episode_rewards=loaded[1],
        episode_spent=loaded[2],
        episode_budget_count=loaded[3],
        episode_shaped_rewards=loaded[4])

    print plotting.EpisodeStats._fields
    print stats_array
    return stats_array


filename = 'a-1_budget-040'

stats_array_loaded = np.load(filename + '.npy')
print stats_array_loaded.shape

print (stats_array_loaded[0].shape)
print (stats_array_loaded[0][:, 10])
print (np.mean(stats_array_loaded[0][:, 10]))

print (stats_array_loaded[1].shape)
print (stats_array_loaded[1][:, 10])
print (np.mean(stats_array_loaded[1][:, 10]))

print (stats_array_loaded[2].shape)
print (stats_array_loaded[2][:, 10])
print (np.mean(stats_array_loaded[2][:, 10]))

print (stats_array_loaded[3].shape)
print (stats_array_loaded[3][:, 500])
print (np.mean(stats_array_loaded[4][:, 10]))
print (np.ravel(stats_array_loaded[3]).shape)

all_reasons_end = np.ravel(stats_array_loaded[3])
unique, counts = np.unique(np.append(all_reasons_end, [0, 1, 2, 3]), return_counts=True)
print unique, counts - 1


# Plot cumm budget spent
fig4 = plt.figure(figsize=(10, 5))

# Data to plot
labels = unique
sizes = counts - 1
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.title("Cummulative budget spent")
plt.show()
#
#
#
#
#
# fig4.savefig(dir + '/cumm-budget-' + 'test' + '.png')

# plotting.plot_episode_stats(stats_array_loaded, label=filename, smoothing_window=10)