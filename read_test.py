from lib import plotting
import numpy as np

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


filename = 'a-1_budget-10'

stats_array_loaded = np.load(filename + '.npy')

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
print (stats_array_loaded[3][:, 10])
print (np.mean(stats_array_loaded[3][:, 10]))

# plotting.plot_episode_stats(stats_array_loaded, label=filename, smoothing_window=10)