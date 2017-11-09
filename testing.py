from program import run
from lib import plotting
import numpy as np
import sys
import pandas as pd

num_episodes = 50
budgets = [80]
reps = 3

results = []


def run_episodes(settings):
    stats_array = plotting.EpisodeStats(
        episode_lengths=np.ones((reps, num_episodes)),
        episode_rewards=np.zeros((reps, num_episodes)),
        episode_spent=np.zeros((reps, num_episodes)))

    # genera el nombre del archivo a partir de las etiquetas
    filename = ''
    for key, value in settings.items():
        filename += key + '-' + str(value) + "_"
    filename = filename[:-1]

    # repite el experimento reps veces
    for r in range(reps):
        stats = run(settings['budget'], num_episodes)

        stats_array.episode_lengths[r] = stats.episode_lengths
        stats_array.episode_rewards[r] = stats.episode_rewards
        stats_array.episode_spent[r] = stats.episode_spent

    # guarda los resultados de los experimentos en un archivo
    to_save = np.array([stats_array.episode_lengths, stats_array.episode_rewards, stats_array.episode_spent])
    np.save(filename+'.npy', to_save)


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
        episode_spent=loaded[2])

    print plotting.EpisodeStats._fields
    print stats_array
    return stats_array


# for b in budgets:
#     run_episodes({'budget': b, 'a':1})

filename = 'a-1_budget-100'

stats_array_loaded = load_stats(filename)

plotting.plot_episode_stats(stats_array_loaded, label=filename, smoothing_window=10)
