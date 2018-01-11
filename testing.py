from program import run
from lib import plotting
from lib.plot_value import plot_value
import numpy as np
import sys
import pandas as pd

num_episodes = 1000
budgets = [100, 60, 40, 20]
reps = 10

results = []


def get_filename(settings):
    # genera el nombre del archivo a partir de las etiquetas
    filename = ''
    for key, value in settings.items():
        filename += key + '-' + str(value) + "_"
    filename = filename[:-1]
    return filename


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


def run_episodes(settings):
    stats_array = plotting.EpisodeStats(
        episode_lengths=np.ones((reps, num_episodes)),
        episode_rewards=np.zeros((reps, num_episodes)),
        episode_shaped_rewards=np.zeros((reps, num_episodes)),
        episode_spent=np.zeros((reps, num_episodes)),
        episode_budget_count=np.zeros((reps, num_episodes))
    )

    filename = get_filename(settings)

    # repite el experimento reps veces
    for r in range(reps):
        stats = run(settings['budget'], num_episodes)

        stats_array.episode_lengths[r] = stats.episode_lengths
        stats_array.episode_rewards[r] = stats.episode_rewards
        stats_array.episode_shaped_rewards[r] = stats.episode_shaped_rewards
        stats_array.episode_spent[r] = stats.episode_spent
        stats_array.episode_budget_count[r] = stats.episode_budget_count

    # guarda los resultados de los experimentos en un archivo
    to_save = np.array(
        [stats_array.episode_lengths,
         stats_array.episode_rewards,
         stats_array.episode_spent,
         stats_array.episode_budget_count,
         stats_array.episode_shaped_rewards])
    np.save(filename + '.npy', to_save)


for b in budgets:
    run_episodes({'budget': b, 'a': 1})


def plot(settings):
    """
    Genera graficas para un experimento
    :return:
    """
    filename = get_filename(settings)
    stats_array_loaded = load_stats(filename)
    plotting.plot_episode_stats(stats_array_loaded, label=filename, smoothing_window=10)


for b in budgets:
    plot({'budget': b, 'a': 1})


# episodes = [100, 500, 950]
# for b in budgets:
#     for ep in episodes:
#         plot_value(b, ep)
