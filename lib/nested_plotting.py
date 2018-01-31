import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

plt.switch_backend('agg')
import sys

EpisodeStats = namedtuple(
    "Stats", ["episode_lengths", "episode_rewards", "episode_spent", "episode_budget_count", "episode_shaped_rewards"])
dir = 'results'




def plot_nested_episode_stats(stats_list, labels, smoothing_window=50, noshow=False):

    # Length #########################################################################

    print "Episode Length over Time"
    fig1 = plt.figure(figsize=(10, 5))
    for i in range(len(stats_list)):
        plt.plot(stats_list[i].episode_lengths.mean(axis=0), label=labels[i])

    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend(loc='best')
    fig1.savefig(dir + '/length-vs-time-' + labels[i] + '.png')

    # Reward #########################################################################

    print "Episode Reward over Time (Smoothed over window size {})".format(smoothing_window)
    fig2 = plt.figure(figsize=(10, 5))

    for i in range(len(stats_list)):
        rewards_smoothed = pd.Series(stats_list[i].episode_rewards.mean(axis=0))\
            .rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed, label=labels[i])

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend(loc='best')
    fig2.savefig(dir + '/reward-vs-time-' + labels[i] + '.png')

    # Acc Budget #########################################################################

    # Plot cumm budget spent
    fig4 = plt.figure(figsize=(10, 5))

    print "Cummulative budget spent"

    for i in range(len(stats_list)):
        plt.plot(np.cumsum(stats_list[i].episode_spent.mean(axis=0)), label=labels[i])

    plt.xlabel("Episode")
    plt.ylabel("Budget")
    plt.title("Cummulative budget spent")
    plt.legend(loc='best')
    fig4.savefig(dir + '/cumm-budget-' + labels[i] + '.png')
