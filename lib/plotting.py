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


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, label = 'fig', smoothing_window=50, noshow=False):

    ####################################################################################

    print "Episode Length over Time"
    print stats.episode_lengths
    print stats.episode_lengths.mean(axis=0)

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths.mean(axis=0))
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        fig1.savefig(dir + '/length-vs-time-' + label + '.png')
        plt.close(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))

    ####################################################################################

    print "Episode Reward over Time (Smoothed over window size {})".format(smoothing_window)
    print stats.episode_rewards
    print stats.episode_rewards.mean(axis=0)

    rewards_smoothed = pd.Series(stats.episode_rewards.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        fig2.savefig(dir + '/reward-vs-time-' + label + '.png')
        plt.close(fig2)

    ####################################################################################

    # Plot the episode reward over time
    fig6 = plt.figure(figsize=(10, 5))

    print "Shaped Episode Reward over Time (Smoothed over window size {})".format(smoothing_window)
    print stats.episode_shaped_rewards
    print stats.episode_shaped_rewards.mean(axis=0)

    shaped_rewards_smoothed = pd.Series(stats.episode_shaped_rewards.mean(axis=0)).rolling(smoothing_window,
                                                                             min_periods=smoothing_window).mean()
    plt.plot(shaped_rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Shaped Reward (Smoothed)")
    plt.title("Episode Shaped Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig6)
    else:
        fig6.savefig(dir + '/shaped_reward-vs-time-' + label + '.png')
        plt.close(fig6)

    ####################################################################################

    # Plot time steps and episode number

    print "Episode per time step"
    print stats.episode_lengths
    print stats.episode_lengths.mean(axis=0)


    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths.mean(axis=0)), np.arange(len(stats.episode_lengths[0])))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        fig3.savefig(dir + '/episode-vs-time-' + label + '.png')
        plt.close(fig3)

    ####################################################################################

    # Plot cumm budget spent
    fig4 = plt.figure(figsize=(10, 5))

    print "Cummulative budget spent"
    print stats.episode_spent
    print stats.episode_spent.mean(axis=0)

    plt.plot(np.cumsum(stats.episode_spent.mean(axis=0)))
    plt.xlabel("Episode")
    plt.ylabel("Budget")
    plt.title("Cummulative budget spent")
    if noshow:
        plt.close(fig4)
    else:
        fig4.savefig(dir + '/cumm-budget-' + label + '.png')
        plt.close(fig4)

    ####################################################################################

    print stats.episode_budget_count.shape
    slices = 4
    slice_size = round(stats.episode_budget_count.shape[1] / slices)
    print slice_size
    fig5, axs = plt.subplots(nrows=2, ncols=2)
    for s in range(slices):
        if s == slices - 1:
            data = stats.episode_budget_count[:, int(s * slice_size):int(stats.episode_budget_count.shape[1])]
        data = stats.episode_budget_count[:, int(s * slice_size):int((s+1) * slice_size)]

        print s, data.shape

        all_reasons_end = np.ravel(data)
        # truco para considerar todos los keys posibles
        unique, counts = np.unique(np.append(all_reasons_end, [0, 1, 2, 3]), return_counts=True)

        # Data to plot
        sizes = counts - 1
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
        explode = (0, 0, 0.2, 0)  # explode 1st slice

        dict = {'0.0': 'init', '1.0': 'env', '2.0': 'exit', '3.0': 'budget'}
        print dict.keys()
        renamed_labels = []
        tmp = 0
        for u, c in zip(unique, counts):
            print u, c, counts[tmp]
            curr_label = dict[str(u)] if counts[tmp] > 0 else ''
            renamed_labels.append(str(u) + ' ' + str(curr_label))
            tmp += 1

        print renamed_labels

        # Plot
        bin_str = '{0:04b}'.format(s)
        axs[int(bin_str[-2]), int(bin_str[-1])].set_title(str(s+1)+'/'+str(slices))
        axs[int(bin_str[-2]), int(bin_str[-1])]
        axs[int(bin_str[-2]), int(bin_str[-1])].pie(sizes, explode=explode, labels=renamed_labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140, labeldistance=1.3)

    # sys.exit(0)
    #
    # all_reasons_end = np.ravel(stats.episode_budget_count)
    # # truco para considerar todos los keys posibles
    # unique, counts = np.unique(np.append(all_reasons_end, [0, 1, 2, 3]), return_counts=True)
    #
    # print "Cummulative budget spent"
    # print stats.episode_spent
    # # para completar el truco le resto a la cuenta 1
    # print unique, counts - 1
    #
    # # Data to plot
    # labels = unique
    # sizes = counts - 1
    # colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    # explode = (0.1, 0, 0, 0)  # explode 1st slice
    #
    # # Plot
    # plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    #         autopct='%1.1f%%', shadow=True, startangle=140)

    plt.suptitle("End reasons")
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    if noshow:
        plt.close(fig5)
    else:
        fig5.savefig(dir + '/no-budget-' + label + '.png')
        plt.close(fig5)

    ####################################################################################

    return fig1, fig2, fig3, fig4, fig5, fig6
