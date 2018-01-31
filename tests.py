import numpy as np
import matplotlib.pyplot as plt

#Setup dummy data
N = 10
ind = np.arange(N)
bars = np.array([100, -10, -20, -30, -20, -5, 15, 20, 20, 20])
t = np.arange(0, 10.0)

#Plot graph with 2 y axes
fig, ax1 = plt.subplots()

#Plot bars
ax1.bar(ind, bars, alpha=0.3)
ax1.set_xlabel('$t$')

# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Instant Reward [r]', color='b')
ax1.set_ylim(bottom=-50, top=110)
[tl.set_color('b') for tl in ax1.get_yticklabels()]

lines = np.cumsum(bars, axis=0)

print bars
print lines

#Set up ax2 to be the second y axis with x shared
ax2 = ax1.twinx()
#Plot a line
ax2.plot(t, lines, 'g-')
ax2.set_ylim(bottom=-50, top=110)

# Make the y-axis label and tick labels match the line color.
ax2.set_ylabel('Accumulated Budget [B]', color='g')
[tl.set_color('g') for tl in ax2.get_yticklabels()]

plt.show()