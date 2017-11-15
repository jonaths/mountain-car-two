#!/usr/bin/python

# My first Atari simulation on the AIgym
# Not much happening yet

import gym
episodes = 500
max_time = 10000

env = gym.make('ChopperCommand-ram-v0')
print env.action_space
print env.observation_space
print env.reward_range
# env.monitor.start('/tmp/atari-experiment-3', force=True)
env.reset()

# for episode in range(episodes):
#     obervation = env.reset()
#     total_reward = 0
#     for time in range(max_time):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print observation, reward, done, info, action
#         total_reward = total_reward + reward
#         if done:
#             print "Episode {}:".format(episode)
#             print "  completed in {} steps".format(time+1)
#             print "  total_reward was {}".format(total_reward)
#             break
#
# env.monitor.close()

