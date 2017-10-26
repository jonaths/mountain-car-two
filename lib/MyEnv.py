import gym
import sys
import numpy as np
import random
import logging

LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
local_logger = logging.getLogger('MyEnv')


class MyEnv:
    def __init__(self, env, budget):
        self.env = gym.envs.make(env)
        self.init_budget = budget

    def get_env(self):
        return self.env

    def reset(self):
        # reset del env
        reset = self.env.reset()
        # el budget actual es el budget inicial
        self.budget = self.init_budget
        # al estado le agrega el budget
        reset = np.append(reset, np.array([self.budget]))
        return reset

    def render(self):
        return self.env.render()

    def step(self, action, i_episode=0, t=1):
        # realiza el paso del env
        next_state, reward, done, c = self.env.step(action)
        # a next_state le agrega el presupuesto
        current_budget = self.update_budget(reward)
        next_state = np.append(next_state, np.array([current_budget]))
        # regresa done si el presupuesto es menor o igual a cero
        done = done or current_budget <= 0
        local_logger.info(
            ' ' +
            str(i_episode).zfill(4) + ' ' +
            str(t).zfill(4) + ' ' +
            "{:.14f}".format(reward) + ' ' +
            "{:.14f}".format(current_budget) + ' ' +
            ("1" if done else "0")
        )
        # regresa la tupla
        return next_state, reward, done, c

    def update_budget(self, reward):
        self.budget = self.budget + reward
        return self.budget

    def calculate_spent(self):
        return self.init_budget - self.budget

    def action_space(self):
        return self.env.action_space

    def observation_space_sample(self):
        """
        Agrega a al muestreo un valor para el presupuesto
        :return:
        """
        random_budget = random.uniform(0, self.init_budget)
        return np.append(
            self.env.observation_space.sample(), np.array([random_budget]))
        # return self.env.observation_space.sample()
