import gym
import sys
import numpy as np
import random
import logging
import math

LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
local_logger = logging.getLogger('MyEnv')


class MyEnv:
    def __init__(self, env, budget):
        self.env = gym.envs.make(env)
        self.init_budget = budget
        self.done_reason = 0

    def get_env(self):
        return self.env

    def reset(self):
        # reset del env
        reset = self.env.reset()
        # el budget actual es el budget inicial
        self.budget = self.init_budget
        self.done_reason = 0
        # al estado le agrega el budget
        reset = np.append(reset, np.array([self.budget]))
        return reset

    def render(self):
        return self.env.render()

    def set_done_reason(self, reason):
        # 0: none
        # 1: env
        # 2: exit
        # 3: budget
        if self.done_reason == 0:
            self.done_reason = reason

    def step(self, action, i_episode=0, t=1):

        # variable que contiene la razon por la cual termina
        budget_end_count = 0

        # accion de env open ai
        next_state, reward, done, c = self.env.step(action)
        if done:
            # terminacion por ambiente
            self.set_done_reason(1)

        # Actualiza el presupuesto con la recompensa actual
        self.update_budget(reward)

        # a next_state le agrega el presupuesto
        next_state = np.append(next_state, np.array([self.budget]))

        # permite terminar antes si se cumplen condiciones de posicion y accion
        if +0.60 <= action[0] < +0.90:
            position = next_state[0]
            if -0.10 <= position < 0.25:
                reward = 40 - math.pow(action[0], 2) * 0.1
                # terminacion por salida anticipada
                self.set_done_reason(2)
            else:
                pass
        else:
            pass

        # verifica si se quedo sin prespuesto
        if self.budget <= 0:
            budget_end_count = 1
            reward = -20 - math.pow(action[0], 2) * 0.1
            # terminacion por presupuesto
            self.set_done_reason(3)

        # guarda el presupuesto en key del arreglo c
        c['episode_budget_count'] = self.done_reason

        def shape_reward(x, b):
            # si b < a esta cantidad entonces se activa la
            # funcion
            offset = 40
            # para valores negativos de x
            # neg = +1.5 * x
            neg = +10.0 * x
            # para valores positivos de x
            # pos = +0.5 * x
            pos = +1.0 * x
            if b <= offset:
                res = pos if x >= 0 else neg
            else:
                res = x
            return res

        # aplica funcion de shaping
        shaped_reward = shape_reward(reward, self.budget)

        # logea informacion
        local_logger.info(
            ' ' +
            str(i_episode).zfill(4) + ' ' +
            str(t).zfill(4) + ' ' +
            "{:.14f}".format(reward) + ' ' +
            "{:.14f}".format(shaped_reward) + ' ' +
            "{:.14f}".format(self.budget) + ' ' +
            ("1" if done else "0")
        )

        # done es True si hay otra cosa en done_reason que no sea 0
        done = False if self.done_reason == 0 else True

        # regresa la tupla
        return next_state, reward, shaped_reward, done, c

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
