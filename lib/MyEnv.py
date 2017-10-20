
import gym


class MyEnv():

    def __init__(self, env, budget):
        self.env = gym.envs.make(env)
        self.init_budget = budget
        return self.env

    def reset(self):
        return self.env.reset

    def render(self):
        return self.env.render

    def step(self):
        return self.env.step
