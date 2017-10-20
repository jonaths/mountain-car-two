
import gym


class MyEnv:

    def __init__(self, env, budget):
        self.env = gym.envs.make(env)
        self.init_budget = budget

    def get_env(self):
        return self.env

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        return self.env.step(action)

    def action_space(self):
        return self.env.action_space

    def observation_space_sample(self):
        return self.env.observation_space.sample()


