import gymnasium as gym
import numpy as np

class WrapperEnv(gym.Env):

  def __init__(self, env_name="CartPole-v1", neural_network=None):
    breakpoint()
    self.env = gym.make(env_name, render_mode="rgb_array")
    self.neural_network = neural_network
    self.state = None

  def reset(self, seed=None, options=None):
    self.state = np.array([np.random.uniform(-4.8, 4.8), np.random.uniform(-100, 100),np.random.uniform(-0.418, 0.418), np.random.uniform(-100, 100) ])
    return self.state, {}

  def step(self, action: np.ndarray):
    reward = 1
    trucated = False
    terminated = False

    next_state = self.neural_network(self.state, action)  # HERE GOES A NeURAL NETWORK THAT TAKES IN STATE + ACTION AND GIVES US NEW STATE
    self.state = next_state

    return next_state, reward, trucated, terminated, {}






WrapperEnv()