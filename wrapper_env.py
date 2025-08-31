import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import TimeLimit
class LearnedCartPoleEnv(CartPoleEnv):

  def __init__(self):
    super().__init__(render_mode=None)

  def neural_network(self, state, action):
    return np.array([0.0, 0.0, 0.0, 0.0])


  def reset(self,*, seed: int | None = None, options: dict | None = None,):
    self.state = np.array([np.random.uniform(-4.8, 4.8), np.random.uniform(-100, 100),np.random.uniform(-0.418, 0.418), np.random.uniform(-100, 100) ])
    return self.state, {}

  def step(self, action: int):
    reward = 1
    next_state = self.neural_network(self.state, action)  # HERE GOES A NeURAL NETWORK THAT TAKES IN STATE + ACTION AND GIVES US NEW STATE
    x, _, theta, _ = next_state

    terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    self.state = next_state

    return next_state, reward, terminated, False, {}

# Wrap our env in a step limit wrapper

env = TimeLimit(LearnedCartPoleEnv(), max_episode_steps=200)
print(env)
print(env.reset())
print(env.step(1))
print(env.step(1))