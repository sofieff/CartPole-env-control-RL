import gymnasium as gym
from stable_baselines3 import DQN
from gymnasium.wrappers import TimeLimit
from wrapper_env import LearnedCartPoleEnv
import numpy as np

# Wrap our env in a step limit wrapper
env = TimeLimit(LearnedCartPoleEnv(), max_episode_steps=200)
model = DQN("MlpPolicy", env, verbose=1)

#train our agent
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

state, info = env.reset()

total_reward = 0

done = False

# let our agent predict on our fake environment
while not done:
    action, _states = model.predict(state, deterministic=True)

    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"Action: {action}, State: {state}, Reward: {reward}, Total Reward: {total_reward}")
    
    done = terminated or truncated



# lets try our trained agent on a real cartpole env to compare... 
real_env = gym.make("CartPole-v1", render_mode="human")

real_state, info = real_env.reset()

real_done = False
real_total_reward = 0

while not real_done:
    # use real_state, not state
    action, _states = model.predict(real_state, deterministic=True)

    # step with real_env
    real_state, reward, terminated, truncated, info = real_env.step(action)
    real_total_reward += reward
    print(f"Action: {action}, State: {real_state}, Reward: {reward}, Total Reward: {real_total_reward}")

    real_done = terminated or truncated

