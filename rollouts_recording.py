import os
import warnings
import ray
import logging
import gymnasium as gym
from plot_utils import visualize_env
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm
import torch
import json

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore") # suppress warnings

# Start Ray with a runtime environment to filter Python warnings.
# Set logging_level to ERROR to suppress INFO and WARNING messages from Ray.
# ray.init(runtime_env={"env_vars": {"PYTHONWARNINGS": "ignore"}}, logging_level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)


# 1 - Build the agent from the config
config = DQNConfig()

# Configure the agent with environment and training settings
config.training(lr=0.0005)  # Set learning rate
config.environment(env="CartPole-v1")  # Specify the environment
config.env_runners(
    num_env_runners=4,  # Number of parallel environment runners
    num_envs_per_env_runner=2  # Environments per runner
)
config.evaluation(
    evaluation_config={"explore": False},  # No exploration during evaluation
    evaluation_duration=10,  # Evaluate for 10 episodes
    evaluation_interval=1,  # Evaluate every training iteration
    evaluation_duration_unit="episodes",
)
config.rl_module(
    model_config={
        'fc_hiddens': [256, 256],  # Two hidden layers with 256 units each
        'fcnet_activation': 'tanh'  # Use tanh activation
    }
)

print("Building agent")
agent = config.build_algo() # build agent from config



# 2 - Interact with the environment and collect data
nr_trainings = 100  # pylint: disable=invalid-name
mean_rewards = []

rollout_data = []  # store the rollout data

print("Starting training")
for _ in range(nr_trainings):
    training_logs = agent.train()
    
    ############ ############ ############ ############
    # collect espisodes as rollouts here for each training iteration
    # 3 - Create CartPole env from Gymnasium and reset to start eval process
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset()
    done = False
    cum_reward = 0

    rl_module = agent.get_module() # get the trained agent

    # rollout_data = [] # store the rollout data - should be out of the loop?

    # Perform a rollout and collect the data
    for idx in range(10): # 10 episodes
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_data = [] # check if this should be here should be out of the loop?

        while not done:
            # get action from agent's policy
            obs_batch = torch.from_numpy(obs).unsqueeze(0)  # Convert to batch format
            action = rl_module.forward_inference({'obs': obs_batch})['actions'].numpy()[0]  # Get action from trained policy

            # step function
            next_obs, reward, terminated, truncated, info = env.step(action=action)
            done = terminated or truncated
            episode_reward += reward

            # collect data for current step
            episode_data.append({ # you shoudl append the training iteration number too, otherwise episodes with same id number will overwrite
                "observation": obs.tolist(),
                "action": action.tolist(),
                "reward": reward,
                "next_observation": next_obs.tolist(),
                "done": done,
                #"training_iteration": training_iteration
            })

            obs = next_obs

        rollout_data.append({
            "episode_id": idx + 1,
            "total_reward": episode_reward,
            "steps": episode_data
        })

        print(f"Rollout single episode {idx+1} completed with total reward: {episode_reward}")
        ############ ############ ############ ############

    mean_total_reward = training_logs['evaluation']['env_runners']['episode_return_mean']
    print(f'mean total reward: {mean_total_reward}')

print("End of training")

pwd = os.getcwd()
agent.save_to_path(pwd)
print(f"Trained Agent saved in {pwd}")

# Save the collected data to a JSON file
output_file = "rollout_data.json"
with open(output_file, "w") as f:
    json.dump(rollout_data, f, indent=4)

print(f"Rollout data saved to {output_file}")

env.close()
ray.shutdown()
print("Rollout complete.")
