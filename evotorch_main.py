import os
import torch
import torch.nn as nn
import numpy as np
from evotorch.neuroevolution import GymNE
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from torch.utils.tensorboard import SummaryWriter
import pygame
from gymnasium import spaces
from argparser import parse_args
from gym_env.evolution_env import EvolutionEnv  # Assuming this import is available


class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate_solution(solution, env_config):
    env = EvolutionEnv(**env_config)
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        obs_visual = torch.tensor(obs["visual"], dtype=torch.float32).view(1, -1)
        obs_boost = torch.tensor(obs["boost_info"], dtype=torch.float32).view(1, -1)
        obs_tensor = torch.cat((obs_visual, obs_boost), dim=1)
        with torch.no_grad():
            action_probs = solution(obs_tensor)
        direction = torch.argmax(action_probs[:, :5]).item()
        boost = torch.argmax(action_probs[:, 5:]).item()
        action = (direction, boost)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


def train_agent(env_config, model_config, log_dir, args):
    observation_space = spaces.Box(low=0, high=1, shape=(env_config["grid_size"], env_config["grid_size"], 3), dtype=np.float32)
    action_space = spaces.Discrete(5)
    input_dim = np.prod(observation_space.shape)
    output_dim = action_space.n
    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = SimpleNet(input_dim, output_dim)

        def forward(self, x):
            return self.net(x)

    problem = GymNE(
        env=lambda: EvolutionEnv(**env_config),
        network="Linear(obs_length, act_length)",  # Linear policy
        observation_normalization=True,  # Normalize the policy inputs
        decrease_rewards_by=5.0,  # Decrease each reward by 5.0
        num_actors="max",  # Use all available CPUs
    )

    searcher = PGPE(
        problem,
        popsize=model_config["population_size"],
        center_learning_rate=0.0075,
        stdev_learning_rate=0.1,
        radius_init=0.27,
    )
    writer = SummaryWriter(log_dir)
    StdOutLogger(searcher)

    for generation in range(model_config["generations"]):
        searcher.step()
        best_fitness = searcher.status["best_fitness"]
        writer.add_scalar("Best Fitness", best_fitness, generation)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    writer.close()


def test_agent(env_config, checkpoint, num_episodes=10, render=False):
    observation_space = spaces.Box(low=0, high=1, shape=(env_config["grid_size"], env_config["grid_size"], 3), dtype=np.float32)
    action_space = spaces.Discrete(5)
    input_dim = np.prod(observation_space["visual"].shape)
    output_dim = sum([space.n for space in action_space])

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = SimpleNet(input_dim, output_dim)

        def forward(self, x):
            return self.net(x)

    solution = torch.load(checkpoint)
    env = EvolutionEnv(**env_config)

    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            if render:
                env.render()
            obs_visual = torch.tensor(obs["visual"], dtype=torch.float32).view(1, -1)
            obs_boost = torch.tensor(obs["boost_info"], dtype=torch.float32).view(1, -1)
            obs_tensor = torch.cat((obs_visual, obs_boost), dim=1)
            with torch.no_grad():
                action_probs = solution(obs_tensor)
            direction = torch.argmax(action_probs[:, :5]).item()
            boost = torch.argmax(action_probs[:, 5:]).item()
            action = (direction, boost)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward


def main():
    args, model_config, env_config = parse_args()
    print("args:", args)
    print("model_config:", model_config)
    print("env_config:", env_config)
    log_dir = os.path.join(os.getcwd(), "training", args["save_name"])

    if args["train"]:
        print("Training mode enabled.")
        train_agent(env_config, model_config, log_dir, args)

    elif args["test"]:
        print("Testing mode enabled.")
        env_config["render_mode"] = "human"
        test_agent(env_config, args["checkpoint"], episodes=args["episodes"], render=True)


if __name__ == "__main__":
    main()
