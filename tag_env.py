import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


class TagEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "custom_tag_v0"}

    def __init__(self, num_prey=1, num_predators=2, grid_size=50, max_steps=400, screen_size=600, predator_speed=1, prey_speed=1, render_mode=None):
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.screen_size = screen_size
        self.predator_speed = predator_speed
        self.prey_speed = prey_speed
        self.render_mode = render_mode

        self.agents = [f"prey_{i}" for i in range(num_prey)] + [f"predator_{i}" for i in range(num_predators)]
        self.possible_agents = self.agents[:]
        self.agent_positions = np.zeros((len(self.agents), 2))
        self.current_step = 0

        # Initialize pygame screen
        self.scale = self.screen_size // self.grid_size
        self.screen = None if self.render_mode != "human" else pygame.display.set_mode((self.screen_size, self.screen_size))

        # Define observation and action spaces
        self.observation_spaces = {
            agent: (
                spaces.Box(low=-1, high=1, shape=(2 * self.num_predators,), dtype=np.float32)  # Two values (x and y) per predator for each prey
                if "prey" in agent
                else spaces.Box(low=-1, high=1, shape=(2 * self.num_prey,), dtype=np.float32)  # Two values (x and y) per prey for each predator
            )
            for agent in self.agents
        }
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}  # 4 directions + no action

        # Track previous distances for rewards
        self.prev_distances = None

    def reset(self, seed=None, return_info=False, options=None):
        self.current_step = 0
        self.agents = self.possible_agents[:]
        self.agent_positions = np.random.rand(len(self.agents), 2) * self.grid_size

        # Initialize previous distances
        prey_positions = self.agent_positions[: self.num_prey]
        predator_positions = self.agent_positions[self.num_prey :]
        self.prev_distances = np.array([[np.linalg.norm(prey_pos - pred_pos) for pred_pos in predator_positions] for prey_pos in prey_positions])

        observations = self.get_observations()
        return observations, {}

    def step(self, actions):
        self.current_step += 1
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Update positions based on actions
        for i, agent in enumerate(self.agents):
            speed = self.prey_speed if "prey" in agent else self.predator_speed

            if actions[agent] == 1:  # move left
                self.agent_positions[i][0] = max(self.agent_positions[i][0] - speed, 0)
            elif actions[agent] == 2:  # move right
                self.agent_positions[i][0] = min(self.agent_positions[i][0] + speed, self.grid_size)
            elif actions[agent] == 3:  # move down
                self.agent_positions[i][1] = max(self.agent_positions[i][1] - speed, 0)
            elif actions[agent] == 4:  # move up
                self.agent_positions[i][1] = min(self.agent_positions[i][1] + speed, self.grid_size)

        if self.render_mode == "human":
            self.render()

        # Compute rewards using the separate reward function
        rewards = self.compute_rewards()

        # Implement truncation condition: if max steps reached
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True

        observations = self.get_observations()

        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        self.screen.fill((255, 255, 255))

        # Draw prey
        for i in range(self.num_prey):
            pygame.draw.circle(self.screen, (0, 255, 0), (self.agent_positions[i] * self.scale).astype(int), self.scale)

        # Draw predators
        for i in range(self.num_predators):
            pygame.draw.circle(self.screen, (255, 0, 0), (self.agent_positions[self.num_prey + i] * self.scale).astype(int), self.scale)

        pygame.display.flip()

    def get_observations(self):
        observations = {}
        prey_positions = self.agent_positions[: self.num_prey]
        predator_positions = self.agent_positions[self.num_prey :]

        for i, agent in enumerate(self.agents):
            if "prey" in agent:
                # Get x and y distances to each predator
                distances = []
                for pred_pos in predator_positions:
                    x_dist = prey_positions[i][0] - pred_pos[0]
                    y_dist = prey_positions[i][1] - pred_pos[1]
                    distances.extend([x_dist, y_dist])
                # Normalize the distances by grid size
                observations[agent] = np.array(distances, dtype=np.float32) / self.grid_size

            elif "predator" in agent:
                # Get x and y distances to each prey
                distances = []
                for prey_pos in prey_positions:
                    x_dist = predator_positions[i - self.num_prey][0] - prey_pos[0]
                    y_dist = predator_positions[i - self.num_prey][1] - prey_pos[1]
                    distances.extend([x_dist, y_dist])
                # Normalize the distances by grid size
                observations[agent] = np.array(distances, dtype=np.float32) / self.grid_size

        return observations

    def compute_rewards(self):
        rewards = {}
        terminated_agents = []

        prey_positions = self.agent_positions[: self.num_prey]
        predator_positions = self.agent_positions[self.num_prey :]

        # Calculate current distances
        distances = np.array([[np.linalg.norm(prey_pos - pred_pos) for pred_pos in predator_positions] for prey_pos in prey_positions])

        prey_rewards = np.zeros(self.num_prey)
        predator_rewards = np.zeros(self.num_predators)

        # Reward structure
        for i in range(self.num_prey):
            closest_predator_idx = np.argmin(distances[i])
            closest_predator_distance = distances[i][closest_predator_idx]
            previous_closest_predator_distance = self.prev_distances[i][closest_predator_idx]

            # Reward prey for increasing the distance from the closest predator
            prey_rewards[i] += closest_predator_distance - previous_closest_predator_distance

            # Reward predator for decreasing the distance to the closest prey
            predator_rewards[closest_predator_idx] += previous_closest_predator_distance - closest_predator_distance

            # # If the predator catches the prey
            # if closest_predator_distance < 1:
            #     prey_rewards[i] -= 100  # Large penalty for getting caught
            #     predator_rewards[closest_predator_idx] += 100  # Large reward for catching prey
            #     terminated_agents.append(self.agents[i])  # Mark prey as terminated

        # Assign rewards to agents
        for i in range(self.num_prey):
            if self.agents[i] not in terminated_agents:
                rewards[self.agents[i]] = prey_rewards[i]

        for i in range(self.num_predators):
            rewards[self.agents[self.num_prey + i]] = predator_rewards[i]

        return rewards

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
