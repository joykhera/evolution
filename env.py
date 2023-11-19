import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TreasureGrid(MultiAgentEnv):
    def __init__(self, num_agents=2, grid_size=5, treasures=2):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.treasures = treasures
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
        )
        # Initialize agent positions as tuples
        self.agent_positions = {
            i: (np.random.randint(grid_size), np.random.randint(grid_size))
            for i in range(num_agents)
        }
        self.treasure_positions = {
            i: (np.random.randint(grid_size), np.random.randint(grid_size))
            for i in range(treasures)
        }

    def reset(self, seed=None, options=None):
        # Reset agent positions as tuples
        self.agent_positions = {
            i: (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            for i in range(self.num_agents)
        }
        self.treasure_positions = {
            i: (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            for i in range(self.treasures)
        }
        return self._get_obs(), {}

    def step(self, action_dict):
        rewards = {}
        for agent_id, action in action_dict.items():
            # Update agent positions based on action
            # This is where you would implement the actual movement logic
            # For now, we'll just set a new random position for simplicity
            self.agent_positions[agent_id] = (
                np.random.randint(self.grid_size),
                np.random.randint(self.grid_size),
            )

            # Check for treasures and assign rewards
            rewards[agent_id] = self._get_reward(agent_id)

        obs = self._get_obs()
        dones = {
            "__all__": all(
                self._agent_found_treasure(agent_id)
                for agent_id in self.agent_positions
            )
        }
        infos = {}
        return obs, rewards, dones, dones, infos

    def _get_obs(self):
        # Returns observation for all agents
        return {
            agent_id: self._get_agent_obs(agent_id)
            for agent_id in range(self.num_agents)
        }

    def _get_agent_obs(self, agent_id):
        # Generate observation for a given agent
        obs = np.zeros((self.grid_size, self.grid_size))
        pos = self.agent_positions[agent_id]
        obs[pos[0]][pos[1]] = 1  # Set the agent's position
        return obs

    def _get_reward(self, agent_id):
        # Compute the reward for a given agent
        if self._agent_found_treasure(agent_id):
            return 1.0  # Found a treasure
        return 0.0  # No treasure found

    def _agent_found_treasure(self, agent_id):
        # Check if the agent found a treasure
        agent_pos = self.agent_positions[agent_id]
        return any(
            agent_pos == treasure_pos
            for treasure_pos in self.treasure_positions.values()
        )
