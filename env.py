import gym
from gym import spaces
import numpy as np
from game.game import Game

class AgarioEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(AgarioEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(5)  # 4 directions
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(100, 100, 3), dtype=np.uint8
        )  # RGB Image
        self.game = Game()

    def step(self, action):
        # Execute one time step within the environment
        self.game.update(action)
        obs = self.game.get_observation()
        reward = self.game.get_reward()
        done = self.game.is_done()
        info = {}  # additional data, not used now
        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.game.reset()
        return self.game.get_observation()  # return initial observation

    def render(self, mode="human"):
        # Render the environment to the screen
        self.game.render()

    def close(self):
        # Close the environment
        self.game.close()
