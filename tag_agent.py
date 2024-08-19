import numpy as np
import pygame


class Agent:

    def __init__(self, agent_type, position, size, speed, color, color_encoding, map_size, view_size, scale):
        self.agent_type = agent_type
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.speed = speed
        self.color = color
        self.color_encoding = color_encoding
        self.map_size = map_size
        self.view_size = view_size
        self.scale = scale

    def move(self, action):
        # print(action, self.position)
        if action == 1:  # move left
            self.position[0] = max(self.position[0] - self.speed, 0)
        elif action == 2:  # move right
            self.position[0] = min(self.position[0] + self.speed, self.map_size)
        elif action == 3:  # move down
            self.position[1] = max(self.position[1] - self.speed, 0)
        elif action == 4:  # move up
            self.position[1] = min(self.position[1] + self.speed, self.map_size)

    def draw(self, screen, render_mode=None, draw_grid=False):
        if render_mode == "human":
            position = (self.position * self.scale).astype(int)
            pygame.draw.circle(screen, self.color, position, self.size * self.scale)

            if draw_grid:
                half_grid = self.scale * self.view_size // 2
                pygame.draw.rect(screen, self.color, (position[0] - half_grid, position[1] - half_grid, 2 * half_grid, 2 * half_grid), 1)
        else:
            position = self.position
            # pygame.draw.circle(screen, self.color_encoding, position, self.size)
            pygame.draw.circle(screen, self.color, position, self.size)

    def get_observation(self, observation_canvas):
        x, y = self.position.astype(int)
        half_grid = self.view_size // 2

        # Determine the observation window bounds within the map
        top_left_x = max(x - half_grid, 0)
        top_left_y = max(y - half_grid, 0)
        bottom_right_x = min(x + half_grid, self.map_size)
        bottom_right_y = min(y + half_grid, self.map_size)

        # Calculate the width and height of the valid observation area
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        # Initialize a blank observation grid with zeros for the RGB channels
        observation_array = np.zeros((self.view_size, self.view_size, 3), dtype=np.uint8)

        if width > 0 and height > 0:
            # Extract the relevant section of the observation canvas
            observation_surface = pygame.Surface((width, height))
            observation_surface.blit(observation_canvas, (0, 0), pygame.Rect(top_left_x, top_left_y, width, height))

            # Resize the observation to fit into the grid
            observation_surface = pygame.transform.scale(observation_surface, (self.view_size, self.view_size))

            # Convert the surface back to a 3D numpy array (10x10x3 for RGB)
            observation_array = pygame.surfarray.array3d(observation_surface)

            # Normalize the observation array to the range [0, 1]
            observation_array = observation_array / 255.0

        return observation_array
