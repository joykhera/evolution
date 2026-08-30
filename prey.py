import numpy as np
import pygame


class Prey:
    def __init__(
        self,
        position,
        size,
        speed,
        color,
        map_size,
        view_size,
        scale,
        water_level,
        max_water_level,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.speed = speed
        self.base_color = color
        self.color = color
        self.map_size = map_size
        self.view_size = view_size
        self.scale = scale
        self.water_level = water_level
        self.max_water_level = max_water_level

    def move(self, action):
        if action == 1:  # move left
            self.position[0] = max(self.position[0] - self.speed, 0)
        elif action == 2:  # move right
            self.position[0] = min(self.position[0] + self.speed, self.map_size)
        elif action == 3:  # move down
            self.position[1] = max(self.position[1] - self.speed, 0)
        elif action == 4:  # move up
            self.position[1] = min(self.position[1] + self.speed, self.map_size)

    def update_water_level(self, water_decrease_rate):
        self.water_level -= water_decrease_rate
        if self.water_level < 0:
            self.water_level = 0
        elif self.water_level > self.max_water_level:
            self.water_level = self.max_water_level

    def update_color(self):
        # Transition from green to yellow as water level decreases
        red = int(255 * (1 - self.water_level / self.max_water_level))
        green = 255
        self.color = (red, green, 0)

    def draw(self, screen, render_mode=None, draw_grid=False):
        if render_mode == "human":
            position = (self.position * self.scale).astype(int)
            pygame.draw.circle(screen, self.color, position, self.size * self.scale)
            if draw_grid:
                half_grid = self.scale * self.view_size // 2
                pygame.draw.rect(
                    screen,
                    self.color,
                    (position[0] - half_grid, position[1] - half_grid, 2 * half_grid, 2 * half_grid),
                    1,
                )
        else:
            position = self.position
            pygame.draw.circle(screen, self.color, position, self.size)

    def get_observation(self, observation_canvas):
        x, y = self.position.astype(int)
        half_grid = self.view_size // 2

        top_left_x = max(x - half_grid, 0)
        top_left_y = max(y - half_grid, 0)
        bottom_right_x = min(x + half_grid, self.map_size)
        bottom_right_y = min(y + half_grid, self.map_size)

        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        observation_array = np.zeros((self.view_size, self.view_size, 3), dtype=np.uint8)

        if width > 0 and height > 0:
            observation_surface = pygame.Surface((width, height))
            observation_surface.blit(observation_canvas, (0, 0), pygame.Rect(top_left_x, top_left_y, width, height))
            observation_surface = pygame.transform.scale(observation_surface, (self.view_size, self.view_size))
            observation_array = pygame.surfarray.array3d(observation_surface)
            observation_array = observation_array.astype(np.float32) / 255.0

        return observation_array
