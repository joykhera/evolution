import random
from game.game_object import GameObject


class Player(GameObject):
    """Player-controlled agent."""

    def __init__(self, position, map_size, size, speed, color):
        super().__init__(position, size, color)
        self.ate = False
        self.map_size = map_size
        self.speed = speed

    def move(self, direction_vector):
        self.position += direction_vector * self.speed
        self.position.x = max(
            self.size, min(self.position.x, self.map_size - self.size)
        )
        self.position.y = max(
            self.size, min(self.position.y, self.map_size - self.size)
        )
