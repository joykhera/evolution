import random
from game.game_object import GameObject


class Food(GameObject):
    """Food particles the player can eat."""

    def __init__(self, map_size, size, color):
        position = (random.randint(0, map_size), random.randint(0, map_size))
        super().__init__(position, size, color)
