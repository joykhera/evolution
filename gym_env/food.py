from gym_env.game_object import GameObject


class Food(GameObject):
    """Food particles the player can eat."""

    def __init__(self, position, size, color):
        super().__init__(position, size, color)
