from env.game_object import GameObject


class Player(GameObject):
    """Player-controlled agent."""

    def __init__(self, position, map_size, size, speed, color):
        super().__init__(position, size, color)
        self.ate = False
        self.map_size = map_size
        self.speed = speed
        self.previous_position = position

    def move(self, direction_vector):
        self.previous_position = self.position
        self.position += direction_vector * self.speed
        self.position.x = max(0, min(self.position.x, self.map_size))
        self.position.y = max(0, min(self.position.y, self.map_size))
