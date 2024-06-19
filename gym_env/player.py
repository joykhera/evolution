import random
from gym_env.game_object import GameObject


class Player(GameObject):
    """Player-controlled agent."""

    def __init__(self, position, map_size, size, speed, color, boost_color, max_speed, boost_duration):
        super().__init__(position, size, color)
        self.ate = False
        self.map_size = map_size
        self.base_speed = speed
        self.current_speed = speed
        self.max_speed = max_speed
        self.boost_duration = boost_duration
        self.boost_active = False
        self.boost_timer = boost_duration
        self.boost_color = boost_color
        self.normal_color = color

    def move(self, direction_vector):
        if self.boost_active:
            self.current_speed = self.max_speed
            self.boost_timer -= 1
        elif not self.boost_active or self.boost_timer <= 0:
            self.boost_active = False
            self.color = self.normal_color
            self.current_speed = self.base_speed

        self.position += direction_vector * self.current_speed
        self.position.x = max(0, min(self.position.x, self.map_size))
        self.position.y = max(0, min(self.position.y, self.map_size))

    def activate_boost(self):
        if self.boost_timer > 0:
            self.boost_active = True
            self.color = self.boost_color
