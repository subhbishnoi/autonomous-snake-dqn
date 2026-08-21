import numpy as np

from src.game.config import (
    ACTION_LEFT,
    ACTION_STRAIGHT,
    ACTION_RIGHT,
    REWARD_COLLISION,
    REWARD_FOOD,
    REWARD_SURVIVAL,
    GRID_WIDTH,
    GRID_HEIGHT,
)

from src.game.snake import Snake


class SnakeEnvironment:
    """
    Reinforcement Learning environment for Snake.

    The agent:
        1. Observes a state
        2. Chooses an action
        3. Environment moves the snake
        4. Environment returns reward and next state
    """

    def __init__(self):
        self.snake = Snake()

        # Number of steps since the last food was eaten.
        self.steps_without_food = 0

        # Prevent the snake from wandering forever.
        self.max_steps_without_food = 100

    def reset(self):
        """Reset the environment and return the initial state."""

        self.snake.reset()
        self.steps_without_food = 0

        return self.get_state()

    def step(self, action):
        """
        Perform one environment step.

        Returns:
            next_state
            reward
            done
        """

        # Convert relative action into a direction.
        self._apply_action(action)

        # Move snake.
        ate_food = self.snake.move()

        self.steps_without_food += 1

        # Check collision.
        done = self.snake.check_collision()

        if done:
            return (
                self.get_state(),
                REWARD_COLLISION,
                True,
            )

        # Calculate reward.
        if ate_food:
            reward = REWARD_FOOD
            self.steps_without_food = 0
        else:
            reward = REWARD_SURVIVAL

        # Prevent endless movement without eating.
        if self.steps_without_food > self.max_steps_without_food:
            done = True
            reward = REWARD_COLLISION

        return (
            self.get_state(),
            reward,
            done,
        )

    def _apply_action(self, action):
        """
        Convert relative action into an absolute direction.

        Actions:
            0 -> LEFT
            1 -> STRAIGHT
            2 -> RIGHT
        """

        current_direction = self.snake.direction

        if action == ACTION_STRAIGHT:
            new_direction = current_direction

        elif action == ACTION_LEFT:
            new_direction = self._turn_left(current_direction)

        elif action == ACTION_RIGHT:
            new_direction = self._turn_right(current_direction)

        else:
            raise ValueError(f"Invalid action: {action}")

        self.snake.set_direction(new_direction)

    @staticmethod
    def _turn_left(direction):
        """Turn 90 degrees left."""

        dx, dy = direction

        return dy, -dx

    @staticmethod
    def _turn_right(direction):
        """Turn 90 degrees right."""

        dx, dy = direction

        return -dy, dx

    def get_state(self):
        """
        Return the 11-dimensional state representation.

        State:

        0  danger straight
        1  danger right
        2  danger left

        3  moving left
        4  moving right
        5  moving up
        6  moving down

        7  food left
        8  food right
        9  food up
        10 food down
        """

        head_x, head_y = self.snake.get_head()

        direction = self.snake.direction

        # Current direction.
        direction_left = direction == (-1, 0)
        direction_right = direction == (1, 0)
        direction_up = direction == (0, -1)
        direction_down = direction == (0, 1)

        # Collision danger.
        danger_straight = self._is_collision_ahead(direction)

        right_direction = self._turn_right(direction)
        left_direction = self._turn_left(direction)

        danger_right = self._is_collision_ahead(right_direction)
        danger_left = self._is_collision_ahead(left_direction)

        # Food position.
        food_x, food_y = self.snake.food

        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(direction_left),
            int(direction_right),
            int(direction_up),
            int(direction_down),

            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
        ]

        return np.array(state, dtype=np.float32)

    def _is_collision_ahead(self, direction):
        """Check whether the next cell causes a collision."""

        head_x, head_y = self.snake.get_head()

        dx, dy = direction

        next_x = head_x + dx
        next_y = head_y + dy

        # Wall collision.
        if (
            next_x < 0
            or next_x >= GRID_WIDTH
            or next_y < 0
            or next_y >= GRID_HEIGHT
        ):
            return True

        # Body collision.
        if (next_x, next_y) in self.snake.body:
            return True

        return False

    def get_score(self):
        """Return current Snake score."""

        return self.snake.score