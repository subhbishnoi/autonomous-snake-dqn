import random

import numpy as np

from src.game.config import (
    Q_LEARNING_RATE,
    Q_GAMMA,
    INITIAL_EPSILON,
    MIN_EPSILON,
    EPSILON_DECAY,
)


class QLearningAgent:
    """
    Tabular Q-Learning agent for the Snake environment.
    """

    def __init__(self, state_size=11, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # Q-table:
        # state -> action values
        self.q_table = {}

        # Learning parameters
        self.learning_rate = Q_LEARNING_RATE
        self.gamma = Q_GAMMA

        # Exploration parameters
        self.epsilon = INITIAL_EPSILON
        self.min_epsilon = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY

    def _state_to_key(self, state):
        """
        Convert NumPy state into a hashable tuple.

        Example:

        [0, 1, 0, 0, 1, ...]
              ↓
        (0, 1, 0, 0, 1, ...)
        """

        return tuple(state.astype(int))

    def _get_q_values(self, state):
        """
        Return Q-values for a state.

        If the state has never been seen before,
        initialize all actions with Q-value 0.
        """

        state_key = self._state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        return self.q_table[state_key]

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy exploration.

        With probability epsilon:
            explore randomly.

        Otherwise:
            exploit the best known action.
        """

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        q_values = self._get_q_values(state)

        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value using the Bellman equation.
        """

        q_values = self._get_q_values(state)

        current_q = q_values[action]

        if done:
            target = reward

        else:
            next_q_values = self._get_q_values(next_state)

            target = reward + self.gamma * np.max(next_q_values)

        # Q-Learning update
        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        q_values[action] = new_q

    def decay_epsilon(self):
        """
        Gradually reduce exploration over time.
        """

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def get_q_table_size(self):
        """Return number of states stored in the Q-table."""

        return len(self.q_table)