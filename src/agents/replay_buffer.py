from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """
    Experience Replay memory for the DQN agent.

    Stores experiences in the form:

        (state, action, reward, next_state, done)

    The agent can later sample random batches
    from this memory for training.
    """

    def __init__(self, capacity=100_000):
        self.capacity = capacity

        self.memory = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store one experience.
        """

        self.memory.append(
            (
                state.copy(),
                action,
                reward,
                next_state.copy(),
                done,
            )
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        """

        batch = random.sample(
            self.memory,
            batch_size,
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        """Return number of stored experiences."""

        return len(self.memory)

    def is_ready(self, batch_size):
        """Check whether enough experiences exist for training."""

        return len(self.memory) >= batch_size