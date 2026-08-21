import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent for the Snake environment.

    Uses:
    - Epsilon-greedy exploration
    - Experience replay
    - Target network
    - Bellman equation
    - Mini-batch gradient updates
    """

    def __init__(
        self,
        state_size=11,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        replay_capacity=100_000,
        batch_size=64,
        target_update_frequency=500,
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.target_update_frequency = (
            target_update_frequency
        )

        self.training_steps = 0

        # Experience replay
        self.memory = ReplayBuffer(
            capacity=replay_capacity
        )

        # Online network
        self.model = self._build_model()

        # Target network
        self.target_model = self._build_model()

        # Initially both networks are identical.
        self.update_target_network()

        self.loss_function = (
            keras.losses.MeanSquaredError()
        )

    def _build_model(self):
        """
        Build the neural network.
        """

        model = keras.Sequential(
            [
                layers.Input(
                    shape=(self.state_size,)
                ),

                layers.Dense(
                    128,
                    activation="relu",
                ),

                layers.Dense(
                    128,
                    activation="relu",
                ),

                layers.Dense(
                    self.action_size,
                    activation="linear",
                ),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate
            )
        )

        return model

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """

        if random.random() < self.epsilon:
            return random.randrange(
                self.action_size
            )

        q_values = self.predict(state)

        return int(np.argmax(q_values))

    def predict(self, state):
        """
        Predict Q-values for one state.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        state = state.reshape(1, -1)

        q_values = self.model(
            state,
            training=False,
        )

        return q_values.numpy()[0]

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store an experience.
        """

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def train_step(self):
        """
        Perform one efficient DQN training step.

        Uses:
            Q(s,a) = current network

        and:

            target = r + gamma * max Q_target(s',a')

        """

        if not self.memory.is_ready(
            self.batch_size
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(
            self.batch_size
        )

        states = tf.convert_to_tensor(
            states,
            dtype=tf.float32,
        )

        next_states = tf.convert_to_tensor(
            next_states,
            dtype=tf.float32,
        )

        actions = tf.convert_to_tensor(
            actions,
            dtype=tf.int32,
        )

        rewards = tf.convert_to_tensor(
            rewards,
            dtype=tf.float32,
        )

        dones = tf.convert_to_tensor(
            dones,
            dtype=tf.float32,
        )

        # Calculate target Q-values using
        # the target network.
        next_q_values = self.target_model(
            next_states,
            training=False,
        )

        max_next_q_values = tf.reduce_max(
            next_q_values,
            axis=1,
        )

        targets = rewards + (
            1.0 - dones
        ) * self.gamma * max_next_q_values

        # Train online network.
        with tf.GradientTape() as tape:

            q_values = self.model(
                states,
                training=True,
            )

            # Select Q-values corresponding
            # to actions actually taken.
            action_mask = tf.one_hot(
                actions,
                self.action_size,
            )

            selected_q_values = tf.reduce_sum(
                q_values * action_mask,
                axis=1,
            )

            loss = self.loss_function(
                targets,
                selected_q_values,
            )

        gradients = tape.gradient(
            loss,
            self.model.trainable_variables,
        )

        self.model.optimizer.apply_gradients(
            zip(
                gradients,
                self.model.trainable_variables,
            )
        )

        self.training_steps += 1

        # Periodically synchronize target network.
        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):
            self.update_target_network()

        return float(loss.numpy())

    def update_target_network(self):
        """
        Copy online network weights
        into the target network.
        """

        self.target_model.set_weights(
            self.model.get_weights()
        )

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        """
        Save the trained online network.
        """

        self.model.save(path)

    def load(self, path):
        """
        Load a trained online network.
        """

        self.model = keras.models.load_model(
            path
        )

        self.target_model = self._build_model()

        self.target_model.set_weights(
            self.model.get_weights()
        )