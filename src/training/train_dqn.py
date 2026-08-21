import os

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DQNAgent
from src.game.environment import SnakeEnvironment


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

NUM_EPISODES = 1000

STATE_SIZE = 11
ACTION_SIZE = 3

LEARNING_RATE = 0.001
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 64

TARGET_UPDATE_FREQUENCY = 500

TRAIN_EVERY = 4


# --------------------------------------------------
# DQN Training
# --------------------------------------------------

def train_dqn(num_episodes=NUM_EPISODES):

    env = SnakeEnvironment()

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,

        learning_rate=LEARNING_RATE,
        gamma=GAMMA,

        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,

        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,

        target_update_frequency=TARGET_UPDATE_FREQUENCY,
    )

    results = []

    for episode in range(1, num_episodes + 1):

        state = env.reset()

        total_reward = 0
        total_loss = 0

        training_steps = 0
        steps = 0

        done = False

        # ------------------------------------------
        # Play one episode
        # ------------------------------------------

        while not done:

            # Agent chooses an action.
            action = agent.choose_action(state)

            # Environment performs the action.
            next_state, reward, done = env.step(action)

            # Store experience.
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # Train every N environment steps.
            if steps % TRAIN_EVERY == 0:

                loss = agent.train_step()

                if loss is not None:
                    total_loss += loss
                    training_steps += 1

            state = next_state

            total_reward += reward
            steps += 1

        # ------------------------------------------
        # Episode finished
        # ------------------------------------------

        agent.decay_epsilon()

        score = env.get_score()

        if training_steps > 0:
            average_loss = (
                total_loss / training_steps
            )
        else:
            average_loss = 0.0

        results.append(
            {
                "episode": episode,
                "score": score,
                "total_reward": total_reward,
                "steps": steps,
                "epsilon": agent.epsilon,
                "loss": average_loss,
                "replay_size": len(agent.memory),
            }
        )

        # ------------------------------------------
        # Progress output
        # ------------------------------------------

        if episode == 1 or episode % 10 == 0:

            recent_scores = [
                result["score"]
                for result in results[-50:]
            ]

            average_score = np.mean(
                recent_scores
            )

            print(
                f"Episode: {episode:4d} | "
                f"Score: {score:3d} | "
                f"Avg Score: {average_score:.2f} | "
                f"Reward: {total_reward:7.2f} | "
                f"Loss: {average_loss:.5f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Memory: {len(agent.memory)}"
            )

    return agent, results


# --------------------------------------------------
# Save training results
# --------------------------------------------------

def save_results(results):

    os.makedirs(
        "results",
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    output_path = (
        "results/dqn_results.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nTraining results saved to: "
        f"{output_path}"
    )


# --------------------------------------------------
# Save trained model
# --------------------------------------------------

def save_model(agent):

    os.makedirs(
        "models/dqn",
        exist_ok=True,
    )

    model_path = (
        "models/dqn/snake_dqn.keras"
    )

    agent.save(model_path)

    print(
        f"DQN model saved to: "
        f"{model_path}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("DQN SNAKE TRAINING")
    print("=" * 60)

    print(f"Episodes: {NUM_EPISODES}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Gamma: {GAMMA}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Replay Capacity: {REPLAY_CAPACITY}")
    print(f"Train Every: {TRAIN_EVERY} steps")
    print(
        f"Target Update: "
        f"every {TARGET_UPDATE_FREQUENCY} steps"
    )

    print("=" * 60)

    agent, results = train_dqn()

    save_results(results)

    save_model(agent)

    print("\nTraining completed successfully.")