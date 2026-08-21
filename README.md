# 🐍 Autonomous Snake Game — Q-Learning & Deep Q-Learning

An autonomous Snake game where reinforcement learning agents learn to play the game through interaction with the environment.

This project implements and compares two reinforcement learning approaches:

- **Tabular Q-Learning**
- **Deep Q-Network (DQN)**

The agents do not receive predefined game strategies. Instead, they learn which actions to take based on rewards received from the environment.

---

## 🚀 Project Overview

The goal of this project is to build an autonomous Snake-playing agent and study the difference between traditional Q-Learning and Deep Q-Learning.

At every step, the agent:

1. Observes the current game state.
2. Selects an action using an epsilon-greedy policy.
3. Receives a reward from the environment.
4. Observes the next state.
5. Updates its knowledge.
6. Repeats the process until the episode ends.

The agent gradually learns to avoid collisions and move toward food.

---

## 🧠 Reinforcement Learning Approaches

### 1. Q-Learning

The Q-Learning agent uses a tabular representation:

```text
State → [Q(left), Q(straight), Q(right)]
The Q-values are updated using the Bellman equation:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

Where:

α = learning rate
γ = discount factor
r = reward
s = current state
a = selected action
s' = next state
2. Deep Q-Network

The DQN agent replaces the Q-table with a neural network.

Game State
    ↓
Dense Layer (128)
    ↓
Dense Layer (128)
    ↓
Output Layer (3)
    ↓
Q-values

The three outputs represent the estimated value of:

[LEFT, STRAIGHT, RIGHT]

The action with the highest predicted Q-value is selected when exploiting the learned policy.

🎯 State Representation

The Snake environment uses an 11-dimensional state representation.

The state contains information about:

Danger straight
Danger left
Danger right
Current movement direction
Relative food position

Example:

[0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]

This compact representation allows the agent to make decisions without directly accessing the entire game board.

🎁 Reward System

The environment uses a custom reward function to encourage useful behavior.

Event	Reward
Food	+10
Survival / movement	+1
Collision	-100

The reward structure encourages the agent to:

Find food
Stay alive
Avoid walls
Avoid self-collision
⚙️ Hyperparameters
Q-Learning
Parameter	Value
Learning Rate	0.001
Discount Factor (γ)	0.95
Initial Epsilon	1.0
Minimum Epsilon	0.01
Epsilon Decay	0.995
Episodes	1,000
DQN
Parameter	Value
Learning Rate	0.001
Discount Factor (γ)	0.95
Batch Size	1,000
Replay Buffer	100,000
Initial Epsilon	1.0
Minimum Epsilon	0.01
Epsilon Decay	0.995
Episodes	1,000
📊 Results

Both agents were trained for 1,000 episodes.

Q-Learning Training
Metric	Result
Maximum Score	19
Mean Score	1.59
Last 100 Episode Average	3.16
Best 100 Episode Average	3.16
Episodes With Food	611 / 1000
Food Acquisition Rate	61.1%
Maximum Reward	122.7
Mean Reward	-6.02
Final Epsilon	0.01
Q-Table States	214
DQN Training
Metric	Result
Maximum Score	60
Mean Score	12.32
Last 100 Episode Average	28.44
Best 100 Episode Average	29.22
Episodes With Food	879 / 1000
Food Acquisition Rate	87.9%
Maximum Reward	455.0
Mean Reward	84.90
Final Epsilon	0.01
Average Training Loss	2.20
🏆 DQN Evaluation

After training, the DQN was evaluated separately for 100 episodes with exploration disabled.

Exploration: OFF
Evaluation Episodes: 100
Metric	Result
Average Score	35.67
Maximum Score	73
Median Score	36
Average Reward	270.44
Average Survival	799.29 steps
Food Acquisition	100%

The evaluation demonstrates that the trained DQN can consistently play the game without random exploration.

📈 Q-Learning vs DQN

The training results show a substantial difference between the two approaches.

Mean Score
Q-Learning : 1.59
DQN        : 12.32

The DQN achieved approximately 7.7× higher mean training score.

Last 100 Episode Average
Q-Learning : 3.16
DQN        : 28.44

The DQN achieved approximately 9× higher average score during the final 100 training episodes.

Food Acquisition
Q-Learning : 61.1%
DQN        : 87.9%

The DQN also demonstrated a substantially higher ability to successfully reach food during training.

📊 Visualizations
Training Performance
Q-Learning

DQN

Q-Learning vs DQN
Average Score Comparison

Best 100 Episode Comparison

Score Comparison

🎮 Autonomous Gameplay

The trained agents can be run directly inside the Pygame environment.

DQN
python -m src.game.play_dqn
Q-Learning
python -m src.game.play_q_learning

The Pygame window displays the autonomous agent playing Snake using its learned policy.
Project Structure
autonomous-snake-dqn/
│
├── assets/
│
├── models/
│   ├── dqn/
│   │   └── .gitkeep
│   │
│   └── q_learning/
│       └── .gitkeep
│
├── notebooks/
│
├── results/
│   ├── plots/
│   │   ├── average_score_comparison.png
│   │   ├── best_100_comparison.png
│   │   └── q_vs_dqn_score.png
│   │
│   ├── dqn_training.png
│   └── q_learning_training.png
│
├── src/
│   │
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   ├── q_learning_agent.py
│   │   └── replay_buffer.py
│   │
│   ├── evaluation/
│   │   └── evaluate_dqn.py
│   │
│   ├── game/
│   │   ├── config.py
│   │   ├── environment.py
│   │   ├── snake.py
│   │   ├── play_dqn.py
│   │   └── play_q_learning.py
│   │
│   ├── training/
│   │   ├── train_dqn.py
│   │   └── train_q_learning.py
│   │
│   ├── utils/
│   │   ├── plot_results.py
│   │   └── plot_dqn_results.py
│   │
│   └── visualization/
│       └── ...
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
🛠️ Technologies
Python
NumPy
Pandas
TensorFlow
Keras
Pygame
Matplotlib
🔬 Reinforcement Learning Concepts

This project demonstrates practical implementation of:

Reinforcement Learning
Q-Learning
Deep Q-Learning
Bellman Equation
Markov Decision Process
Epsilon-Greedy Exploration
Experience Replay
Neural Network Function Approximation
Reward Engineering
Hyperparameter Tuning
Model Evaluation
Training Visualization
💡 Key Learning

The main difference observed in this project is how the two approaches represent and learn the value function.

Q-Learning
State → Q-Table → Action

Q-Learning works well when the state space is small and discrete, but its performance becomes limited as the number of possible states grows.

DQN
State → Neural Network → Q-values → Action

DQN uses a neural network to approximate the Q-function, allowing it to generalize across states rather than maintaining an explicit Q-value for every state.

In this Snake environment, the DQN achieved significantly higher scores and food acquisition rates than the tabular Q-Learning agent.

🚀 Future Improvements

Potential improvements include:

Double DQN
Dueling DQN
Prioritized Experience Replay
Target Network
Larger state representation
Curriculum learning
Reward shaping experiments
Hyperparameter optimization
GPU/WSL2 training
Training video generation
TensorBoard experiment tracking
👨‍💻 Author

Subhash Bishnoi

B.Tech — Artificial Intelligence & Machine Learning

⭐ Project Goal

This project was built to understand how reinforcement learning agents can learn gameplay behavior through trial and error and to compare traditional tabular Q-Learning with neural-network-based Deep Q-Learning.
