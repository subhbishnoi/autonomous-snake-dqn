# 🐍 Autonomous Snake Game — Q-Learning & Deep Q-Learning

An autonomous Snake game where Reinforcement Learning agents learn to play through trial and error.

This project implements and compares two approaches:

* **Tabular Q-Learning**
* **Deep Q-Network (DQN)**

The agents interact with a custom Snake environment built using **Python and Pygame**, receive rewards and penalties from the environment, and progressively improve their gameplay through reinforcement learning.

---

## 🎯 Project Objective

The goal is not to program the snake with predefined rules.

Instead, the agent starts with little or no knowledge of the game and learns a strategy through repeated interaction:

```text
             ┌──────────────────────┐
             │   Snake Environment  │
             │       Pygame         │
             └──────────┬───────────┘
                        │
                        ▼
                   State
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        Q-Learning               DQN
        Q-Table             Neural Network
              │                   │
              └─────────┬─────────┘
                        │
                        ▼
                     Action
                        │
                        ▼
                Snake Environment
                        │
                        ▼
                     Reward
                        │
                        ▼
                      Learn
                        │
                        └──────────► Repeat
```

The objective is to maximize cumulative reward while learning to:

* Find food
* Avoid collisions
* Navigate efficiently
* Survive for longer
* Improve its average score

---

# 🚀 Features

* 🤖 Autonomous Snake agent
* 🧠 Tabular Q-Learning implementation
* 🧠 Deep Q-Network implementation
* 🎮 Custom Pygame environment
* 🎯 ε-greedy exploration
* 🔄 Experience Replay
* 🎯 DQN Target Network
* 🎁 Configurable reward function
* 📊 Training metrics
* 📈 Learning/convergence curves
* ⚖️ Q-Learning vs DQN comparison
* 💾 Trained model saving/loading
* 🔬 Reproducible experiments

---

# 🧠 Reinforcement Learning Setup

The Snake environment follows the standard reinforcement learning loop:

```text
State
  ↓
Agent
  ↓
Action
  ↓
Environment
  ↓
Reward + Next State
  ↓
Agent learns
  ↓
Next step
```

At every time step, the agent observes the current state and chooses an action.

The environment then returns:

```text
(state, action, reward, next_state, done)
```

---

# 🎮 Environment

The game environment is implemented using **Pygame**.

The agent can choose between three relative actions:

```text
LEFT
STRAIGHT
RIGHT
```

The action space is deliberately kept small so that the learning problem focuses on navigation and decision-making rather than unnecessary complexity.

---

# 🧩 State Representation

The agent receives a compact representation of the game state rather than raw game pixels.

The state contains information about:

### Collision danger

```text
Danger straight
Danger right
Danger left
```

### Current direction

```text
Moving left
Moving right
Moving up
Moving down
```

### Food location

```text
Food left
Food right
Food up
Food down
```

The resulting state vector contains **11 features**.

Example:

```text
[
    danger_straight,
    danger_right,
    danger_left,

    direction_left,
    direction_right,
    direction_up,
    direction_down,

    food_left,
    food_right,
    food_up,
    food_down
]
```

---

# 🎁 Reward Function

The environment uses reward shaping to encourage useful behavior.

| Event                        | Reward |
| ---------------------------- | -----: |
| 🍎 Food collected            |    +10 |
| 🐍 Survival / valid movement |     +1 |
| 💥 Collision                 |   -100 |

The reward values are configurable so different reward strategies can be tested experimentally.

---

# 🧠 Q-Learning

The first agent uses traditional tabular Q-Learning.

The agent maintains a Q-table representing the expected value of taking an action in a particular state.

```text
State
  │
  ▼
Q-Table
  │
  ├── Left
  ├── Straight
  └── Right
          │
          ▼
      Best Action
```

The Q-Learning agent uses an ε-greedy strategy to balance:

* Exploration
* Exploitation

---

# 🤖 Deep Q-Network

The second agent replaces the Q-table with a neural network.

```text
11-Dimensional State
        │
        ▼
Dense Layer
256 neurons
        │
        ▼
ReLU
        │
        ▼
Dense Layer
256 neurons
        │
        ▼
ReLU
        │
        ▼
Output Layer
3 Q-values
```

The three outputs represent the estimated value of:

```text
Q(left)
Q(straight)
Q(right)
```

The action with the highest predicted Q-value is selected during exploitation.

---

# 🔄 Experience Replay

The DQN stores previous experiences in a replay buffer:

```text
(state,
 action,
 reward,
 next_state,
 done)
```

The agent randomly samples batches from this memory during training.

This helps reduce correlations between consecutive experiences and improves training stability.

Initial replay memory:

```text
100,000 experiences
```

---

# 🎯 Target Network

The DQN uses two neural networks:

```text
Policy Network
      │
      └── Predicts current Q-values

Target Network
      │
      └── Generates stable target Q-values
```

The target network is periodically synchronized with the policy network.

This helps stabilize DQN training.

---

# ⚙️ Hyperparameters

The initial experiment uses:

| Parameter         |           Value |
| ----------------- | --------------: |
| Learning Rate     |           0.001 |
| Discount Factor γ |            0.95 |
| Batch Size        |            1000 |
| Replay Memory     |         100,000 |
| Optimizer         |            Adam |
| Exploration       |        ε-greedy |
| Training          | 1,000+ episodes |

These values will be configurable so that we can perform controlled experiments.

---

# 📊 Evaluation

The agents will be evaluated using:

* Average score
* Maximum score
* Average reward
* Win rate
* Episode length
* Training loss
* Exploration rate
* Convergence speed

The project will compare:

```text
                 Q-Learning        DQN
Average Score       --              --
Maximum Score       --              --
Win Rate            --              --
Avg Reward          --              --
Convergence         --              --
```

> Final performance numbers will be generated from the actual training and evaluation runs.

---

# 📈 Results

Training automatically generates performance visualizations.

### Q-Learning Training

### DQN Training

### Algorithm Comparison

---

# 🏆 Benchmark

The project will benchmark both agents under the same environment and evaluation conditions.

The final benchmark will report measured results such as:

```text
Episodes trained:       1,000+
Average score:          --
Maximum score:          --
Win rate:               --
Convergence episode:    --
```

All reported performance metrics will be generated from the experiment logs.

---

# 🛠️ Tech Stack

### Programming

* Python

### Reinforcement Learning

* Q-Learning
* Deep Q-Learning
* Experience Replay
* ε-greedy exploration
* Target Networks
* Reward Engineering

### Machine Learning

* TensorFlow / Keras
* NumPy

### Environment

* Pygame

### Data & Visualization

* Pandas
* Matplotlib

### Development

* Git
* GitHub
* vs code

---

# 📂 Project Structure

```text
autonomous-snake-dqn/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── assets/
│   ├── demo.gif
│   ├── q_learning_training.png
│   ├── dqn_training.png
│   ├── comparison.png
│   └── architecture.png
│
├── models/
│   ├── q_learning/
│   └── dqn/
│
├── results/
│   ├── q_learning_results.csv
│   ├── dqn_results.csv
│   └── comparison_results.csv
│
├── notebooks/
│   └── training_analysis.ipynb
│
└── src/
    │
    ├── game/
    │   ├── config.py
    │   ├── snake.py
    │   └── environment.py
    │
    ├── agents/
    │   ├── q_learning_agent.py
    │   ├── dqn_agent.py
    │   └── replay_memory.py
    │
    ├── training/
    │   ├── train_q_learning.py
    │   ├── train_dqn.py
    │   └── evaluate.py
    │
    └── utils/
        ├── metrics.py
        ├── plotting.py
        └── model_utils.py
```

---

# ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/autonomous-snake-dqn.git

cd autonomous-snake-dqn
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🎮 Running the Project

## Train Q-Learning

```bash
python -m src.training.train_q_learning
```

## Train DQN

```bash
python -m src.training.train_dqn
```

## Evaluate Agents

```bash
python -m src.training.evaluate
```

---

# 🔬 Experiments

The project is designed for controlled experimentation.

Possible experiments include:

### Reward Engineering

Compare different reward strategies.

### Hyperparameter Tuning

Test:

```text
Learning Rate
Gamma
Batch Size
Replay Memory
Epsilon Decay
```

### Algorithm Comparison

Compare:

```text
Q-Learning
vs
DQN
```

using identical environments and evaluation criteria.

---

# 🔮 Future Improvements

Potential extensions:

* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* CNN-based visual input
* Curriculum Learning
* TensorBoard experiment tracking
* Automated hyperparameter optimization
* Dockerized training
* GPU-accelerated experiments

---

# 📚 Concepts Demonstrated

This project demonstrates practical understanding of:

1. Reinforcement Learning
2. Markov Decision Processes
3. Q-Learning
4. Deep Q-Networks
5. Bellman equation
6. Exploration vs exploitation
7. Reward engineering
8. Experience Replay
9. Target Networks
10. Hyperparameter tuning
11. Model evaluation
12. Training convergence
13. Reinforcement Learning environments

---

# 👨‍💻 Author

**Subhash Bishnoi**

B.Tech — Artificial Intelligence & Machine Learning

Focused on Machine Learning, Deep Learning, Reinforcement Learning, and ML Engineering.

---

# ⭐ Acknowledgements

This project was developed as a practical exploration of Reinforcement Learning and Deep Reinforcement Learning through an interactive game environment.

If you found the project useful, consider giving the repository a ⭐.
