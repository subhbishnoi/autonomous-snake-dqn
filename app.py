
from pathlib import Path
import pickle
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent
from src.game.environment import SnakeEnvironment


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "dqn" / "snake_dqn.keras"
Q_TABLE_PATH = BASE_DIR / "models" / "q_learning" / "q_table.pkl"

TEMPLATES_DIR = BASE_DIR / "templates"
RESULTS_DIR = BASE_DIR / "results"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Autonomous Snake AI",
    description="Play Snake using trained DQN and Q-learning agents.",
    version="1.0.0",
)


# Serve comparison graphs and other results.
# No /static mount is needed because the HTML has
# its CSS and JavaScript embedded.
if RESULTS_DIR.is_dir():
    app.mount(
        "/results",
        StaticFiles(directory=str(RESULTS_DIR)),
        name="results",
    )


# --------------------------------------------------
# Load trained agents
# --------------------------------------------------

def load_dqn_agent():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"DQN model not found: {MODEL_PATH}"
        )

    agent = DQNAgent(
        state_size=11,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        replay_capacity=100_000,
        batch_size=64,
        target_update_frequency=500,
    )

    agent.load(str(MODEL_PATH))
    agent.epsilon = 0.0

    return agent


def load_q_learning_agent():
    if not Q_TABLE_PATH.is_file():
        raise FileNotFoundError(
            f"Q-learning table not found: {Q_TABLE_PATH}"
        )

    agent = QLearningAgent(
        state_size=11,
        action_size=3,
    )

    with open(Q_TABLE_PATH, "rb") as file:
        agent.q_table = pickle.load(file)

    agent.epsilon = 0.0

    return agent


# Load both agents when the app starts.
# If either model is missing or invalid, startup fails
# with a clear error in the Render logs.
dqn_agent = load_dqn_agent()
q_agent = load_q_learning_agent()


# --------------------------------------------------
# Game sessions
# --------------------------------------------------

games = {}


class StartRequest(BaseModel):
    agent: str


class StepRequest(BaseModel):
    game_id: str


# --------------------------------------------------
# Helper: prepare game response
# --------------------------------------------------

def game_state(game_id: str):
    game = games[game_id]
    env = game["env"]

    return {
        "game_id": game_id,
        "agent": game["agent_name"],
        "score": env.get_score(),
        "steps": game["steps"],
        "done": game["done"],
        "snake": [
            {"x": int(x), "y": int(y)}
            for x, y in env.snake.body
        ],
        "food": {
            "x": int(env.snake.food[0]),
            "y": int(env.snake.food[1]),
        },
    }


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    index_file = TEMPLATES_DIR / "index.html"

    if not index_file.is_file():
        raise HTTPException(
            status_code=500,
            detail="templates/index.html was not found.",
        )

    return FileResponse(index_file)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dqn_loaded": dqn_agent is not None,
        "q_learning_loaded": q_agent is not None,
    }


@app.post("/api/start")
def start_game(request: StartRequest):
    agent_name = request.agent.strip().lower()

    if agent_name == "dqn":
        agent = dqn_agent
    elif agent_name in ("q_learning", "q-learning"):
        agent = q_agent
        agent_name = "q_learning"
    else:
        raise HTTPException(
            status_code=400,
            detail="Agent must be 'dqn' or 'q_learning'.",
        )

    game_id = str(uuid.uuid4())

    env = SnakeEnvironment()
    state = env.reset()

    games[game_id] = {
        "env": env,
        "agent": agent,
        "agent_name": agent_name,
        "state": state,
        "steps": 0,
        "done": False,
    }

    return game_state(game_id)


@app.post("/api/step")
def step_game(request: StepRequest):
    game_id = request.game_id

    if game_id not in games:
        raise HTTPException(
            status_code=404,
            detail="Game session not found.",
        )

    game = games[game_id]

    if game["done"]:
        return game_state(game_id)

    agent = game["agent"]
    state = game["state"]

    action = agent.choose_action(state)

    next_state, reward, done = game["env"].step(action)

    game["state"] = next_state
    game["steps"] += 1
    game["done"] = done

    return game_state(game_id)