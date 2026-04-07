---
title: Dynamic Route Optimizer
emoji: 🚚
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Dynamic Route Optimizer

A dynamic logistics simulation built on the OpenEnv specification. This environment challenges Large Language Models (LLMs) and RL agents to manage delivery fleets, prioritize package deadlines, and instantly reroute trucks when unpredictable obstacles (like severe weather) block the roads.

## Configuration (.env)

Before running the project or the inference scripts, set up your environment variables by copying `.env.example` to `.env`. 

The following variables are available to configure API clients and the environment behavior:

```dotenv
# LLM Provider Configuration (used in inference.py)
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here

# Docker and Environment Configuration
IMAGE_NAME=dynamic-routing-env

# Task Difficulty/Scenario Selection
# Options: 
# - easy_avoid_blockage
# - medium_reroute_fleet 
# - hard_storm_logistics
DRO_TASK=easy_avoid_blockage
```

## Quick Start

The simplest way to use the Dynamic Route Optimizer environment is through the `DynamicRouteEnv` client:

> **NOTE:** Ensure Docker is installed and running before attempting to create an environment from a Docker image. The environment requires 4GB of available RAM for optimal performance.

```python
from client import DynamicRouteEnv
from models import DynamicRouteAction

try:
    # Create environment from Docker image
    env = DynamicRouteEnv.from_docker_image("dynamic-route-env:latest")

    # Reset (generates a brand new map and scenario)
    result = env.reset()
    print(f"Time Step: {result.observation.time_step}")
    print(f"Trucks Available: {len(result.observation.trucks)}")

    # Send a step - Provide route updates for the trucks
    # Format: {"truck_id": "TRK_123", "new_route_order": ["Node_1_2", "Node_4_5"]}
    
    # Let's say we have TRK_100 and valid nodes Node_50_50 and Node_20_20
    action = DynamicRouteAction(
        route_updates=[
            {
                "truck_id": "TRK_100", 
                "new_route_order": ["Node_50_50", "Node_20_20"]
            }
        ]
    )
    
    result = env.step(action)
    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")

finally:
    # Always clean up
    env.close()
```

## How the Simulation Works

1. **Procedural Maps:** Calling `reset()` generates a new set of random Nodes and mathematically connects them via Manhattan Distance.

> **NOTE:** Each call to `reset()` creates a completely new map with different node positions and connections. The random seed can be controlled for reproducibility.

2. **Strict Validation:** The system strictly validates agent route proposals. The proposed nodes must exactly match the internal nodes randomly generated during the current episode.

3. **Unexpected Events:** Certain road segments will become blocked (e.g., storms). Trucks attempting to travel on these blocked paths will be forced to stop, requiring the agent to `step` again with a routed bypass.

> **NOTE:** Blocked edges are bidirectional. If Node_A → Node_B is blocked, then Node_B → Node_A is also blocked in the same step.

4. **Scoring:** The final score (0.0 to 1.0) depends on maximizing on-time deliveries and minimizing inefficient travel.

## Testing in the Web UI (Custom UI)

If you deploy the environment to Hugging Face Spaces or run it locally with the web interface enabled (on `/web`), you can interact with the simulation directly in your browser.

> **NOTE:** The web interface provides real-time visualization of the fleet state, including truck locations, package assignments, and dynamic events. Use the "Get State" button to retrieve current episode metadata including step count and episode ID.

Here is exactly how to submit actions in the Web UI:

1. **Click Reset First:** You must begin an episode by clicking `Reset`. If you try to step before resetting, the UI will throw an error: `Error: No simulation loaded! Click RESET first.`

> **ERROR:** Attempting to send a step action without calling reset first will fail with status 400 and the message "No simulation loaded! Click RESET first."

2. **Copy Valid IDs:** Look at the `trucks` list in the Observation box to get a valid `truck_id` (e.g., `TRK_143`). Look at the `distances` dictionary to get valid node names (e.g., `Node_50_50`). 
3. **Format Your Input:** In the **Action** input box, you must provide a valid JSON object that matches the action schema.

**Example Valid JSON Action Input:**
```json
{
  "route_updates": [
    {
      "truck_id": "TRK_143",
      "new_route_order": [
        "Node_50_50",
        "Node_12_8"
      ]
    }
  ]
}
```

### Common Web UI Errors to Avoid:

> **ERROR:** Empty Updates — If you send `{}` or omit the `route_updates` array, you get: `Error: route_updates is empty! Provide truck_id + new_route_order...`

> **ERROR:** Invalid Truck ID — If you guess a truck ID incorrectly, you get: `Error: Truck 'TRK_999' not found! Valid IDs: [...]`

> **ERROR:** Invalid Node — If you type a node that wasn't generated in *this* round's map, you get: `Error: Node 'Node_99_99' not found in city map! Valid nodes: [...]`

> **ERROR:** Driving into a Storm — If your route crosses the dynamic roadblock mentioned in the `event` object, the truck will stall and you'll see a warning in the Observation's `error` metadata field: `Warning: Truck TRK_143 blocked: Node_A → Node_B. Reroute!`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

> **NOTE:** The Docker build process will install all dependencies via `uv sync` and create a virtual environment within the container. This typically takes 2-5 minutes depending on your internet connection and system resources.

```bash
# From the Dynamic_Routing directory
docker build -t dynamic-routing-env:latest -f Dockerfile .
```

## Running LLM Inference

The repository includes a comprehensive `inference.py` script that connects to the environment, processes the observation, prompts an LLM to make routing decisions, and submits the optimized route back to the environment.

1. Ensure your `.env` file is configured with your API keys and exact `IMAGE_NAME`.
2. Run the script:

```bash
python inference.py
```

This script will automatically stand up the Docker container, run through an entire episode matching your `DRO_TASK` difficulty, log the LLM's thought process, and output the final reward and evaluation score.

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`).
2. Prepare a custom build for Hugging Face Docker space (enables web interface).
3. Upload to Hugging Face.

## Changing Difficulties

The environment ships with 3 difficulty tasks:
- `easy_avoid_blockage`: 1 truck, 3 packages, 50x50 grid
- `medium_reroute_fleet`: 3 trucks, 9 packages, 100x100 grid
- `hard_storm_logistics`: 5 trucks, 20 packages, 200x200 grid

To switch datasets, set the `DRO_TASK` environment variable when starting the server:

```bash
export DRO_TASK="hard_storm_logistics"
uvicorn server.app:app --reload
```

> **NOTE:** The difficulty level determines the size of the map, number of vehicles, and package complexity. Restart the server after changing `DRO_TASK` for the change to take effect. You can also pass `episode_id=f"task:{task_name}"` when calling the client's `reset()` method to switch tasks dynamically.

## Environment Details

### Action (`DynamicRouteAction`)
- `route_updates`: A list containing dictionaries with `truck_id` and `new_route_order` for the fleet.

### Observation (`DynamicRouteObservation`)
- `time_step`: Current simulation time.
- `trucks`: List of truck states (location, assigned packages, pending route order).
- `packages`: Status and deadlines for all packages.
- `event`: Contains the `traffic_delays` and `blocked_edges` for real-time avoidance.
- `distances`: The full procedurally generated dictionary matrix tracking distances between all generated nodes.

## Development & Testing

Run the server locally for development without Docker:

```bash
uvicorn server.app:app --reload
```

> **NOTE:** When running locally, ensure all dependencies are installed via `pip install -r requirements.txt` or `uv sync`. The `--reload` flag automatically restarts the server when you modify Python files, which is useful for development but should not be used in production.

## Environment Server APIs

Alongside standard OpenEnv standard endpoints (`/env/reset`, `/env/step`), the server exposes:

- **`GET /current_state`**: Retrieves the current global state of the simulation outside normal step loops, returning:
  ```json
  {
    "episode_id": "task:easy_avoid_blockage",
    "step_count": 0,
    "observation": { ... }
  }
  ```

## Project Structure

```
Dynamic_Routing/
├── client.py                # Python client for communicating with env
├── models.py                # Action/Observation schemas
├── tasks.py                 # Task Generator & Grading logic
├── inference.py             # LLM Testing Script that acts as an Agent
├── openenv.yaml             # Manifest file
├── pyproject.toml           # Modern Python packaging configuration
├── Dockerfile               # Docker container specification
├── .env.example             # Configuration template
└── server/
    ├── app.py               # REST FastAPI Wrapper & Custom Endpoints
    ├── Dynamic_Routing_environment.py  # Core simulation logic
    └── index.html           # Embedded Javascript/HTML Web Interface
```
