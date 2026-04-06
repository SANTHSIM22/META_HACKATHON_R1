---
title: Dynamic Route Optimizer
emoji: 🚛
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

## Quick Start

The simplest way to use the Dynamic Route Optimizer environment is through the `DynamicRouteEnv` client:

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
2. **Strict Validation:** The system strictly validates agent route proposals. The proposed nodes must exactly match the internal nodes randomly generated during the current episode.
3. **Unexpected Events:** Certain road segments will become blocked (e.g., storms). Trucks attempting to travel on these blocked paths will be forced to stop, requiring the agent to `step` again with a routed bypass.
4. **Scoring:** The final score (0.0 to 1.0) depends on maximizing on-time deliveries and minimizing inefficient travel.

## Testing in the Web UI (Gradio)

If you deploy the environment to Hugging Face Spaces or run it locally with the web interface enabled (on `/web`), you can interact with the simulation directly in your browser.

Here is exactly how to submit actions in the Web UI:

1. **Click Reset First:** You must begin an episode by clicking `Reset`. If you try to step before resetting, the UI will throw an error: `❌ No simulation loaded! Click RESET first.`
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
* **Empty Updates:** If you send `{}` or omit the `route_updates` array, you get: `❌ route_updates is empty! Provide truck_id + new_route_order...`
* **Invalid Truck ID:** If you guess a truck ID incorrectly, you get: `❌ Truck 'TRK_999' not found! Valid IDs: [...]`
* **Invalid Node:** If you type a node that wasn't generated in *this* round's map, you get: `❌ Node 'Node_99_99' not found in city map! Valid nodes: [...]`
* **Driving into a Storm:** If your route crosses the dynamic roadblock mentioned in the `event` object, the truck will stall and you'll see a warning in the Observation's `error` metadata field: `🚧 Truck TRK_143 blocked: Node_A → Node_B. Reroute!`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the Dynamic_Routing directory
docker build -t dynamic-route-env:latest -f server/Dockerfile .
```

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

To switch datasets, open `server/Dynamic_Routing_environment.py` and modify the `ACTIVE_TASK` constant, then restart your server.

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

## Project Structure

```
Dynamic_Routing/
├── client.py                # Python client for communicating with env
├── models.py                # Action/Observation schemas
├── tasks.py                 # Task Generator & Grading logic
├── openenv.yaml             # Manifest file
├── requirements.txt         # Dependencies
└── server/
    ├── Dynamic_Routing_environment.py  # Core simulation logic
    ├── app.py               # REST/WebSocket FastAPI Wrapper
    └── Dockerfile           # Docker configuration
```
