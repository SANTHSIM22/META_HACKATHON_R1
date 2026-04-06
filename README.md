# 🚚 Dynamic Route Optimizer (Meta Hackathon)

Welcome to the **Dynamic Route Optimizer**! This project is an AI-training environment (built on the OpenEnv spec) designed to teach Large Language Models (LLMs) or Reinforcement Learning agents how to manage real-time logistics and delivery fleets.

If you are new to this project, this guide will explain exactly how the simulation works in plain English.

---

## 🎮 What is this Simulation?

Imagine you are a dispatcher for a delivery company. You have trucks, packages with deadlines, and a city map. Your job is to route your trucks efficiently so every package is delivered on time. 

But there's a twist: **The simulation is dynamic.** Roadblocks (like severe storms) will suddenly appear and block certain paths. The agent must instantly adapt, reroute the trucks, and still make the deliveries.

### The 3 Core Mechanics:

1. **The Map Changes Every Round:**
   Every time you hit the "Reset" button, the system generates a completely brand new map. It drops random "Nodes" (intersections) onto a grid, connects them all, and calculates the exact travel time between them (factoring in realistic, random traffic delays).

2. **Actions Must Follow the Rules of the Map:**
   The agent doesn't send "Left/Right" commands. It sends a specific route plan like `["Node_4_12", "Node_45_2"]`. The system strictly validates these routes against the nodes that were randomly generated for that specific round. If the agent hallucinates a fake node, the move is rejected!

3. **Scoring & Penalties:**
   The ultimate goal is a score of `1.0` (100%). You get points for on-time deliveries, but your score drops if:
   * Packages arrive late.
   * You take inefficient paths (wasting time).
   * You attempt to drive through a road that the system explicitly reported as "blocked".

---

## 🛠️ How it Works Under the Hood

The environment is built using Python, FastAPI, and OpenEnv. Here's a breakdown of the process:

### 1. Resetting the Game (`tasks.py`)
When a new episode starts, `tasks.py` runs the `generate_state()` function. This creates the "Truth" for the current game:
* **Depot & Destinations:** Random `(x, y)` grid coordinates (e.g., `Node_12_34`).
* **Distances:** Calculates the time it takes to travel between every single node using "Manhattan Distance" (like driving on city blocks) plus a bit of randomness for traffic.
* **Packages:** Assigns random deadlines and destinations.
* **Events:** Chooses two random nodes to block off (representing a storm or crash).

### 2. The Agent Makes a Move (`models.py`)
The AI looks at the map and sends a `DynamicRouteAction`. This action tells the system:
* Which truck is moving (`TRK_123`).
* What its new path is (`["Node_12_34", "Node_99_1"]`).

### 3. The Server Checks the Move (`Dynamic_Routing_environment.py`)
The server receives the move and verifies it:
* Did the AI use real nodes from *this specific round*? 
* Is the truck trying to drive into a blocked storm path?
If valid, the server calculates the travel time, updates the truck's location, checks if any packages were dropped off, and moves the clock forward.

### 4. Grading
When the game ends (max steps reached or all packages delivered), the system calculates the final grade based on the difficulty level (Easy, Medium, Hard).

---

## 🚀 Getting Started

If you want to run this locally and test the agent:

**1. Build the Docker Container:**
```bash
cd Dynamic_Routing
docker build -t dynamic-router-env:latest -f server/Dockerfile .
```

**2. Run the Server:**
You can deploy it directly via Docker, or if you are developing and want to run it natively in Python:
```bash
uvicorn server.app:app --reload
```
This will start the environment server on port 8000.

**3. Test the Agent:**
Run the inference script to watch the AI try to solve the routing problem in real-time!
```bash
python client.py
# or if you have a test UI configured, open /web in your browser!
```
