---
title: Dynamic Routing Environment Server
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Dynamic Route Optimizer (Meta Hackathon)

A sophisticated AI-training environment for managing real-time logistics and delivery fleet operations, built on the OpenEnv specification. This project enables training Large Language Models (LLMs) and Reinforcement Learning agents to solve complex vehicle routing problems with dynamic constraints and real-world conditions.

---

## Overview

The Dynamic Route Optimizer simulates a real-time logistics dispatch scenario where:
- Multiple trucks must deliver packages to destinations within specified deadlines
- Road conditions change dynamically (blockages, traffic delays)
- Fuel constraints limit truck operations
- Load transfers between trucks offer optimization opportunities
- Agents must adapt routing strategies in response to environmental changes

### Key Objectives

- Minimize total delivery time
- Ensure all packages meet delivery deadlines
- Optimize fuel consumption
- Navigate around dynamic road blockages
- Maximize cumulative reward through efficient route planning

> **NOTE:** While reset values appear random, they are procedurally generated using a specific logic: nodes are placed as coordinates on a grid (e.g., 50x50 or 200x200 depending on difficulty), and distances are calculated using **Manhattan distance** with slight randomized traffic noise. Trucks are deliberately initialized with shuffled, inefficient routes to challenge the agent. You can use AI to create the route updates based on the raw JSON observation.

---

## Architecture

### Core Components

**1. Environment Generator (`tasks.py`)**
- Generates randomized problem instances across three difficulty levels:
  - Easy: 1 truck, 3 packages
  - Medium: 3 trucks, 9 packages
  - Hard: 5 trucks, 20 packages
- Creates grid-based city maps with Manhattan distance calculations
- assigns random traffic delays and dynamic event blockages
- Specifies fuel stations and package deadlines

**2. Action Handler (`models.py`)**
- Defines action schema: route updates and load transfers
- Validates truck IDs, node identifiers, and route feasibility
- Enforces package assignment constraints
- Type-safe request/response models using Pydantic

**3. Simulation Engine (`Dynamic_Routing_environment.py`)**
- Maintains global simulation state across HTTP requests
- Validates route legality against blocked edges
- Tracks vehicle fuel consumption and refueling
- Manages package delivery status and deadline compliance
- Calculates step-wise rewards based on efficiency metrics
- Handles concurrent sessions while maintaining episode isolation

**4. Web Interface (`server/app.py`)**
- FastAPI-based HTTP server with WebSocket support
- RESTful endpoints for reset, step, and state queries
- Integrated dashboard for real-time monitoring
- Custom state endpoint for episode tracking

**5. Client Library (`client.py`)**
- Synchronous wrapper around OpenEnv WebSocket protocol
- Handles message serialization and state parsing
- Provides convenient interface for training scripts

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- 4GB minimum RAM available

### Installation

1. Navigate to the project directory:
```bash
cd Dynamic_Routing
```

2. Install dependencies using uv:
```bash
uv sync
```

### Running the Server

#### Option A: Docker Deployment (Recommended)

Build and run the container:
```bash
docker build -t dynamic-routing-env .
docker run --env-file .env -p 8000:8000 dynamic-routing-env
```

#### Option B: Local Development

Start the FastAPI server:
```bash
cd Dynamic_Routing
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`

### Accessing the Dashboard

Open your browser and navigate to:
```
http://localhost:8000/web
```

The interactive dashboard provides:
- Session initialization with difficulty selection
- Real-time observation visualization
- Manual action dispatch interface
- Episode status and reward tracking
- **Get State button** to retrieve current episode_id and step_count

---

## API Endpoints

### Core Endpoints

**POST /reset**
- Initializes a new episode
- Parameters: `episode_id` (optional, format: `task:easy_avoid_blockage`)
- Returns: Initial observation and episode metadata

**POST /step**
- Executes one environment step
- Body: `{ "action": { "route_updates": [...], "load_transfers": [...] } }`
- Returns: Updated observation, reward, and done status

**GET /current_state**
- Retrieves episode_id and step_count
- Returns: Current episode state with full observation

**GET /state**
- Retrieves raw episode state from the OpenEnv framework
- Returns: State object with episode_id and step_count

**GET /web**
- Serves interactive web dashboard
- HTML-based interface for manual testing

---

## Inference Scripts

### Cloud LLM Inference (`inference.py`)

Uses cloud-based LLM providers (OpenAI-compatible APIs):

```bash
# Configure environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export OPENAI_API_KEY="your-api-key"
export DRO_TASK="easy_avoid_blockage"
export IMAGE_NAME="dynamic-routing-env"

# Run inference
python Dynamic_Routing/inference.py
```

Output format:
```
[START] task=easy_avoid_blockage env=Dynamic_Routing model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"route_updates":[{"truck_id":"TRK_XX","new_route_order":["Node_XX"]}]} reward=0.75 done=false error=null
[END] success=true steps=5 score=0.92 rewards=0.75,0.80,0.85,0.88,0.92
```

Features:
- Automatically spawns Docker container with environment
- Maintains multi-turn conversations with LLM for error recovery
- Logs detailed step metadata including observations and rewards
- Computes final episode score against difficulty-adjusted thresholds
- Supports environment variable override via `.env` file

---

## Data Structures

### State Object
```python
{
    "episode_id": "task:easy_avoid_blockage",
    "step_count": 5,
    "observation": {
        "time_step": 250,
        "trucks": [...],
        "packages": [...],
        "event": {...},
        "distances": {...},
        "fuel_stations": [...],
        "metadata": {...}
    },
    "reward": 0.75,
    "done": false
}
```

### Action Schema
```python
{
    "route_updates": [
        {
            "truck_id": "TRK_001",
            "new_route_order": ["Node_0_0", "Node_1_2", "Node_3_4"]
        }
    ],
    "load_transfers": [
        {
            "from_truck_id": "TRK_001",
            "to_truck_id": "TRK_002",
            "package_id": "PKG_123"
        }
    ]
}
```

---

## Simulation Mechanics

### Route Validation
- All node identifiers must exist in the current graph
- No two consecutive nodes in a route can form a blocked edge
- Route must include all destinations for assigned packages

### Fuel System
- Each unit of travel distance costs 0.15 fuel units
- Trucks automatically refuel at fuel stations if fuel is below 20 units or insufficient for remaining route
- Stranded trucks cannot move; packages must be transferred to mobile trucks

### Reward Calculation
- Step reward = 1.0 - (travel_time / 150.0), clamped to [0, 1]
- Episode reward averages across all trucks that moved in the step
- Final score = total_reward / (max_steps * 1.0)

### Difficulty Profiles

| Difficulty | Trucks | Packages | Typical Complexity |
|------------|--------|----------|-------------------|
| Easy       | 1      | 3        | Single vehicle routing |
| Medium     | 3      | 9        | Multi-vehicle coordination |
| Hard       | 5      | 20       | Fleet optimization under constraints |

---

## Configuration

### Environment Variables

```env
# API Configuration
API_BASE_URL=https://api.openai.com/v1
API_KEY=your-api-key-here
MODEL_NAME=gpt-4
HF_TOKEN=huggingface-token-optional

# Task Configuration
DRO_TASK=easy_avoid_blockage
IMAGE_NAME=dynamic-routing-env

# LLM Parameters
TEMPERATURE=0.2
MAX_TOKENS=2048
```

---

## Project Structure

```
Dynamic_Routing/
├── server/
│   ├── app.py                          # FastAPI application
│   ├── Dynamic_Routing_environment.py   # Core simulation engine
│   ├── index.html                      # Web dashboard
│   └── requirements.txt
├── client.py                           # Client library
├── models.py                           # Data models and schemas
├── tasks.py                            # Problem generation
├── inference.py                        # LLM inference script
├── Dockerfile                          # Container configuration
├── pyproject.toml                      # Project metadata
└── README.md                           # This file
```

---

## Testing

### Manual Testing via Dashboard

1. Navigate to http://localhost:8000/web
2. Select difficulty level
3. Click "Initialize Session"
4. Click "Get State" to view current episode status
5. Enter action JSON in the dispatch area
6. Click "Send Step" to execute action
7. Monitor reward and episode status in the Raw Response tab

### Automated Testing

Run inference scripts with validation:

```bash
python inference.py 2>&1 | grep -E '\[END\]'
```

Expected output shows final score, steps taken, and success status.

---

## Troubleshooting

### Server Won't Start

Verify port 8000 is available:
```bash
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

### Docker Build Fails

Ensure Dockerfile is in the Dynamic_Routing directory:
```bash
ls -la Dynamic_Routing/Dockerfile
```

### State Not Updating

Confirm environment variables are set:
```bash
echo $DRO_TASK
python -c "import os; print(os.getenv('DRO_TASK'))"
```

---

## Performance Considerations

- Simulation maintains global state in memory; ideal for single concurrent session
- Step latency: 50-200ms depending on action validation complexity
- Container startup time: 2-5 seconds
- Memory footprint: approximately 350MB per active environment

---

## Contributing

When contributing to this project:
1. Maintain OpenEnv specification compliance
2. Add comprehensive docstrings to new functions
3. Include error handling for all user inputs
4. Test changes with multiple difficulty levels
5. Update README with any new features or configuration options

---

## License

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the BSD-style license found in the LICENSE file.
