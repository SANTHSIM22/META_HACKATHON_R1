# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dynamic Route Optimizer - Task Generator & Grader.

3 Tasks of increasing difficulty, each with its own grader:

  Task 1 - easy_avoid_blockage  : 1 truck,  3 packages, 50x50 grid
  Task 2 - medium_reroute_fleet : 3 trucks, 9 packages, 100x100 grid
  Task 3 - hard_storm_logistics : 5 trucks, 20 packages, 200x200 grid

Each task has a .grade(state) method that returns a score in [0.0, 1.0].
"""

import copy
import random
from typing import Dict, Any


class DynamicTask:
    """
    Procedurally generates a logistics scenario and grades the final state.

    Grading criteria per task:
      - Easy  : Did the agent avoid the blocked path and deliver 3 packages?
      - Medium: Did the agent reroute 3 trucks efficiently across 9 packages?
      - Hard  : Did the agent handle storm chaos across 5 trucks, 20 packages?

    All grades return a float in [0.0, 1.0].
    """

    def __init__(
        self,
        name: str,
        difficulty: str,
        num_trucks: int,
        num_packages: int,
        grid_size: int = 100,
    ) -> None:
        self.name = name
        self.difficulty = difficulty
        self.num_trucks = num_trucks
        self.num_packages = num_packages
        self.grid_size = grid_size
        self.initial_state: Dict[str, Any] = {}


    def generate_state(self) -> Dict[str, Any]:
        """Procedurally build a fresh simulation state."""

        # 1. Unique node coordinates
        depot = (
            f"Node_{random.randint(0, self.grid_size)}"
            f"_{random.randint(0, self.grid_size)}"
        )
        destinations: set[str] = set()
        while len(destinations) < self.num_packages:
            destinations.add(
                f"Node_{random.randint(0, self.grid_size)}"
                f"_{random.randint(0, self.grid_size)}"
            )
        dest_list = list(destinations)
        all_nodes = [depot] + dest_list

        # 2. Manhattan distance matrix with light traffic noise
        def calc_distance(n1: str, n2: str) -> int:
            _, x1, y1 = n1.split("_")
            _, x2, y2 = n2.split("_")
            base = abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))
            return max(1, base + random.randint(-2, 5))

        distances: Dict[str, Dict[str, int]] = {
            n1: {n2: calc_distance(n1, n2) for n2 in all_nodes if n1 != n2}
            for n1 in all_nodes
        }

        # 3. Packages
        packages = [
            {
                "id": f"PKG_{random.randint(1000, 9999)}",
                "destination": dest_list[i],
                "deadline": random.randint(150, 400),
                "status": "pending",
            }
            for i in range(self.num_packages)
        ]

        # 4. Trucks with shuffled (inefficient) default routes
        trucks = []
        packages_per_truck = self.num_packages // self.num_trucks
        for i in range(self.num_trucks):
            start = i * packages_per_truck
            end = (i + 1) * packages_per_truck
            assigned = [p["id"] for p in packages[start:end]]
            shuffled_dest = list(
                {p["destination"] for p in packages if p["id"] in assigned}
            )
            random.shuffle(shuffled_dest)
            trucks.append(
                {
                    "id": f"TRK_{random.randint(100, 999)}",
                    "current_location": depot,
                    "route_order": shuffled_dest,
                    "assigned_packages": assigned,
                    "fuel": float(random.randint(40, 100)),
                    "fuel_capacity": 100.0,
                }
            )

        # 5. Disruptive event
        node_a = random.choice(all_nodes)
        node_b = random.choice([n for n in all_nodes if n != node_a])

        # 6. Fuel stations
        num_stations = max(1, self.num_packages // 3)
        fuel_stations = random.sample(all_nodes, num_stations)

        self.initial_state = {
            "time_step": 0,
            "trucks": trucks,
            "packages": packages,
            "event": {
                "description": f"Severe weather blocking {node_a} to {node_b}",
                "blocked_edges": [[node_a, node_b], [node_b, node_a]],
                "traffic_delays": {
                    random.choice(dest_list): random.randint(20, 50)
                },
            },
            "distances": distances,
            "fuel_stations": fuel_stations,
        }
        return copy.deepcopy(self.initial_state)

    def grade(self, state: Dict[str, Any]) -> float:
        """
        Grade the final simulation state.

        Scoring formula (same logic, scaled expectations per difficulty):
            base  = on_time_deliveries / total_packages    [0.0 - 1.0]
            penalty = time_elapsed * time_penalty_rate
            late_penalty = late_count * per_late_penalty
            score = base - penalty - late_penalty          clamped [0.0, 1.0]

        Difficulty scaling:
            Easy   : lenient time penalty (0.0002), low late penalty (0.05)
            Medium : moderate time penalty (0.0005), moderate late (0.10)
            Hard   : strict time penalty (0.0008), harsh late penalty (0.15)

        Args:
            state: Current simulation state dict.

        Returns:
            Float in [0.0, 1.0].
        """
        total = len(state["packages"])
        if total == 0:
            return 0.01

        on_time = sum(1 for p in state["packages"] if p["status"] == "delivered")
        late    = sum(1 for p in state["packages"] if p["status"] == "late")

        weights = {
            "easy":   {"time_penalty": 0.0002, "late_penalty": 0.05},
            "medium": {"time_penalty": 0.0005, "late_penalty": 0.10},
            "hard":   {"time_penalty": 0.0008, "late_penalty": 0.15},
        }
        w = weights.get(self.difficulty, weights["medium"])

        base         = on_time / total
        time_penalty = state.get("time_step", 0) * w["time_penalty"]
        late_penalty = late * w["late_penalty"]

        score = base - time_penalty - late_penalty
        return max(0.01, min(0.99, score))



TASKS: Dict[str, DynamicTask] = {
    # Task 1 — Easy
    # Objective : 1 truck avoids 1 blocked road and delivers 3 packages
    # Success   : All 3 packages delivered on time = 1.0
    # Failure   : Any package late or blocked path used = < 0.5
    "easy_avoid_blockage": DynamicTask(
        name="easy_avoid_blockage",
        difficulty="easy",
        num_trucks=1,
        num_packages=3,
        grid_size=50,
    ),

    # Task 2 — Medium
    # Objective : 3 trucks coordinate to deliver 9 packages across a larger city
    # Success   : All packages delivered on time with efficient routes = 1.0
    # Failure   : Late deliveries or blocked paths reduce score proportionally
    "medium_reroute_fleet": DynamicTask(
        name="medium_reroute_fleet",
        difficulty="medium",
        num_trucks=3,
        num_packages=9,
        grid_size=100,
    ),

    # Task 3 — Hard
    # Objective : 5 trucks handle storm chaos, 20 packages, large 200x200 city
    # Success   : Majority of packages delivered on time despite disruptions = 1.0
    # Failure   : Poor routing under pressure leads to cascading late deliveries
    "hard_storm_logistics": DynamicTask(
        name="hard_storm_logistics",
        difficulty="hard",
        num_trucks=5,
        num_packages=20,
        grid_size=200,
    ),
}