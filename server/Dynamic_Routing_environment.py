import copy
import os
from uuid import uuid4
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

load_dotenv()

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DynamicRouteAction, DynamicRouteObservation
    from ..tasks import TASKS
except ImportError:
    from models import DynamicRouteAction, DynamicRouteObservation
    from tasks import TASKS

raw_task = os.getenv("DRO_TASK", "easy_avoid_blockage")
ACTIVE_TASK = raw_task.strip("\"'")
# "easy_avoid_blockage"  → 1 truck,  3 packages
# "medium_reroute_fleet" → 3 trucks, 9 packages
# "hard_storm_logistics" → 5 trucks, 20 packages
GLOBAL_SIM: Dict[str, Any] = {}
GLOBAL_STEP_COUNT: int = 0
GLOBAL_EPISODE_ID: str = ""


class DynamicRoutingEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    SUPPORTED_TASKS = list(TASKS.keys())

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = TASKS[ACTIVE_TASK]
        self.max_steps: int = 10

    # RESET
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DynamicRouteObservation:
        global GLOBAL_SIM, GLOBAL_STEP_COUNT, ACTIVE_TASK, GLOBAL_EPISODE_ID

        with open("reset_args.txt", "w") as f:
            f.write(f"episode_id: {episode_id}, seed: {seed}, kwargs: {kwargs}\n")

        if episode_id and episode_id.startswith("task:"):
            task_key = episode_id.split("task:")[1]
            if task_key in TASKS:
                ACTIVE_TASK = task_key
                self._task = TASKS[ACTIVE_TASK]
            episode_id = str(uuid4())

        GLOBAL_STEP_COUNT = 0
        GLOBAL_EPISODE_ID = episode_id or str(uuid4())
        self._state = State(episode_id=GLOBAL_EPISODE_ID, step_count=0)
        GLOBAL_SIM = self._task.generate_state()

        with open("reset_args.txt", "a") as f:
            f.write(
                f"GENERATED TASK NAME: {self._task.name}, TRUCKS: {self._task.num_trucks}\n"
            )
        return self._build_obs()

    def state(self) -> State:
        """Returns the current structured State object."""
        return self._state

    # STEP
    def step(self, action: DynamicRouteAction) -> DynamicRouteObservation:
        global GLOBAL_SIM, GLOBAL_STEP_COUNT

        if not GLOBAL_SIM:
            raise ValueError(" No simulation loaded! Click RESET first.")

        if not action.route_updates:
            truck_ids = [t["id"] for t in GLOBAL_SIM["trucks"]]
            raise ValueError(
                f" route_updates is empty! "
                f"Truck IDs: {truck_ids}. "
                f"Provide truck_id + new_route_order for each truck."
            )

        truck_map = {t["id"]: t for t in GLOBAL_SIM["trucks"]}
        valid_nodes = set(GLOBAL_SIM["distances"].keys())

        for update in action.route_updates:
            if update.truck_id not in truck_map:
                raise ValueError(
                    f" Truck '{update.truck_id}' not found! "
                    f"Valid IDs: {list(truck_map.keys())}. "
                    f"No trucks moved."
                )
            for node in update.new_route_order:
                if node not in valid_nodes:
                    raise ValueError(
                        f" Node '{node}' not found in city map! "
                        f"Valid nodes: {sorted(valid_nodes)}. "
                        f"No trucks moved."
                    )

        GLOBAL_STEP_COUNT += 1
        self._state.step_count = GLOBAL_STEP_COUNT

        error_messages: List[str] = []
        step_rewards: List[float] = []

        for update in action.route_updates:
            truck_map[update.truck_id]["route_order"] = list(update.new_route_order)

        transfer_count = 0
        if hasattr(action, "load_transfers") and action.load_transfers:
            for transfer in action.load_transfers:
                t1 = truck_map.get(transfer.from_truck_id)
                t2 = truck_map.get(transfer.to_truck_id)
                if not t1 or not t2:
                    error_messages.append(f" Transfer failed: Truck not found.")
                    continue
                if t1["current_location"] != t2["current_location"]:
                    error_messages.append(
                        f" Transfer failed: {t1['id']} and {t2['id']} not at same location."
                    )
                    continue
                if transfer.package_id not in t1["assigned_packages"]:
                    error_messages.append(f" Transfer failed: Package not assigned.")
                    continue

                pkg = next(
                    (
                        p
                        for p in GLOBAL_SIM["packages"]
                        if p["id"] == transfer.package_id
                    ),
                    None,
                )
                pkg_dest = pkg["destination"] if pkg else "Unknown"

                t1["assigned_packages"].remove(transfer.package_id)
                t2["assigned_packages"].append(transfer.package_id)
                transfer_count += 1
                error_messages.append(
                    f" Transferred: {transfer.package_id} (Route Dest: {pkg_dest}) from {t1['id']} to {t2['id']}."
                )

        distances = GLOBAL_SIM["distances"]

        def get_fuel_needed(route, start_loc):
            if not route:
                return 0
            cost = 0
            curr = start_loc
            for nxt in route:
                cost += distances.get(curr, {}).get(nxt, 0) * 0.15
                curr = nxt
            return cost

        fuel_stations = GLOBAL_SIM.get("fuel_stations", [])
        for truck in GLOBAL_SIM["trucks"]:
            fuel_needed = get_fuel_needed(
                truck["route_order"], truck["current_location"]
            )

            if truck["current_location"] in fuel_stations and (
                truck["fuel"] < fuel_needed or truck["fuel"] <= 20.0
            ):
                truck["fuel"] = truck.get("fuel_capacity", 100.0)
                fuel_stations.remove(truck["current_location"])
                error_messages.append(
                    f" Truck {truck['id']} fully refueled at {truck['current_location']}! Station is now EMPTY/CLOSED."
                )

        event = GLOBAL_SIM.get("event") or {}
        blocked = event.get("blocked_edges", [])
        delays = event.get("traffic_delays", {})
        package_map = {p["id"]: p for p in GLOBAL_SIM["packages"]}

        for truck in GLOBAL_SIM["trucks"]:
            if not truck["route_order"]:

                continue

            current_loc = truck["current_location"]
            next_stop = truck["route_order"][0]
            current_time = GLOBAL_SIM["time_step"]

            if [current_loc, next_stop] in blocked or [
                next_stop,
                current_loc,
            ] in blocked:
                error_messages.append(
                    f"🚧 Truck {truck['id']} blocked: "
                    f"{current_loc} → {next_stop}. Reroute!"
                )

                continue

            travel_time = distances.get(current_loc, {}).get(
                next_stop, 999
            ) + delays.get(next_stop, 0)

            fuel_cost = travel_time * 0.15
            current_fuel = truck.get("fuel", 100.0)
            if current_fuel < fuel_cost:
                error_messages.append(
                    f" Truck {truck['id']} ran out of fuel! Needed {fuel_cost:.1f} but has {current_fuel:.1f}. Stranded at {current_loc}."
                )
                continue

            truck["fuel"] = current_fuel - fuel_cost
            current_time += travel_time

            hop_efficiency = 1.0 - (travel_time / 150.0)
            step_rewards.append(max(0.0, min(1.0, hop_efficiency)))

            for pkg_id in truck["assigned_packages"]:
                pkg = package_map.get(pkg_id)
                if (
                    pkg
                    and pkg["destination"] == next_stop
                    and pkg["status"] == "pending"
                ):
                    pkg["status"] = (
                        "delivered" if current_time <= pkg["deadline"] else "late"
                    )

            truck["current_location"] = next_stop
            truck["route_order"] = truck["route_order"][1:]
            fuel_needed = get_fuel_needed(truck["route_order"], next_stop)
            if next_stop in fuel_stations and (
                truck["fuel"] < fuel_needed or truck["fuel"] <= 20.0
            ):
                truck["fuel"] = truck.get("fuel_capacity", 100.0)
                fuel_stations.remove(next_stop)
                error_messages.append(
                    f" Truck {truck['id']} fully refueled at {next_stop}! Station is now EMPTY/CLOSED."
                )

        raw_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        reward = raw_reward

        load_time_penalty = transfer_count * 15
        GLOBAL_SIM["time_step"] += 50 + load_time_penalty

        all_done = all(
            p["status"] in ("delivered", "late") for p in GLOBAL_SIM["packages"]
        )
        done = all_done or GLOBAL_STEP_COUNT >= self.max_steps

        error_msg = " | ".join(error_messages) if error_messages else None

        return self._build_obs(done=done, reward=reward, error=error_msg)

    # GRADE
    def grade_task(self, task_id: str) -> float:
        global GLOBAL_SIM
        mapping = {
            "easy": "easy_avoid_blockage",
            "medium": "medium_reroute_fleet",
            "hard": "hard_storm_logistics",
        }
        key = mapping.get(task_id, task_id)
        if key == ACTIVE_TASK and GLOBAL_SIM:
            score = self._task.grade(GLOBAL_SIM)
            return max(0.01, min(0.99, score))
        return 0.01

    def _build_obs(
        self,
        done: bool = False,
        reward: float = 0.0,
        error: Optional[str] = None,
    ) -> DynamicRouteObservation:
        global GLOBAL_SIM
        if not GLOBAL_SIM:
            return DynamicRouteObservation(
                time_step=0,
                trucks=[],
                packages=[],
                event=None,
                distances={},
                fuel_stations=[],
                done=False,
                reward=0.0,
                metadata={"task": ACTIVE_TASK, "error": " Call Reset first!"},
            )
        return DynamicRouteObservation(
            time_step=GLOBAL_SIM["time_step"],
            trucks=[copy.deepcopy(t) for t in GLOBAL_SIM["trucks"]],
            packages=[copy.deepcopy(p) for p in GLOBAL_SIM["packages"]],
            event=copy.deepcopy(GLOBAL_SIM.get("event")),
            distances=copy.deepcopy(GLOBAL_SIM["distances"]),
            fuel_stations=copy.deepcopy(GLOBAL_SIM.get("fuel_stations", [])),
            done=done,
            reward=reward,
            metadata={"task": ACTIVE_TASK, "error": error},
        )

    # STATE
    @property
    def state(self) -> State:
        return self._state
