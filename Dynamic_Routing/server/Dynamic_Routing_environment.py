import copy
from uuid import uuid4
from typing import Dict, Any, Optional, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DynamicRouteAction, DynamicRouteObservation
    from ..tasks import TASKS
except ImportError:
    from models import DynamicRouteAction, DynamicRouteObservation
    from tasks import TASKS

# ── CHANGE THIS LINE TO SWITCH TASKS ─────────────────────────────────────────
ACTIVE_TASK = "easy_avoid_blockage"
# "easy_avoid_blockage"  → 1 truck,  3 packages
# "medium_reroute_fleet" → 3 trucks, 9 packages
# "hard_storm_logistics" → 5 trucks, 20 packages
# ─────────────────────────────────────────────────────────────────────────────

# 🚨 GLOBAL STATE — survives between HTTP requests 🚨
GLOBAL_SIM: Dict[str, Any] = {}
GLOBAL_STEP_COUNT: int = 0


class DynamicRoutingEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task  = TASKS[ACTIVE_TASK]
        self.max_steps: int = 10

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self) -> DynamicRouteObservation:
        global GLOBAL_SIM, GLOBAL_STEP_COUNT
        GLOBAL_STEP_COUNT = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        GLOBAL_SIM  = self._task.generate_state()
        return self._build_obs()

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action: DynamicRouteAction) -> DynamicRouteObservation:  # type: ignore[override]
        global GLOBAL_SIM, GLOBAL_STEP_COUNT

        # GUARD 1: Must reset first
        if not GLOBAL_SIM:
            raise ValueError(
                "❌ No simulation loaded! Click RESET first."
            )

        # GUARD 2: route_updates mandatory
        if not action.route_updates:
            truck_ids = [t["id"] for t in GLOBAL_SIM["trucks"]]
            raise ValueError(
                f"❌ route_updates is empty! "
                f"Truck IDs: {truck_ids}. "
                f"Provide truck_id + new_route_order for each truck."
            )

        # GUARD 3: Validate truck IDs and nodes
        truck_map   = {t["id"]: t for t in GLOBAL_SIM["trucks"]}
        valid_nodes = set(GLOBAL_SIM["distances"].keys())

        for update in action.route_updates:
            if update.truck_id not in truck_map:
                raise ValueError(
                    f"❌ Truck '{update.truck_id}' not found! "
                    f"Valid IDs: {list(truck_map.keys())}. "
                    f"No trucks moved."
                )
            for node in update.new_route_order:
                if node not in valid_nodes:
                    raise ValueError(
                        f"❌ Node '{node}' not found in city map! "
                        f"Valid nodes: {sorted(valid_nodes)}. "
                        f"No trucks moved."
                    )

        # ── All guards passed — apply updates ─────────────────────────────
        GLOBAL_STEP_COUNT += 1
        self._state.step_count = GLOBAL_STEP_COUNT

        # Apply ALL route updates first before simulating
        for update in action.route_updates:
            truck_map[update.truck_id]["route_order"] = list(update.new_route_order)

        # ── Simulate exactly ONE hop per truck ────────────────────────────
        distances   = GLOBAL_SIM["distances"]
        event       = GLOBAL_SIM.get("event") or {}
        blocked     = event.get("blocked_edges", [])
        delays      = event.get("traffic_delays", {})
        package_map = {p["id"]: p for p in GLOBAL_SIM["packages"]}

        # FIX: collect rewards from ALL trucks, average at the end
        step_rewards: List[float] = []
        error_messages: List[str] = []

        for truck in GLOBAL_SIM["trucks"]:
            if not truck["route_order"]:
                # Truck finished all stops — nothing to do
                continue

            current_loc  = truck["current_location"]
            next_stop    = truck["route_order"][0]
            current_time = GLOBAL_SIM["time_step"]

            # FIX: Check blockage BEFORE moving
            if (
                [current_loc, next_stop] in blocked
                or [next_stop, current_loc] in blocked
            ):
                error_messages.append(
                    f"🚧 Truck {truck['id']} blocked: "
                    f"{current_loc} → {next_stop}. Reroute!"
                )
                # FIX: Truck stays put, route_order unchanged (don't remove the stop)
                continue

            # Calculate travel time
            travel_time   = (
                distances.get(current_loc, {}).get(next_stop, 999)
                + delays.get(next_stop, 0)
            )
            current_time += travel_time

            # FIX: Collect reward per truck, average later
            hop_efficiency = 1.0 - (travel_time / 150.0)
            step_rewards.append(max(0.0, min(1.0, hop_efficiency)))

            # FIX: Check package delivery at next_stop (not current_loc)
            for pkg_id in truck["assigned_packages"]:
                pkg = package_map.get(pkg_id)
                if (
                    pkg
                    and pkg["destination"] == next_stop  # ← correct destination
                    and pkg["status"] == "pending"
                ):
                    pkg["status"] = (
                        "delivered" if current_time <= pkg["deadline"] else "late"
                    )

            # FIX: Move truck to next_stop and remove it from route_order
            truck["current_location"] = next_stop
            truck["route_order"]      = truck["route_order"][1:]  # remove first hop only

        # FIX: Average reward across ALL trucks that moved this step
        reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

        # Advance simulation time
        GLOBAL_SIM["time_step"] += 50

        # Check episode completion
        all_done = all(
            p["status"] in ("delivered", "late") for p in GLOBAL_SIM["packages"]
        )
        done = all_done or GLOBAL_STEP_COUNT >= self.max_steps

        # Combine all error messages
        error_msg = " | ".join(error_messages) if error_messages else None

        return self._build_obs(done=done, reward=reward, error=error_msg)

    # ── grader ────────────────────────────────────────────────────────────

    def grade_task(self, task_id: str) -> float:
        global GLOBAL_SIM
        mapping = {
            "easy":   "easy_avoid_blockage",
            "medium": "medium_reroute_fleet",
            "hard":   "hard_storm_logistics",
        }
        key = mapping.get(task_id, task_id)
        if key == ACTIVE_TASK and GLOBAL_SIM:
            return self._task.grade(GLOBAL_SIM)
        return 0.0

    # ── build observation ─────────────────────────────────────────────────

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
                done=False,
                reward=0.0,
                metadata={"task": ACTIVE_TASK, "error": "❌ Call Reset first!"},
            )
        return DynamicRouteObservation(
            time_step=GLOBAL_SIM["time_step"],
            trucks=[copy.deepcopy(t) for t in GLOBAL_SIM["trucks"]],
            packages=[copy.deepcopy(p) for p in GLOBAL_SIM["packages"]],
            event=copy.deepcopy(GLOBAL_SIM.get("event")),
            distances=copy.deepcopy(GLOBAL_SIM["distances"]),
            done=done,
            reward=reward,
            metadata={"task": ACTIVE_TASK, "error": error},
        )

    @property
    def state(self) -> State:
        return self._state