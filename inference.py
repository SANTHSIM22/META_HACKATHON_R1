"""
Inference Script — Dynamic Route Optimizer
==========================================
MANDATORY ENV VARS:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier.
    HF_TOKEN          Your Hugging Face / API key.
    IMAGE_NAME        The name of the local Docker image.

STDOUT FORMAT:
    [START] task=<task_name> env=Dynamic_Routing model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from models import (
    DynamicRouteAction,
    DynamicRouteObservation,
    RouteUpdate,
    LoadTransfer,
)
from client import DynamicRouteEnv


IMAGE_NAME = os.getenv("IMAGE_NAME", "dynamic-routing-env")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


TASKS_TO_RUN = ["easy_avoid_blockage", "medium_reroute_fleet", "hard_storm_logistics"]
BENCHMARK = "Dynamic_Routing"
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5


TEMPERATURE = 0.2
MAX_TOKENS = 2048

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert AI logistics controller managing a delivery fleet.

    You will receive a JSON observation containing:
      - time_step      : current simulation time
      - trucks         : list of trucks with id, current_location, route_order, assigned_packages, fuel, fuel_capacity
      - packages       : list of packages with id, destination, deadline, status
      - event          : a disruptive event with blocked_edges and traffic_delays
      - distances      : travel time matrix between all nodes
      - fuel_stations  : list of station nodes where trucks automatically refuel
      - metadata.error : error message from last step (null if no error)

    YOUR JOB:
      Return a JSON object with route updates for trucks, and any optional load transfers.
      
    RESPONSE FORMAT:
    {
      "route_updates": [
        {
          "truck_id": "TRK_XXX",
          "new_route_order": ["Node_A_B", "Node_C_D"]
        }
      ],
      "load_transfers": [
        {
          "from_truck_id": "TRK_111",
          "to_truck_id": "TRK_222",
          "package_id": "PKG_999"
        }
      ]
    }

    CRITICAL RULES:
      1. NEVER use a blocked edge (listed in event.blocked_edges).
      2. A route is a sequence of NODES. Check if consecutive nodes form a blocked edge.
      3. Each truck must visit ALL destinations of its assigned packages.
      4. Deliver packages before their deadlines. Minimize total travel time.
      5. WATCH YOUR FUEL! Traveling costs fuel. You DO NOT need to visit a fuel station unless your current fuel is insufficient to complete your remaining route. ONLY if fuel is critically low, inject a `fuel_stations` node into your `new_route_order` to refuel. Do not visit fuel stations unnecessarily.
      6. TRANSFER LOADS ONLY IF EFFICIENT: You MAY use `load_transfers` to transfer a package from one truck to another if it significantly saves overall delivery time. Both trucks must meet at the EXACT same node. Loading takes 15 minutes, so ONLY transfer if the receiving truck can cut travel time by more than 15 minutes compared to the original truck. Otherwise, do not transfer.
      7. If a truck reaches 0 fuel, it is STRANDED. You must use `load_transfers` to assign an active truck to its location node to pick up its packages.
      8. Use EXACT truck IDs, node names, and package IDs from the observation.
      9. Return ONLY valid JSON, no markdown, no explanation.

    If metadata.error is present, analyze the error and fix it in your next response.
"""
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    action_str = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(obs: DynamicRouteObservation, step: int) -> str:
    """Build an enhanced user prompt with route planning guidance."""
    data = {
        "time_step": obs.time_step,
        "trucks": [t.model_dump() for t in obs.trucks],
        "packages": [p.model_dump() for p in obs.packages],
        "event": obs.event.model_dump() if obs.event else None,
        "distances": obs.distances,
        "fuel_stations": obs.fuel_stations,
        "metadata": obs.metadata if obs.metadata else {},
    }

    blocked_edges = obs.event.blocked_edges if obs.event else []

    truck_guidance = []
    for truck in obs.trucks:

        destinations = []
        for pkg in obs.packages:
            if (
                pkg.id in truck.assigned_packages
                and pkg.destination not in destinations
            ):
                destinations.append(pkg.destination)

        guidance = f"\n  {truck.id}:"
        guidance += f"\n    - Current location: {truck.current_location}"
        guidance += f"\n    - Fuel Check: {truck.fuel:.1f} / {truck.fuel_capacity:.1f}"
        guidance += f"\n    - Must visit: {destinations}"
        guidance += f"\n    - Current route: {truck.route_order}"
        truck_guidance.append(guidance)

    return textwrap.dedent(
        f"""
        Current observation (Step {step}):
        {json.dumps(data, indent=2)}

        BLOCKED EDGES (NEVER USE):
        {json.dumps(blocked_edges, indent=2)}

        TRUCK REQUIREMENTS:
        {''.join(truck_guidance)}

        CRITICAL INSTRUCTIONS:
        1. Check if ANY consecutive nodes in your route form a blocked edge
        2. For example, if ["A", "B"] is blocked, route [..., "A", "B", ...] is INVALID
        3. Find alternative paths using intermediate nodes from the distances matrix
        4. Each truck MUST visit ALL its package destinations
        5. Use EXACT node and truck IDs from the observation above
        6. Fuel stations are strictly OPTIONAL. Only detour to a `fuel_stations` node if a truck does not have enough fuel to finish its remaining route.
        7. Evaluate possible load transfers. ONLY return a `load_transfers` object if meeting at a node saves more than 15 minutes of combined travel time.

        Output ONLY the JSON object with route_updates for ALL trucks.
    """
    ).strip()


def parse_llm_response(text: str) -> Optional[DynamicRouteAction]:
    """Parse LLM response into a DynamicRouteAction."""
    try:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")

            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        data = json.loads(cleaned)

        updates = []
        for u in data.get("route_updates", []):
            updates.append(
                RouteUpdate(
                    truck_id=u["truck_id"], new_route_order=u["new_route_order"]
                )
            )

        transfers = []
        for t in data.get("load_transfers", []):
            transfers.append(
                LoadTransfer(
                    from_truck_id=t["from_truck_id"],
                    to_truck_id=t["to_truck_id"],
                    package_id=t["package_id"],
                )
            )

        return DynamicRouteAction(route_updates=updates, load_transfers=transfers)

    except Exception as e:
        print(f"[DEBUG] Parse failed: {e}", file=sys.stderr, flush=True)
        return None


async def get_model_action(
    client: OpenAI, obs: DynamicRouteObservation, step: int, conversation: List[dict]
) -> tuple[Optional[DynamicRouteAction], str]:
    """
    Get action from the model.
    Runs formatting and sync generation inside a thread to
    prevent freezing the asyncio event loop and causing WebSocket
    1011 Keepalive Ping Timeouts.
    """
    user_prompt = build_user_prompt(obs, step)
    conversation.append({"role": "user", "content": user_prompt})

    def run_sync():
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip(), None
        except Exception as e:
            return "", str(e)

    llm_output, err = await asyncio.to_thread(run_sync)

    if err:
        print(f"[DEBUG] Model request failed: {err}", file=sys.stderr, flush=True)
        return None, ""

    conversation.append({"role": "assistant", "content": llm_output})

    action = parse_llm_response(llm_output)
    return action, llm_output


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    from client import DynamicRouteEnv

    image_to_run = (
        IMAGE_NAME if ":" in IMAGE_NAME.split("/")[-1] else f"{IMAGE_NAME}:latest"
    )

    for current_task in TASKS_TO_RUN:
        env = await DynamicRouteEnv.from_docker_image(
            image_to_run, env_vars={"DRO_TASK": current_task}
        )

        log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        steps_taken = 0
        final_score = 0.01
        success = False

        try:

            result = await env.reset(episode_id=f"task:{current_task}")
            obs = result.observation

            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

            for step in range(1, MAX_STEPS + 1):
                steps_taken = step

                action, llm_output = await get_model_action(
                    client, obs, step, conversation
                )

                if action is None:

                    error_msg = "Could not parse JSON from LLM output"
                    action_str = "null"
                    rewards.append(0.01)

                    log_step(
                        step=step,
                        action=action_str,
                        reward=0.01,
                        done=False,
                        error=error_msg,
                    )

                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Your last response was not valid JSON. "
                                "Output ONLY a JSON object with this exact structure:\n"
                                '{"route_updates": [{"truck_id": "TRK_001", "new_route_order": ["Node_A", "Node_B"]}]}'
                            ),
                        }
                    )
                    continue

                action_str = json.dumps(action.model_dump(), separators=(",", ":"))

                try:
                    result = await env.step(action)
                    obs = result.observation
                    raw_step_reward = result.reward or 0.01
                    reward = max(0.01, min(0.99, raw_step_reward))
                    done = result.done
                    error = obs.metadata.get("error") if obs.metadata else None

                    rewards.append(reward)

                    log_step(
                        step=step,
                        action=action_str,
                        reward=reward,
                        done=done,
                        error=error,
                    )

                    if done:
                        break

                    if error:
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    f"ERROR from environment: {error}\n\n"
                                    f"Common fixes:\n"
                                    f"- Use EXACT truck IDs from the observation\n"
                                    f"- Use EXACT node names from the observation\n"
                                    f"- Ensure route doesn't use blocked edges\n"
                                    f"- Include ALL destinations for each truck's packages\n\n"
                                    f"Provide corrected route updates."
                                ),
                            }
                        )

                except Exception as step_err:

                    err_str = str(step_err)
                    rewards.append(0.01)

                    log_step(
                        step=step,
                        action=action_str,
                        reward=0.01,
                        done=False,
                        error=err_str,
                    )

                    conversation.append(
                        {
                            "role": "user",
                            "content": f"Environment error: {err_str}\nPlease fix and retry.",
                        }
                    )

        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr, flush=True)

        finally:            
            # Always ensure a valid score between (0, 1) is submitted, even on crash
            total_reward = sum(rewards)
            max_reward = float(steps_taken) if steps_taken > 0 else 1.0
            raw_score = total_reward / max_reward
            final_score = max(0.01, min(0.99, raw_score))
            success = final_score >= SUCCESS_SCORE_THRESHOLD
            
            if hasattr(env, "submit_task_score"):
                try:
                    await env.submit_task_score(final_score)
                except Exception as e:
                    print(f"[DEBUG] submit_task_score error: {e}", file=sys.stderr, flush=True)
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)

            log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
