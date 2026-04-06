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

from models import DynamicRouteAction, DynamicRouteObservation, RouteUpdate, LoadTransfer
from client import DynamicRouteEnv

# ── config ────────────────────────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME", "dynamic-routing-env")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Task to run — change this or set DRO_TASK env var
TASK_NAME  = os.getenv("DRO_TASK", "easy_avoid_blockage")
BENCHMARK  = "Dynamic_Routing"
MAX_STEPS  = 10
SUCCESS_SCORE_THRESHOLD = 0.5

# LLM parameters
TEMPERATURE = 0.2
MAX_TOKENS  = 2048

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
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
      6. If a truck reaches 0 fuel, it is STRANDED. You must use `load_transfers` to assign an active truck to its location node to pick up its packages.
      7. Use EXACT truck IDs, node names, and package IDs from the observation.
      8. Return ONLY valid JSON, no markdown, no explanation.

    If metadata.error is present, analyze the error and fix it in your next response.
""").strip()


# ── logging helpers ───────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_metadata(step: int, obs: DynamicRouteObservation, reward: float, done: bool) -> None:
    """Log detailed metadata for each step."""
    print(f"\n=== METADATA FOR STEP {step} ===", flush=True)
    
    step_data = {
        "observation": {
            "time_step": obs.time_step,
            "trucks": [t.model_dump() for t in obs.trucks],
            "packages": [p.model_dump() for p in obs.packages],
            "event": obs.event.model_dump() if obs.event else None,
            "distances": obs.distances,
            "fuel_stations": obs.fuel_stations,
            "metadata": obs.metadata if obs.metadata else {},
        },
        "reward": reward,
        "done": done,
    }
    
    print(json.dumps(step_data, indent=2), flush=True)
    print("=" * 50 + "\n", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ───────────────────────────────────────────────────────────
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
    
    # Build truck-specific guidance
    truck_guidance = []
    for truck in obs.trucks:
        # Get destinations for this truck
        destinations = []
        for pkg in obs.packages:
            if pkg.id in truck.assigned_packages and pkg.destination not in destinations:
                destinations.append(pkg.destination)
        
        guidance = f"\n  {truck.id}:"
        guidance += f"\n    - Current location: {truck.current_location}"
        guidance += f"\n    - Fuel Check: {truck.fuel:.1f} / {truck.fuel_capacity:.1f}"
        guidance += f"\n    - Must visit: {destinations}"
        guidance += f"\n    - Current route: {truck.route_order}"
        truck_guidance.append(guidance)
    
    return textwrap.dedent(f"""
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

        Output ONLY the JSON object with route_updates for ALL trucks.
    """).strip()


def parse_llm_response(text: str) -> Optional[DynamicRouteAction]:
    """Parse LLM response into a DynamicRouteAction."""
    try:
        cleaned = text.strip()
        
        # Remove markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
            # Handle ```json prefix
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        
        data = json.loads(cleaned)
        
        # Parse route updates
        updates = []
        for u in data.get("route_updates", []):
            updates.append(
                RouteUpdate(
                    truck_id=u["truck_id"],
                    new_route_order=u["new_route_order"]
                )
            )
            
        transfers = []
        for t in data.get("load_transfers", []):
            transfers.append(
                LoadTransfer(
                    from_truck_id=t["from_truck_id"],
                    to_truck_id=t["to_truck_id"],
                    package_id=t["package_id"]
                )
            )
        
        return DynamicRouteAction(route_updates=updates, load_transfers=transfers)
    
    except Exception as e:
        print(f"[DEBUG] Parse failed: {e}", flush=True)
        return None


def get_model_action(
    client: OpenAI,
    obs: DynamicRouteObservation,
    step: int,
    conversation: List[dict]
) -> tuple[Optional[DynamicRouteAction], str]:
    """
    Get action from the model.
    Returns (action, llm_output) tuple.
    """
    user_prompt = build_user_prompt(obs, step)
    conversation.append({"role": "user", "content": user_prompt})
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        
        llm_output = (completion.choices[0].message.content or "").strip()
        conversation.append({"role": "assistant", "content": llm_output})
        
        action = parse_llm_response(llm_output)
        return action, llm_output
    
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None, ""


# ── main loop ─────────────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await DynamicRouteEnv.from_docker_image(IMAGE_NAME)

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    try:
        # Reset environment
        result = await env.reset()
        obs = result.observation
        
        # Initialize conversation with system prompt
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Episode loop
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            
            # Get action from model
            action, llm_output = get_model_action(client, obs, step, conversation)
            
            if action is None:
                # LLM gave unparseable output
                error_msg = "Could not parse JSON from LLM output"
                action_str = "null"
                rewards.append(0.0)
                
                log_step(
                    step=step,
                    action=action_str,
                    reward=0.0,
                    done=False,
                    error=error_msg
                )
                
                # Log metadata even for failed parse
                log_metadata(step=step, obs=obs, reward=0.0, done=False)
                
                # Ask LLM to retry
                conversation.append({
                    "role": "user",
                    "content": (
                        "Your last response was not valid JSON. "
                        "Output ONLY a JSON object with this exact structure:\n"
                        '{"route_updates": [{"truck_id": "TRK_001", "new_route_order": ["Node_A", "Node_B"]}]}'
                    )
                })
                continue
            
            # Format action for logging
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            
            # Step environment
            try:
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = obs.metadata.get("error") if obs.metadata else None
                
                rewards.append(reward)
                
                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error
                )
                
                # Log detailed metadata for this step
                log_metadata(step=step, obs=obs, reward=reward, done=done)
                
                if done:
                    break
                
                # If there was an error, give feedback to the model
                if error:
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"ERROR from environment: {error}\n\n"
                            f"Common fixes:\n"
                            f"- Use EXACT truck IDs from the observation\n"
                            f"- Use EXACT node names from the observation\n"
                            f"- Ensure route doesn't use blocked edges\n"
                            f"- Include ALL destinations for each truck's packages\n\n"
                            f"Provide corrected route updates."
                        )
                    })
            
            except Exception as step_err:
                # Environment error
                err_str = str(step_err)
                rewards.append(0.0)
                
                log_step(
                    step=step,
                    action=action_str,
                    reward=0.0,
                    done=False,
                    error=err_str
                )
                
                # Log metadata even for environment errors
                log_metadata(step=step, obs=obs, reward=0.0, done=False)
                
                conversation.append({
                    "role": "user",
                    "content": f"Environment error: {err_str}\nPlease fix and retry."
                })

        # Compute final score
        total_reward = sum(rewards)
        max_reward = MAX_STEPS * 1.0
        final_score = min(1.0, total_reward / max_reward) if max_reward > 0 else 0.0
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards
        )


if __name__ == "__main__":
    asyncio.run(main())