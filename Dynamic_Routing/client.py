# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic Routing Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import DynamicRouteAction, DynamicRouteObservation


class DynamicRouteEnv(
    EnvClient[DynamicRouteAction, DynamicRouteObservation, State]
):
    """
    Client for the Dynamic Routing Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DynamicRoutingEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(DynamicRouteAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DynamicRoutingEnv.from_docker_image("Dynamic_Routing-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DynamicRouteAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DynamicRouteAction) -> Dict:
        """
        Convert DynamicRouteAction to JSON payload for step message.

        Args:
            action: DynamicRouteAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {}
        if action.route_updates is not None:
            payload["route_updates"] = [
                {"truck_id": ru.truck_id, "new_route_order": ru.new_route_order} 
                if hasattr(ru, "truck_id") else ru 
                for ru in action.route_updates
            ]
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DynamicRouteObservation]:
        """
        Parse server response into StepResult[DynamicRouteObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DynamicRouteObservation
        """
        obs_data = payload.get("observation", {})
        observation = DynamicRouteObservation(
            time_step=obs_data.get("time_step", 0),
            trucks=obs_data.get("trucks", []),
            packages=obs_data.get("packages", []),
            event=obs_data.get("event", None),
            distances=obs_data.get("distances", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
