try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run: uv sync") from e

import os
import sys

try:
    from ..models import DynamicRouteAction, DynamicRouteObservation
    from .Dynamic_Routing_environment import DynamicRoutingEnvironment
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import DynamicRouteAction, DynamicRouteObservation
    from server.Dynamic_Routing_environment import DynamicRoutingEnvironment

app = create_app(
    DynamicRoutingEnvironment,
    DynamicRouteAction,
    DynamicRouteObservation,
    env_name="Dynamic_Routing",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)