try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run: uv sync") from e

import os
import sys

# Force disable the default Gradio playground
os.environ["ENABLE_WEB_INTERFACE"] = "false"

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

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.get("/web", response_class=HTMLResponse)
async def custom_ui(request: Request):
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    def main():
        uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
    main()

from fastapi.encoders import jsonable_encoder

@app.get("/current_state")
async def current_state(request: Request):
    try:
        from .Dynamic_Routing_environment import GLOBAL_EPISODE_ID, GLOBAL_STEP_COUNT, DynamicRoutingEnvironment
    except ImportError:
        from server.Dynamic_Routing_environment import GLOBAL_EPISODE_ID, GLOBAL_STEP_COUNT, DynamicRoutingEnvironment

    env = DynamicRoutingEnvironment()
    obs = env._build_obs()

    return JSONResponse(content={
        "episode_id": GLOBAL_EPISODE_ID,
        "step_count": GLOBAL_STEP_COUNT,
        "observation": jsonable_encoder(obs)
    })


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)