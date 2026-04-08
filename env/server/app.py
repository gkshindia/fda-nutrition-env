import logging
import os
import threading
import time

from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from openenv.core.env_server.serialization import deserialize_action, serialize_observation
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import ResetRequest, ResetResponse, StepRequest, StepResponse
from env.models import FDAAction, FDAObservation
from env.server.environment import FDAEnvironment, TASKS

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fda.server")

app = create_app(FDAEnvironment, FDAAction, FDAObservation)

# Keep one shared environment instance for HTTP /reset -> /step -> /state flows.
# OpenEnv's default HTTP handlers are stateless (new env per request), which
# causes /state to look stale and /step to run without the prior /reset state.
_http_env = FDAEnvironment()
_http_env_lock = threading.Lock()


def _replace_stateless_openenv_routes() -> None:
    routes_to_replace = {
        ("POST", "/reset"),
        ("POST", "/step"),
        ("GET", "/state"),
    }
    kept_routes = []
    for route in app.router.routes:
        if not isinstance(route, APIRoute):
            kept_routes.append(route)
            continue
        if any((method, route.path) in routes_to_replace for method in route.methods):
            continue
        kept_routes.append(route)
    app.router.routes = kept_routes


_replace_stateless_openenv_routes()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("→ %s %s", request.method, request.url.path)
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("← %s %s  %d  (%.0f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response

_static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
def serve_root():
    index = os.path.join(_static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "FDA Nutrition Facts Panel API", "docs": "/docs"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GraderRequest(BaseModel):
    task_id: str
    seed: int | None = None
    actions: list[dict]


@app.get("/tasks")
def get_tasks():
    task_list = list(TASKS.values())
    logger.info("GET /tasks → %d tasks", len(task_list))
    return {"tasks": task_list, "action_schema": FDAAction.model_json_schema()}


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
def reset_env(request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    kwargs = request.model_dump(exclude_unset=True)
    with _http_env_lock:
        observation = _http_env.reset(**kwargs)
    return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
def step_env(request: StepRequest) -> StepResponse:
    try:
        action = deserialize_action(request.action, FDAAction)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(),
        ) from exc

    kwargs = request.model_dump(exclude_unset=True, exclude={"action"})
    with _http_env_lock:
        if not _http_env.state.ground_truth:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No active episode. Call POST /reset before POST /step.",
            )
        observation = _http_env.step(action, **kwargs)
    return StepResponse(**serialize_observation(observation))


@app.get("/state")
def get_state():
    with _http_env_lock:
        s = _http_env.state
        return {
            "episode_id": s.episode_id,
            "task_id": s.task_id,
            "difficulty": s.difficulty,
            "step_count": s.step_count,
            "max_steps": s.max_steps,
            "best_score": s.best_score,
            "completed": s.completed,
        }


@app.post("/grader")
def grader(req: GraderRequest):
    logger.info("GRADER  task=%s seed=%s actions=%d", req.task_id, req.seed, len(req.actions))
    env = FDAEnvironment()
    env.reset(task_id=req.task_id, seed=req.seed)
    for action_payload in req.actions:
        obs = env.step(FDAAction(**action_payload))
        if obs.done:
            break
    score = env.grader_score
    logger.info("GRADER  task=%s → score=%.3f steps=%d", req.task_id, score, env.state.step_count)
    return {
        "task_id": req.task_id,
        "grader_score": score,
        "steps_taken": env.state.step_count,
    }


@app.post("/baseline")
def run_baseline():
    logger.info("BASELINE  starting full run across all tasks")
    from baseline import run_baseline_agent
    scores = run_baseline_agent()
    logger.info("BASELINE  done — %s", scores)
    return {"scores": scores, "status": "completed"}


class BaselineRunRequest(BaseModel):
    task_id: str
    seed: int | None = None


@app.post("/baseline/run")
def run_baseline_task(req: BaselineRunRequest):
    from baseline import run_baseline_task as _run
    logger.info("BASELINE/RUN  task=%s — starting", req.task_id)
    result = _run(req.task_id, req.seed)
    logger.info(
        "BASELINE/RUN  task=%s → score=%.3f steps=%d",
        req.task_id, result.get("grader_score", 0), result.get("steps_taken", 0),
    )
    return result


def main():
    import uvicorn
    uvicorn.run("env.server.app:app", host="0.0.0.0", port=7860, reload=False)
