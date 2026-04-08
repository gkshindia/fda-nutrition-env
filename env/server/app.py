import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openenv.core.env_server.http_server import create_app
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


@app.post("/grader")
def grader(req: GraderRequest):
    logger.info("GRADER  task=%s seed=%s actions=%d", req.task_id, req.seed, len(req.actions))
    env = FDAEnvironment()
    env.reset(task_id=req.task_id, seed=req.seed)
    for action_payload in req.actions:
        obs = env.step(FDAAction(label=action_payload))
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
