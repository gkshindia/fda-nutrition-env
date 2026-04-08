from fastapi import FastAPI
from pydantic import BaseModel
from openenv.core.env_server.http_server import create_app
from env.models import FDAAction, FDAObservation
from env.server.environment import FDAEnvironment, TASKS

app = create_app(FDAEnvironment, FDAAction, FDAObservation)


class GraderRequest(BaseModel):
    task_id: str
    seed: int | None = None
    actions: list[dict]


@app.get("/tasks")
def get_tasks():
    task_list = list(TASKS.values())
    return {"tasks": task_list, "action_schema": FDAAction.model_json_schema()}


@app.post("/grader")
def grader(req: GraderRequest):
    env = FDAEnvironment()
    env.reset(task_id=req.task_id, seed=req.seed)
    for action_payload in req.actions:
        obs = env.step(FDAAction(label=action_payload))
        if obs.done:
            break
    return {
        "task_id": req.task_id,
        "grader_score": env.grader_score,
        "steps_taken": env.state.step_count,
    }


@app.post("/baseline")
def run_baseline():
    from baseline import run_baseline_agent
    scores = run_baseline_agent()
    return {"scores": scores, "status": "completed"}


def main():
    import uvicorn
    uvicorn.run("env.server.app:app", host="0.0.0.0", port=7860, reload=False)
