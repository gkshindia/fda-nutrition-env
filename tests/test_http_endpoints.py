from __future__ import annotations

from fastapi.testclient import TestClient

import env.server.app as app_module
from env.server.environment import FDAEnvironment


def test_http_reset_step_state_persistence():
    # Recreate shared HTTP env for deterministic test state.
    app_module._http_env = FDAEnvironment()

    client = TestClient(app_module.app)

    reset_resp = client.post("/reset", json={"task_id": "task_easy", "seed": 42})
    assert reset_resp.status_code == 200
    reset_payload = reset_resp.json()

    state_after_reset = client.get("/state")
    assert state_after_reset.status_code == 200
    state_payload = state_after_reset.json()
    reset_episode_id = state_payload["episode_id"]
    assert state_payload["task_id"] == "task_easy"
    assert state_payload["step_count"] == 0

    draft_label = reset_payload["observation"]["draft_label"]
    step_resp = client.post("/step", json={"action": {"label": draft_label}})
    assert step_resp.status_code == 200

    state_after_step = client.get("/state")
    assert state_after_step.status_code == 200
    state_after_step_payload = state_after_step.json()
    assert state_after_step_payload["episode_id"] == reset_episode_id
    assert state_after_step_payload["step_count"] == 1
