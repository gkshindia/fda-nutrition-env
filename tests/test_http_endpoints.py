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


def test_http_rejects_invalid_task_ids():
    app_module._http_env = FDAEnvironment()
    client = TestClient(app_module.app)

    reset_resp = client.post("/reset", json={"task_id": "does_not_exist", "seed": 42})
    assert reset_resp.status_code == 404
    assert "Unknown task_id" in reset_resp.json()["detail"]

    grader_resp = client.post(
        "/grader",
        json={"task_id": "does_not_exist", "seed": 42, "actions": []},
    )
    assert grader_resp.status_code == 404
    assert "Unknown task_id" in grader_resp.json()["detail"]

    baseline_resp = client.post(
        "/baseline/run",
        json={"task_id": "does_not_exist", "seed": 42},
    )
    assert baseline_resp.status_code == 404
    assert "Unknown task_id" in baseline_resp.json()["detail"]


def test_openapi_reset_request_includes_task_id():
    client = TestClient(app_module.app)
    spec = client.get("/openapi.json")
    assert spec.status_code == 200

    reset_schema = spec.json()["components"]["schemas"]["ResetRequest"]
    assert "task_id" in reset_schema["properties"]


def test_mcp_route_not_exposed():
    client = TestClient(app_module.app)

    openapi = client.get("/openapi.json").json()
    assert "/mcp" not in openapi["paths"]

    mcp_resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert mcp_resp.status_code == 404
