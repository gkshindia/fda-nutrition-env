"""
HTTP endpoint integration tests for the 5-phase FDA environment.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import env.server.app as app_module
from env.server.environment import FDAEnvironment


def test_http_reset_step_state_persistence():
    """Full 5-phase HTTP flow: reset → 5 steps → verify state at each stage."""
    app_module._http_env = FDAEnvironment()
    client = TestClient(app_module.app)

    # Reset
    reset_resp = client.post("/reset", json={"task_id": "task_easy", "seed": 42})
    assert reset_resp.status_code == 200
    reset_payload = reset_resp.json()
    assert reset_payload["observation"]["phase"] == 1

    # State after reset
    state = client.get("/state").json()
    assert state["task_id"] == "task_easy"
    assert state["step_count"] == 0
    assert state["current_phase"] == 1
    episode_id = state["episode_id"]

    # Step through phases 1-4 (not done)
    for phase in range(1, 5):
        step_resp = client.post("/step", json={
            "action": {
                "phase": phase,
                "nutrients": {},
                "percent_dvs": {},
                "energy_kcal": 0,
                "ingredient_list": [],
                "compound_ingredient_sublists": {},
            },
        })
        assert step_resp.status_code == 200
        resp = step_resp.json()
        assert resp["done"] is False
        assert resp["observation"]["phase"] == phase + 1

    # Verify state mid-episode
    state = client.get("/state").json()
    assert state["episode_id"] == episode_id
    assert state["step_count"] == 4
    assert state["current_phase"] == 5

    # Phase 5 (done)
    step_resp = client.post("/step", json={
        "action": {
            "phase": 5,
            "declared_type_size_inch": 0.125,
            "health_claims": [],
            "consistency_violations": [],
        },
    })
    assert step_resp.status_code == 200
    resp = step_resp.json()
    assert resp["done"] is True
    assert resp["reward"] is not None

    # State after completion
    state = client.get("/state").json()
    assert state["completed"] is True
    assert state["step_count"] == 5


def test_http_step_before_reset_returns_409():
    """Step without reset should return 409 Conflict."""
    app_module._http_env = FDAEnvironment()
    client = TestClient(app_module.app)

    step_resp = client.post("/step", json={
        "action": {"phase": 1, "food_category": "test", "racc_g": 30.0},
    })
    assert step_resp.status_code == 409


def test_http_grader_replays_five_phases():
    """Grader endpoint should replay a full 5-action episode."""
    app_module._http_env = FDAEnvironment()
    client = TestClient(app_module.app)

    actions = [
        {"phase": 1, "food_category": "test", "racc_g": 30.0, "household_measure": "1 serving"},
        {"phase": 2, "label_format": "single_column", "serving_size_g": 30.0, "serving_declaration_text": "1 serving (30g)"},
        {"phase": 3, "nutrients": {}, "percent_dvs": {}, "energy_kcal": 0},
        {"phase": 4, "ingredient_list": [], "compound_ingredient_sublists": {}},
        {"phase": 5, "declared_type_size_inch": 0.125, "health_claims": [], "consistency_violations": []},
    ]

    grader_resp = client.post("/grader", json={
        "task_id": "task_easy",
        "seed": 42,
        "actions": actions,
    })
    assert grader_resp.status_code == 200
    result = grader_resp.json()
    assert "grader_score" in result
    assert 0.0 <= result["grader_score"] <= 1.0
    assert result["steps_taken"] == 5


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


def test_http_state_includes_current_phase():
    app_module._http_env = FDAEnvironment()
    client = TestClient(app_module.app)

    client.post("/reset", json={"task_id": "task_easy", "seed": 42})
    state = client.get("/state").json()
    assert "current_phase" in state
    assert state["current_phase"] == 1
    assert "phase_scores" in state


def test_openapi_reset_request_includes_task_id():
    client = TestClient(app_module.app)
    spec = client.get("/openapi.json")
    assert spec.status_code == 200

    reset_schema = spec.json()["components"]["schemas"]["ResetRequest"]
    assert "task_id" in reset_schema["properties"]


def test_mcp_route_exposed():
    client = TestClient(app_module.app)

    mcp_resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert mcp_resp.status_code == 200
