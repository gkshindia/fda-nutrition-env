"""
Integration tests for FDAEnvironment — 5-phase sequential episode flow.
"""
from __future__ import annotations

import pytest

from env.server.environment import FDAEnvironment
from env.models import FDAAction, FDAObservation


@pytest.fixture
def env():
    return FDAEnvironment()


# ── Test 1: reset returns valid phase 1 observation ──────────────────────────

@pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
def test_reset_returns_observation(env, task_id):
    obs = env.reset(task_id=task_id, seed=42)
    assert isinstance(obs, FDAObservation)
    assert len(obs.text) > 0
    assert obs.phase == 1
    assert obs.done is False
    assert "food_category_description" in obs.phase_data
    assert obs.prior_submissions == {}


# ── Test 2: full 5-step episode reaches done ─────────────────────────────────

def test_full_five_step_episode(env):
    obs = env.reset(task_id="task_easy", seed=42)
    assert obs.phase == 1

    # Phase 1
    obs = env.step(FDAAction(phase=1, food_category="test", racc_g=30.0, household_measure="1 serving"))
    assert obs.phase == 2
    assert obs.done is False

    # Phase 2
    obs = env.step(FDAAction(phase=2, label_format="single_column", serving_size_g=30.0, serving_declaration_text="1 serving (30g)"))
    assert obs.phase == 3
    assert obs.done is False

    # Phase 3
    obs = env.step(FDAAction(phase=3, nutrients={}, percent_dvs={}, energy_kcal=0))
    assert obs.phase == 4
    assert obs.done is False

    # Phase 4
    obs = env.step(FDAAction(phase=4, ingredient_list=[], compound_ingredient_sublists={}))
    assert obs.phase == 5
    assert obs.done is False

    # Phase 5
    obs = env.step(FDAAction(phase=5, declared_type_size_inch=0.125, health_claims=[], consistency_violations=[]))
    assert obs.done is True
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


# ── Test 3: phase mismatch raises error ──────────────────────────────────────

def test_phase_mismatch_raises(env):
    env.reset(task_id="task_easy", seed=42)
    with pytest.raises(ValueError, match="Phase mismatch"):
        env.step(FDAAction(phase=3, nutrients={}, percent_dvs={}, energy_kcal=0))


# ── Test 4: step before reset raises error ───────────────────────────────────

def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError, match="not initialized"):
        env.step(FDAAction(phase=1, food_category="test", racc_g=30.0))


# ── Test 5: perfect score ─────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
def test_perfect_score(env, task_id):
    env.reset(task_id=task_id, seed=42)
    gt = env.state.ground_truth

    # Phase 1
    p1 = gt["phase_1"]
    env.step(FDAAction(
        phase=1,
        food_category=p1["food_category"],
        racc_g=p1["racc_g"],
        household_measure=p1["household_measure"],
    ))

    # Phase 2
    p2 = gt["phase_2"]
    env.step(FDAAction(
        phase=2,
        label_format=p2["label_format"],
        serving_size_g=p2["serving_size_g"],
        serving_declaration_text=p2["serving_declaration_text"],
    ))

    # Phase 3
    p3 = gt["phase_3"]
    env.step(FDAAction(
        phase=3,
        nutrients=p3["per_serving_rounded"],
        percent_dvs=p3["percent_dvs"],
        energy_kcal=p3["atwater_kcal_declared"],
    ))

    # Phase 4
    p4 = gt["phase_4"]
    env.step(FDAAction(
        phase=4,
        ingredient_list=p4["ingredient_order"],
        compound_ingredient_sublists=p4.get("compound_sublists", {}),
    ))

    # Phase 5
    p5 = gt["phase_5"]
    env.step(FDAAction(
        phase=5,
        declared_type_size_inch=p5["min_type_size_inch"],
        health_claims=p5.get("valid_health_claims", []),
        consistency_violations=p5.get("consistency_violations", []),
    ))

    assert env.grader_score == pytest.approx(1.0, abs=0.001)


# ── Test 6: garbage score ─────────────────────────────────────────────────────

def test_garbage_score(env):
    env.reset(task_id="task_easy", seed=42)

    env.step(FDAAction(phase=1))
    env.step(FDAAction(phase=2))
    env.step(FDAAction(phase=3, nutrients={}, percent_dvs={}, energy_kcal=0))
    env.step(FDAAction(phase=4, ingredient_list=[], compound_ingredient_sublists={}))
    env.step(FDAAction(phase=5, declared_type_size_inch=0, health_claims=[], consistency_violations=[]))

    assert env.grader_score < 0.3


# ── Test 7: state tracks step count and phase ────────────────────────────────

def test_state_tracks_step_and_phase(env):
    env.reset(task_id="task_easy", seed=42)
    assert env.state.step_count == 0
    assert env.state.current_phase == 1

    env.step(FDAAction(phase=1, food_category="test", racc_g=30.0))
    assert env.state.step_count == 1
    assert env.state.current_phase == 2


# ── Test 8: seed reproducibility ──────────────────────────────────────────────

def test_seed_reproducibility():
    env1 = FDAEnvironment()
    env2 = FDAEnvironment()
    obs1 = env1.reset(task_id="task_hard", seed=99)
    obs2 = env2.reset(task_id="task_hard", seed=99)
    assert obs1.text == obs2.text
    assert obs1.phase_data == obs2.phase_data

    # Same actions → same scores
    for phase in range(1, 6):
        action = FDAAction(phase=phase, nutrients={}, percent_dvs={}, energy_kcal=0,
                           ingredient_list=[], compound_ingredient_sublists={},
                           health_claims=[], consistency_violations=[],
                           declared_type_size_inch=0)
        env1.step(action)
        env2.step(action)
    assert env1.grader_score == env2.grader_score


# ── Test 9: invalid task id ───────────────────────────────────────────────────

def test_invalid_task_id_raises():
    env = FDAEnvironment()
    with pytest.raises(ValueError, match="Unknown task_id"):
        env.reset(task_id="does_not_exist", seed=42)


# ── Test 10: prior submissions visible in later phases ────────────────────────

def test_prior_submissions_visible(env):
    env.reset(task_id="task_easy", seed=42)

    # Phase 1 — no prior submissions
    obs = env.step(FDAAction(phase=1, food_category="test", racc_g=30.0, household_measure="1 serving"))
    assert 1 in obs.prior_submissions  # Phase 1 submission visible

    # Phase 2 — prior submissions include phase 1
    obs = env.step(FDAAction(phase=2, label_format="single_column", serving_size_g=30.0))
    assert 1 in obs.prior_submissions
    assert 2 in obs.prior_submissions


# ── Test 11: phase scores recorded in state ───────────────────────────────────

def test_phase_scores_recorded(env):
    env.reset(task_id="task_easy", seed=42)
    gt = env.state.ground_truth

    # Submit perfect phase 1
    p1 = gt["phase_1"]
    env.step(FDAAction(
        phase=1,
        food_category=p1["food_category"],
        racc_g=p1["racc_g"],
        household_measure=p1["household_measure"],
    ))

    assert 1 in env.state.phase_scores
    assert env.state.phase_scores[1] == pytest.approx(1.0, abs=0.001)


# ── Test 12: completed flag set after phase 5 ────────────────────────────────

def test_completed_flag(env):
    env.reset(task_id="task_easy", seed=42)
    assert env.state.completed is False

    for phase in range(1, 6):
        env.step(FDAAction(phase=phase, nutrients={}, percent_dvs={}, energy_kcal=0,
                           ingredient_list=[], compound_ingredient_sublists={},
                           health_claims=[], consistency_violations=[],
                           declared_type_size_inch=0))

    assert env.state.completed is True
