"""
Integration tests for FDAEnvironment — reset/step/grader_score cycle.
"""
from __future__ import annotations

import pytest

from env.server.environment import FDAEnvironment
from env.models import FDAAction, FDAObservation
from core.episode_generator import _build_correct_label
from core.grader import grade


@pytest.fixture
def env():
    return FDAEnvironment()


# ── Test 1: reset returns valid observation for all tasks ──────────────────────

@pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
def test_reset_returns_observation(env, task_id):
    obs = env.reset(task_id=task_id, seed=42)
    assert isinstance(obs, FDAObservation)
    assert len(obs.text) > 0
    assert len(obs.draft_label) > 0
    assert "nutrients" in obs.draft_label
    assert len(obs.episode_context) > 0
    assert "lab_nutrients" in obs.episode_context
    assert obs.done is False


# ── Test 2: final_submission step returns done with reward in [0, 1] ──────────

def test_step_final_submission_returns_done(env):
    obs = env.reset(task_id="task_easy", seed=42)
    result = env.step(FDAAction(label=obs.draft_label, final_submission=True))
    assert result.done is True
    assert 0.0 <= result.reward <= 1.0


# ── Test 2b: non-final step returns feedback (not done) ──────────────────────

def test_step_non_final_returns_feedback(env):
    obs = env.reset(task_id="task_easy", seed=42)
    result = env.step(FDAAction(label=obs.draft_label))
    assert result.done is False
    assert result.reward is not None
    assert "Group scores" in result.text
    assert "Incorrect fields" in result.text or "All fields correct" in result.text


# ── Test 3: perfect score ─────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
def test_perfect_score(env, task_id):
    env.reset(task_id=task_id, seed=42)
    gt = env.state.ground_truth
    correct = _build_correct_label(
        rounded=gt["per_serving_rounded"],
        percent_dvs=gt["percent_dvs"],
        ingredient_order=gt["ingredient_order"],
        serving_g=gt["serving_size_g"],
        declared_type_size_inch=gt["min_type_size_inch"],
        label_format=gt["label_format"],
    )
    env.step(FDAAction(label=correct, final_submission=True))
    assert env.grader_score == pytest.approx(1.0, abs=0.001)


# ── Test 4: garbage score ─────────────────────────────────────────────────────

def test_garbage_score(env):
    env.reset(task_id="task_easy", seed=42)
    env.step(FDAAction(label={}, final_submission=True))
    assert env.grader_score < 0.3


# ── Test 5: draft passthrough matches grader directly ──────────────────────────

def test_draft_passthrough_score(env):
    obs = env.reset(task_id="task_medium", seed=42)
    env.step(FDAAction(label=obs.draft_label, final_submission=True))
    env_score = env.grader_score
    direct_score = grade(obs.draft_label, env.state.ground_truth)["score"]
    assert env_score == pytest.approx(direct_score, abs=0.001)


# ── Test 6: state tracks step count ───────────────────────────────────────────

def test_state_tracks_step_count(env):
    env.reset(task_id="task_easy", seed=42)
    assert env.state.step_count == 0
    env.step(FDAAction(label={}))
    assert env.state.step_count == 1


# ── Test 7: seed reproducibility ──────────────────────────────────────────────

def test_seed_reproducibility():
    env1 = FDAEnvironment()
    env2 = FDAEnvironment()
    obs1 = env1.reset(task_id="task_hard", seed=99)
    obs2 = env2.reset(task_id="task_hard", seed=99)
    assert obs1.text == obs2.text
    assert obs1.draft_label == obs2.draft_label

    env1.step(FDAAction(label=obs1.draft_label, final_submission=True))
    env2.step(FDAAction(label=obs2.draft_label, final_submission=True))
    assert env1.grader_score == env2.grader_score


# ── Test 8: multi-step improves score ────────────────────────────────────────

def test_multi_step_revision_improves_score(env):
    """Submit garbage first, then correct label — final score should be 1.0."""
    env.reset(task_id="task_easy", seed=42)
    gt = env.state.ground_truth

    # Step 1: submit garbage (non-final)
    result1 = env.step(FDAAction(label={}))
    assert result1.done is False
    assert result1.reward < 0.3

    # Step 2: submit correct label — auto-stops because score >= 0.90
    correct = _build_correct_label(
        rounded=gt["per_serving_rounded"],
        percent_dvs=gt["percent_dvs"],
        ingredient_order=gt["ingredient_order"],
        serving_g=gt["serving_size_g"],
        declared_type_size_inch=gt["min_type_size_inch"],
        label_format=gt["label_format"],
    )
    result2 = env.step(FDAAction(label=correct))  # no final_submission needed
    assert result2.done is True  # auto-stopped: score 1.0 >= 0.90
    assert result2.reward == pytest.approx(1.0, abs=0.001)


# ── Test 9: plateau detection auto-finalizes ─────────────────────────────────

def test_plateau_detection_auto_finalizes(env):
    """When score doesn't improve for 2 consecutive steps, auto-finalize."""
    env.reset(task_id="task_easy", seed=42)
    # Step 1: garbage — sets baseline score
    result1 = env.step(FDAAction(label={}))
    assert result1.done is False
    # Step 2: same garbage — no improvement (count=1)
    result2 = env.step(FDAAction(label={}))
    assert result2.done is False
    # Step 3: same garbage — no improvement (count=2) → plateau → done
    result3 = env.step(FDAAction(label={}))
    assert result3.done is True


# ── Test 10: score threshold auto-stop ────────────────────────────────────────

def test_score_threshold_auto_stop(env):
    """Environment auto-finalizes when score >= 0.90, even on step 1."""
    env.reset(task_id="task_easy", seed=42)
    gt = env.state.ground_truth
    correct = _build_correct_label(
        rounded=gt["per_serving_rounded"],
        percent_dvs=gt["percent_dvs"],
        ingredient_order=gt["ingredient_order"],
        serving_g=gt["serving_size_g"],
        declared_type_size_inch=gt["min_type_size_inch"],
        label_format=gt["label_format"],
    )
    # Perfect label on step 1 — should auto-stop without final_submission
    result = env.step(FDAAction(label=correct))
    assert result.done is True
    assert env.state.step_count == 1  # stopped on first step


# ── Test 11: keep best label across steps ─────────────────────────────────────

def test_keep_best_label(env):
    """Grader score reflects the best submission, not the last one."""
    env.reset(task_id="task_easy", seed=42)
    gt = env.state.ground_truth
    correct = _build_correct_label(
        rounded=gt["per_serving_rounded"],
        percent_dvs=gt["percent_dvs"],
        ingredient_order=gt["ingredient_order"],
        serving_g=gt["serving_size_g"],
        declared_type_size_inch=gt["min_type_size_inch"],
        label_format=gt["label_format"],
    )
    # Step 1: perfect label
    env.step(FDAAction(label=correct))
    # Auto-stopped because score >= 0.90, but if it didn't:
    # the best_label should still be the correct one
    assert env.state.best_score == pytest.approx(1.0, abs=0.001)
    assert env.grader_score == pytest.approx(1.0, abs=0.001)
