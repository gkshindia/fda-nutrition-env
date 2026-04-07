"""
Tests for core/grader.py — FDA Nutrition Facts Panel label grader.
"""
from __future__ import annotations

import copy

import pytest

from core.episode_generator import generate_episode, _build_correct_label
from core.grader import grade
from data.regulatory_tables import compute_atwater_calories, round_calories


# ── Helpers ────────────────────────────────────────────────────────────────────

def _correct_label(ep: dict) -> dict:
    """Build the correct label from an episode's ground truth."""
    gt = ep["ground_truth"]
    return _build_correct_label(
        rounded=gt["per_serving_rounded"],
        percent_dvs=gt["percent_dvs"],
        ingredient_order=gt["ingredient_order"],
        serving_g=gt["serving_size_g"],
        declared_type_size_inch=gt["min_type_size_inch"],
        label_format=gt["label_format"],
    )


@pytest.fixture
def easy_episode():
    return generate_episode("easy", seed=42)


@pytest.fixture
def medium_episode():
    return generate_episode("medium", seed=42)


@pytest.fixture
def hard_episode():
    return generate_episode("hard", seed=42)


# ── Test 1: Perfect score ──────────────────────────────────────────────────────

def test_perfect_score(easy_episode):
    label = _correct_label(easy_episode)
    result = grade(label, easy_episode["ground_truth"])
    assert result["score"] == pytest.approx(1.0, abs=0.001)
    for group, s in result["group_scores"].items():
        assert s == pytest.approx(1.0, abs=0.001), f"{group} not 1.0"


# ── Test 2: Draft-as-is easy (not constant) ───────────────────────────────────

def test_draft_as_is_easy(easy_episode):
    result = grade(easy_episode["draft_label"], easy_episode["ground_truth"])
    assert 0.5 < result["score"] < 0.95, f"Expected 0.5–0.95, got {result['score']}"


# ── Test 3: Difficulty variance ────────────────────────────────────────────────

def test_difficulty_variance():
    easy = generate_episode("easy", seed=7)
    hard = generate_episode("hard", seed=7)
    easy_score = grade(easy["draft_label"], easy["ground_truth"])["score"]
    hard_score = grade(hard["draft_label"], hard["ground_truth"])["score"]
    assert easy_score > hard_score, (
        f"Easy draft ({easy_score:.3f}) should score higher than hard draft ({hard_score:.3f})"
    )


# ── Test 4: Missing nutrients key ─────────────────────────────────────────────

def test_missing_nutrients(easy_episode):
    label = _correct_label(easy_episode)
    del label["nutrients"]
    result = grade(label, easy_episode["ground_truth"])
    assert result["group_scores"]["nutrients"] == 0.0
    assert result["group_scores"]["atwater_consistency"] == 0.0
    assert result["score"] <= 0.65


# ── Test 5: None %DV handling ─────────────────────────────────────────────────

def test_none_dv_handling(easy_episode):
    gt = easy_episode["ground_truth"]
    label = _correct_label(easy_episode)
    # Find a DV that is None (trans_fat has no DV → not in percent_dvs)
    # All present DVs should score 1.0 when correct
    result = grade(label, gt)
    assert result["group_scores"]["percent_dvs"] == pytest.approx(1.0, abs=0.001)


# ── Test 6: Atwater consistent ─────────────────────────────────────────────────

def test_atwater_consistent(easy_episode):
    label = _correct_label(easy_episode)
    result = grade(label, easy_episode["ground_truth"])
    assert result["group_scores"]["atwater_consistency"] == pytest.approx(1.0, abs=0.001)


# ── Test 7: Atwater inconsistent ───────────────────────────────────────────────

def test_atwater_inconsistent(easy_episode):
    label = _correct_label(easy_episode)
    # Shift declared calories way off from Atwater expectation
    label["nutrients"]["energy_kcal"] += 50
    result = grade(label, easy_episode["ground_truth"])
    assert result["group_scores"]["atwater_consistency"] == 0.0


# ── Test 8: Type size at minimum ───────────────────────────────────────────────

def test_type_size_at_minimum(easy_episode):
    label = _correct_label(easy_episode)
    # Correct label already declares exactly at minimum
    result = grade(label, easy_episode["ground_truth"])
    assert result["group_scores"]["declared_type_size_inch"] == 1.0


# ── Test 9: Type size below ────────────────────────────────────────────────────

def test_type_size_below(easy_episode):
    label = _correct_label(easy_episode)
    label["declared_type_size_inch"] = 0.01  # way below any minimum
    result = grade(label, easy_episode["ground_truth"])
    assert result["group_scores"]["declared_type_size_inch"] == 0.0


# ── Test 10: Ingredient case insensitive ───────────────────────────────────────

def test_ingredient_case_insensitive(easy_episode):
    label = _correct_label(easy_episode)
    # Capitalize all ingredient names
    label["ingredient_list"] = [s.upper() for s in label["ingredient_list"]]
    result = grade(label, easy_episode["ground_truth"])
    assert result["group_scores"]["ingredient_list"] == 1.0


# ── Test 11: Health claims wrong ───────────────────────────────────────────────

def test_health_claims_wrong(easy_episode):
    label = _correct_label(easy_episode)
    label["health_claims"] = ["Excellent source of fiber"]
    result = grade(label, easy_episode["ground_truth"])
    assert result["group_scores"]["health_claims"] == 0.0


# ── Test 12: Determinism ──────────────────────────────────────────────────────

def test_determinism(easy_episode):
    label = easy_episode["draft_label"]
    gt = easy_episode["ground_truth"]
    r1 = grade(label, gt)
    r2 = grade(label, gt)
    assert r1["score"] == r2["score"]
    assert r1["group_scores"] == r2["group_scores"]
