"""
FDA Nutrition Facts Panel — Label Grader
=========================================
Pure-Python, deterministic grader. Compares an agent's corrected label
against episode ground truth and returns a score in [0.0, 1.0].

Usage:
    from core.grader import grade
    result = grade(agent_label, ground_truth)
    score  = result["score"]           # 0.0–1.0
    groups = result["group_scores"]    # per-group breakdown
    fields = result["field_details"]   # per-field correct/agent/expected
"""
from __future__ import annotations

from data.regulatory_tables import compute_atwater_calories, round_calories


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "serving_size_g": 0.10,
    "nutrients": 0.30,
    "percent_dvs": 0.20,
    "ingredient_list": 0.15,
    "declared_type_size_inch": 0.10,
    "health_claims": 0.10,
    "atwater_consistency": 0.05,
}

# 14 nutrient keys in label order (energy_kcal scored here + atwater group)
_NUTRIENT_KEYS = [
    "energy_kcal",
    "protein_g",
    "total_fat_g",
    "saturated_fat_g",
    "trans_fat_g",
    "cholesterol_mg",
    "total_carbohydrate_g",
    "dietary_fiber_g",
    "total_sugars_g",
    "added_sugars_g",
    "sodium_mg",
    "potassium_mg",
    "calcium_mg",
    "iron_mg",
    "vitamin_d_mcg",
]


# ─────────────────────────────────────────────────────────────────────────────
# SAFE CAST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GROUP SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_serving_size(agent_label: dict, ground_truth: dict) -> tuple[float, dict]:
    expected = ground_truth["serving_size_g"]
    agent_val = _safe_float(agent_label.get("serving_size_g"))
    if agent_val is not None and abs(agent_val - expected) <= 0.01:
        correct = True
    else:
        correct = False
    return (
        1.0 if correct else 0.0,
        {"serving_size_g": {"correct": correct, "agent": agent_val, "expected": expected}},
    )


def _score_nutrients(agent_label: dict, ground_truth: dict) -> tuple[float, dict]:
    gt_rounded = ground_truth["per_serving_rounded"]
    agent_nutrients = agent_label.get("nutrients", {})
    n_correct = 0
    details = {}
    for key in _NUTRIENT_KEYS:
        expected = gt_rounded[key]
        agent_val = _safe_float(agent_nutrients.get(key))
        if agent_val is not None and abs(agent_val - expected) < 0.01:
            ok = True
            n_correct += 1
        else:
            ok = False
        details[f"nutrients.{key}"] = {
            "correct": ok,
            "agent": agent_val,
            "expected": expected,
        }
    n_total = len(_NUTRIENT_KEYS)
    return n_correct / n_total, details


def _score_percent_dvs(agent_label: dict, ground_truth: dict) -> tuple[float, dict]:
    gt_dvs = ground_truth["percent_dvs"]
    agent_dvs = agent_label.get("percent_dvs", {})
    n_correct = 0
    details = {}
    for key, expected in gt_dvs.items():
        agent_val = agent_dvs.get(key)
        # None == None is correct; int == int is correct
        if expected is None:
            ok = agent_val is None
        else:
            agent_int = _safe_int(agent_val)
            ok = agent_int is not None and agent_int == int(expected)
        if ok:
            n_correct += 1
        details[f"percent_dvs.{key}"] = {
            "correct": ok,
            "agent": agent_val,
            "expected": expected,
        }
    n_total = len(gt_dvs)
    if n_total == 0:
        return 1.0, details
    return n_correct / n_total, details


def _score_ingredient_list(agent_label: dict, ground_truth: dict) -> tuple[float, dict]:
    expected = ground_truth["ingredient_order"]
    agent_list = agent_label.get("ingredient_list", [])
    # Case-insensitive comparison
    agent_lower = [s.lower().strip() for s in agent_list] if isinstance(agent_list, list) else []
    expected_lower = [s.lower().strip() for s in expected]
    ok = agent_lower == expected_lower
    return (
        1.0 if ok else 0.0,
        {"ingredient_list": {"correct": ok, "agent": agent_list, "expected": expected}},
    )


def _score_type_size(agent_label: dict, ground_truth: dict) -> tuple[float, dict]:
    min_size = ground_truth["min_type_size_inch"]
    agent_val = _safe_float(agent_label.get("declared_type_size_inch"))
    if agent_val is not None and agent_val >= min_size - 0.001:
        ok = True
    else:
        ok = False
    return (
        1.0 if ok else 0.0,
        {"declared_type_size_inch": {
            "correct": ok,
            "agent": agent_val,
            "expected_min": min_size,
        }},
    )


def _score_health_claims(agent_label: dict) -> tuple[float, dict]:
    # Correct label never has health claims (_build_correct_label sets [])
    expected: list[str] = []
    agent_claims = agent_label.get("health_claims", [])
    if not isinstance(agent_claims, list):
        agent_claims = []
    ok = set(agent_claims) == set(expected)
    return (
        1.0 if ok else 0.0,
        {"health_claims": {"correct": ok, "agent": agent_claims, "expected": expected}},
    )


def _score_atwater_consistency(agent_label: dict) -> tuple[float, dict]:
    agent_nutrients = agent_label.get("nutrients", {})
    fat = _safe_float(agent_nutrients.get("total_fat_g"))
    carb = _safe_float(agent_nutrients.get("total_carbohydrate_g"))
    protein = _safe_float(agent_nutrients.get("protein_g"))
    agent_kcal = _safe_float(agent_nutrients.get("energy_kcal"))

    if any(v is None for v in (fat, carb, protein, agent_kcal)):
        return (
            0.0,
            {"atwater_consistency": {
                "correct": False,
                "agent_kcal": agent_kcal,
                "expected_kcal": None,
                "note": "missing macro or kcal value",
            }},
        )

    expected_kcal = round_calories(compute_atwater_calories(fat, carb, protein))
    ok = abs(agent_kcal - expected_kcal) <= 5
    return (
        1.0 if ok else 0.0,
        {"atwater_consistency": {
            "correct": ok,
            "agent_kcal": agent_kcal,
            "expected_kcal": expected_kcal,
        }},
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def grade(agent_label: dict, ground_truth: dict) -> dict:
    """
    Score an agent's corrected label against ground truth.

    Args:
        agent_label:  The label dict submitted by the agent (same schema as draft_label).
        ground_truth: The ground_truth dict from generate_episode().

    Returns:
        {
            "score": float,               # weighted total in [0.0, 1.0]
            "group_scores": {str: float},  # per-group scores
            "field_details": {str: dict},  # per-field breakdown
        }
    """
    group_scores: dict[str, float] = {}
    field_details: dict[str, dict] = {}

    # 1. Serving size
    s, d = _score_serving_size(agent_label, ground_truth)
    group_scores["serving_size_g"] = s
    field_details.update(d)

    # 2. Nutrients
    s, d = _score_nutrients(agent_label, ground_truth)
    group_scores["nutrients"] = s
    field_details.update(d)

    # 3. %DV
    s, d = _score_percent_dvs(agent_label, ground_truth)
    group_scores["percent_dvs"] = s
    field_details.update(d)

    # 4. Ingredient list
    s, d = _score_ingredient_list(agent_label, ground_truth)
    group_scores["ingredient_list"] = s
    field_details.update(d)

    # 5. Type size
    s, d = _score_type_size(agent_label, ground_truth)
    group_scores["declared_type_size_inch"] = s
    field_details.update(d)

    # 6. Health claims
    s, d = _score_health_claims(agent_label)
    group_scores["health_claims"] = s
    field_details.update(d)

    # 7. Atwater consistency
    s, d = _score_atwater_consistency(agent_label)
    group_scores["atwater_consistency"] = s
    field_details.update(d)

    # Weighted total
    score = sum(WEIGHTS[k] * group_scores[k] for k in WEIGHTS)

    return {
        "score": score,
        "group_scores": group_scores,
        "field_details": field_details,
    }
