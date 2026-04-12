"""
FDA Nutrition Facts Panel — Per-Phase Grader (v2)
=================================================
Scores each of the 5 sequential phases independently.
Phase 3 includes dual-scoring (absolute vs. propagation).

Usage:
    from core.grader_v2 import grade_phase, grade_episode, PHASE_WEIGHTS

    result = grade_phase(phase=1, action_dict=..., ground_truth=..., prior_actions={})
    score = result["score"]       # 0.0–1.0 for this phase
    fields = result["details"]    # per-field breakdown

    final = grade_episode(phase_scores={1: 0.9, 2: 0.8, ...})  # weighted 0.0–1.0
"""
from __future__ import annotations

import math
from typing import Any, Optional

from data.regulatory_tables import (
    compute_atwater_calories,
    compute_percent_dv,
    compute_pdp_area,
    lookup_min_type_size,
    round_calories,
    round_total_fat,
    round_saturated_fat,
    round_trans_fat,
    round_cholesterol,
    round_sodium,
    round_total_carbohydrate,
    round_dietary_fiber,
    round_total_sugars,
    round_added_sugars,
    round_protein,
    round_vitamin_d,
    round_calcium,
    round_iron,
    round_potassium,
)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

PHASE_WEIGHTS: dict[int, float] = {
    1: 0.15,   # Category + RACC + household measure
    2: 0.15,   # Format + serving size + declaration
    3: 0.35,   # Nutrients + %DVs + Atwater (heaviest — core math)
    4: 0.15,   # Ingredient list + compound sublists
    5: 0.20,   # PDP/type size + health claims + consistency audit
}


# ─────────────────────────────────────────────────────────────────────────────
# SAFE CAST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NUTRIENT PIPELINE (for dual-scoring recomputation)
# ─────────────────────────────────────────────────────────────────────────────

_ROUNDING_DISPATCH: dict[str, callable] = {
    "energy_kcal":          round_calories,
    "protein_g":            round_protein,
    "total_fat_g":          round_total_fat,
    "saturated_fat_g":      round_saturated_fat,
    "trans_fat_g":          round_trans_fat,
    "cholesterol_mg":       lambda v: round_cholesterol(v)[0],
    "total_carbohydrate_g": round_total_carbohydrate,
    "dietary_fiber_g":      round_dietary_fiber,
    "total_sugars_g":       round_total_sugars,
    "added_sugars_g":       round_added_sugars,
    "sodium_mg":            round_sodium,
    "potassium_mg":         round_potassium,
    "calcium_mg":           round_calcium,
    "iron_mg":              round_iron,
    "vitamin_d_mcg":        round_vitamin_d,
}

_NUTRIENT_KEYS = list(_ROUNDING_DISPATCH.keys())

_DV_MAP = {
    "total_fat_g":          "total_fat",
    "saturated_fat_g":      "saturated_fat",
    "cholesterol_mg":       "cholesterol",
    "sodium_mg":            "sodium",
    "total_carbohydrate_g": "total_carbohydrate",
    "dietary_fiber_g":      "dietary_fiber",
    "added_sugars_g":       "added_sugars",
    "protein_g":            "protein",
    "vitamin_d_mcg":        "vitamin_d",
    "calcium_mg":           "calcium",
    "iron_mg":              "iron",
    "potassium_mg":         "potassium",
}


def _recompute_nutrients_for_serving(
    lab_nutrients: dict[str, float],
    lab_sample_g: float,
    agent_serving_g: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, Optional[float]]]:
    """
    Recompute rounded nutrients and %DVs using the agent's serving size.
    Used for dual-scoring in Phase 3.

    Returns:
        (scaled_unrounded, rounded, percent_dvs)
    """
    scale = agent_serving_g / lab_sample_g
    scaled = {k: v * scale for k, v in lab_nutrients.items()}

    # Atwater on pre-rounded
    atwater_raw = compute_atwater_calories(
        total_fat_g=scaled.get("total_fat_g", 0.0),
        total_carb_g=scaled.get("total_carbohydrate_g", 0.0),
        protein_g=scaled.get("protein_g", 0.0),
    )

    rounded = {}
    for key, fn in _ROUNDING_DISPATCH.items():
        rounded[key] = fn(scaled.get(key, 0.0))
    rounded["energy_kcal"] = round_calories(atwater_raw)

    percent_dvs = {}
    for field, dv_key in _DV_MAP.items():
        percent_dvs[field] = compute_percent_dv(dv_key, rounded.get(field, 0.0))

    return scaled, rounded, percent_dvs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 GRADER — Category + RACC + Household Measure
# ─────────────────────────────────────────────────────────────────────────────

def _grade_phase_1(action: dict, ground_truth: dict) -> dict:
    gt = ground_truth["phase_1"]
    details = {}

    # Food category (case-insensitive exact match)
    agent_cat = (action.get("food_category") or "").strip().lower()
    expected_cat = gt["food_category"].strip().lower()
    cat_ok = agent_cat == expected_cat
    details["food_category"] = {
        "correct": cat_ok,
        "agent": action.get("food_category"),
        "expected": gt["food_category"],
    }

    # RACC in grams (tolerance 0.01)
    agent_racc = _safe_float(action.get("racc_g"))
    expected_racc = gt["racc_g"]
    racc_ok = agent_racc is not None and abs(agent_racc - expected_racc) <= 0.01
    details["racc_g"] = {
        "correct": racc_ok,
        "agent": agent_racc,
        "expected": expected_racc,
    }

    # Household measure (case-insensitive)
    agent_hm = (action.get("household_measure") or "").strip().lower()
    expected_hm = gt["household_measure"].strip().lower()
    hm_ok = agent_hm == expected_hm
    details["household_measure"] = {
        "correct": hm_ok,
        "agent": action.get("household_measure"),
        "expected": gt["household_measure"],
    }

    # Weighted: category=0.4, racc=0.4, household_measure=0.2
    score = (0.4 * cat_ok) + (0.4 * racc_ok) + (0.2 * hm_ok)

    return {"score": score, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 GRADER — Format + Serving Size + Declaration
# ─────────────────────────────────────────────────────────────────────────────

def _grade_phase_2(action: dict, ground_truth: dict, phase_1_action: dict) -> dict:
    gt = ground_truth["phase_2"]
    details = {}

    # Label format (exact match)
    agent_format = (action.get("label_format") or "").strip().lower()
    expected_format = gt["label_format"].strip().lower()
    format_ok = agent_format == expected_format
    details["label_format"] = {
        "correct": format_ok,
        "agent": action.get("label_format"),
        "expected": gt["label_format"],
    }

    # Serving size (tolerance 0.01)
    agent_ss = _safe_float(action.get("serving_size_g"))
    expected_ss = gt["serving_size_g"]
    ss_ok = agent_ss is not None and abs(agent_ss - expected_ss) <= 0.01
    details["serving_size_g"] = {
        "correct": ss_ok,
        "agent": agent_ss,
        "expected": expected_ss,
    }

    # Serving declaration text (case-insensitive)
    agent_decl = (action.get("serving_declaration_text") or "").strip().lower()
    expected_decl = gt["serving_declaration_text"].strip().lower()
    decl_ok = agent_decl == expected_decl
    details["serving_declaration_text"] = {
        "correct": decl_ok,
        "agent": action.get("serving_declaration_text"),
        "expected": gt["serving_declaration_text"],
    }

    # Weighted: format=0.3, serving_size=0.5, declaration=0.2
    score = (0.3 * format_ok) + (0.5 * ss_ok) + (0.2 * decl_ok)

    return {"score": score, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 GRADER — Nutrients + %DVs + Atwater (DUAL SCORING)
# ─────────────────────────────────────────────────────────────────────────────

def _score_nutrients_against(agent_nutrients: dict, expected_rounded: dict) -> tuple[float, dict]:
    """Score agent nutrients against a set of expected rounded values."""
    correct_count = 0
    details = {}
    for key in _NUTRIENT_KEYS:
        expected = expected_rounded.get(key, 0.0)
        agent_val = _safe_float(agent_nutrients.get(key))
        ok = agent_val is not None and abs(agent_val - expected) < 0.01
        if ok:
            correct_count += 1
        details[f"nutrients.{key}"] = {
            "correct": ok,
            "agent": agent_val,
            "expected": expected,
        }
    return correct_count / len(_NUTRIENT_KEYS), details


def _score_dvs_against(agent_dvs: dict, expected_dvs: dict) -> tuple[float, dict]:
    """Score agent %DVs against expected values."""
    correct_count = 0
    details = {}
    total = 0
    for key, expected in expected_dvs.items():
        total += 1
        agent_val = agent_dvs.get(key)
        if expected is None:
            ok = agent_val is None
        else:
            agent_int = _safe_int(agent_val)
            ok = agent_int is not None and agent_int == int(expected)
        if ok:
            correct_count += 1
        details[f"percent_dvs.{key}"] = {
            "correct": ok,
            "agent": agent_val,
            "expected": expected,
        }
    if total == 0:
        return 1.0, details
    return correct_count / total, details


def _grade_phase_3(
    action: dict,
    ground_truth: dict,
    phase_1_action: dict,
    phase_2_action: dict,
) -> dict:
    gt = ground_truth["phase_3"]
    agent_nutrients = action.get("nutrients") or {}
    agent_dvs = action.get("percent_dvs") or {}

    # --- Absolute scoring (against ground truth) ---
    nutrient_score_gt, nutrient_details_gt = _score_nutrients_against(
        agent_nutrients, gt["per_serving_rounded"],
    )
    dv_score_gt, dv_details_gt = _score_dvs_against(
        agent_dvs, gt["percent_dvs"],
    )

    # Atwater consistency (agent's own macros should be self-consistent)
    fat = _safe_float(agent_nutrients.get("total_fat_g"))
    carb = _safe_float(agent_nutrients.get("total_carbohydrate_g"))
    protein = _safe_float(agent_nutrients.get("protein_g"))
    agent_kcal = _safe_float(action.get("energy_kcal"))
    # Also check nutrients dict for energy_kcal
    if agent_kcal is None:
        agent_kcal = _safe_float(agent_nutrients.get("energy_kcal"))

    atwater_ok = False
    atwater_details = {}
    if all(v is not None for v in (fat, carb, protein, agent_kcal)):
        expected_kcal = round_calories(compute_atwater_calories(fat, carb, protein))
        atwater_ok = abs(agent_kcal - expected_kcal) <= 5
        atwater_details = {
            "correct": atwater_ok,
            "agent_kcal": agent_kcal,
            "expected_kcal": expected_kcal,
        }
    else:
        atwater_details = {
            "correct": False,
            "agent_kcal": agent_kcal,
            "expected_kcal": None,
            "note": "missing macro or kcal value",
        }

    gt_score = (0.50 * nutrient_score_gt) + (0.35 * dv_score_gt) + (0.15 * atwater_ok)

    # --- Propagation scoring (against recomputed from agent's serving size) ---
    agent_serving = _safe_float(phase_2_action.get("serving_size_g"))
    lab_nutrients = ground_truth.get("lab_nutrients_raw", {})
    lab_sample_g = ground_truth.get("lab_sample_size_g", 1.0)

    propagation_score = 0.0
    if agent_serving is not None and agent_serving > 0 and lab_nutrients:
        _, recomputed_rounded, recomputed_dvs = _recompute_nutrients_for_serving(
            lab_nutrients, lab_sample_g, agent_serving,
        )
        nutrient_score_prop, _ = _score_nutrients_against(
            agent_nutrients, recomputed_rounded,
        )
        dv_score_prop, _ = _score_dvs_against(agent_dvs, recomputed_dvs)
        propagation_score = (0.50 * nutrient_score_prop) + (0.35 * dv_score_prop) + (0.15 * atwater_ok)

    # Final phase 3 score: best of absolute or half-credit for correct math from wrong serving
    final_score = max(gt_score, 0.5 * propagation_score)

    details = {}
    details.update(nutrient_details_gt)
    details.update(dv_details_gt)
    details["atwater_consistency"] = atwater_details
    details["_gt_score"] = gt_score
    details["_propagation_score"] = propagation_score

    return {"score": final_score, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 GRADER — Ingredient List + Compound Sublists
# ─────────────────────────────────────────────────────────────────────────────

def _grade_phase_4(action: dict, ground_truth: dict, phase_3_action: dict) -> dict:
    gt = ground_truth["phase_4"]
    details = {}

    # Ingredient order (case-insensitive exact ordered match)
    expected_order = gt["ingredient_order"]
    agent_list = action.get("ingredient_list") or []
    agent_lower = [s.lower().strip() for s in agent_list] if isinstance(agent_list, list) else []
    expected_lower = [s.lower().strip() for s in expected_order]
    order_ok = agent_lower == expected_lower
    details["ingredient_list"] = {
        "correct": order_ok,
        "agent": agent_list,
        "expected": expected_order,
    }

    # Compound ingredient sublists
    expected_sublists = gt.get("compound_sublists", {})
    agent_sublists = action.get("compound_ingredient_sublists") or {}

    compound_score = 1.0
    if expected_sublists:
        correct_compounds = 0
        total_compounds = len(expected_sublists)
        for compound_name, expected_subs in expected_sublists.items():
            agent_subs = agent_sublists.get(compound_name, [])
            agent_subs_lower = [s.lower().strip() for s in agent_subs]
            expected_subs_lower = [s.lower().strip() for s in expected_subs]
            match = agent_subs_lower == expected_subs_lower
            if match:
                correct_compounds += 1
            details[f"compound.{compound_name}"] = {
                "correct": match,
                "agent": agent_subs,
                "expected": expected_subs,
            }
        compound_score = correct_compounds / total_compounds if total_compounds > 0 else 1.0
    else:
        # No compound ingredients expected — agent should submit empty
        if agent_sublists:
            compound_score = 0.5  # small penalty for hallucinating sublists
            details["compound_sublists"] = {
                "correct": False,
                "agent": agent_sublists,
                "expected": {},
                "note": "no compound ingredients expected",
            }

    # Weighted: order=0.7, compound_sublists=0.3
    score = (0.7 * order_ok) + (0.3 * compound_score)

    return {"score": score, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 GRADER — PDP/Type Size + Health Claims + Consistency Audit
# ─────────────────────────────────────────────────────────────────────────────

def _grade_phase_5(action: dict, ground_truth: dict, all_prior_actions: dict) -> dict:
    gt = ground_truth["phase_5"]
    details = {}

    # Type size (must meet minimum for PDP area)
    min_size = gt["min_type_size_inch"]
    agent_ts = _safe_float(action.get("declared_type_size_inch"))
    ts_ok = agent_ts is not None and agent_ts >= min_size - 0.001
    details["declared_type_size_inch"] = {
        "correct": ts_ok,
        "agent": agent_ts,
        "expected_min": min_size,
    }

    # Health claims (should be empty unless substantiated)
    expected_claims = gt.get("valid_health_claims", [])
    agent_claims = action.get("health_claims") or []
    if not isinstance(agent_claims, list):
        agent_claims = []
    # Case-insensitive set comparison
    agent_claims_lower = {c.lower().strip() for c in agent_claims}
    expected_claims_lower = {c.lower().strip() for c in expected_claims}
    claims_ok = agent_claims_lower == expected_claims_lower
    details["health_claims"] = {
        "correct": claims_ok,
        "agent": agent_claims,
        "expected": expected_claims,
    }

    # Consistency violations — check if agent identified expected violations
    expected_violations = gt.get("consistency_violations", [])
    agent_violations = action.get("consistency_violations") or []

    if expected_violations:
        # Score based on how many expected violations were identified
        found = 0
        for expected_v in expected_violations:
            ev_lower = expected_v.lower().strip()
            for av in agent_violations:
                if ev_lower in av.lower().strip() or av.lower().strip() in ev_lower:
                    found += 1
                    break
        violation_score = found / len(expected_violations)
        # Penalty for false positives (hallucinated violations)
        false_positives = max(0, len(agent_violations) - found)
        false_positive_penalty = min(0.3, false_positives * 0.1)
        violation_score = max(0.0, violation_score - false_positive_penalty)
    else:
        # No violations expected — agent should submit empty list
        if agent_violations:
            violation_score = max(0.0, 1.0 - len(agent_violations) * 0.2)  # penalty for false positives
        else:
            violation_score = 1.0

    details["consistency_violations"] = {
        "correct": violation_score >= 0.99,
        "agent": agent_violations,
        "expected": expected_violations,
        "score": violation_score,
    }

    # Weighted: type_size=0.30, health_claims=0.30, consistency=0.40
    score = (0.30 * ts_ok) + (0.30 * claims_ok) + (0.40 * violation_score)

    return {"score": score, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def grade_phase(
    phase: int,
    action_dict: dict,
    ground_truth: dict,
    prior_actions: dict[int, dict],
) -> dict:
    """
    Grade a single phase submission.

    Args:
        phase: Phase number (1-5)
        action_dict: The agent's submission for this phase
        ground_truth: Full episode ground truth (with phase_1..phase_5 sub-dicts)
        prior_actions: Agent's prior phase submissions {phase_num: action_dict}

    Returns:
        {"score": float, "details": dict}
    """
    if phase == 1:
        return _grade_phase_1(action_dict, ground_truth)
    elif phase == 2:
        return _grade_phase_2(action_dict, ground_truth, prior_actions.get(1, {}))
    elif phase == 3:
        return _grade_phase_3(
            action_dict, ground_truth,
            prior_actions.get(1, {}), prior_actions.get(2, {}),
        )
    elif phase == 4:
        return _grade_phase_4(action_dict, ground_truth, prior_actions.get(3, {}))
    elif phase == 5:
        return _grade_phase_5(action_dict, ground_truth, prior_actions)
    else:
        raise ValueError(f"Invalid phase: {phase}")


def grade_episode(phase_scores: dict[int, float]) -> float:
    """Compute weighted episode score from per-phase scores."""
    total = 0.0
    for phase_num, weight in PHASE_WEIGHTS.items():
        total += weight * phase_scores.get(phase_num, 0.0)
    return total
