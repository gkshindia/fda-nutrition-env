"""
Tests for _build_ground_truth() in core/episode_generator.py.

Fixtures are hand-calculated; no production pipeline functions are called
to compute expected values (except for compute_atwater_calories / round_calories /
compute_percent_dv / compute_pdp_area, which are tested separately in
regulatory_tables.py self-tests and trusted here as ground truth).

All float assertions use pytest.approx(abs=0.01) unless a tighter tolerance is noted.

Tests 4 and 5 are regression tests for two known bugs:
  Bug 1 (test 4): ingredient order used as-added weight, not finished weight.
  Bug 2 (test 5): 2% compound-ingredient threshold used as-added total, not finished total.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from data.regulatory_tables import (
    compute_atwater_calories,
    compute_pdp_area,
    compute_percent_dv,
    round_calories,
)
from data.seed_products import SEED_PRODUCTS
from core.episode_generator import (
    _build_ground_truth,
    _compute_lab_nutrients,
    _compute_recipe_nutrients,
)

# ── Shared stubs ────────────────────────────────────────────────────────────────

_RACC_STUB = SimpleNamespace(category="Test Category", racc_g=30.0)

_CONTAINER_RECT = {
    "shape": "rectangular",
    "height_in": 5.0,
    "width_in": 3.0,
    "num_display_panels": 3,
}


def _build(
    recipe: list[dict],
    *,
    moisture_loss_pct: float = 0.0,
    serving_g: float = 30.0,
    lab_sample_g: float = 100.0,
    container: dict | None = None,
    lab_nutrients_override: dict | None = None,
) -> dict:
    """Convenience wrapper: derive lab_nutrients from recipe then call _build_ground_truth."""
    if container is None:
        container = _CONTAINER_RECT
    nutrients_per_100g, _, _, _ = _compute_recipe_nutrients(recipe, moisture_loss_pct)
    lab_nutrients = _compute_lab_nutrients(nutrients_per_100g, lab_sample_g)
    if lab_nutrients_override:
        lab_nutrients.update(lab_nutrients_override)
    return _build_ground_truth(
        racc_entry=_RACC_STUB,
        serving_g=serving_g,
        dual_column=False,
        label_format="single_column",
        lab_nutrients=lab_nutrients,
        lab_sample_g=lab_sample_g,
        recipe=recipe,
        container=container,
        moisture_loss_pct=moisture_loss_pct,
    )


# ── Test 1: Scaling ─────────────────────────────────────────────────────────────

def test_scaling():
    """
    Per-serving unrounded value = (seed_product_per_100g / 100) * serving_g.
    Verified for protein via: lab_sample=50g, serving=28g, almonds only.
    """
    recipe = [
        {
            "ingredient_slug": "almonds",
            "weight_as_added_g": 100.0,
            "moisture_contributing": True,
            "is_compound": False,
        }
    ]
    result = _build(recipe, serving_g=28.0, lab_sample_g=50.0)

    # almonds protein = 21.15 g/100g
    # lab_protein  = 21.15/100 * 50  = 10.575
    # scaled       = 10.575 * (28/50) = 5.922  (= 21.15 * 0.28)
    expected_protein = SEED_PRODUCTS["almonds"]["protein_g"] / 100.0 * 28.0
    assert result["per_serving_scaled_unrounded"]["protein_g"] == pytest.approx(
        expected_protein, abs=0.01
    )


# ── Test 2: Atwater calories on pre-rounded scaled macros ───────────────────────

def test_atwater_on_prerounded_values():
    """
    atwater_kcal_declared = round_calories(9F + 4C + 4P) where F/C/P are
    the PRE-ROUNDED scaled macros — NOT the USDA energy_kcal field.

    For almonds at 28g serving, USDA energy_kcal gives ~162 kcal after rounding,
    while Atwater gives ~170 kcal. The declared value must be the Atwater result.
    """
    recipe = [
        {
            "ingredient_slug": "almonds",
            "weight_as_added_g": 100.0,
            "moisture_contributing": True,
            "is_compound": False,
        }
    ]
    result = _build(recipe, serving_g=28.0, lab_sample_g=100.0)

    scaled = result["per_serving_scaled_unrounded"]
    expected_raw = compute_atwater_calories(
        total_fat_g=scaled["total_fat_g"],
        total_carb_g=scaled["total_carbohydrate_g"],
        protein_g=scaled["protein_g"],
    )
    expected_kcal = round_calories(expected_raw)

    assert result["atwater_kcal_raw"] == pytest.approx(expected_raw, abs=0.1)
    assert result["atwater_kcal_declared"] == pytest.approx(expected_kcal, abs=1.0)
    assert result["per_serving_rounded"]["energy_kcal"] == pytest.approx(expected_kcal, abs=1.0)


# ── Test 3: Rounding uses round-half-up, not Python banker's rounding ───────────

def test_rounding_half_up_not_bankers():
    """
    sodium = 12.5 mg (exactly at a 5 mg increment boundary).

    round-half-up  : floor(12.5/5 + 0.5) * 5 = floor(3.0) * 5 = 15 mg  ← correct
    Python round() : round(12.5/5) * 5 = round(2.5) * 5 = 2 * 5 = 10 mg ← wrong
    """
    recipe = [
        {
            "ingredient_slug": "almonds",
            "weight_as_added_g": 100.0,
            "moisture_contributing": True,
            "is_compound": False,
        }
    ]
    # serving = lab_sample → scale = 1.0; override sodium to land exactly on boundary
    result = _build(
        recipe,
        serving_g=100.0,
        lab_sample_g=100.0,
        lab_nutrients_override={"sodium_mg": 12.5},
    )
    assert result["per_serving_rounded"]["sodium_mg"] == 15


# ── Test 4: Ingredient order uses finished weight, not as-added (Bug 1) ─────────

def test_ingredient_order_with_moisture_loss():
    """
    After moisture evaporation, finished weights differ from as-added weights.
    Ingredient list order must reflect finished weights (descending).

    Recipe: oats 60g (moisture-contributing) + honey 55g (not moisture-contributing)
    moisture_loss_pct = 10%

    Hand-calc:
      mc_total     = 60g
      total_ml     = 115 * 0.10 = 11.5g
      oats_finished  = 60 - (60/60) * 11.5 = 48.5g
      honey_finished = 55g  (unchanged — not moisture-contributing)

    Correct order : ["honey", "oats"]   (55 > 48.5)
    Buggy order   : ["oats", "honey"]   (60 > 55, as-added)
    """
    recipe = [
        {
            "ingredient_slug": "oats",
            "weight_as_added_g": 60.0,
            "moisture_contributing": True,
            "is_compound": False,
        },
        {
            "ingredient_slug": "honey",
            "weight_as_added_g": 55.0,
            "moisture_contributing": False,
            "is_compound": False,
        },
    ]
    result = _build(recipe, moisture_loss_pct=10.0)
    assert result["ingredient_order"] == ["honey", "oats"]


# ── Test 5: 2% threshold denominator uses finished weight total (Bug 2) ─────────

def test_2pct_threshold_uses_finished_weight():
    """
    A compound ingredient that is ≤2% of total as-added weight but >2% of
    total finished weight must NOT appear in ingredients_below_2pct.

    Recipe: oats 100g (moisture-contributing) + dark_chocolate 2g (not moisture-contributing, compound)
    moisture_loss_pct = 5%

    Hand-calc:
      mc_total       = 100g (oats only)
      total_as_added = 102g
      total_ml       = 102 * 0.05 = 5.1g
      oats_finished       = 100 - 5.1 = 94.9g
      dark_choc_finished  = 2.0g          (unchanged)
      total_finished      = 96.9g

    dark_choc / total_as_added = 2.0/102  = 1.96% ≤2%  → buggy code INCLUDES it
    dark_choc / total_finished = 2.0/96.9 = 2.06% >2%  → correct code EXCLUDES it
    """
    recipe = [
        {
            "ingredient_slug": "oats",
            "weight_as_added_g": 100.0,
            "moisture_contributing": True,
            "is_compound": False,
        },
        {
            "ingredient_slug": "dark_chocolate",
            "weight_as_added_g": 2.0,
            "moisture_contributing": False,
            "is_compound": True,
        },
    ]
    result = _build(recipe, moisture_loss_pct=5.0)
    assert "dark_chocolate" not in result["ingredients_below_2pct"]


# ── Test 6: %DV computed from rounded (declared) values ─────────────────────────

def test_percent_dv_from_rounded_values():
    """%DV for total_fat equals compute_percent_dv('total_fat', rounded_fat)."""
    recipe = [
        {
            "ingredient_slug": "almonds",
            "weight_as_added_g": 100.0,
            "moisture_contributing": True,
            "is_compound": False,
        }
    ]
    result = _build(recipe, serving_g=30.0, lab_sample_g=100.0)
    rounded_fat = result["per_serving_rounded"]["total_fat_g"]
    expected_dv = compute_percent_dv("total_fat", rounded_fat)
    assert result["percent_dvs"]["total_fat_g"] == pytest.approx(expected_dv, abs=0.1)


# ── Test 7: PDP area rectangular ────────────────────────────────────────────────

def test_pdp_area_rectangular():
    """PDP area for a rectangular container = height × width (21 CFR 101.1(b))."""
    recipe = [
        {
            "ingredient_slug": "almonds",
            "weight_as_added_g": 100.0,
            "moisture_contributing": True,
            "is_compound": False,
        }
    ]
    result = _build(recipe, container=_CONTAINER_RECT)
    expected_area = compute_pdp_area("rectangular", 5.0, width_in=3.0)  # = 15.0 sq in
    assert result["pdp_area_sq_in"] == pytest.approx(expected_area, abs=0.01)
