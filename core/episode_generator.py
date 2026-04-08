"""
FDA Nutrition Facts Panel — Episode Generator
==============================================
Produces fully-specified episodes for the OpenEnv environment.

Usage:
    from core.episode_generator import generate_episode
    episode = generate_episode("easy", seed=42)
    episode = generate_episode("medium", seed=7)
    episode = generate_episode("hard", seed=99)

Episode dict schema documented in generate_episode() docstring.
All randomness flows through numpy.random.RandomState(seed).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from data.regulatory_tables import (
    RACC_TABLE,
    RACC_BY_CATEGORY,
    DAILY_REFERENCE_VALUES,
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
    compute_atwater_calories,
    compute_percent_dv,
    compute_pdp_area,
    lookup_min_type_size,
    discrete_unit_serving_size_branch,
    bulk_package_format_branch,
    _round_to_nearest,
)
from data.seed_products import SEED_PRODUCTS


# ─────────────────────────────────────────────────────────────────────────────
# NUTRIENT PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Maps seed_products field names → rounding function
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

# All nutrient keys in label order
_NUTRIENT_KEYS = list(_ROUNDING_DISPATCH.keys())


def _rhu(x: float, inc: float) -> float:
    """Round-half-up to increment. floor(x/inc + 0.5) * inc."""
    return math.floor(x / inc + 0.5) * inc


def _compute_recipe_nutrients(
    recipe: list[dict],
    moisture_loss_pct: float,
) -> tuple[dict[str, float], float, float, float]:
    """
    Compute per-100g finished-product nutrient values from a recipe.

    Args:
        recipe: list of {ingredient_slug, weight_as_added_g, moisture_contributing}
        moisture_loss_pct: percentage of moisture lost during processing (0-100)

    Returns:
        (nutrients_per_100g, total_as_added_g, finished_weight_g, added_sugars_per_100g)
    """
    totals: dict[str, float] = {k: 0.0 for k in _NUTRIENT_KEYS}
    total_as_added = sum(r["weight_as_added_g"] for r in recipe)
    pure_sugar_g = sum(
        r["weight_as_added_g"]
        for r in recipe
        if SEED_PRODUCTS[r["ingredient_slug"]].get("is_pure_sugar", False)
    )

    for item in recipe:
        slug = item["ingredient_slug"]
        w = item["weight_as_added_g"]
        prod = SEED_PRODUCTS[slug]
        for key in _NUTRIENT_KEYS:
            if key == "added_sugars_g":
                continue  # handled separately
            per_100 = prod.get(key, 0.0)
            totals[key] += (w / 100.0) * per_100

    # Apply moisture loss to non-nutrient mass only
    # Nutrients (solids) don't evaporate; only water does.
    moisture_loss_g = total_as_added * (moisture_loss_pct / 100.0)
    finished_weight = total_as_added - moisture_loss_g

    nutrients_per_100g: dict[str, float] = {}
    for key in _NUTRIENT_KEYS:
        if key == "added_sugars_g":
            continue
        raw = totals[key]
        nutrients_per_100g[key] = (raw / finished_weight) * 100.0

    # added_sugars: use weight of is_pure_sugar ingredients as added
    added_sugars_per_100g = (pure_sugar_g / finished_weight) * 100.0
    nutrients_per_100g["added_sugars_g"] = added_sugars_per_100g

    return nutrients_per_100g, total_as_added, finished_weight, added_sugars_per_100g


def _compute_lab_nutrients(
    nutrients_per_100g: dict[str, float],
    lab_sample_g: float,
) -> dict[str, float]:
    """Lab measures the sample: lab_nutrient = (per_100g / 100) * lab_sample_g."""
    return {k: (v / 100.0) * lab_sample_g for k, v in nutrients_per_100g.items()}


def _scale_to_serving(
    lab_nutrients: dict[str, float],
    lab_sample_g: float,
    serving_g: float,
) -> dict[str, float]:
    """Scale lab measurement to per-serving. Scale BEFORE rounding."""
    scale = serving_g / lab_sample_g
    return {k: v * scale for k, v in lab_nutrients.items()}


def _round_all(scaled: dict[str, float]) -> dict[str, float]:
    """Apply regulatory rounding to each nutrient."""
    result: dict[str, float] = {}
    for key, fn in _ROUNDING_DISPATCH.items():
        result[key] = fn(scaled.get(key, 0.0))
    return result


def _compute_percent_dvs(rounded: dict[str, float]) -> dict[str, Optional[float]]:
    dv_map = {
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
    result: dict[str, Optional[float]] = {}
    for field, dv_key in dv_map.items():
        result[field] = compute_percent_dv(dv_key, rounded.get(field, 0.0))
    return result


def _build_ground_truth(
    racc_entry,
    serving_g: float,
    dual_column: bool,
    label_format: str,
    lab_nutrients: dict[str, float],
    lab_sample_g: float,
    recipe: list[dict],
    container: dict,
    moisture_loss_pct: float = 0.0,
) -> dict:
    """Assemble the complete ground truth dict for an episode."""
    scaled = _scale_to_serving(lab_nutrients, lab_sample_g, serving_g)

    # Atwater on pre-rounded scaled values (per CFR: apply to actual amounts)
    atwater_kcal_raw = compute_atwater_calories(
        total_fat_g=scaled["total_fat_g"],
        total_carb_g=scaled["total_carbohydrate_g"],
        protein_g=scaled["protein_g"],
    )
    atwater_kcal_declared = round_calories(atwater_kcal_raw)

    rounded = _round_all(scaled)
    rounded["energy_kcal"] = atwater_kcal_declared  # override with Atwater result
    percent_dvs = _compute_percent_dvs(rounded)

    # Per-ingredient finished weight: moisture distributed among moisture-contributing
    # ingredients proportionally. Non-contributing ingredients retain as-added weight.
    _mc_total = sum(r["weight_as_added_g"] for r in recipe if r["moisture_contributing"])
    _total_as_added = sum(r["weight_as_added_g"] for r in recipe)
    _total_ml = _total_as_added * (moisture_loss_pct / 100.0)

    def _fw(r: dict) -> float:
        if r["moisture_contributing"] and _mc_total > 0:
            return r["weight_as_added_g"] - (r["weight_as_added_g"] / _mc_total) * _total_ml
        return r["weight_as_added_g"]

    # Ingredient order: sort by finished weight descending (21 CFR 101.4(a)(1))
    sorted_recipe = sorted(recipe, key=_fw, reverse=True)
    ingredient_order = [r["ingredient_slug"] for r in sorted_recipe]

    # PDP / type size
    shape = container["shape"]
    if shape == "cylindrical":
        pdp_area = compute_pdp_area(shape, container["height_in"],
                                    diameter_in=container["diameter_in"])
    else:
        pdp_area = compute_pdp_area(shape, container["height_in"],
                                    width_in=container["width_in"])
    type_tier = lookup_min_type_size(pdp_area)

    # 2% threshold for compound ingredients — denominator is total finished weight
    total_finished_w = sum(_fw(r) for r in recipe)
    two_pct_slugs = [
        r["ingredient_slug"]
        for r in recipe
        if r.get("is_compound", False)
        and (_fw(r) / total_finished_w) <= 0.02
    ]

    return {
        "racc_category": racc_entry.category,
        "racc_g": racc_entry.racc_g,
        "serving_size_g": serving_g,
        "label_format": label_format,
        "dual_column_required": dual_column,
        "per_serving_scaled_unrounded": scaled,
        "per_serving_rounded": rounded,
        "atwater_kcal_raw": atwater_kcal_raw,
        "atwater_kcal_declared": atwater_kcal_declared,
        "percent_dvs": percent_dvs,
        "ingredient_order": ingredient_order,
        "pdp_area_sq_in": pdp_area,
        "min_type_size_inch": type_tier.min_type_size_inch,
        "min_type_size_description": type_tier.description,
        "ingredients_below_2pct": two_pct_slugs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE TEMPLATES
# Pre-defines the structural choices for each difficulty tier.
# ─────────────────────────────────────────────────────────────────────────────

# Each template: {
#   racc_category, food_category_description, physical_form,
#   ingredient_pool, n_ingredients, compound_indices,
#   ambiguous_categories (medium/hard), correct_racc_category
# }

_EASY_TEMPLATES = [
    {
        # Nuts/seeds RACC (30g) — high fat, protein, fiber; many non-zero %DVs
        "racc_category": "Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole",
        "food_category_description": "Roasted mixed nut and seed trail mix",
        "physical_form": "bulk",
        "ingredient_pool": ["almonds", "walnuts", "flaxseed", "chia_seeds", "salt"],
        "n_ingredients": 5,
        "compound_indices": [],
        "weight_alpha_overrides": {"salt": 0.05},  # salt ≈0.7% of batch (realistic seasoning)
    },
    {
        # Nut butters RACC (32g) — peanut-dominant, added sugars from honey
        "racc_category": "Nut and seed butters, pastes, or creams",
        "food_category_description": "Creamy peanut butter with honey and oats",
        "physical_form": "bulk",
        "ingredient_pool": ["peanut_butter", "oats", "honey", "almonds", "salt"],
        "n_ingredients": 5,
        "compound_indices": [],
        "weight_alpha_overrides": {"salt": 0.05},  # salt ≈0.7% of batch (realistic seasoning)
    },
    {
        # Nut butters RACC (32g) — dark_chocolate as compound ingredient (cocoa, sugar, cocoa butter)
        "racc_category": "Nut and seed butters, pastes, or creams",
        "food_category_description": "Dark chocolate walnut spread",
        "physical_form": "bulk",
        "ingredient_pool": ["dark_chocolate", "walnuts", "canola_oil", "brown_sugar", "salt"],
        "n_ingredients": 5,
        "compound_indices": [0],  # dark_chocolate contains sub-ingredients
        "weight_alpha_overrides": {"salt": 0.05},  # salt ≈0.7% of batch (realistic seasoning)
    },
    {
        # Hot cereal RACC (40g) — oat-dominant, moisture from whole_milk
        "racc_category": "Breakfast cereals (hot cereal type), hominy grits",
        "food_category_description": "Hearty oat and flaxseed hot cereal",
        "physical_form": "bulk",
        "ingredient_pool": ["oats", "flaxseed", "honey", "whole_milk", "salt"],
        "n_ingredients": 5,
        "compound_indices": [],
        "weight_alpha_overrides": {"salt": 0.05},  # salt ≈0.7% of batch (realistic seasoning)
    },
    {
        # Fats/oils RACC (14g) — small serving, high sat-fat from coconut; tests trans_fat rounding
        "racc_category": "Butter, margarine, oil, shortening",
        "food_category_description": "Coconut and canola cooking oil blend",
        "physical_form": "bulk",
        "ingredient_pool": ["coconut_oil", "butter", "canola_oil", "salt"],
        "n_ingredients": 4,
        "compound_indices": [],
        "weight_alpha_overrides": {"salt": 0.05},  # salt ≈0.7% of batch (realistic seasoning)
    },
    {
        # Cookies RACC (30g) — discrete_unit; teaches the discrete-unit serving-size branch
        "racc_category": "Cookies",
        "food_category_description": "Classic butter shortbread cookies",
        "physical_form": "discrete_unit",
        "ingredient_pool": ["all_purpose_flour", "butter", "egg_whole_raw", "brown_sugar", "salt"],
        "n_ingredients": 5,
        "compound_indices": [],
        "weight_alpha_overrides": {"salt": 0.05},  # salt ≈0.7% of batch (realistic seasoning)
    },
]

_MEDIUM_TEMPLATES = [
    {
        # Cookies (30g) vs Grain-based bars (40g) — 10g RACC gap; discrete_unit borderline weight
        "correct_racc_category": "Grain-based bars with or without filling or coating, "
                                 "e.g., breakfast bars, granola bars, rice cereal bars",
        "ambiguous_categories": [
            "Cookies",
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
        ],
        "food_category_description": "Chewy oat and honey bar with dried fruit",
        "physical_form": "discrete_unit",
        "ingredient_pool": ["oats", "honey", "raisins", "canola_oil",
                            "all_purpose_flour", "brown_sugar"],
        "n_ingredients": 5,
        "compound_indices": [2],  # raisins — coated dried fruit, has sub-ingredient list
    },
    {
        # Nuts/seeds (30g) vs Snacks (30g) — same RACC, but grader checks declared category name
        "correct_racc_category": "Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole",
        "ambiguous_categories": [
            "Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole",
            "All varieties of snacks: chips, pretzels, popcorn, extruded snacks, "
            "fruit and vegetable-based snacks, grain-based snack mixes",
        ],
        "food_category_description": "Mixed nuts and dried fruit snack blend",
        "physical_form": "bulk",
        "ingredient_pool": ["almonds", "walnuts", "raisins", "dried_cranberries",
                            "dark_chocolate", "brown_sugar"],
        "n_ingredients": 5,
        "compound_indices": [4],  # dark_chocolate as compound
    },
    {
        # RTF cereal ≥43g/cup (60g) vs Grain-based bars (40g) — 20g RACC gap; moisture loss
        "correct_racc_category":
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
        "ambiguous_categories": [
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
        ],
        "food_category_description": "Honey almond toasted granola cereal",
        "physical_form": "bulk",
        "ingredient_pool": ["oats", "almonds", "honey", "canola_oil", "brown_sugar", "flaxseed"],
        "n_ingredients": 5,
        "compound_indices": [2],  # honey often carries additives → compound
        "moisture_loss_range": (8.0, 12.0),
    },
    {
        # Grain-based bars (40g) vs Brownies (40g) — same RACC, different label statement;
        # discrete_unit borderline weight; moisture loss adds finished-weight ordering challenge
        "correct_racc_category": "Grain-based bars with or without filling or coating, "
                                 "e.g., breakfast bars, granola bars, rice cereal bars",
        "ambiguous_categories": [
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
            "Brownies",
        ],
        "food_category_description": "Chewy walnut raisin oat energy bar",
        "physical_form": "discrete_unit",
        "ingredient_pool": ["oats", "walnuts", "raisins", "honey", "all_purpose_flour", "brown_sugar"],
        "n_ingredients": 5,
        "compound_indices": [2],  # raisins — compound dried fruit
        "moisture_loss_range": (5.0, 10.0),
    },
    {
        # Cookies (30g) vs All other candies (30g) — same RACC; agent must reason from food description
        "correct_racc_category": "Cookies",
        "ambiguous_categories": [
            "Cookies",
            "All other candies",
        ],
        "food_category_description": "No-bake dark chocolate almond energy bites",
        "physical_form": "discrete_unit",
        "ingredient_pool": ["dark_chocolate", "almonds", "oats", "honey", "chia_seeds", "salt"],
        "n_ingredients": 5,
        "compound_indices": [0],  # dark_chocolate is compound
        "moisture_loss_range": (3.0, 7.0),
    },
    {
        # Nuts/seeds (30g) vs Dried fruit (40g) — 10g RACC gap; correct category depends on
        # which ingredient class dominates by finished weight (almonds > dried_cranberries)
        "correct_racc_category":
            "Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole",
        "ambiguous_categories": [
            "Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole",
            "Dried fruit",
        ],
        "food_category_description": "Almond and dried cranberry snack mix with honey glaze",
        "physical_form": "bulk",
        "ingredient_pool": ["almonds", "dried_cranberries", "walnuts", "honey", "chia_seeds", "salt"],
        "n_ingredients": 5,
        "compound_indices": [3],  # honey acts as compound coating on fruit/nuts
        "moisture_loss_range": (2.0, 5.0),
    },
]

_HARD_TEMPLATES = [
    {
        # Cookies (30g) vs Grain-based bars (40g) vs All other candies (30g) — two 30g options
        "correct_racc_category": "Cookies",
        "ambiguous_categories": [
            "Cookies",
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
            "All other candies",
        ],
        "food_category_description": "Dark chocolate oat cluster with seeds and honey",
        "physical_form": "discrete_unit",
        "ingredient_pool": ["oats", "dark_chocolate", "honey", "flaxseed",
                            "chia_seeds", "brown_sugar", "canola_oil", "all_purpose_flour"],
        "n_ingredients": 7,
        "compound_indices": [1, 5],  # dark_chocolate and brown_sugar as compounds
        "moisture_loss_range": (15.0, 20.0),
        "health_claim": "Excellent source of fiber",  # may or may not be supported
    },
    {
        # RTF ≥43g/cup (60g) vs Hot cereal (40g) vs Grain-based bars (40g) — two 40g options
        "correct_racc_category":
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
        "ambiguous_categories": [
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
            "Breakfast cereals (hot cereal type), hominy grits",
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
        ],
        "food_category_description": "Toasted whole grain clusters with seeds and dried fruit",
        "physical_form": "bulk",
        "ingredient_pool": ["oats", "flaxseed", "chia_seeds", "raisins",
                            "dried_cranberries", "honey", "brown_sugar", "canola_oil"],
        "n_ingredients": 7,
        "compound_indices": [3, 5],  # raisins and honey as compounds
        "moisture_loss_range": (15.0, 20.0),
        "health_claim": "Good source of omega-3",  # unsupported (no DV for omega-3)
    },
    {
        # Grain-based bars (40g) vs Nut butters (32g) vs Cookies (30g) — three different RACCs
        "correct_racc_category": "Grain-based bars with or without filling or coating, "
                                 "e.g., breakfast bars, granola bars, rice cereal bars",
        "ambiguous_categories": [
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
            "Nut and seed butters, pastes, or creams",
            "Cookies",
        ],
        "food_category_description": "Dark chocolate peanut butter protein energy bar",
        "physical_form": "discrete_unit",
        "ingredient_pool": ["peanut_butter", "oats", "dark_chocolate", "honey",
                            "whey_protein_isolate", "brown_sugar", "canola_oil", "almonds"],
        "n_ingredients": 7,
        "compound_indices": [2, 5],  # dark_chocolate and brown_sugar as compounds
        "moisture_loss_range": (10.0, 15.0),
        "health_claim": "High protein",
    },
    {
        # Cookies (30g) vs Grain-based bars (40g) vs RTF ≥43g/cup (60g) — widest RACC spread (30/40/60)
        "correct_racc_category": "Grain-based bars with or without filling or coating, "
                                 "e.g., breakfast bars, granola bars, rice cereal bars",
        "ambiguous_categories": [
            "Cookies",
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
        ],
        "food_category_description": "Almond chia seed crunch bar with honey and oats",
        "physical_form": "discrete_unit",
        "ingredient_pool": ["almonds", "chia_seeds", "oats", "honey", "flaxseed",
                            "brown_sugar", "canola_oil", "all_purpose_flour"],
        "n_ingredients": 7,
        "compound_indices": [3, 5],  # honey and brown_sugar as compounds
        "moisture_loss_range": (12.0, 17.0),
        "health_claim": "Excellent source of omega-3 fatty acids",  # unsupported (no DV)
    },
    {
        # RTF ≥43g/cup (60g) vs Grain-based bars (40g) vs Hot cereal (40g)
        # Two 40g options make it harder to distinguish; dried fruit tips the finished-weight ordering
        "correct_racc_category":
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
        "ambiguous_categories": [
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
            "Breakfast cereals (hot cereal type), hominy grits",
        ],
        "food_category_description": "Crunchy walnut flaxseed granola cereal with dried fruit",
        "physical_form": "bulk",
        "ingredient_pool": ["oats", "walnuts", "flaxseed", "honey", "dried_cranberries",
                            "canola_oil", "brown_sugar", "salt"],
        "n_ingredients": 7,
        "compound_indices": [3, 6],  # honey and brown_sugar as compounds
        "moisture_loss_range": (15.0, 20.0),
        "health_claim": "Excellent source of fiber",
    },
    {
        # RTF ≥43g/cup (60g) vs Grain-based bars (40g) vs Cookies (30g)
        # whey_protein_isolate creates a nutrient profile that can misleadingly support "high protein"
        "correct_racc_category":
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
        "ambiguous_categories": [
            "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
            "Grain-based bars with or without filling or coating, "
            "e.g., breakfast bars, granola bars, rice cereal bars",
            "Cookies",
        ],
        "food_category_description": "High-protein whey oat cluster with almonds and honey",
        "physical_form": "bulk",
        "ingredient_pool": ["oats", "whey_protein_isolate", "almonds", "honey",
                            "flaxseed", "brown_sugar", "canola_oil", "chia_seeds"],
        "n_ingredients": 7,
        "compound_indices": [3, 5],  # honey and brown_sugar as compounds
        "moisture_loss_range": (12.0, 18.0),
        "health_claim": "Good source of protein",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CONTAINER GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _make_container(rng: np.random.RandomState, physical_form: str,
                    total_package_g: float) -> dict:
    """Generate plausible container dimensions."""
    shape = rng.choice(["rectangular", "cylindrical"])
    if shape == "rectangular":
        height_in = float(rng.uniform(3.0, 7.0))
        width_in = float(rng.uniform(2.0, 4.5))
        num_panels = int(rng.choice([3, 4, 6]))
        return {
            "shape": "rectangular",
            "height_in": round(height_in, 2),
            "width_in": round(width_in, 2),
            "num_display_panels": num_panels,
        }
    else:
        height_in = float(rng.uniform(3.0, 7.0))
        diameter_in = float(rng.uniform(2.0, 4.0))
        num_panels = int(rng.choice([2, 3]))
        return {
            "shape": "cylindrical",
            "height_in": round(height_in, 2),
            "diameter_in": round(diameter_in, 2),
            "num_display_panels": num_panels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DRAFT LABEL + ERROR INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_correct_label(
    rounded: dict[str, float],
    percent_dvs: dict[str, Optional[float]],
    ingredient_order: list[str],
    serving_g: float,
    declared_type_size_inch: float,
    label_format: str,
) -> dict:
    """Build a fully-correct label dict."""
    return {
        "serving_size_g": serving_g,
        "label_format": label_format,
        "declared_type_size_inch": declared_type_size_inch,
        "nutrients": dict(rounded),
        "percent_dvs": dict(percent_dvs),
        "ingredient_list": list(ingredient_order),
        "health_claims": [],
    }


def _inject_errors_easy(
    rng: np.random.RandomState,
    label: dict,
    ground_truth: dict,
) -> tuple[dict, list[dict]]:
    """
    5 injected errors for easy:
    1. One wrong %DV (small delta — easy to overlook)
    2. A second wrong %DV on a different nutrient
    3. Wrong ingredient order (swap 2 adjacent)
    4. Type size below minimum
    5. One wrong nutrient rounding
    """
    errors = []
    label = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
             for k, v in label.items()}

    # Errors 1-2: two wrong %DVs on different nutrients
    dv_fields = [f for f, v in label["percent_dvs"].items() if v is not None and v > 0]
    chosen_dv = rng.choice(dv_fields, size=min(2, len(dv_fields)), replace=False)
    for field in chosen_dv:
        correct = label["percent_dvs"][field]
        delta = int(rng.choice([-10, -5, 5, 10]))
        wrong = max(0, int(correct) + delta)
        label["percent_dvs"][field] = wrong
        errors.append({
            "type": "wrong_percent_dv",
            "field_path": f"percent_dvs.{field}",
            "injected_value": wrong,
            "correct_value": int(correct),
        })

    # Error 3: swap two adjacent ingredients in the list
    ingr = label["ingredient_list"]
    if len(ingr) >= 2:
        idx = int(rng.randint(0, len(ingr) - 1))
        ingr[idx], ingr[idx + 1] = ingr[idx + 1], ingr[idx]
        errors.append({
            "type": "wrong_ingredient_order",
            "field_path": "ingredient_list",
            "injected_value": list(ingr),
            "correct_value": list(ground_truth["ingredient_order"]),
        })

    # Error 4: type size below minimum
    min_size = ground_truth["min_type_size_inch"]
    wrong_size = round(min_size * 0.5, 4)
    label["declared_type_size_inch"] = wrong_size
    errors.append({
        "type": "type_size_violation",
        "field_path": "declared_type_size_inch",
        "injected_value": wrong_size,
        "correct_value": min_size,
    })

    # Error 5: one wrong nutrient rounding
    rounding_targets = ["total_fat_g", "total_carbohydrate_g", "protein_g", "sodium_mg"]
    field = str(rng.choice(rounding_targets))
    correct = label["nutrients"][field]
    if field.endswith("_mg"):
        wrong = correct + float(rng.choice([-5.0, 5.0]))
    else:
        wrong = correct + float(rng.choice([-0.5, 0.5, -1.0, 1.0]))
    wrong = max(0.0, wrong)
    label["nutrients"][field] = wrong
    errors.append({
        "type": "wrong_rounding",
        "field_path": f"nutrients.{field}",
        "injected_value": wrong,
        "correct_value": correct,
    })

    return label, errors


def _inject_errors_medium(
    rng: np.random.RandomState,
    label: dict,
    ground_truth: dict,
) -> tuple[dict, list[dict]]:
    """
    7 injected errors for medium:
    1-3. Three wrong nutrient rounding values
    4-5. Two wrong %DVs (one on a nutrient with wrong rounding — cascades)
    6. Wrong ingredient order (non-adjacent swap)
    7. Cross-step inconsistency: serving size on label ≠ RACC decision
    """
    errors = []
    label = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
             for k, v in label.items()}

    # Errors 1-3: three wrong nutrient roundings
    rounding_targets = ["total_fat_g", "total_carbohydrate_g", "protein_g",
                        "dietary_fiber_g", "sodium_mg", "saturated_fat_g"]
    chosen = rng.choice(rounding_targets, size=min(3, len(rounding_targets)), replace=False)
    injected_rounding_fields = set()
    for field in chosen:
        correct = label["nutrients"][field]
        if field.endswith("_mg"):
            wrong = correct + float(rng.choice([-5.0, 5.0, -10.0, 10.0]))
        else:
            wrong = correct + float(rng.choice([-0.5, 0.5, -1.0, 1.0]))
        wrong = max(0.0, wrong)
        label["nutrients"][field] = wrong
        injected_rounding_fields.add(field)
        errors.append({
            "type": "wrong_rounding",
            "field_path": f"nutrients.{field}",
            "injected_value": wrong,
            "correct_value": correct,
        })

    # Errors 4-5: two wrong %DVs — one on a field that was already rounded wrong (cascade)
    dv_fields = [f for f, v in label["percent_dvs"].items() if v is not None and v > 0]
    cascaded = [f for f in dv_fields if f in injected_rounding_fields]
    independent = [f for f in dv_fields if f not in injected_rounding_fields]
    # Pick one cascaded and one independent if possible
    dv_pick = []
    if cascaded:
        dv_pick.append(str(rng.choice(cascaded)))
    if independent:
        dv_pick.append(str(rng.choice([f for f in independent if f not in dv_pick])))
    dv_pick = dv_pick[:2]
    for field in dv_pick:
        correct = label["percent_dvs"][field]
        delta = int(rng.choice([-10, -5, 5, 10]))
        wrong = max(0, int(correct) + delta)
        label["percent_dvs"][field] = wrong
        errors.append({
            "type": "wrong_percent_dv",
            "field_path": f"percent_dvs.{field}",
            "injected_value": wrong,
            "correct_value": int(correct),
        })

    # Error 6: wrong ingredient order — swap two non-adjacent items
    ingr = label["ingredient_list"]
    if len(ingr) >= 3:
        i, j = 0, len(ingr) - 1  # swap first and last
        ingr[i], ingr[j] = ingr[j], ingr[i]
    elif len(ingr) >= 2:
        ingr[0], ingr[1] = ingr[1], ingr[0]
    errors.append({
        "type": "wrong_ingredient_order",
        "field_path": "ingredient_list",
        "injected_value": list(ingr),
        "correct_value": list(ground_truth["ingredient_order"]),
    })

    # Error 7: cross-step inconsistency — serving size off by 5g
    correct_ss = label["serving_size_g"]
    wrong_ss = correct_ss + float(rng.choice([-5.0, 5.0]))
    label["serving_size_g"] = wrong_ss
    errors.append({
        "type": "cross_step_inconsistency",
        "field_path": "serving_size_g",
        "injected_value": wrong_ss,
        "correct_value": correct_ss,
        "note": "Serving size on label does not match RACC decision tree result",
    })

    return label, errors


def _inject_errors_hard(
    rng: np.random.RandomState,
    label: dict,
    ground_truth: dict,
    health_claim: str,
) -> tuple[dict, list[dict]]:
    """
    10 injected errors for hard:
    1-4. Four wrong nutrient roundings (at rounding boundaries)
    5.   Unsupported front-of-pack health claim
    6.   Atwater inconsistency in declared calories
    7.   Wrong ingredient order (full reversal of two non-trivial positions)
    8-9. Two wrong %DVs — one cascades from wrong rounding, one independent
    10.  Cross-step serving size inconsistency (cascades to all per-serving values)
    """
    errors = []
    label = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
             for k, v in label.items()}

    # Errors 1-4: four wrong nutrient roundings
    rounding_targets = ["total_fat_g", "saturated_fat_g", "total_carbohydrate_g",
                        "dietary_fiber_g", "protein_g", "sodium_mg", "potassium_mg"]
    chosen = rng.choice(rounding_targets, size=min(4, len(rounding_targets)), replace=False)
    injected_rounding_fields = set()
    for field in chosen:
        correct = label["nutrients"][field]
        if field.endswith("_mg"):
            wrong = correct + float(rng.choice([-10.0, 10.0, -5.0, 5.0]))
        else:
            wrong = correct + float(rng.choice([-1.0, 1.0, -0.5, 0.5]))
        wrong = max(0.0, wrong)
        label["nutrients"][field] = wrong
        injected_rounding_fields.add(field)
        errors.append({
            "type": "wrong_rounding",
            "field_path": f"nutrients.{field}",
            "injected_value": wrong,
            "correct_value": correct,
        })

    # Error 5: unsupported health claim on PDP
    label["health_claims"] = [health_claim]
    errors.append({
        "type": "unsupported_health_claim",
        "field_path": "health_claims",
        "injected_value": [health_claim],
        "correct_value": [],
        "note": f"Claim '{health_claim}' is not supported by computed nutrient values "
                f"or references a nutrient without an established DV.",
    })

    # Error 6: Atwater inconsistency — wrong declared calories
    wrong_fat  = label["nutrients"].get("total_fat_g",          ground_truth["per_serving_rounded"]["total_fat_g"])
    wrong_carb = label["nutrients"].get("total_carbohydrate_g", ground_truth["per_serving_rounded"]["total_carbohydrate_g"])
    wrong_prot = label["nutrients"].get("protein_g",            ground_truth["per_serving_rounded"]["protein_g"])
    atwater_from_wrong = compute_atwater_calories(wrong_fat, wrong_carb, wrong_prot)
    kcal_offset = float(rng.choice([-20.0, 20.0, 30.0, -30.0, 40.0, -40.0]))
    wrong_kcal = max(0.0, round_calories(atwater_from_wrong) + kcal_offset)
    correct_kcal = ground_truth["atwater_kcal_declared"]
    label["nutrients"]["energy_kcal"] = wrong_kcal
    errors.append({
        "type": "atwater_inconsistency",
        "field_path": "nutrients.energy_kcal",
        "injected_value": wrong_kcal,
        "correct_value": correct_kcal,
        "note": "Declared calories are inconsistent with Atwater calculation from declared macros.",
    })

    # Error 7: wrong ingredient order — rotate list by 2 positions
    ingr = label["ingredient_list"]
    if len(ingr) >= 3:
        ingr[:] = ingr[2:] + ingr[:2]  # rotate left by 2
    errors.append({
        "type": "wrong_ingredient_order",
        "field_path": "ingredient_list",
        "injected_value": list(ingr),
        "correct_value": list(ground_truth["ingredient_order"]),
    })

    # Errors 8-9: two wrong %DVs
    dv_fields = [f for f, v in label["percent_dvs"].items() if v is not None and v > 0]
    cascaded   = [f for f in dv_fields if f in injected_rounding_fields]
    independent = [f for f in dv_fields if f not in injected_rounding_fields]
    dv_pick: list[str] = []
    if cascaded:
        dv_pick.append(str(rng.choice(cascaded)))
    if independent:
        remaining = [f for f in independent if f not in dv_pick]
        if remaining:
            dv_pick.append(str(rng.choice(remaining)))
    for field in dv_pick[:2]:
        correct = label["percent_dvs"][field]
        delta = int(rng.choice([-15, 15, -10, 10, -20, 20]))
        wrong = max(0, int(correct) + delta)
        label["percent_dvs"][field] = wrong
        errors.append({
            "type": "wrong_percent_dv",
            "field_path": f"percent_dvs.{field}",
            "injected_value": wrong,
            "correct_value": int(correct),
            "note": "Cross-step: %DV inconsistent with declared rounded value.",
        })

    # Error 10: cross-step serving size inconsistency
    correct_ss = label["serving_size_g"]
    wrong_ss = correct_ss + float(rng.choice([-10.0, 10.0, -5.0, 5.0]))
    label["serving_size_g"] = max(1.0, wrong_ss)
    errors.append({
        "type": "cross_step_inconsistency",
        "field_path": "serving_size_g",
        "injected_value": max(1.0, wrong_ss),
        "correct_value": correct_ss,
        "note": "Serving size does not match RACC decision tree — all per-serving values cascade.",
    })

    return label, errors


# ─────────────────────────────────────────────────────────────────────────────
# SERVING SIZE DETERMINATION
# ─────────────────────────────────────────────────────────────────────────────

def _determine_serving(
    racc_entry,
    physical_form: str,
    unit_weight_g: Optional[float],
    total_package_g: float,
) -> tuple[float, str, bool]:
    """
    Returns (serving_size_g, label_format, dual_column_required).
    """
    racc_g = racc_entry.racc_g

    if physical_form == "discrete_unit":
        branch_result = discrete_unit_serving_size_branch(unit_weight_g, racc_g)
        dual = branch_result["dual_column_required"]
        serving_g = branch_result.get("serving_size_g",
                                      branch_result.get("serving_size_g_option_1",
                                                        unit_weight_g))
        if dual:
            fmt = "dual_column"
        else:
            fmt = "single_column"
    else:
        # bulk
        branch_result = bulk_package_format_branch(total_package_g, racc_g)
        dual = branch_result["dual_column_required"]
        fmt_key = branch_result["format"]
        if fmt_key == "single_serving_container":
            serving_g = total_package_g
            fmt = "single_serving_container"
        elif fmt_key == "dual_column_required":
            serving_g = _rhu(racc_g, 5.0)
            fmt = "dual_column"
        else:
            serving_g = _rhu(racc_g, 5.0)
            fmt = "single_column"

    return serving_g, fmt, dual


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_episode(
    difficulty: str,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate one FDA Nutrition Facts Panel episode.

    Args:
        difficulty: "easy" | "medium" | "hard"
        seed: integer seed for numpy.random.RandomState; None = random

    Returns:
        Episode dict with keys:
            food_category_description  — natural language, ambiguous at medium/hard
            physical_form              — "discrete_unit" | "bulk"
            unit_weight_g              — float (discrete_unit only, else None)
            total_package_weight_g     — float
            lab_sample_size_g          — float ≠ serving_size for medium/hard
            lab_nutrients              — dict of measured nutrients on the lab sample
            recipe                     — list of ingredient dicts
            moisture_loss_pct          — float (0 for easy)
            container                  — {shape, height_in, ...}
            draft_label                — label dict with injected errors
            ground_truth               — correct answer dict
            difficulty                 — echo of input
            episode_seed               — int seed used
    """
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(f"difficulty must be 'easy', 'medium', or 'hard'; got {difficulty!r}")

    # Resolve seed
    if seed is None:
        seed = int(np.random.randint(0, 2**31))
    rng = np.random.RandomState(seed)

    # ── 1. Select template ────────────────────────────────────────────────────
    if difficulty == "easy":
        templates = _EASY_TEMPLATES
    elif difficulty == "medium":
        templates = _MEDIUM_TEMPLATES
    else:
        templates = _HARD_TEMPLATES

    tmpl = templates[int(rng.randint(0, len(templates)))]

    # ── 2. Resolve RACC entry ─────────────────────────────────────────────────
    if difficulty == "easy":
        racc_category = tmpl["racc_category"]
    else:
        racc_category = tmpl["correct_racc_category"]
    racc_entry = RACC_BY_CATEGORY[racc_category]
    racc_g = racc_entry.racc_g

    # ── 3. Build recipe ───────────────────────────────────────────────────────
    pool = tmpl["ingredient_pool"]
    n = tmpl["n_ingredients"]
    chosen_slugs = pool[:n]  # always take first n (order is curated)

    # Distribute weights: first ingredient dominates, rest are smaller
    # Total recipe batch = 100-300g (reasonable kitchen batch)
    batch_base = float(rng.uniform(150.0, 250.0))

    # Weights: draw from Dirichlet then scale to batch_base.
    # weight_alpha_overrides lets templates give seasonings/condiments (e.g. salt)
    # a tiny alpha so they stay at culinary-realistic fractions (<1%) rather than
    # the default ~12% that Dirichlet(1) would produce for a 5-ingredient recipe.
    alpha = np.ones(n)
    alpha[0] = 4.0  # first ingredient is dominant
    alpha_overrides = tmpl.get("weight_alpha_overrides", {})
    for i, slug in enumerate(chosen_slugs):
        if slug in alpha_overrides:
            alpha[i] = alpha_overrides[slug]
    fractions = rng.dirichlet(alpha)
    weights_raw = fractions * batch_base

    # For hard: ensure one compound ingredient is exactly at 2.0% of total
    compound_indices = tmpl.get("compound_indices", [])
    if difficulty == "hard" and len(compound_indices) >= 1:
        ci = compound_indices[0]
        total = float(weights_raw.sum())
        weights_raw[ci] = total * 0.020  # exactly 2.0%
        # Re-normalize non-compound ingredients proportionally
        other_indices = [i for i in range(n) if i != ci]
        other_total = sum(weights_raw[i] for i in other_indices)
        needed = total - weights_raw[ci]
        for i in other_indices:
            weights_raw[i] = weights_raw[i] / other_total * needed

    recipe = []
    for i, slug in enumerate(chosen_slugs):
        recipe.append({
            "ingredient_slug": slug,
            "weight_as_added_g": round(float(weights_raw[i]), 2),
            "moisture_contributing": SEED_PRODUCTS[slug].get("is_pure_sugar", False) is False,
            "is_compound": i in compound_indices,
        })

    # ── 4. Moisture loss ──────────────────────────────────────────────────────
    if difficulty == "hard":
        ml_range = tmpl.get("moisture_loss_range", (15.0, 20.0))
        moisture_loss_pct = float(rng.uniform(*ml_range))
    elif difficulty == "medium":
        # Medium templates may opt in to moisture loss via moisture_loss_range key
        ml_range = tmpl.get("moisture_loss_range", (0.0, 0.0))
        moisture_loss_pct = float(rng.uniform(*ml_range)) if ml_range[1] > 0 else 0.0
    else:
        moisture_loss_pct = 0.0

    # ── 5. Compute finished-product nutrients per 100g ────────────────────────
    nutrients_per_100g, total_as_added, finished_weight, _ = _compute_recipe_nutrients(
        recipe, moisture_loss_pct
    )

    # ── 6. Determine physical form and package weight ─────────────────────────
    physical_form = tmpl["physical_form"]

    if physical_form == "discrete_unit":
        # Unit weight scaled to put us in the right RACC tier per difficulty
        if difficulty == "easy":
            # Unit in 67-100% of RACC → branch C (1 unit serving)
            ratio = float(rng.uniform(0.67, 1.00))
            unit_weight_g = round(_rhu(racc_g * ratio, 1.0), 1)
            n_units = int(rng.randint(8, 16))
        elif difficulty == "medium":
            # Unit in 100-200% of RACC → branch C (borderline)
            ratio = float(rng.uniform(1.00, 1.80))
            unit_weight_g = round(_rhu(racc_g * ratio, 1.0), 1)
            n_units = int(rng.randint(4, 10))
        else:  # hard
            # Unit in 200-300% of RACC → branch D (dual column)
            ratio = float(rng.uniform(2.01, 2.90))
            unit_weight_g = round(_rhu(racc_g * ratio, 1.0), 1)
            n_units = int(rng.randint(3, 8))
        total_package_weight_g = round(unit_weight_g * n_units, 1)
    else:
        # Bulk
        unit_weight_g = None
        if difficulty == "easy":
            # <= 150% RACC → single-serving
            ratio = float(rng.uniform(1.10, 1.50))
        elif difficulty == "medium":
            # ~180-220% RACC → near dual-column boundary
            ratio = float(rng.uniform(1.80, 2.20))
        else:
            # 200-290% RACC → dual-column
            ratio = float(rng.uniform(2.01, 2.90))
        total_package_weight_g = round(_rhu(racc_g * ratio, 5.0), 1)

    # ── 7. Determine serving size ─────────────────────────────────────────────
    serving_g, label_format, dual_column = _determine_serving(
        racc_entry, physical_form, unit_weight_g, total_package_weight_g
    )

    # ── 8. Lab sample size ────────────────────────────────────────────────────
    if difficulty == "easy":
        # Lab sample = serving size (scale factor = 1.0 exactly)
        lab_sample_size_g = serving_g
    elif difficulty == "medium":
        # Non-integer scale factor: lab sample ≠ serving
        # Choose a lab sample that gives a non-integer scale factor ≥ 1.2
        # e.g., lab = 25g, serving = 40g → scale = 1.6
        scale_factor = float(rng.choice([1.25, 1.5, 1.6, 1.75, 2.0, 2.25]))
        lab_sample_size_g = round(serving_g / scale_factor, 1)
        if lab_sample_size_g == serving_g:
            lab_sample_size_g = round(serving_g / 1.5, 1)
    else:  # hard
        # Scale factor >= 2.5
        scale_factor = float(rng.uniform(2.5, 4.0))
        lab_sample_size_g = round(serving_g / scale_factor, 2)
        if lab_sample_size_g == serving_g:
            lab_sample_size_g = round(serving_g / 3.0, 2)

    # Ensure lab_sample > 0
    lab_sample_size_g = max(lab_sample_size_g, 1.0)

    # ── 9. Compute lab nutrients ──────────────────────────────────────────────
    lab_nutrients = _compute_lab_nutrients(nutrients_per_100g, lab_sample_size_g)

    # ── 10. Nudge boundary nutrients for medium/hard ──────────────────────────
    if difficulty in ("medium", "hard"):
        # Scale lab nutrients to serving to find pre-round values, then nudge
        # some to land near rounding boundaries
        n_boundary = 2 if difficulty == "medium" else 3
        boundary_targets = ["total_fat_g", "total_carbohydrate_g", "sodium_mg",
                            "protein_g", "dietary_fiber_g"]
        chosen_boundary = rng.choice(boundary_targets,
                                     size=min(n_boundary, len(boundary_targets)),
                                     replace=False)
        for field in chosen_boundary:
            # Compute current per-serving value
            per_serving = lab_nutrients[field] * (serving_g / lab_sample_size_g)
            # Nudge to the nearest rounding boundary
            if field.endswith("_mg"):
                boundary = _rhu(per_serving, 5.0)
                # Put it exactly at the boundary ± 0.1mg
                nudge = float(rng.choice([-0.1, 0.1]))
                target_per_serving = boundary + nudge
            elif field == "total_fat_g" and per_serving > 5.0:
                boundary = _rhu(per_serving, 1.0)
                nudge = float(rng.choice([-0.05, 0.05]))
                target_per_serving = boundary + nudge
            else:
                boundary = _rhu(per_serving, 1.0)
                nudge = float(rng.choice([-0.05, 0.05]))
                target_per_serving = boundary + nudge
            target_per_serving = max(0.0, target_per_serving)
            # Back-compute the lab nutrient needed
            lab_nutrients[field] = target_per_serving * (lab_sample_size_g / serving_g)

    # ── 11. Ground truth ──────────────────────────────────────────────────────
    container = _make_container(rng, physical_form, total_package_weight_g)

    ground_truth = _build_ground_truth(
        racc_entry=racc_entry,
        serving_g=serving_g,
        dual_column=dual_column,
        label_format=label_format,
        lab_nutrients=lab_nutrients,
        lab_sample_g=lab_sample_size_g,
        recipe=recipe,
        container=container,
        moisture_loss_pct=moisture_loss_pct,
    )

    # Add racc candidates for medium/hard
    if difficulty in ("medium", "hard"):
        ground_truth["ambiguous_racc_candidates"] = [
            {
                "category": cat,
                "racc_g": RACC_BY_CATEGORY[cat].racc_g,
                "is_correct": cat == racc_category,
            }
            for cat in tmpl["ambiguous_categories"]
        ]

    # ── 12. Draft label with injected errors ─────────────────────────────────
    # Correct type size = minimum required (declare exactly at minimum)
    correct_type_size = ground_truth["min_type_size_inch"]

    correct_label = _build_correct_label(
        rounded=ground_truth["per_serving_rounded"],
        percent_dvs=ground_truth["percent_dvs"],
        ingredient_order=ground_truth["ingredient_order"],
        serving_g=serving_g,
        declared_type_size_inch=correct_type_size,
        label_format=label_format,
    )

    if difficulty == "easy":
        draft_label, injected_errors = _inject_errors_easy(rng, correct_label, ground_truth)
    elif difficulty == "medium":
        draft_label, injected_errors = _inject_errors_medium(rng, correct_label, ground_truth)
    else:
        health_claim = tmpl.get("health_claim", "Excellent source of fiber")
        draft_label, injected_errors = _inject_errors_hard(
            rng, correct_label, ground_truth, health_claim
        )

    ground_truth["injected_errors"] = injected_errors

    # ── 13. Assemble episode ──────────────────────────────────────────────────
    food_category_description = tmpl["food_category_description"]

    episode = {
        "food_category_description": food_category_description,
        "physical_form": physical_form,
        "unit_weight_g": unit_weight_g,
        "total_package_weight_g": total_package_weight_g,
        "lab_sample_size_g": lab_sample_size_g,
        "lab_nutrients": lab_nutrients,
        "recipe": recipe,
        "moisture_loss_pct": moisture_loss_pct,
        "container": container,
        "draft_label": draft_label,
        "ground_truth": ground_truth,
        "difficulty": difficulty,
        "episode_seed": seed,
    }

    return episode
