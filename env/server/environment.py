"""
FDA Nutrition Facts Panel — OpenEnv Environment
================================================
Wraps episode generation + grading into the OpenEnv reset/step/grader cycle.

One-step interaction:
  1. Agent calls reset(task_id=...) → receives draft label + episode context
  2. Agent calls step(action) with corrected label → done=True
  3. grader_score returns the grade() result
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from core.episode_generator import generate_episode
from core.grader import grade
from env.models import FDAAction, FDAObservation, FDAState


# ─────────────────────────────────────────────────────────────────────────────
# TASKS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

TASKS = {
    "task_easy": {
        "task_id": "task_easy",
        "name": "Easy — Fix label errors",
        "description": "A Nutrition Facts label with 3 injected errors "
        "(wrong %DV, wrong ingredient order, type size violation). "
        "Correct all errors and submit the fixed label.",
        "difficulty": "easy",
        "max_steps": 1,
    },
    "task_medium": {
        "task_id": "task_medium",
        "name": "Medium — Fix label errors with cross-step issues",
        "description": "A Nutrition Facts label with 5 injected errors "
        "including wrong nutrient rounding, wrong %DV, wrong ingredient "
        "order, and a serving size inconsistency. Correct all errors.",
        "difficulty": "medium",
        "max_steps": 1,
    },
    "task_hard": {
        "task_id": "task_hard",
        "name": "Hard — Fix label errors with cascading issues",
        "description": "A Nutrition Facts label with 7 injected errors "
        "including wrong nutrient roundings, an unsupported health claim, "
        "Atwater calorie inconsistency, wrong ingredient order, and wrong "
        "%DV. Correct all errors.",
        "difficulty": "hard",
        "max_steps": 1,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are reviewing a draft FDA Nutrition Facts label for a food product.
The draft label contains errors. Your job is to identify and correct ALL errors,
then submit the corrected label as JSON.

The corrected label must have these keys:
  serving_size_g          — correct serving size in grams
  label_format            — "single_column", "dual_column", or "single_serving_container"
  declared_type_size_inch — minimum type size (must meet 21 CFR 101.9(d) requirement)
  nutrients               — dict of 15 nutrient values (FDA-rounded declared values)
  percent_dvs             — dict of %Daily Value for each nutrient (integer or null)
  ingredient_list         — list of ingredient slugs in descending weight order
  health_claims           — list of valid front-of-pack health claims (empty if none valid)

Key FDA rules:
  - Serving size comes from 21 CFR 101.12 RACC table + decision tree
  - Nutrients must be rounded per 21 CFR 101.9(c) (NOT Python round())
  - Calories = Atwater formula on PRE-ROUNDED scaled macros: 9×fat + 4×carb + 4×protein
  - %DV = round_half_up(rounded_value / DRV × 100)
  - Ingredients listed in descending order by FINISHED weight (after moisture loss)
  - Type size minimum depends on PDP area (21 CFR 101.9(d))
  - Health claims must be substantiated by nutrient values
"""


def _build_episode_context(episode: dict) -> dict:
    """Extract the subset of episode data visible to the agent."""
    return {
        "food_category_description": episode["food_category_description"],
        "physical_form": episode["physical_form"],
        "unit_weight_g": episode.get("unit_weight_g"),
        "total_package_weight_g": episode["total_package_weight_g"],
        "lab_sample_size_g": episode["lab_sample_size_g"],
        "lab_nutrients": episode["lab_nutrients"],
        "recipe": episode["recipe"],
        "moisture_loss_pct": episode["moisture_loss_pct"],
        "container": episode["container"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class FDAEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = FDAState()
        self._episode: dict | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "task_easy",
        **kwargs: Any,
    ) -> FDAObservation:
        task = TASKS.get(task_id, TASKS["task_easy"])
        difficulty = task["difficulty"]

        episode = generate_episode(difficulty, seed=seed)
        self._episode = episode

        self._state = FDAState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task["task_id"],
            difficulty=difficulty,
            max_steps=task["max_steps"],
            ground_truth=episode["ground_truth"],
            agent_label=None,
            completed=False,
        )

        context = _build_episode_context(episode)

        prompt = (
            f"{_SYSTEM_PROMPT}\n"
            f"--- PRODUCT ---\n"
            f"{json.dumps(context, indent=2)}\n\n"
            f"--- DRAFT LABEL (contains errors) ---\n"
            f"{json.dumps(episode['draft_label'], indent=2)}\n\n"
            f"Submit your corrected label as JSON in the 'label' field of your action."
        )

        return FDAObservation(
            text=prompt,
            draft_label=episode["draft_label"],
            episode_context=context,
        )

    def step(
        self,
        action: FDAAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FDAObservation:
        self._state.step_count += 1
        self._state.agent_label = action.label
        self._state.completed = True

        result = grade(action.label, self._state.ground_truth)
        score = result["score"]

        return FDAObservation(
            text=f"Label submitted. Score: {score:.3f}",
            done=True,
            reward=score,
        )

    @property
    def state(self) -> FDAState:
        return self._state

    @property
    def grader_score(self) -> float:
        if self._state.agent_label is None:
            return 0.0
        return grade(self._state.agent_label, self._state.ground_truth)["score"]
