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

# Safety cap: maximum steps before forced termination, regardless of score.
MAX_STEPS_SAFETY_CAP = 10

TASKS = {
    "task_easy": {
        "task_id": "task_easy",
        "name": "Easy — Fix label errors",
        "description": "A Nutrition Facts label with 3 injected errors "
        "(wrong %DV, wrong ingredient order, type size violation). "
        "Correct all errors and submit the fixed label. "
        "You will receive feedback after each attempt. "
        "Episode ends when score >= 0.90 or you set final_submission=true.",
        "difficulty": "easy",
        "max_steps": MAX_STEPS_SAFETY_CAP,
    },
    "task_medium": {
        "task_id": "task_medium",
        "name": "Medium — Fix label errors with cross-step issues",
        "description": "A Nutrition Facts label with 5 injected errors "
        "including wrong nutrient rounding, wrong %DV, wrong ingredient "
        "order, and a serving size inconsistency. Correct all errors. "
        "You will receive feedback after each attempt. "
        "Episode ends when score >= 0.90 or you set final_submission=true.",
        "difficulty": "medium",
        "max_steps": MAX_STEPS_SAFETY_CAP,
    },
    "task_hard": {
        "task_id": "task_hard",
        "name": "Hard — Fix label errors with cascading issues",
        "description": "A Nutrition Facts label with 7 injected errors "
        "including wrong nutrient roundings, an unsupported health claim, "
        "Atwater calorie inconsistency, wrong ingredient order, and wrong "
        "%DV. Correct all errors. "
        "You will receive feedback after each attempt. "
        "Episode ends when score >= 0.90 or you set final_submission=true.",
        "difficulty": "hard",
        "max_steps": MAX_STEPS_SAFETY_CAP,
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


# Score threshold for early termination. When an agent's submission scores
# at or above this value, the environment auto-finalizes the episode.
# Rationale: FDA compliance tolerances (21 CFR 101.9) allow ±20% for most
# nutrients (Class I ≤120%, Class II ≥80%). A score of 0.90 means nearly all
# fields are correct — remaining errors are within real-world FDA tolerance.
EARLY_STOP_SCORE_THRESHOLD = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_feedback_text(
    grader_result: dict, step_number: int, max_steps: int,
) -> str:
    """
    Build human-readable feedback from grader result.

    Reveals which groups and fields are wrong, but NOT the expected values.
    """
    score = grader_result["score"]
    group_scores = grader_result["group_scores"]
    field_details = grader_result["field_details"]

    lines = [
        f"Score: {score:.3f} (step {step_number} of {max_steps})",
        f"You have {max_steps - step_number} revision(s) remaining. "
        f"Set final_submission=true to submit early.",
        "",
        "Group scores:",
    ]

    for group_name, group_score in group_scores.items():
        if group_score >= 1.0:
            status = "OK"
        else:
            status = "NEEDS FIX"
        lines.append(f"  {group_name}: {group_score:.2f} — {status}")

    # Collect incorrect fields (without showing expected values)
    incorrect_fields = [
        field_name
        for field_name, detail in field_details.items()
        if not detail.get("correct", False)
    ]

    if incorrect_fields:
        lines.append("")
        lines.append("Incorrect fields (expected values hidden):")
        for field_name in incorrect_fields:
            lines.append(f"  - {field_name}")
    else:
        lines.append("")
        lines.append("All fields correct! Set final_submission=true to lock in your score.")

    return "\n".join(lines)


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

        result = grade(action.label, self._state.ground_truth)
        score = result["score"]

        # Track best submission
        if score > self._state.best_score:
            self._state.best_score = score
            self._state.best_label = action.label
            self._state.no_improvement_count = 0
        else:
            self._state.no_improvement_count += 1

        # Always keep the best label as the active one for grading
        self._state.agent_label = self._state.best_label

        # Plateau detection: no improvement for 2 consecutive steps
        plateau = self._state.no_improvement_count >= 2

        is_final = (
            action.final_submission
            or self._state.step_count >= self._state.max_steps
            or score >= EARLY_STOP_SCORE_THRESHOLD
            or plateau
        )

        if is_final:
            self._state.completed = True
            best = self._state.best_score
            return FDAObservation(
                text=f"Label submitted. Final score: {best:.3f}"
                + (f" (best of {self._state.step_count} attempts)"
                   if self._state.step_count > 1 else ""),
                done=True,
                reward=best,
            )

        # Non-final step: return feedback without revealing expected values
        feedback_text = _build_feedback_text(
            result, self._state.step_count, self._state.max_steps,
        )
        return FDAObservation(
            text=feedback_text,
            draft_label=action.label,  # echo back what agent submitted
            episode_context={},
            done=False,
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
