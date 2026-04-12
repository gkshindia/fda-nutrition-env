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
import logging
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

logger = logging.getLogger("fda.env")

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
        "description": "A Nutrition Facts label with 5 injected errors: "
        "two wrong %DVs, wrong ingredient order, type size below minimum, "
        "and one wrong nutrient rounding. "
        "Correct all errors and submit the fixed label. "
        "You will receive feedback after each attempt. "
        "Episode ends when you set final_submission=true or reach the step limit.",
        "difficulty": "easy",
        "max_steps": MAX_STEPS_SAFETY_CAP,
    },
    "task_medium": {
        "task_id": "task_medium",
        "name": "Medium — Fix label errors with cascading issues",
        "description": "A Nutrition Facts label with 7 injected errors: "
        "three wrong nutrient roundings, two wrong %DVs (one cascades from "
        "a wrong rounding), wrong ingredient order (non-adjacent swap), "
        "and a serving size inconsistency. Correct all errors. "
        "You will receive feedback after each attempt. "
        "Episode ends when you set final_submission=true or reach the step limit.",
        "difficulty": "medium",
        "max_steps": MAX_STEPS_SAFETY_CAP,
    },
    "task_hard": {
        "task_id": "task_hard",
        "name": "Hard — Fix label errors with full cascade",
        "description": "A Nutrition Facts label with 10 injected errors: "
        "four wrong nutrient roundings, an unsupported health claim, "
        "Atwater calorie inconsistency, wrong ingredient order (rotated), "
        "two wrong %DVs, and a serving size inconsistency that cascades "
        "to all per-serving values. Correct all errors. "
        "You will receive feedback after each attempt. "
        "Episode ends when you set final_submission=true or reach the step limit.",
        "difficulty": "hard",
        "max_steps": MAX_STEPS_SAFETY_CAP,
    },
}


def validate_task_id(task_id: str) -> dict:
    """Return the task config for a valid task id or raise a clear error."""
    task = TASKS.get(task_id)
    if task is None:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_id {task_id!r}. Valid task_ids: {valid}")
    return task


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
# FEEDBACK BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_feedback_text(
    grader_result: dict, step_number: int, max_steps: int,
) -> str:
    """
    Build human-readable feedback from grader result.

    Reveals which groups and fields are wrong without leaking hidden answers.
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

    def summarize_issue(field_name: str, detail: dict) -> str:
        if field_name == "declared_type_size_inch":
            minimum = detail.get("expected_min")
            if minimum is not None:
                return f"submitted {detail.get('agent')}; must be at least {minimum}"
            return f"submitted {detail.get('agent')}; declared type size is invalid"

        if field_name == "atwater_consistency":
            if detail.get("expected_kcal") is not None:
                return (
                    f"declared calories {detail.get('agent_kcal')} are inconsistent "
                    "with fat/carbohydrate/protein values"
                )
            return "missing fat, carbohydrate, protein, or calories needed for Atwater check"

        agent_val = detail.get("agent")
        if field_name == "ingredient_list":
            return f"submitted {agent_val}; ingredient order is incorrect"
        return f"submitted {agent_val}; value is incorrect"

    # Collect incorrect fields without leaking exact expected values.
    incorrect_fields = {
        field_name: detail
        for field_name, detail in field_details.items()
        if not detail.get("correct", False)
    }

    if incorrect_fields:
        lines.append("")
        lines.append("Incorrect fields:")
        for field_name, detail in incorrect_fields.items():
            lines.append(f"  - {field_name}: {summarize_issue(field_name, detail)}")
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
        task = validate_task_id(task_id)
        difficulty = task["difficulty"]

        episode = generate_episode(difficulty, seed=seed)
        self._episode = episode

        logger.info(
            "RESET  task=%s difficulty=%s seed=%s product=%r",
            task_id, difficulty, seed,
            episode.get("food_category_description", "?"),
        )

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
        context["episode_seed"] = episode["episode_seed"]  # expose for /grader replay

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
            metadata={
                "seed": episode["episode_seed"],
                "episode_id": self._state.episode_id,
            },
        )

    def step(
        self,
        action: FDAAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FDAObservation:
        if not self._state.ground_truth:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")

        self._state.step_count += 1

        result = grade(action.label, self._state.ground_truth)
        score = result["score"]

        # Track best submission
        improved = score > self._state.best_score
        if improved:
            self._state.best_score = score
            self._state.best_label = action.label
            self._state.no_improvement_count = 0
        else:
            self._state.no_improvement_count += 1

        # Always keep the best label as the active one for grading
        self._state.agent_label = self._state.best_label

        # Plateau detection: no improvement for 3 consecutive steps
        plateau = self._state.no_improvement_count >= 3

        is_final = (
            action.final_submission
            or self._state.step_count >= self._state.max_steps
            or plateau
        )

        stop_reason = (
            "final_submission" if action.final_submission
            else "max_steps"  if self._state.step_count >= self._state.max_steps
            else "plateau"    if plateau
            else None
        )

        logger.info(
            "STEP %d/%d  task=%s  score=%.3f (best=%.3f)  %s%s",
            self._state.step_count, self._state.max_steps,
            self._state.task_id, score, self._state.best_score,
            "↑ improved" if improved else "↔ no change",
            f"  → DONE ({stop_reason})" if is_final else "",
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
