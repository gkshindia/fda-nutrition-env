"""
FDA Nutrition Facts Panel — OpenEnv Environment (v2: 5-Phase Sequential)
========================================================================
Each episode has 5 phases. Each step() call advances to the next phase.
No revision loops — one shot per phase.

Phase flow:
  reset()   → phase=1 observation (food description + package info)
  step(1)   → grade phase 1, return phase 2 observation
  step(2)   → grade phase 2, return phase 3 observation
  step(3)   → grade phase 3 (dual-scored), return phase 4 observation
  step(4)   → grade phase 4, return phase 5 observation
  step(5)   → grade phase 5, done=True, reward = weighted episode score
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from core.episode_generator import generate_episode
from core.grader_v2 import grade_phase, grade_episode, PHASE_WEIGHTS
from env.models import FDAAction, FDAObservation, FDAState

logger = logging.getLogger("fda.env")


# ─────────────────────────────────────────────────────────────────────────────
# TASKS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

TASKS = {
    "task_easy": {
        "task_id": "task_easy",
        "name": "Easy — FDA Label Compliance Pipeline",
        "description": (
            "Navigate the 5-step FDA Nutrition Facts label compliance pipeline: "
            "(1) classify the food category and determine RACC, "
            "(2) select label format via the decision tree, "
            "(3) compute nutrient values with FDA rounding and Atwater calories, "
            "(4) determine ingredient order after moisture loss, "
            "(5) audit PDP type size and health claims. "
            "Easy: unambiguous category, scale factor = 1.0, no moisture loss."
        ),
        "difficulty": "easy",
        "max_steps": 5,
    },
    "task_medium": {
        "task_id": "task_medium",
        "name": "Medium — FDA Label Compliance with Ambiguity",
        "description": (
            "Same 5-step pipeline but with added difficulty: "
            "ambiguous food category (2 plausible RACCs), "
            "non-integer scaling factor (lab ≠ serving size), "
            "nutrients at rounding boundaries, "
            "and moisture loss affecting ingredient order."
        ),
        "difficulty": "medium",
        "max_steps": 5,
    },
    "task_hard": {
        "task_id": "task_hard",
        "name": "Hard — Full Cascade FDA Label Compliance",
        "description": (
            "Same 5-step pipeline at maximum difficulty: "
            "3 ambiguous food categories, scale factor ≥ 2.5, "
            "15-20% moisture loss, compound ingredients at the 2% threshold, "
            "nutrients at rounding boundaries, and unsupported health claims. "
            "Wrong category in Step 1 cascades through all subsequent steps."
        ),
        "difficulty": "hard",
        "max_steps": 5,
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
# PHASE OBSERVATION BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_PROMPTS = {
    1: (
        "## Phase 1: Food Category Classification & RACC Determination\n\n"
        "Classify this food product into the correct FDA category from 21 CFR 101.12 Table 2.\n"
        "Determine the RACC (Reference Amount Customarily Consumed) in grams.\n"
        "Provide the household measure declaration (e.g., '1 bar', '2 tbsp').\n\n"
        "Return JSON with keys: food_category, racc_g, household_measure"
    ),
    2: (
        "## Phase 2: Label Format Selection\n\n"
        "Using the RACC from Phase 1, navigate the decision tree to determine:\n"
        "- Label format: single_column, dual_column, or single_serving_container\n"
        "- Serving size in grams\n"
        "- Serving declaration text (e.g., '1 bar (40g)')\n\n"
        "Decision tree rules:\n"
        "- Discrete units: if unit weight is 50-200% of RACC, one unit is the serving\n"
        "- Bulk: package ≤200% RACC → single serving container; "
        "200-300% → dual column; >300% → single column\n\n"
        "Return JSON with keys: label_format, serving_size_g, serving_declaration_text"
    ),
    3: (
        "## Phase 3: Nutrient Math, Rounding, & Calorie Consistency\n\n"
        "Scale each lab nutrient to per-serving, apply FDA rounding (round-half-UP), "
        "compute %DV, and verify Atwater calorie consistency.\n\n"
        "Steps:\n"
        "1. Scale: nutrient_per_serving = (lab_nutrient / lab_sample_size_g) × serving_size_g\n"
        "2. Round per nutrient-specific FDA rules (21 CFR 101.9(c))\n"
        "3. Atwater: energy_kcal = round_calories(9×UNROUNDED_fat + 4×UNROUNDED_carb + 4×UNROUNDED_protein)\n"
        "4. %DV = floor(rounded_value / DRV × 100 + 0.5)\n\n"
        "Return JSON with keys: nutrients (dict of 15 values), percent_dvs (dict), energy_kcal"
    ),
    4: (
        "## Phase 4: Ingredient List & Compound Ingredients\n\n"
        "Determine the correct ingredient order by finished weight (after moisture loss).\n"
        "For compound ingredients (>2% of finished weight), list sub-ingredients.\n\n"
        "Moisture loss formula:\n"
        "- total_moisture_loss = total_as_added × (moisture_loss_pct / 100)\n"
        "- Distribute proportionally among moisture_contributing ingredients only\n"
        "- finished_weight_i = as_added_i − (as_added_i / mc_total) × total_moisture_loss\n"
        "- Sort descending by finished weight\n\n"
        "Return JSON with keys: ingredient_list (ordered list), "
        "compound_ingredient_sublists (dict of compound → sub-ingredients)"
    ),
    5: (
        "## Phase 5: Global Consistency Audit\n\n"
        "Review the complete label for cross-step consistency:\n"
        "1. Compute PDP area from container dimensions → determine minimum type size\n"
        "2. Validate health claims against computed nutrient values\n"
        "3. Identify any cross-step consistency violations\n\n"
        "PDP area: rectangular = h × w; cylindrical = h × (π × d) / num_panels\n"
        "Type size thresholds: ≤5 sq in → 0.0625\", 5-25 → 0.125\", "
        "25-100 → 0.1875\", >100 → 0.25\"\n\n"
        "Return JSON with keys: declared_type_size_inch, health_claims (list), "
        "consistency_violations (list of identified issues)"
    ),
}


def _build_phase_observation(
    phase: int,
    episode: dict,
    state: FDAState,
    difficulty: str,
    phase_result: dict | None = None,
) -> FDAObservation:
    """Build the observation for a given phase."""
    phase_data: dict[str, Any] = {}

    if phase == 1:
        phase_data = {
            "food_category_description": episode["food_category_description"],
            "physical_form": episode["physical_form"],
            "unit_weight_g": episode.get("unit_weight_g"),
            "total_package_weight_g": episode["total_package_weight_g"],
        }
        # For medium/hard, include ambiguous candidates
        gt = episode["ground_truth"]
        if "ambiguous_racc_candidates" in gt:
            phase_data["ambiguous_racc_candidates"] = gt["ambiguous_racc_candidates"]

    elif phase == 2:
        phase_data = {
            "physical_form": episode["physical_form"],
            "unit_weight_g": episode.get("unit_weight_g"),
            "total_package_weight_g": episode["total_package_weight_g"],
        }

    elif phase == 3:
        phase_data = {
            "lab_nutrients": episode["lab_nutrients"],
            "lab_sample_size_g": episode["lab_sample_size_g"],
            # The agent uses their own serving_size_g from Phase 2
        }

    elif phase == 4:
        phase_data = {
            "recipe": episode["recipe"],
            "moisture_loss_pct": episode["moisture_loss_pct"],
        }

    elif phase == 5:
        phase_data = {
            "container": episode["container"],
        }

    # Build feedback text from previous phase result
    feedback = ""
    if phase_result is not None:
        feedback = _build_phase_feedback(phase - 1, phase_result, difficulty)

    prompt = _PHASE_PROMPTS[phase]
    text = f"{prompt}\n\n"
    if feedback:
        text += f"--- FEEDBACK FROM PHASE {phase - 1} ---\n{feedback}\n\n"
    text += f"--- PHASE {phase} DATA ---\n{json.dumps(phase_data, indent=2)}"

    return FDAObservation(
        text=text,
        phase=phase,
        phase_data=phase_data,
        prior_submissions=dict(state.phase_actions),
        reward=phase_result["score"] if phase_result else None,
        done=False,
    )


def _build_phase_feedback(
    completed_phase: int,
    result: dict,
    difficulty: str,
) -> str:
    """Build feedback text for a completed phase. Detail level depends on difficulty."""
    score = result["score"]
    details = result.get("details", {})

    if difficulty == "hard":
        # Hard: score only, no field details
        return f"Phase {completed_phase} score: {score:.3f}"

    if difficulty == "medium":
        # Medium: which fields are wrong, not expected values
        lines = [f"Phase {completed_phase} score: {score:.3f}"]
        for field_name, detail in details.items():
            if field_name.startswith("_"):
                continue
            if not detail.get("correct", False):
                lines.append(f"  - {field_name}: INCORRECT")
            else:
                lines.append(f"  - {field_name}: OK")
        return "\n".join(lines)

    # Easy: full breakdown
    lines = [f"Phase {completed_phase} score: {score:.3f}"]
    for field_name, detail in details.items():
        if field_name.startswith("_"):
            continue
        if detail.get("correct", False):
            lines.append(f"  - {field_name}: OK")
        else:
            agent_val = detail.get("agent")
            lines.append(f"  - {field_name}: INCORRECT (submitted {agent_val})")
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
            current_phase=1,
            phase_actions={},
            phase_scores={},
            phase_details={},
            completed=False,
        )

        obs = _build_phase_observation(
            phase=1,
            episode=episode,
            state=self._state,
            difficulty=difficulty,
        )
        obs.metadata = {
            "seed": episode["episode_seed"],
            "episode_id": self._state.episode_id,
        }
        return obs

    def step(
        self,
        action: FDAAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FDAObservation:
        if not self._state.ground_truth:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")

        current_phase = self._state.current_phase
        if action.phase != current_phase:
            raise ValueError(
                f"Phase mismatch: action.phase={action.phase} but "
                f"current_phase={current_phase}. Submit for phase {current_phase}."
            )

        self._state.step_count += 1

        # Extract phase-relevant fields from action
        action_dict = action.model_dump(exclude_none=True, exclude={"phase"})

        # Grade this phase
        result = grade_phase(
            phase=current_phase,
            action_dict=action_dict,
            ground_truth=self._state.ground_truth,
            prior_actions=self._state.phase_actions,
        )

        phase_score = result["score"]
        self._state.phase_actions[current_phase] = action_dict
        self._state.phase_scores[current_phase] = phase_score
        self._state.phase_details[current_phase] = result.get("details", {})

        logger.info(
            "STEP phase=%d/%d  task=%s  phase_score=%.3f",
            current_phase, self._state.max_steps,
            self._state.task_id, phase_score,
        )

        if current_phase >= 5:
            # Episode complete
            self._state.completed = True
            episode_score = grade_episode(self._state.phase_scores)

            logger.info(
                "DONE  task=%s  episode_score=%.3f  phase_scores=%s",
                self._state.task_id, episode_score,
                {k: f"{v:.3f}" for k, v in self._state.phase_scores.items()},
            )

            return FDAObservation(
                text=(
                    f"Episode complete. Final score: {episode_score:.3f}\n"
                    f"Phase scores: {json.dumps({k: round(v, 3) for k, v in self._state.phase_scores.items()})}"
                ),
                phase=5,
                done=True,
                reward=episode_score,
            )

        # Advance to next phase
        next_phase = current_phase + 1
        self._state.current_phase = next_phase

        obs = _build_phase_observation(
            phase=next_phase,
            episode=self._episode,
            state=self._state,
            difficulty=self._state.difficulty,
            phase_result=result,
        )
        return obs

    @property
    def state(self) -> FDAState:
        return self._state

    @property
    def grader_score(self) -> float:
        if not self._state.phase_scores:
            return 0.0
        return grade_episode(self._state.phase_scores)
