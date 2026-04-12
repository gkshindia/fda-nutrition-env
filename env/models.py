from __future__ import annotations

from typing import Any, Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class FDAAction(Action):
    """Agent submits phase-specific data for the current step."""
    phase: int = Field(
        ...,
        ge=1,
        le=5,
        description="Which phase this submission is for (1-5). Must match current_phase.",
    )

    # Phase 1 — Food Category Classification & RACC
    food_category: Optional[str] = Field(
        default=None,
        description="FDA food category name from 21 CFR 101.12 Table 2",
    )
    racc_g: Optional[float] = Field(
        default=None,
        description="Reference Amount Customarily Consumed in grams",
    )
    household_measure: Optional[str] = Field(
        default=None,
        description="Household measure declaration (e.g., '1 bar', '2 tbsp')",
    )

    # Phase 2 — Label Format Selection
    label_format: Optional[str] = Field(
        default=None,
        description="Label format: 'single_column', 'dual_column', or 'single_serving_container'",
    )
    serving_size_g: Optional[float] = Field(
        default=None,
        description="Declared serving size in grams",
    )
    serving_declaration_text: Optional[str] = Field(
        default=None,
        description="Serving size declaration text (e.g., '1 bar (40g)')",
    )

    # Phase 3 — Nutrient Math, Rounding, Calorie Consistency
    nutrients: Optional[dict[str, Any]] = Field(
        default=None,
        description="Dict of 15 nutrient values (FDA-rounded declared values)",
    )
    percent_dvs: Optional[dict[str, Any]] = Field(
        default=None,
        description="Dict of %Daily Value for each nutrient (integer or null)",
    )
    energy_kcal: Optional[float] = Field(
        default=None,
        description="Declared calories via Atwater formula",
    )

    # Phase 4 — Ingredient List
    ingredient_list: Optional[list[str]] = Field(
        default=None,
        description="Ingredients in descending order by finished weight",
    )
    compound_ingredient_sublists: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Sub-ingredients for compound ingredients (e.g., {'dark_chocolate': ['cocoa mass', 'sugar', ...]})",
    )

    # Phase 5 — Global Consistency Audit
    declared_type_size_inch: Optional[float] = Field(
        default=None,
        description="Minimum type size declared on the label (must meet PDP area requirement)",
    )
    health_claims: Optional[list[str]] = Field(
        default=None,
        description="Valid front-of-pack health claims (empty list if none substantiated)",
    )
    consistency_violations: Optional[list[str]] = Field(
        default=None,
        description="Cross-step consistency violations identified by the agent",
    )
    corrections: Optional[dict[str, Any]] = Field(
        default=None,
        description="Corrections to prior phase outputs (e.g., revised serving_size_g)",
    )


class FDAObservation(Observation):
    """Observation returned to the agent: phase-specific data + prior submissions."""
    text: str = Field(default="", description="Human-readable phase prompt")
    phase: int = Field(default=0, description="Current phase number (1-5)")
    phase_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured inputs for the current phase",
    )
    prior_submissions: dict[int, dict] = Field(
        default_factory=dict,
        description="Agent's own prior phase submissions (keyed by phase number)",
    )


class FDAState(State):
    """Internal environment state for one episode."""
    task_id: str = Field(default="task_easy")
    difficulty: str = Field(default="easy")
    max_steps: int = Field(default=5)
    current_phase: int = Field(default=0)
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    phase_actions: dict[int, dict] = Field(
        default_factory=dict,
        description="Agent submissions keyed by phase number",
    )
    phase_scores: dict[int, float] = Field(
        default_factory=dict,
        description="Per-phase scores",
    )
    phase_details: dict[int, dict] = Field(
        default_factory=dict,
        description="Per-phase grading details",
    )
    completed: bool = Field(default=False)
