from __future__ import annotations

from typing import Any, Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class FDAAction(Action):
    """Agent submits a corrected label as JSON."""
    label: dict[str, Any] = Field(
        default_factory=dict,
        description="Corrected Nutrition Facts label dict with keys: "
        "serving_size_g, label_format, declared_type_size_inch, "
        "nutrients, percent_dvs, ingredient_list, health_claims",
    )


class FDAObservation(Observation):
    """Observation returned to the agent: episode context + draft label."""
    text: str = Field(default="", description="Human-readable task prompt")
    draft_label: dict[str, Any] = Field(
        default_factory=dict,
        description="Draft label with injected errors for the agent to correct",
    )
    episode_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Episode data visible to the agent (food description, "
        "lab nutrients, recipe, container, etc.)",
    )


class FDAState(State):
    """Internal environment state for one episode."""
    task_id: str = Field(default="task_easy")
    difficulty: str = Field(default="easy")
    max_steps: int = Field(default=1)
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    agent_label: Optional[dict[str, Any]] = Field(default=None)
    completed: bool = Field(default=False)
