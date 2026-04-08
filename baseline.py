"""
FDA Nutrition Facts Panel — Baseline Agent
===========================================
Submits corrected Nutrition Facts labels using an OpenAI-compatible LLM.

Required env vars (hackathon):
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — model identifier (default: gpt-4o-mini)
    HF_TOKEN      — auth token for HuggingFace inference endpoints

Optional env vars:
    ENV_BASE_URL  — environment server URL (default: http://localhost:7860)
    OPENAI_API_KEY — for direct OpenAI usage

Inference log format (hackathon requirement):
    [START] task=<name> env=fda-nutrition-env model=<name>
    [STEP]  step=<n> action=<str> reward=<float> done=<bool> error=<null|str>
    [END]   success=<bool> steps=<n> score=<float> rewards=<float,...>
"""
from __future__ import annotations

import json
import os
import re
import sys

import dotenv
import httpx
from openai import OpenAI

# Load .env file from project root (does not override existing shell env vars)
dotenv.load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _env(name: str, default: str | None = None) -> str | None:
    """Read env var, treating empty strings as unset."""
    value = os.getenv(name, "")
    return value.strip() if value.strip() else default

ENVIRONMENT_BASE_URL = _env("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = _env("API_BASE_URL")
MODEL_NAME = _env("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = _env("HF_TOKEN")
OPENAI_API_KEY = _env("OPENAI_API_KEY")

# Fixed seeds per task for reproducibility
TASK_SEEDS = {
    "task_easy": 100,
    "task_medium": 200,
    "task_hard": 300,
}

ENVIRONMENT_NAME = "fda-nutrition-env"


# ─────────────────────────────────────────────────────────────────────────────
# FDA REGULATORY CONTEXT (included in the LLM prompt)
# ─────────────────────────────────────────────────────────────────────────────

FDA_REGULATORY_RULES = """\
## FDA Nutrition Facts Label — Regulatory Rules

You are correcting a draft Nutrition Facts label. Apply these rules exactly.

### 1. Serving Size
- Serving size comes from 21 CFR 101.12 RACC (Reference Amounts Customarily Consumed).
- For bulk products: serving_size_g = RACC.
- For discrete units: if one unit is 50%-200% of RACC, one unit IS the serving.
- Check that the label's serving_size_g matches the RACC for the product category.

### 2. Nutrient Scaling
- Scale each lab nutrient from lab_sample_size_g to serving_size_g:
    nutrient_per_serving = (lab_nutrient / lab_sample_size_g) * serving_size_g

### 3. Regulatory Rounding (21 CFR 101.9(c))
Use round-half-UP (NOT Python's round which uses banker's rounding).
Round-half-up: floor(value / increment + 0.5) * increment

Rounding tiers by nutrient:
- Calories: <5→0, 5-50→nearest 5, >50→nearest 10
- Total fat: <0.5→0, 0.5-5→nearest 0.5g, >5→nearest 1g
- Saturated fat: <0.5→0, 0.5-5→nearest 0.5g, >5→nearest 1g
- Trans fat: <0.5→0, 0.5-5→nearest 0.5g, >5→nearest 1g
- Cholesterol: <2→0, 2-5→"less than 5mg", ≥5→nearest 5mg
- Sodium: <5→0, 5-140→nearest 5mg, >140→nearest 10mg
- Total carbohydrate: <0.5→0, 0.5-1→"less than 1g", ≥1→nearest 1g
- Dietary fiber: <0.5→0, 0.5-1→"less than 1g", ≥1→nearest 1g
- Total sugars: <0.5→0, 0.5-1→"less than 1g", ≥1→nearest 1g
- Added sugars: <0.5→0, 0.5-1→"less than 1g", ≥1→nearest 1g
- Protein: <0.5→0, 0.5-1→"less than 1g", ≥1→nearest 1g
- Vitamin D: <0.1→0, ≥0.1→nearest 0.1mcg (but declare in mcg, round to 0.1)
- Calcium: <5→0, 5-140→nearest 10mg, >140→nearest 20mg
- Iron: <0.1→0, ≥0.1→nearest 0.1mg (but >0.5 round to nearest 0.1)
- Potassium: <5→0, 5-140→nearest 5mg, >140→nearest 10mg

### 4. Atwater Calories (21 CFR 101.9(c)(1)(i))
- Declared calories = round_calories(9 × fat + 4 × carb + 4 × protein)
- Use the UNROUNDED (pre-rounding) scaled macros for this calculation, NOT the rounded values.
- The declared energy_kcal must match the Atwater result, not the USDA energy field.

### 5. %Daily Value (21 CFR 101.9(c)(8))
- %DV = round_half_up(rounded_declared_value / DRV × 100) to nearest integer
- Daily Reference Values (2020 update):
    total_fat: 78g, saturated_fat: 20g, cholesterol: 300mg, sodium: 2300mg,
    total_carbohydrate: 275g, dietary_fiber: 28g, added_sugars: 50g, protein: 50g,
    vitamin_d: 20mcg, calcium: 1300mg, iron: 18mg, potassium: 4700mg
- trans_fat and total_sugars have NO established DV → %DV is null for these.

### 6. Ingredient Order (21 CFR 101.4(a)(1))
- List ingredients in DESCENDING order by FINISHED weight (after moisture loss).
- Moisture loss only affects moisture-contributing ingredients.
- finished_weight = as_added_weight - (as_added_weight / mc_total) × total_moisture_loss
  where mc_total = sum of moisture-contributing ingredient weights,
  total_moisture_loss = total_as_added × (moisture_loss_pct / 100)
- Non-moisture-contributing ingredients keep their as-added weight.

### 7. PDP Area & Type Size (21 CFR 101.9(d))
- Rectangular: PDP area = height × width
- Cylindrical: PDP area = height × (π × diameter) / num_display_panels
- Minimum type size depends on PDP area:
    ≤5 sq in: 1/16 inch (0.0625)
    5-25 sq in: 1/8 inch (0.125)
    25-100 sq in: 3/16 inch (0.1875)
    >100 sq in: 1/4 inch (0.25)
- declared_type_size_inch must be ≥ the minimum for the PDP area.

### 8. Health Claims
- Only include health claims that are substantiated by the product's nutrient values.
- If a claim references a nutrient that doesn't meet thresholds, remove it.
- When in doubt, remove the claim (set health_claims to empty list []).
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _create_openai_client() -> OpenAI:
    """Create an OpenAI client respecting hackathon env vars."""
    # When API_BASE_URL is set, we're hitting an external endpoint (e.g. HF inference)
    # and should use HF_TOKEN as the API key.
    # When API_BASE_URL is NOT set, we're hitting OpenAI directly and need OPENAI_API_KEY.
    if API_BASE_URL:
        api_key = HF_TOKEN or OPENAI_API_KEY or "not-needed"
        return OpenAI(api_key=api_key, base_url=API_BASE_URL)
    else:
        api_key = OPENAI_API_KEY or HF_TOKEN or "not-needed"
        return OpenAI(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_from_response(response_text: str) -> dict | None:
    """
    Try to extract a JSON dict from the LLM's response.

    Attempts in order:
    1. Direct json.loads on the full response
    2. Extract from ```json ... ``` markdown code block
    3. Extract from ``` ... ``` code block
    4. Find the first { ... } substring and parse it
    """
    # Attempt 1: direct parse
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: markdown json code block
    json_block_match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
    if json_block_match:
        try:
            parsed = json.loads(json_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # Attempt 3: generic code block
    code_block_match = re.search(r"```\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # Attempt 4: find outermost { ... }
    brace_start = response_text.find("{")
    if brace_start != -1:
        # Find matching closing brace by counting depth
        depth = 0
        for i in range(brace_start, len(response_text)):
            if response_text[i] == "{":
                depth += 1
            elif response_text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(response_text[brace_start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

    return None


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_prompt(observation_data: dict) -> str:
    """
    Build the user prompt from the observation returned by /reset.

    The observation contains:
    - text: full task prompt (includes episode context + draft label)
    - draft_label: the label dict with injected errors
    - episode_context: recipe, lab nutrients, container, etc.
    """
    episode_context = observation_data.get("episode_context", {})
    draft_label = observation_data.get("draft_label", {})

    prompt = (
        "Here is the episode data and draft label. "
        "Identify all errors in the draft label and return the CORRECTED label as a single JSON object.\n\n"
        f"## Episode Context\n```json\n{json.dumps(episode_context, indent=2)}\n```\n\n"
        f"## Draft Label (contains errors)\n```json\n{json.dumps(draft_label, indent=2)}\n```\n\n"
        "## Instructions\n"
        "1. Compute the correct serving size from the RACC table for this food category.\n"
        "2. Scale each lab nutrient: nutrient_per_serving = (lab_value / lab_sample_size_g) * serving_size_g\n"
        "3. Apply FDA regulatory rounding to each scaled nutrient (use round-half-up, NOT Python round).\n"
        "4. Compute Atwater calories from the UNROUNDED scaled fat, carb, protein: "
        "energy_kcal = round_calories(9*fat + 4*carb + 4*protein)\n"
        "5. Compute %DV for each nutrient using the rounded declared values.\n"
        "6. Order ingredients by FINISHED weight (after moisture loss), descending.\n"
        "7. Check declared_type_size_inch meets the PDP area minimum.\n"
        "8. Remove any unsupported health claims.\n\n"
        "Return ONLY the corrected JSON label with these exact keys:\n"
        "serving_size_g, label_format, declared_type_size_inch, nutrients, percent_dvs, "
        "ingredient_list, health_claims\n\n"
        "Return ONLY valid JSON. No explanation, no markdown outside the JSON."
    )
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _log_start(task_id: str, model_name: str) -> None:
    print(f"[START] task={task_id} env={ENVIRONMENT_NAME} model={model_name}")


def _log_step(step_number: int, action_json: str, reward: float, done: bool, error: str | None = None) -> None:
    error_str = "null" if error is None else error
    print(
        f"[STEP]  step={step_number} "
        f"action={action_json} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error_str}"
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_str}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_agent() -> dict[str, float]:
    """
    Run the baseline agent against all tasks.

    For each task:
    1. Reset the environment with a fixed seed
    2. Send the draft label + FDA rules to the LLM
    3. Parse the LLM's corrected label
    4. Submit to grader for scoring

    Returns:
        {"task_easy": 0.73, "task_medium": 0.65, "task_hard": 0.55}
    """
    http_client = httpx.Client(base_url=ENVIRONMENT_BASE_URL, timeout=60)
    openai_client = _create_openai_client()

    # Fetch task list
    tasks_response = http_client.get("/tasks")
    tasks_response.raise_for_status()
    task_list = tasks_response.json()["tasks"]

    task_scores: dict[str, float] = {}

    for task in task_list:
        task_id = task["task_id"]
        seed = TASK_SEEDS.get(task_id, 42)

        _log_start(task_id, MODEL_NAME)

        # Step 1: Reset environment to get the observation
        reset_response = http_client.post(
            "/reset",
            json={"task_id": task_id, "seed": seed},
        )
        reset_response.raise_for_status()
        # OpenEnv wraps the observation: {"observation": {...}, "reward": ..., "done": ...}
        observation_data = reset_response.json().get("observation", reset_response.json())

        # Step 2: Build prompt and call the LLM
        user_prompt = _build_user_prompt(observation_data)

        try:
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": FDA_REGULATORY_RULES},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            llm_response_text = completion.choices[0].message.content or ""
        except Exception as exception:
            llm_response_text = ""
            print(f"  [WARNING] LLM call failed: {exception}", file=sys.stderr)

        # Step 3: Parse the corrected label from LLM response
        corrected_label = _extract_json_from_response(llm_response_text)
        if corrected_label is None:
            print(f"  [WARNING] Could not parse JSON from LLM response for {task_id}", file=sys.stderr)
            corrected_label = {}

        # Step 4: Submit to grader with the same seed for reproducibility
        action_json_string = json.dumps(corrected_label, separators=(",", ":"))

        grader_response = http_client.post(
            "/grader",
            json={
                "task_id": task_id,
                "seed": seed,
                "actions": [corrected_label],
            },
        )
        grader_response.raise_for_status()
        grader_result = grader_response.json()
        grader_score = grader_result["grader_score"]

        # Log in hackathon format
        _log_step(
            step_number=1,
            action_json=action_json_string,
            reward=grader_score,
            done=True,
        )
        _log_end(
            success=grader_score > 0.0,
            steps=1,
            score=grader_score,
            rewards=[grader_score],
        )

        task_scores[task_id] = grader_score
        print(f"  → {task_id}: score={grader_score:.3f}", file=sys.stderr)

    http_client.close()
    return task_scores


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scores = run_baseline_agent()
    print(json.dumps(scores, indent=2), file=sys.stderr)
