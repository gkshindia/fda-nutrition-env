"""
FDA Nutrition Facts Panel — Baseline Agent
===========================================
Submits corrected Nutrition Facts labels using an OpenAI-compatible LLM.

Required env vars (hackathon):
    API_BASE_URL  — OpenAI-compatible API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME    — model identifier (default: gpt-5.4-mini)
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
import logging
import os
import re
import sys

import dotenv
import httpx
from openai import OpenAI

from core.grader import grade
from env.models import FDAAction
from env.server.environment import FDAEnvironment

logger = logging.getLogger("fda.baseline")

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
API_BASE_URL = _env("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = _env("MODEL_NAME", "gpt-5.4-mini")
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
# RACC KEYWORD LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _guess_racc_grams(food_description: str, physical_form: str) -> tuple[float, str]:
    """
    Keyword-match food_category_description to the most likely RACC in grams.
    Returns (racc_g, matched_category_name).
    Rules ordered by specificity; first match wins.
    """
    desc = food_description.lower()

    # Oils and fats (14g) — checked before "butter" to avoid nut butter false match
    if any(kw in desc for kw in ("cooking oil", "oil blend", "shortening")):
        return 14.0, "Butter, margarine, oil, shortening"

    # Grain-based bars (40g) — " bar" is a strong, reliable signal
    if " bar" in desc or desc.startswith("bar ") or "energy bar" in desc or "protein bar" in desc:
        return 40.0, "Grain-based bars with or without filling or coating"

    # RTE cereal ≥43g/cup (60g) — granola or clusters not already tagged as bar
    if any(kw in desc for kw in ("granola", "cluster", "whole grain cluster")):
        return 60.0, "Breakfast cereals, ready-to-eat, weighing 43 g or more per cup"

    # Hot cereal (40g)
    if any(kw in desc for kw in ("hot cereal", "oatmeal", "porridge")):
        return 40.0, "Breakfast cereals (hot cereal type), hominy grits"

    # Nut/seed butters (32g) — compound phrases first, then single-word "spread"/"paste"
    if any(kw in desc for kw in ("peanut butter", "almond butter", "nut butter",
                                  "walnut spread", "chocolate spread")):
        return 32.0, "Nut and seed butters, pastes, or creams"
    if ("spread" in desc or "paste" in desc
            or ("butter" in desc and "shortbread" not in desc and "cooking" not in desc)):
        if any(kw in desc for kw in ("nut", "almond", "peanut", "walnut", "chocolate")):
            return 32.0, "Nut and seed butters, pastes, or creams"

    # Cookies (30g)
    if any(kw in desc for kw in ("cookie", "shortbread", "biscuit", "bites", "energy bites")):
        return 30.0, "Cookies"

    # Trail mix / snack mix / nuts-and-seeds (30g)
    if any(kw in desc for kw in ("trail mix", "snack mix", "snack blend")):
        return 30.0, "Nuts, seeds and mixtures, all types"
    if ("nut" in desc or "seed" in desc) and "bar" not in desc and "butter" not in desc:
        return 30.0, "Nuts, seeds and mixtures, all types"

    # Default: 30g (snacks, candies, cookies)
    return 30.0, "Snacks / Cookies / Candies (default)"


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
    api_base_url_from_env = bool(os.getenv("API_BASE_URL", "").strip())

    # If API_BASE_URL was explicitly provided, prefer HF_TOKEN auth for that endpoint.
    # Otherwise use the OpenAI default endpoint with OPENAI_API_KEY auth.
    if api_base_url_from_env:
        api_key = HF_TOKEN or OPENAI_API_KEY or "not-needed"
        logger.info("LLM client → %s  model=%s", API_BASE_URL, MODEL_NAME)
    else:
        api_key = OPENAI_API_KEY or HF_TOKEN or "not-needed"
        logger.info("LLM client → %s  model=%s", API_BASE_URL, MODEL_NAME)

    return OpenAI(api_key=api_key, base_url=API_BASE_URL, timeout=90.0)


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

def _precompute_scaled_nutrients(episode_context: dict, serving_g: float) -> dict:
    """
    Pre-compute scaled (unrounded) nutrient values per serving.

    This removes the scaling math burden from the LLM so it only
    needs to apply FDA rounding rules.
    """
    lab = episode_context.get("lab_nutrients", {})
    lab_sample = episode_context.get("lab_sample_size_g", 1.0)
    scale = serving_g / lab_sample
    return {k: v * scale for k, v in lab.items()}


def _precompute_ingredient_order(episode_context: dict) -> list[str]:
    """
    Pre-compute ingredient order by finished weight (after moisture loss).
    """
    recipe = episode_context.get("recipe", [])
    moisture_loss_pct = episode_context.get("moisture_loss_pct", 0.0)
    total_as_added = sum(ing["weight_as_added_g"] for ing in recipe)
    total_moisture_loss = total_as_added * (moisture_loss_pct / 100.0)
    moisture_contributing_total = sum(
        ing["weight_as_added_g"] for ing in recipe if ing.get("moisture_contributing", False)
    )

    finished = []
    for ing in recipe:
        w = ing["weight_as_added_g"]
        if ing.get("moisture_contributing", False) and moisture_contributing_total > 0:
            w = w - (w / moisture_contributing_total) * total_moisture_loss
        finished.append((ing["ingredient_slug"], w))

    finished.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in finished]


def _compute_pdp_and_type_size(episode_context: dict) -> tuple[float, float]:
    """Compute PDP area and minimum type size from container info."""
    import math
    container = episode_context.get("container", {})
    shape = container.get("shape", "rectangular")
    height = container.get("height_in", 0)

    if shape == "cylindrical":
        diameter = container.get("diameter_in", 0)
        num_panels = container.get("num_display_panels", 1)
        pdp_area = height * (math.pi * diameter) / num_panels
    else:
        width = container.get("width_in", 0)
        pdp_area = height * width

    if pdp_area <= 5:
        min_type = 0.0625
    elif pdp_area <= 25:
        min_type = 0.125
    elif pdp_area <= 100:
        min_type = 0.1875
    else:
        min_type = 0.25

    return pdp_area, min_type


def _build_user_prompt(observation_data: dict) -> str:
    """
    Build the initial user prompt. Raw episode data only — no pre-computed answers.
    The agent must reason through scaling, ingredient ordering, and PDP area itself.
    """
    episode_context = observation_data.get("episode_context", {})
    draft_label = observation_data.get("draft_label", {})

    prompt = (
        "You are correcting a draft FDA Nutrition Facts label. The draft contains errors.\n\n"
        f"## Episode Context\n```json\n{json.dumps(episode_context, indent=2)}\n```\n\n"
        f"## Draft Label (contains errors)\n```json\n{json.dumps(draft_label, indent=2)}\n```\n\n"
        "## YOUR TASK\n\n"
        "1. **Serving size**: Determine the correct serving_size_g from 21 CFR 101.12 RACC for this food category.\n"
        "   - For bulk products: serving = RACC. For discrete units: if one unit is 50-200% of RACC, one unit IS the serving.\n"
        "   - For single_serving_container: serving = total_package_weight_g.\n\n"
        "2. **Nutrients**: Scale each lab nutrient to your serving size:\n"
        "     nutrient_per_serving = (lab_nutrient / lab_sample_size_g) * serving_size_g\n"
        "   Then apply FDA round-half-UP (floor(value/increment + 0.5) * increment):\n"
        "   - Calories: <5→0, 5-50→nearest 5, >50→nearest 10\n"
        "   - Fat/sat fat/trans fat: <0.5→0, 0.5-5→nearest 0.5, >5→nearest 1\n"
        "   - Cholesterol: <2→0, ≥5→nearest 5mg\n"
        "   - Sodium: <5→0, 5-140→nearest 5, >140→nearest 10\n"
        "   - Carb/fiber/sugars/added sugars/protein: <0.5→0, ≥1→nearest 1\n"
        "   - Vitamin D: nearest 0.1mcg. Iron: nearest 0.1mg\n"
        "   - Calcium: <5→0, 5-140→nearest 10, >140→nearest 20\n"
        "   - Potassium: <5→0, 5-140→nearest 5, >140→nearest 10\n\n"
        "3. **Atwater calories**: energy_kcal = round_calories(9 × UNROUNDED_fat + 4 × UNROUNDED_carb + 4 × UNROUNDED_protein)\n"
        "   Use UNROUNDED scaled values, not the rounded declared values.\n\n"
        "4. **%DV**: %DV = floor(rounded_declared_value / DRV × 100 + 0.5)\n"
        "   DRVs: fat=78g, sat_fat=20g, cholesterol=300mg, sodium=2300mg, carb=275g,\n"
        "   fiber=28g, added_sugars=50g, protein=50g, vit_d=20mcg, calcium=1300mg,\n"
        "   iron=18mg, potassium=4700mg. trans_fat and total_sugars → null.\n\n"
        "5. **Ingredient list**: Sort ingredients by FINISHED weight (after moisture loss).\n"
        "   - total_moisture_loss = total_as_added_weight × (moisture_loss_pct / 100)\n"
        "   - Distribute moisture loss proportionally among moisture_contributing ingredients only.\n"
        "   - finished_weight_i = as_added_i − (as_added_i / mc_total) × total_moisture_loss\n"
        "   - Non-moisture-contributing ingredients keep their as-added weight.\n"
        "   - Sort descending by finished weight.\n\n"
        "6. **Type size**: Compute PDP area from container dimensions:\n"
        "   - Rectangular: area = height_in × width_in\n"
        "   - Cylindrical: area = height_in × (π × diameter_in) / num_display_panels\n"
        "   - Min type size: ≤5 sq in → 0.0625\", 5-25 → 0.125\", 25-100 → 0.1875\", >100 → 0.25\"\n"
        "   - declared_type_size_inch must be ≥ minimum.\n\n"
        "7. **Health claims**: Set to [] unless clearly substantiated by nutrient values.\n\n"
        "Return ONLY a JSON object with keys: serving_size_g, label_format, declared_type_size_inch,\n"
        "nutrients, percent_dvs, ingredient_list, health_claims.\n"
        "No explanation, no markdown wrapping.\n\n"
        "You will receive feedback after each attempt. Use all available steps to improve your score."
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
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.2f} "
        f"rewards={rewards_str}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _build_revision_prompt(previous_label: dict, feedback_text: str) -> str:
    """
    Build a follow-up prompt that includes the agent's previous submission
    and the environment's feedback on what was wrong.

    Key strategy: tell the model to NOT change fields that are already correct,
    and focus only on the incorrect ones.
    """
    return (
        "Your previous submission had errors. Here is the feedback:\n\n"
        f"## Feedback\n```\n{feedback_text}\n```\n\n"
        f"## Your Previous Label\n```json\n{json.dumps(previous_label, indent=2)}\n```\n\n"
        "## CRITICAL INSTRUCTIONS\n\n"
        "**DO NOT change fields that are already marked OK.** Only fix the incorrect fields.\n\n"
        "Common mistakes to check:\n"
        "- **serving_size_g wrong?** You may have the wrong RACC category. "
        "Key RACC values: RTE granola/cereal clusters = 60g, grain bars = 40g, hot cereal = 40g, "
        "nut butters/spreads = 32g, cookies/nuts/snacks/candy = 30g, oils = 14g. "
        "If serving_size changes, ALL nutrients and %DVs must be recalculated.\n"
        "- **nutrients wrong?** Re-check: scaled_value = (lab_value / lab_sample_size_g) * serving_size_g. "
        "Then apply the correct FDA rounding tier. Remember rounding is round-half-UP.\n"
        "- **percent_dvs wrong?** Recompute: %DV = floor(rounded_nutrient / DRV * 100 + 0.5). "
        "trans_fat and total_sugars must be null.\n"
        "- **ingredient_list wrong?** Use the pre-computed order from the first message. "
        "Do NOT re-sort yourself.\n"
        "- **atwater_consistency wrong?** energy_kcal = round_calories(9*UNROUNDED_fat + 4*UNROUNDED_carb + 4*UNROUNDED_protein). "
        "Use the UNROUNDED scaled values from the pre-computed table.\n"
        "- **health_claims wrong?** Set to []. Always.\n\n"
        "Return ONLY the corrected JSON. Keep all correct fields unchanged."
    )


def run_baseline_agent() -> dict[str, float]:
    """
    Run the baseline agent against all tasks with multi-step feedback.

    For each task:
    1. Reset the environment with a fixed seed
    2. Send the draft label + FDA rules to the LLM
    3. Parse the LLM's corrected label and submit via /step
    4. If not done, feed back the grader feedback and let the LLM revise
    5. Repeat until done, then submit all actions to /grader

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
        max_steps = task.get("max_steps", 1)
        seed = TASK_SEEDS.get(task_id, 42)

        _log_start(task_id, MODEL_NAME)

        # Use local environment for multi-step loop (HTTP endpoints are stateless)
        local_env = FDAEnvironment()
        obs = local_env.reset(task_id=task_id, seed=seed)

        observation_data = {
            "text": obs.text,
            "draft_label": obs.draft_label,
            "episode_context": obs.episode_context,
        }

        # Build initial prompt
        user_prompt = _build_user_prompt(observation_data)
        messages = [
            {"role": "system", "content": FDA_REGULATORY_RULES},
            {"role": "user", "content": user_prompt},
        ]

        all_actions: list[dict] = []
        all_rewards: list[float] = []
        done = False
        step_number = 0

        while not done:
            step_number += 1

            # Call the LLM
            try:
                completion = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_completion_tokens=4096,
                    temperature=0.0,
                )
                llm_response_text = completion.choices[0].message.content or ""
            except Exception as exception:
                llm_response_text = ""
                logger.warning("LLM call failed at step %d: %s", step_number, exception)

            # Parse the corrected label
            corrected_label = _extract_json_from_response(llm_response_text)
            if corrected_label is None:
                logger.warning("Could not parse JSON from LLM response — task=%s step=%d", task_id, step_number)
                corrected_label = {}

            all_actions.append({"label": corrected_label, "final_submission": done})

            # Submit to local environment
            step_result = local_env.step(FDAAction(label=corrected_label))
            step_reward = step_result.reward if step_result.reward is not None else 0.0
            done = step_result.done

            all_rewards.append(step_reward)

            action_json_string = json.dumps(corrected_label, separators=(",", ":"))
            _log_step(
                step_number=step_number,
                action_json=action_json_string,
                reward=step_reward,
                done=done,
            )

            if not done:
                # Build revision prompt from feedback and add to conversation
                feedback_text = step_result.text
                revision_prompt = _build_revision_prompt(corrected_label, feedback_text)
                messages.append({"role": "assistant", "content": llm_response_text})
                messages.append({"role": "user", "content": revision_prompt})
                logger.info("task=%s step=%d score=%.3f → revising", task_id, step_number, step_reward)

        # Final grader call for official score
        grader_response = http_client.post(
            "/grader",
            json={
                "task_id": task_id,
                "seed": seed,
                "actions": all_actions,
            },
        )
        grader_response.raise_for_status()
        grader_result = grader_response.json()
        grader_score = grader_result["grader_score"]

        _log_end(
            success=grader_score > 0.0,
            steps=step_number,
            score=grader_score,
            rewards=all_rewards,
        )

        task_scores[task_id] = grader_score
        logger.info("task=%s DONE  final_score=%.3f  steps=%d", task_id, grader_score, step_number)

    http_client.close()
    return task_scores


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TASK RUNNER (for UI /baseline/run endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_task(task_id: str, seed: int | None = None) -> dict:
    """
    Run the baseline agent for a single task and return full results for the UI.

    Returns:
        {
            "task_id": str,
            "seed": int,
            "grader_score": float,
            "steps_taken": int,
            "rewards": list[float],
            "draft_label": dict,
            "corrected_label": dict,   # label from the best-scoring step
            "episode_context": dict,
        }
    """
    logger.info("run_baseline_task  task=%s seed=%s  model=%s", task_id, seed, MODEL_NAME)
    openai_client = _create_openai_client()
    local_env = FDAEnvironment()
    obs = local_env.reset(task_id=task_id, seed=seed)
    _log_start(task_id, MODEL_NAME)

    draft_label = obs.draft_label
    episode_context = obs.episode_context
    actual_seed = obs.metadata.get("seed", seed)

    observation_data = {
        "text": obs.text,
        "draft_label": draft_label,
        "episode_context": episode_context,
    }

    messages = [
        {"role": "system", "content": FDA_REGULATORY_RULES},
        {"role": "user", "content": _build_user_prompt(observation_data)},
    ]

    all_rewards: list[float] = []
    best_score = 0.0
    best_label: dict = {}
    done = False
    step_number = 0

    while not done:
        step_number += 1
        logger.info("  LLM call step=%d — sending request to %s", step_number, API_BASE_URL or "openai")
        llm_error: str | None = None

        try:
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_completion_tokens=4096,
                temperature=0.0,
            )
            llm_response_text = completion.choices[0].message.content or ""
            logger.info("  LLM call step=%d — response received (%d chars)", step_number, len(llm_response_text))
        except Exception as exception:
            llm_response_text = ""
            llm_error = str(exception)
            logger.warning("LLM call failed at step %d: %s", step_number, exception)

        corrected_label = _extract_json_from_response(llm_response_text) or {}
        step_result = local_env.step(FDAAction(label=corrected_label))
        step_reward = step_result.reward if step_result.reward is not None else 0.0
        done = step_result.done
        all_rewards.append(step_reward)
        action_json_string = json.dumps(corrected_label, separators=(",", ":"))
        _log_step(
            step_number=step_number,
            action_json=action_json_string,
            reward=step_reward,
            done=done,
            error=llm_error,
        )

        improved = step_reward > best_score
        if improved:
            best_score = step_reward
            best_label = corrected_label

        logger.info(
            "  step=%d  score=%.3f  best=%.3f  %s",
            step_number, step_reward, best_score,
            "↑ improved" if improved else "↔ no change",
        )

        if not done:
            revision_prompt = _build_revision_prompt(corrected_label, step_result.text)
            messages.append({"role": "assistant", "content": llm_response_text})
            messages.append({"role": "user", "content": revision_prompt})

    # Get detailed group scores from final grading
    ground_truth = local_env.state.ground_truth
    grader_result = grade(best_label, ground_truth)
    _log_end(
        success=best_score > 0.0,
        steps=step_number,
        score=best_score,
        rewards=all_rewards,
    )

    logger.info("run_baseline_task  task=%s DONE  score=%.3f  steps=%d", task_id, best_score, step_number)
    return {
        "task_id": task_id,
        "seed": actual_seed,
        "grader_score": best_score,
        "steps_taken": step_number,
        "rewards": all_rewards,
        "draft_label": draft_label,
        "corrected_label": best_label,
        "episode_context": episode_context,
        "group_scores": grader_result["group_scores"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scores = run_baseline_agent()
    print(json.dumps(scores, indent=2), file=sys.stderr)
