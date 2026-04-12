"""
FDA Nutrition Facts Panel — Baseline Agent (5-Phase Sequential)
===============================================================
Submits one action per phase using an OpenAI-compatible LLM.
All interaction is via HTTP — no local FDAEnvironment.

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
# LLM CLIENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _create_openai_client() -> OpenAI:
    """Create an OpenAI client respecting hackathon env vars."""
    api_base_url_from_env = bool(os.getenv("API_BASE_URL", "").strip())

    if api_base_url_from_env:
        api_key = HF_TOKEN or OPENAI_API_KEY or "not-needed"
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
# PHASE-SPECIFIC SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PHASE_1_SYSTEM = """\
You are an FDA regulatory expert classifying food products.

Your task: Given a food product description, classify it into the correct FDA category
from 21 CFR 101.12 Table 2 and determine the RACC (Reference Amount Customarily Consumed).

Key RACC categories and values:
- Breakfast cereals (hot cereal type), hominy grits: 40g
- Breakfast cereals, ready-to-eat, weighing 43g or more per cup (granola, clusters): 60g
- Grain-based bars with or without filling or coating: 40g
- Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole: 30g
- Nut and seed butters, pastes, or creams: 32g
- Cookies: 30g
- Butter, margarine, oil, shortening: 14g
- Candies: 30g (hard candies 15g)

For household measure:
- Bulk products: use standard measures like "1/4 cup", "2 tbsp"
- Discrete units: use unit description like "1 bar", "3 cookies"

Return ONLY a JSON object with keys: food_category, racc_g, household_measure
No explanation, no markdown wrapping."""

PHASE_2_SYSTEM = """\
You are an FDA regulatory expert determining label format and serving size.

Decision tree for label format:
1. Discrete units (bars, cookies, etc.):
   - If one unit weight is 50-200% of RACC → one unit is the serving (single_column)
   - If one unit weight is <50% or >200% of RACC → use bulk rules below
2. Bulk products:
   - Package weight ≤ 200% RACC → single_serving_container (serving = package weight)
   - Package weight 200-300% RACC → dual_column (serving = RACC)
   - Package weight > 300% RACC → single_column (serving = RACC)

Serving declaration format: "{household_measure} ({serving_size_g}g)"
For discrete units: "1 bar (40g)", for bulk: "1/4 cup (30g)"

Return ONLY a JSON object with keys: label_format, serving_size_g, serving_declaration_text
No explanation, no markdown wrapping."""

PHASE_3_SYSTEM = """\
You are an FDA regulatory expert computing nutrient values for a Nutrition Facts label.

Steps:
1. Scale each lab nutrient to per-serving:
   nutrient_per_serving = (lab_nutrient / lab_sample_size_g) × serving_size_g

2. Apply FDA rounding (round-half-UP = floor(value/increment + 0.5) * increment):
   - Calories: <5→0, 5-50→nearest 5, >50→nearest 10
   - Total fat/saturated fat/trans fat: <0.5→0, 0.5-5→nearest 0.5g, >5→nearest 1g
   - Cholesterol: <2→0, 2-5→"less than 5mg" (use 0), ≥5→nearest 5mg
   - Sodium: <5→0, 5-140→nearest 5mg, >140→nearest 10mg
   - Total carb/dietary fiber/total sugars/added sugars/protein: <0.5→0, 0.5-1→<1g (use 0), ≥1→nearest 1g
   - Vitamin D: nearest 0.1mcg (use 0 if <0.1)
   - Calcium: <5→0, 5-140→nearest 10mg, >140→nearest 20mg
   - Iron: nearest 0.1mg (use 0 if <0.1)
   - Potassium: <5→0, 5-140→nearest 5mg, >140→nearest 10mg

3. Atwater calories (use UNROUNDED scaled values, NOT rounded):
   energy_kcal = round_calories(9 × fat_g + 4 × carb_g + 4 × protein_g)

4. %DV = floor(rounded_value / DRV × 100 + 0.5) as integer
   DRVs: total_fat=78g, saturated_fat=20g, cholesterol=300mg, sodium=2300mg,
   total_carbohydrate=275g, dietary_fiber=28g, added_sugars=50g, protein=50g,
   vitamin_d=20mcg, calcium=1300mg, iron=18mg, potassium=4700mg
   trans_fat and total_sugars → null (no established DV)

Return ONLY a JSON object with keys:
- nutrients: dict with keys energy_kcal, protein_g, total_fat_g, saturated_fat_g, trans_fat_g,
  cholesterol_mg, total_carbohydrate_g, dietary_fiber_g, total_sugars_g, added_sugars_g,
  sodium_mg, potassium_mg, calcium_mg, iron_mg, vitamin_d_mcg
- percent_dvs: dict with same keys (null for trans_fat_g and total_sugars_g)
- energy_kcal: the Atwater-computed calorie value
No explanation, no markdown wrapping."""

PHASE_4_SYSTEM = """\
You are an FDA regulatory expert determining ingredient order for a food label.

Rules (21 CFR 101.4(a)(1)):
- List ingredients in DESCENDING order by FINISHED weight (after moisture loss)
- Moisture loss calculation:
  1. total_moisture_loss = total_as_added_weight × (moisture_loss_pct / 100)
  2. Distribute loss proportionally among moisture_contributing ingredients ONLY
  3. finished_weight_i = as_added_i - (as_added_i / mc_total) × total_moisture_loss
     where mc_total = sum of all moisture_contributing ingredient weights
  4. Non-moisture-contributing ingredients keep their as_added weight
  5. Sort descending by finished weight

- Compound ingredients (>2% of finished weight): list sub-ingredients in parentheses
- Ingredients ≤2% of finished weight: may be listed in any order with "Contains 2% or less of:"

Return ONLY a JSON object with keys:
- ingredient_list: ordered list of ingredient names (descending by finished weight)
- compound_ingredient_sublists: dict mapping compound ingredient names to their sub-ingredient lists
No explanation, no markdown wrapping."""

PHASE_5_SYSTEM = """\
You are an FDA regulatory expert performing a final consistency audit on a Nutrition Facts label.

Tasks:
1. PDP Area & Type Size:
   - Rectangular: PDP area = height_in × width_in
   - Cylindrical: PDP area = height_in × (π × diameter_in) / num_display_panels
   - Minimum type size thresholds:
     ≤5 sq in → 0.0625" (1/16")
     5-25 sq in → 0.125" (1/8")
     25-100 sq in → 0.1875" (3/16")
     >100 sq in → 0.25" (1/4")

2. Health Claims:
   - Only include claims substantiated by the product's computed nutrient values
   - When in doubt, use empty list []

3. Consistency Violations:
   - Check for cross-step inconsistencies (e.g., serving size doesn't match RACC for declared category)
   - Report only genuine violations you can verify from the data
   - Do NOT hallucinate violations — empty list [] is better than false positives

Return ONLY a JSON object with keys:
- declared_type_size_inch: minimum type size for the PDP area (float)
- health_claims: list of substantiated health claims (usually [])
- consistency_violations: list of identified cross-step issues (usually [])
No explanation, no markdown wrapping."""

_PHASE_SYSTEMS = {
    1: PHASE_1_SYSTEM,
    2: PHASE_2_SYSTEM,
    3: PHASE_3_SYSTEM,
    4: PHASE_4_SYSTEM,
    5: PHASE_5_SYSTEM,
}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE-SPECIFIC USER PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_phase_user_prompt(phase: int, observation: dict, prior_submissions: dict) -> str:
    """Build the user prompt for a given phase from the observation data."""
    phase_data = observation.get("phase_data", {})
    text = observation.get("text", "")

    if phase == 1:
        return (
            f"Classify this food product and determine the RACC:\n\n"
            f"Food description: {phase_data.get('food_category_description', 'unknown')}\n"
            f"Physical form: {phase_data.get('physical_form', 'unknown')}\n"
            f"Unit weight: {phase_data.get('unit_weight_g', 'N/A')}g\n"
            f"Total package weight: {phase_data.get('total_package_weight_g', 'unknown')}g\n"
        )

    if phase == 2:
        p1 = prior_submissions.get(1, {})
        return (
            f"Determine the label format and serving size.\n\n"
            f"From Phase 1:\n"
            f"  Food category: {p1.get('food_category', 'unknown')}\n"
            f"  RACC: {p1.get('racc_g', 'unknown')}g\n"
            f"  Household measure: {p1.get('household_measure', 'unknown')}\n\n"
            f"Product info:\n"
            f"  Physical form: {phase_data.get('physical_form', 'unknown')}\n"
            f"  Unit weight: {phase_data.get('unit_weight_g', 'N/A')}g\n"
            f"  Total package weight: {phase_data.get('total_package_weight_g', 'unknown')}g\n"
        )

    if phase == 3:
        p2 = prior_submissions.get(2, {})
        return (
            f"Compute nutrient values for the label.\n\n"
            f"From Phase 2:\n"
            f"  Serving size: {p2.get('serving_size_g', 'unknown')}g\n\n"
            f"Lab data:\n"
            f"  Lab sample size: {phase_data.get('lab_sample_size_g', 'unknown')}g\n"
            f"  Lab nutrients:\n```json\n{json.dumps(phase_data.get('lab_nutrients', {}), indent=2)}\n```\n"
        )

    if phase == 4:
        return (
            f"Determine ingredient order after moisture loss.\n\n"
            f"Recipe:\n```json\n{json.dumps(phase_data.get('recipe', []), indent=2)}\n```\n\n"
            f"Moisture loss: {phase_data.get('moisture_loss_pct', 0)}%\n"
        )

    if phase == 5:
        return (
            f"Perform the final consistency audit.\n\n"
            f"Container:\n```json\n{json.dumps(phase_data.get('container', {}), indent=2)}\n```\n\n"
            f"Prior submissions:\n```json\n{json.dumps(prior_submissions, indent=2)}\n```\n"
        )

    return text


# ─────────────────────────────────────────────────────────────────────────────
# ACTION BUILDER (add phase number to parsed JSON)
# ─────────────────────────────────────────────────────────────────────────────

def _build_action_for_phase(phase: int, parsed_json: dict) -> dict:
    """Wrap parsed LLM output into a phase-tagged action dict."""
    action = {"phase": phase}
    action.update(parsed_json)
    return action


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
# MAIN AGENT LOOP (HTTP-only, 5-phase)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_agent() -> dict[str, float]:
    """
    Run the baseline agent against all tasks using the 5-phase HTTP flow.

    For each task:
    1. POST /reset → get phase 1 observation
    2. For each phase 1-5:
       a. Build phase-specific prompt
       b. Call LLM
       c. Parse JSON response
       d. POST /step with phase-tagged action
    3. POST /grader to get official score

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

        # Reset environment
        reset_response = http_client.post("/reset", json={
            "task_id": task_id,
            "seed": seed,
        })
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        observation = reset_data["observation"]

        all_actions: list[dict] = []
        all_rewards: list[float] = []
        prior_submissions: dict = {}
        done = False

        for phase in range(1, 6):
            if done:
                break

            # Build phase-specific prompt
            system_prompt = _PHASE_SYSTEMS[phase]
            user_prompt = _build_phase_user_prompt(phase, observation, prior_submissions)

            # Include feedback from previous phase if available
            feedback_text = observation.get("text", "")
            if phase > 1 and "FEEDBACK" in feedback_text:
                user_prompt = f"{feedback_text}\n\n---\n\n{user_prompt}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM
            llm_error: str | None = None
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
                llm_error = str(exception)
                logger.warning("LLM call failed at phase %d: %s", phase, exception)

            # Parse response
            parsed = _extract_json_from_response(llm_response_text)
            if parsed is None:
                logger.warning("Could not parse JSON from LLM — task=%s phase=%d", task_id, phase)
                parsed = {}

            # Build action with phase tag
            action = _build_action_for_phase(phase, parsed)
            all_actions.append(action)

            # Track prior submissions for context in later phases
            prior_submissions[phase] = parsed

            # Submit to environment
            step_response = http_client.post("/step", json={"action": action})
            step_response.raise_for_status()
            step_data = step_response.json()

            reward = step_data.get("reward") or 0.0
            done = step_data.get("done", False)
            observation = step_data.get("observation", {})
            all_rewards.append(reward)

            action_json_string = json.dumps(parsed, separators=(",", ":"))
            _log_step(
                step_number=phase,
                action_json=action_json_string,
                reward=reward,
                done=done,
                error=llm_error,
            )

            logger.info("task=%s phase=%d reward=%.3f done=%s", task_id, phase, reward, done)

        # Get official grader score
        grader_response = http_client.post("/grader", json={
            "task_id": task_id,
            "seed": seed,
            "actions": all_actions,
        })
        grader_response.raise_for_status()
        grader_result = grader_response.json()
        grader_score = grader_result["grader_score"]

        _log_end(
            success=grader_score > 0.0,
            steps=len(all_actions),
            score=grader_score,
            rewards=all_rewards,
        )

        task_scores[task_id] = grader_score
        logger.info("task=%s DONE  final_score=%.3f  steps=%d", task_id, grader_score, len(all_actions))

    http_client.close()
    return task_scores


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TASK RUNNER (for /baseline/run endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_task(task_id: str, seed: int | None = None) -> dict:
    """
    Run the baseline agent for a single task via HTTP and return results.

    Returns:
        {
            "task_id": str,
            "seed": int,
            "grader_score": float,
            "steps_taken": int,
            "rewards": list[float],
            "phase_scores": dict,
        }
    """
    logger.info("run_baseline_task  task=%s seed=%s  model=%s", task_id, seed, MODEL_NAME)
    http_client = httpx.Client(base_url=ENVIRONMENT_BASE_URL, timeout=60)
    openai_client = _create_openai_client()

    _log_start(task_id, MODEL_NAME)

    actual_seed = seed or TASK_SEEDS.get(task_id, 42)

    # Reset
    reset_response = http_client.post("/reset", json={
        "task_id": task_id,
        "seed": actual_seed,
    })
    reset_response.raise_for_status()
    observation = reset_response.json()["observation"]

    all_actions: list[dict] = []
    all_rewards: list[float] = []
    prior_submissions: dict = {}
    done = False

    for phase in range(1, 6):
        if done:
            break

        system_prompt = _PHASE_SYSTEMS[phase]
        user_prompt = _build_phase_user_prompt(phase, observation, prior_submissions)

        feedback_text = observation.get("text", "")
        if phase > 1 and "FEEDBACK" in feedback_text:
            user_prompt = f"{feedback_text}\n\n---\n\n{user_prompt}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_error: str | None = None
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
            llm_error = str(exception)
            logger.warning("LLM call failed at phase %d: %s", phase, exception)

        parsed = _extract_json_from_response(llm_response_text) or {}
        action = _build_action_for_phase(phase, parsed)
        all_actions.append(action)
        prior_submissions[phase] = parsed

        step_response = http_client.post("/step", json={"action": action})
        step_response.raise_for_status()
        step_data = step_response.json()

        reward = step_data.get("reward") or 0.0
        done = step_data.get("done", False)
        observation = step_data.get("observation", {})
        all_rewards.append(reward)

        action_json_string = json.dumps(parsed, separators=(",", ":"))
        _log_step(
            step_number=phase,
            action_json=action_json_string,
            reward=reward,
            done=done,
            error=llm_error,
        )

    # Get state for phase scores
    state_response = http_client.get("/state")
    state_data = state_response.json() if state_response.status_code == 200 else {}

    # Get official grader score
    grader_response = http_client.post("/grader", json={
        "task_id": task_id,
        "seed": actual_seed,
        "actions": all_actions,
    })
    grader_response.raise_for_status()
    grader_score = grader_response.json()["grader_score"]

    _log_end(
        success=grader_score > 0.0,
        steps=len(all_actions),
        score=grader_score,
        rewards=all_rewards,
    )

    http_client.close()
    logger.info("run_baseline_task  task=%s DONE  score=%.3f  steps=%d", task_id, grader_score, len(all_actions))
    return {
        "task_id": task_id,
        "seed": actual_seed,
        "grader_score": grader_score,
        "steps_taken": len(all_actions),
        "rewards": all_rewards,
        "phase_scores": state_data.get("phase_scores", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scores = run_baseline_agent()
    print(json.dumps(scores, indent=2), file=sys.stderr)
