"""
Inference entrypoint for hackathon evaluation.
================================================
Uses the OpenEnv SDK to spin up the FDA environment container,
then runs the 5-phase baseline agent against all tasks.

Required env vars (set by the evaluator):
    API_BASE_URL       OpenAI-compatible API endpoint
    MODEL_NAME         Model identifier
    HF_TOKEN           Auth token for the LLM endpoint
    IMAGE_NAME         Docker image name for the environment

Optional:
    OPENAI_API_KEY     For direct OpenAI usage (fallback)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

from openai import OpenAI

from env import FDAAction, FDAEnv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _env(name: str, default: str | None = None) -> str | None:
    """Read env var, treating empty strings as unset."""
    value = os.getenv(name, "")
    return value.strip() if value.strip() else default


IMAGE_NAME = _env("IMAGE_NAME")
ENV_BASE_URL = _env("ENV_BASE_URL", "http://localhost:7860")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = _env("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = _env("MODEL_NAME", "gpt-4o-mini")
ENVIRONMENT_NAME = "fda-nutrition-env"

TASK_SEEDS = {
    "task_easy": 100,
    "task_medium": 200,
    "task_hard": 300,
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _create_openai_client() -> OpenAI:
    """Create an OpenAI client using evaluator-provided credentials."""
    return OpenAI(api_key=API_KEY or "not-needed", base_url=API_BASE_URL, timeout=90.0)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS FROM BASELINE (prompts, JSON extraction, action builder)
# ─────────────────────────────────────────────────────────────────────────────

from baseline import (
    _extract_json_from_response,
    _build_phase_user_prompt,
    _build_action_for_phase,
    _PHASE_SYSTEMS,
)


# ─────────────────────────────────────────────────────────────────────────────
# STDOUT LOGGING (hackathon format)
# ─────────────────────────────────────────────────────────────────────────────

def _log_start(task_id: str, model_name: str) -> None:
    print(f"[START] task={task_id} env={ENVIRONMENT_NAME} model={model_name}", flush=True)


def _log_step(step: int, action_json: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = "null" if error is None else error
    print(
        f"[STEP]  step={step} "
        f"action={action_json} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error_str}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def run_task(env: FDAEnv, openai_client: OpenAI, task_id: str, seed: int) -> float:
    """Run one task through the 5-phase pipeline. Returns grader score."""
    _log_start(task_id, MODEL_NAME)

    result = await env.reset(task_id=task_id, seed=seed)
    observation = {
        "text": result.observation.text,
        "phase": result.observation.phase,
        "phase_data": result.observation.phase_data,
        "prior_submissions": result.observation.prior_submissions,
    }

    all_rewards: list[float] = []
    prior_submissions: dict = {}
    done = False
    steps_taken = 0

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

        parsed = _extract_json_from_response(llm_response_text)
        if parsed is None:
            parsed = {}

        action_dict = _build_action_for_phase(phase, parsed)
        prior_submissions[phase] = parsed

        result = await env.step(FDAAction(**action_dict))

        reward = result.reward or 0.0
        done = result.done
        all_rewards.append(reward)
        steps_taken = phase

        observation = {
            "text": result.observation.text,
            "phase": result.observation.phase,
            "phase_data": result.observation.phase_data,
            "prior_submissions": result.observation.prior_submissions,
        }

        action_json_string = json.dumps(parsed, separators=(",", ":"))
        _log_step(
            step=phase,
            action_json=action_json_string,
            reward=reward,
            done=done,
            error=llm_error,
        )

    # Final score is the reward from the last step (phase 5)
    final_score = all_rewards[-1] if all_rewards else 0.0

    _log_end(
        success=final_score > 0.0,
        steps=steps_taken,
        score=final_score,
        rewards=all_rewards,
    )

    return final_score


async def main() -> int:
    openai_client = _create_openai_client()

    if IMAGE_NAME:
        env = await FDAEnv.from_docker_image(IMAGE_NAME)
    else:
        env = FDAEnv(base_url=ENV_BASE_URL)

    task_scores: dict[str, float] = {}

    try:
        for task_id, seed in TASK_SEEDS.items():
            score = await run_task(env, openai_client, task_id, seed)
            task_scores[task_id] = score
    finally:
        try:
            await env.close()
        except Exception:
            pass

    print(json.dumps(task_scores, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
