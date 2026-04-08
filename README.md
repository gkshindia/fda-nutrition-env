# FDA Nutrition Facts Panel — OpenEnv Environment

## What is this?

Imagine you're a quality checker at a food company. You get handed a draft Nutrition Facts label (the thing on the back of every food package) — but someone made mistakes on it. Wrong calorie count, ingredients in the wrong order, wrong percentages.

Your job: find and fix all the mistakes so the label complies with FDA regulations.

**This project turns that task into a game for AI agents.**

1. The **environment** (server) generates a fake food product with a deliberately broken label
2. The **agent** (an LLM like GPT-4o) looks at the broken label and tries to fix it
3. The **grader** scores how well the agent did (0% to 100%)
4. If the score isn't good enough, the agent gets feedback ("these fields are wrong") and tries again
5. The episode ends when the agent scores **90%+** or hits the 10-step safety cap

The "RL" here follows the standard reinforcement learning loop — observe, act, get reward, repeat — but instead of training a robot to walk, we're testing whether an AI can understand and apply FDA food labeling regulations.

## How it works

```
Agent                           Server (:7860)
  |                                |
  |-- POST /reset --------------->|  Generate episode (broken label + hidden answer)
  |<-- draft label + context -----|
  |                                |
  |  (Agent reasons about FDA      |
  |   rules and fixes errors)      |
  |                                |
  |-- POST /step {corrected label}>|  Grade correction against ground truth
  |<-- feedback (score + wrong     |
  |    fields, no answers shown)   |
  |                                |
  |  ... agent revises and         |
  |      resubmits until ...       |
  |                                |
  |  score >= 0.90  --> done!      |
  |  final_submission=true --> done!|
  |  10 steps hit   --> done!      |
```

The agent keeps revising until it reaches FDA-acceptable quality (score >= 0.90), decides to stop (`final_submission=true`), or exhausts the 10-step safety cap.

## Grading

The grader checks 7 categories, each with a fixed weight:

| Group | Weight | What it checks |
|---|---|---|
| `serving_size_g` | 10% | Exact match (tolerance 0.01g) |
| `nutrients` | 30% | 15 nutrient fields vs FDA-rounded ground truth |
| `percent_dvs` | 20% | Integer %Daily Value match per nutrient |
| `ingredient_list` | 15% | Exact ordered list (case-insensitive) |
| `declared_type_size_inch` | 10% | Must meet minimum type size for PDP area |
| `health_claims` | 10% | Must be empty (no unsupported claims) |
| `atwater_consistency` | 5% | Declared kcal matches Atwater formula |

Final score = weighted sum, in [0.0, 1.0].

## Difficulty levels

| Aspect | Easy | Medium | Hard |
|---|---|---|---|
| Errors injected | 3 | 5 | 7 |
| Lab sample = serving? | Yes | No (scale 1.25-2.25x) | No (scale 2.5-4.0x) |
| Moisture loss | 0% | 0-12% | 15-20% |
| RACC category | Unambiguous | Ambiguous (2 options) | Ambiguous (3 options) |
| Boundary nudging | None | 2 nutrients | 3 nutrients |

## Quickstart

### Run the server

```bash
uv run server          # starts FastAPI on port 7860
```

### Run the baseline agent

```bash
uv run python baseline.py    # needs server running + OPENAI_API_KEY in .env
```

### Run tests

```bash
uv run pytest          # 34 tests, no API key needed
```

## Environment variables

Loaded from `.env` via `python-dotenv` in `baseline.py`.

| Variable | Default | Purpose |
|---|---|---|
| `ENV_BASE_URL` | `http://localhost:7860` | Base URL for baseline agent |
| `API_BASE_URL` | -- | OpenAI-compatible API base (hackathon evaluator sets this) |
| `MODEL_NAME` | `gpt-4o-mini` | Model to use for baseline |
| `HF_TOKEN` | -- | HuggingFace token |
| `OPENAI_API_KEY` | -- | Required when using OpenAI models directly |

**Priority**: When `API_BASE_URL` is set, use `HF_TOKEN` as api_key. When unset, use `OPENAI_API_KEY`.

## Project layout

```
fda-nutrition-env/
  data/
    regulatory_tables.py      # Ground truth: RACC table, rounding fns, DRVs, Atwater
    seed_products.py          # 20 base ingredients, per-100g USDA SR Legacy nutrients
  core/
    episode_generator.py      # generate_episode(difficulty, seed) -> episode dict
    grader.py                 # grade(agent_label, ground_truth) -> score 0-1
  env/
    models.py                 # FDAAction, FDAObservation, FDAState pydantic types
    client.py                 # FDAEnv -- EnvClient subclass
    server/
      app.py                  # FastAPI app + /tasks, /grader, /baseline routes
      environment.py          # FDAEnvironment class + TASKS registry
  baseline.py                 # GPT-4o-mini baseline agent (multi-step with feedback)
  tests/
    test_ground_truth.py      # Scaling, rounding, Atwater, ingredients, PDP
    test_grader.py            # Scoring, variance, edge cases, determinism
    test_environment.py       # Reset/step cycle, multi-step, auto-stop
  server/
    Dockerfile                # Docker image for HF Spaces deployment
  openenv.yaml                # spec_version: 1, port: 7860
  pyproject.toml
```
