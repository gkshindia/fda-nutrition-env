---
title: FDA Nutrition Facts Panel Environment
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# FDA Nutrition Facts Panel — OpenEnv RL Environment

## What problem does this solve?

Every packaged food product sold in the United States must carry a Nutrition Facts label. The format, numbers, rounding rules, and ingredient order are all governed by federal law — specifically **21 CFR 101.9** and **21 CFR 101.12**. Getting it wrong means FDA warning letters, product recalls, and costly reformulations.

Right now, compliance is handled by humans — trained regulatory specialists who verify labels manually, or expensive third-party compliance software. The process is slow, error-prone, and doesn't scale. A food company launching 50 new SKUs a year needs 50 label reviews. Larger manufacturers deal with hundreds.

**This project asks: can an AI agent learn to do that review automatically?**

It frames FDA nutrition label compliance as a **reinforcement learning environment** — a structured game where an AI agent gets a broken label, tries to fix it, receives feedback on what's still wrong, and keeps improving across multiple attempts. The environment scores every correction against the legally correct answer, field by field.

---

## Why does this matter for companies?

### Food manufacturers
- Catch label errors before products ship — not after an FDA inspection
- Reduce dependence on expensive regulatory consultants for routine label checks
- Scale compliance review across hundreds of SKUs without adding headcount

### Retailers and contract manufacturers
- Verify supplier-submitted labels automatically before accepting product
- Audit label accuracy across a private label portfolio

### Regulatory technology companies
- Train and benchmark LLM-based compliance agents on a rigorous, deterministic task
- Compare model versions: does GPT-5 get serving size right more often than GPT-4?

### AI researchers
- A clean, grounded RL benchmark with real regulatory ground truth
- Partial reward signal across 7 independent field groups — not just pass/fail
- Seeded reproducibility: same seed = same episode every time, enabling fair comparison

---

## How it works — plain English

Think of this like a graded exam for an AI.

**Step 1 — The exam begins (`POST /reset`)**

The server generates a fake food product: say, a "Honey Almond Granola Bar." It calculates the legally correct Nutrition Facts label using real FDA rules and USDA nutrient data. Then it deliberately injects errors — wrong calorie count, ingredients in the wrong order, a %Daily Value that doesn't match the nutrient, a health claim the product doesn't actually qualify for.

The AI agent receives the product information (ingredients, lab measurements, container dimensions) and the broken draft label. Its job: identify and fix every error.

**Step 2 — The agent submits a correction (`POST /step`)**

The agent sends back its corrected label as JSON. The server grades it immediately against the hidden correct answer — no peeking allowed. It returns a score between 0 and 1, plus field-by-field feedback: "your sodium rounding is wrong, your ingredient order is wrong, your %DV for calcium is wrong."

**Step 3 — The agent iterates**

The agent reads the feedback, figures out what it got wrong, and submits a revised label. This repeats until the agent either decides it's done (`final_submission=true`), runs out of attempts (10-step cap), or stops improving (3 consecutive steps with no score increase).

**Step 4 — Final score**

The grader returns the best score achieved across all attempts, broken down by category.

```
Agent                           Server (:7860)
  |                                |
  |-- POST /reset --------------->|  Generate food product + inject errors
  |<-- broken draft label --------|
  |                                |
  |  (Agent reasons: "sodium       |
  |   should be 140mg, not 130mg") |
  |                                |
  |-- POST /step {corrected} ---->|  Grade vs. hidden ground truth
  |<-- score=0.61, feedback -------|  "ingredient_list: NEEDS FIX"
  |                                |  "percent_dvs: NEEDS FIX"
  |                                |
  |-- POST /step {revised} ------->|  Grade revised attempt
  |<-- score=0.78, feedback -------|  "ingredient_list: OK"
  |                                |  "percent_dvs: NEEDS FIX"
  |                                |
  |-- POST /step {revised} ------->|
  |<-- score=0.94, done=true ------|  Agent submits final
```

---

## What the grader checks

The grader evaluates 7 independent categories. Each contributes a fixed percentage to the final score:

| Category | Weight | What it checks |
|---|---|---|
| `serving_size_g` | 10% | Must match the RACC (Reference Amount Customarily Consumed) for the food category per 21 CFR 101.12 |
| `nutrients` | 30% | All 15 nutrient values must be scaled correctly from lab measurements and rounded per 21 CFR 101.9(c) |
| `percent_dvs` | 20% | Each %Daily Value must be computed from the rounded declared value using the 2020 DRV table |
| `ingredient_list` | 15% | Ingredients must be ordered by finished weight (after moisture loss), descending |
| `declared_type_size_inch` | 10% | Font size must meet the minimum for the label's PDP (principal display panel) area |
| `health_claims` | 10% | Claims like "high in fiber" must only appear if the nutrient values actually qualify |
| `atwater_consistency` | 5% | Declared calories must match the Atwater formula: 9×fat + 4×carb + 4×protein (applied to pre-rounded values) |

Score = weighted sum, 0.0 to 1.0. Partial credit is awarded per category — an agent that gets nutrients right but messes up ingredient order still scores ~0.65.

---

## What makes this hard

### The math is unforgiving
FDA rounding uses "round-half-up" — not Python's built-in `round()` which uses banker's rounding. At boundary values (e.g. exactly 2.5mg sodium), the two methods give different answers. The grader checks exact compliance.

### Cascading errors
On medium and hard difficulty, a wrong serving size cascades to all 15 nutrient values, all 12 %DVs, and the calorie count. One wrong input creates 28+ wrong outputs. The agent must identify the root cause, not patch each symptom.

### Moisture loss changes ingredient order
When a product is baked or cooked, water evaporates. The ingredient that was heaviest before baking may not be heaviest after. Ingredients must be sorted by *finished* weight — which requires tracking moisture loss proportionally across all moisture-contributing ingredients.

### Atwater calories require pre-rounded inputs
The FDA rule says: compute calories from the *unrounded* scaled nutrient values, then round. If the agent rounds first and applies Atwater second, it gets a different (wrong) calorie count. A subtle ordering error that's easy to miss.

---

## Difficulty levels

| | Easy | Medium | Hard |
|---|---|---|---|
| Errors injected | 5 | 7 | 10 |
| Scaling required | No (lab = serving size) | Yes (1.25–2.25x) | Yes (2.5–4.0x) |
| Moisture loss | 0% | 0–12% | 15–20% |
| Cascading errors | No | 1 cascade | Full cascade (serving size → all fields) |
| Error types | Wrong %DVs, ingredient swap, type size, rounding | Above + serving size inconsistency, non-adjacent ingredient swap | Above + Atwater inconsistency, unsupported health claim, ingredient rotation |

---

## API endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/reset` | POST | Start a new episode. Returns the broken draft label and product context. |
| `/step` | POST | Submit a corrected label. Returns score, feedback, and done flag. |
| `/state` | GET | Check current episode status (step count, best score, task). |
| `/tasks` | GET | List available tasks with descriptions and difficulty levels. |
| `/grader` | POST | Replay a full episode from seed + actions. Returns final grader score. |
| `/baseline/run` | POST | Run the GPT-4o-mini baseline agent on a task. Returns full trajectory. |
| `/docs` | GET | Auto-generated API documentation (FastAPI/Swagger). |

---

## Running it locally

**Requirements**: Python 3.11+, `uv`

```bash
# Clone and install
git clone https://github.com/your-org/fda-nutrition-env
cd fda-nutrition-env
uv sync

# Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# Start the server
uv run server
# → Server running at http://localhost:7860
# → API docs at http://localhost:7860/docs
# → Dashboard at http://localhost:7860

# Run the baseline agent (in a separate terminal)
uv run python baseline.py

# Run tests
uv run pytest
```

---

## Project layout

```
fda-nutrition-env/
│
├── data/
│   ├── regulatory_tables.py      # The legal ground truth — RACC table, FDA rounding
│   │                             # functions, Daily Reference Values, Atwater factors,
│   │                             # PDP area thresholds. All values from 21 CFR 101.9/101.12.
│   │                             # Do not modify without updating source citation.
│   │
│   ├── seed_products.py          # 20 base food ingredients with per-100g nutrient profiles
│   │                             # sourced from USDA SR Legacy 2018 database. Used by the
│   │                             # episode generator to build realistic food recipes.
│   │
│   └── _build_seed_products.py   # One-time script that generated seed_products.py from
│                                 # USDA CSV files. Not used in production.
│
├── core/
│   ├── episode_generator.py      # The heart of the environment. Given a difficulty level
│   │                             # and a random seed, generates a complete food episode:
│   │                             # picks ingredients, calculates the correct label, then
│   │                             # injects errors according to difficulty. Returns both
│   │                             # the broken draft and the hidden correct answer.
│   │
│   └── grader.py                 # Deterministic label grader. Compares an agent's submitted
│                                 # label against ground truth across 7 field groups.
│                                 # Returns a score 0.0–1.0 with per-group breakdown.
│
├── env/
│   ├── models.py                 # Pydantic data types: FDAAction (what the agent sends),
│   │                             # FDAObservation (what the server returns), FDAState
│   │                             # (internal episode state tracking).
│   │
│   ├── client.py                 # Python client for agents — wraps the HTTP API into
│   │                             # a clean reset()/step() interface.
│   │
│   └── server/
│       ├── app.py                # FastAPI application. Defines all HTTP endpoints,
│       │                         # manages the shared environment instance with thread
│       │                         # locking, serves the static UI dashboard.
│       │
│       └── environment.py        # FDAEnvironment class — implements the OpenEnv interface.
│                                 # Handles multi-step episode lifecycle: reset, step,
│                                 # plateau detection, best-score tracking.
│
├── baseline.py                   # Reference AI agent using GPT-4o-mini (or any OpenAI-
│                                 # compatible model). Implements the multi-step loop:
│                                 # calls /reset, sends label to LLM, submits to /step,
│                                 # feeds back grader feedback, repeats until done.
│
├── tests/
│   ├── test_ground_truth.py      # 7 tests — nutrient scaling, FDA rounding correctness,
│   │                             # Atwater formula, ingredient ordering, PDP area calculation
│   │
│   ├── test_grader.py            # 12 tests — scoring accuracy, partial credit, edge cases,
│   │                             # determinism across seeds, garbage input handling
│   │
│   ├── test_environment.py       # 11 tests — full reset/step cycle, multi-step improvement,
│   │                             # plateau detection, best-label tracking, seed reproducibility
│   │
│   └── test_http_endpoints.py    # HTTP integration test — verifies /reset → /step → /state
│                                 # persistence across requests using FastAPI TestClient
│
├── static/
│   └── index.html                # Single-page dashboard. Shows API Explorer (manual
│                                 # reset/step), visual Nutrition Facts label comparison
│                                 # (draft vs. corrected), and grader breakdown panel.
│                                 # "Run LLM Agent" button triggers /baseline/run live.
│
├── server/
│   └── Dockerfile                # Docker image for HuggingFace Spaces deployment
│
├── openenv.yaml                  # OpenEnv spec declaration (spec_version: 1, port: 7860)
└── pyproject.toml                # Dependencies, entry points (uv run server)
```

---

## The reinforcement learning loop

This environment follows the standard RL loop from the OpenEnv specification:

```
Environment (this repo)          Agent (baseline.py or your own)
       │                                   │
       │──── observation (broken label) ──>│
       │                                   │  Agent reads label, applies
       │                                   │  FDA rules, produces correction
       │<─── action (corrected label) ─────│
       │                                   │
       │  Grade → reward (0.0–1.0)         │
       │  Build feedback text              │
       │                                   │
       │──── reward + feedback ───────────>│  Agent uses feedback to
       │                                   │  improve next submission
       │<─── action (revised label) ───────│
       │        ...repeat...               │
```

The environment never calls the LLM — it only scores what the agent sends. The agent never sees the correct answer — only which fields are wrong and by how much. This mirrors real-world compliance: a QA reviewer tells you "this is wrong" without handing you the answer sheet.

---

## Baseline scores (GPT-4o-mini)

| Task | Score | Typical steps |
|---|---|---|
| Easy | ~0.76 | 2–4 |
| Medium | ~0.42 | 3–6 |
| Hard | ~0.33 | 4–8 |

The gap between easy and hard reflects real difficulty: hard episodes require the agent to reason about serving size cascades, moisture loss, and Atwater calorie computation from scratch — without the answer being handed to it.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for direct OpenAI usage |
| `API_BASE_URL` | — | OpenAI-compatible API base (set by hackathon evaluator) |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | — | Auth token when using HuggingFace inference endpoints |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL for the baseline agent |

When `API_BASE_URL` is set, `HF_TOKEN` is used as the API key. When unset, `OPENAI_API_KEY` is used. Empty strings in `.env` are treated as unset.
