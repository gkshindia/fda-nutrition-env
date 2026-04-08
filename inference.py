"""Inference entrypoint for hackathon evaluation.

This module intentionally delegates to the baseline agent implementation.
"""

from __future__ import annotations

import json
import sys

from baseline import run_baseline_agent


def main() -> int:
    scores = run_baseline_agent()
    # Keep summary output off stdout so START/STEP/END logs remain clean.
    print(json.dumps(scores, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
