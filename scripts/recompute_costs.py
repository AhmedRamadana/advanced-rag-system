"""
Cost-correction utility: recalculates every cost_usd figure in an
existing results/results.json using the CURRENT pricing in config.py,
based purely on the token counts already recorded from the real run.

Why this script exists:
The official 10-question run was executed while config.py's pricing
constants (PRICE_INPUT_PER_1M / PRICE_OUTPUT_PER_1M) still held the
strong model's prices (1.50 / 7.50), even though the actual model being
called at that time was gemini-3.5-flash-lite (a config.py update that
changed which model NAME was used never updated the price constants to
match). This produced accurate token counts (read directly from the
API's real usage_metadata) but inflated cost figures (computed against
the wrong price table).

Why we fix this by recomputing from stored tokens instead of re-running
the pipeline:
The input/output token counts already recorded in results.json are
real, correct, and independent of pricing - only the derived cost_usd
values need correcting. Re-running the entire 10-question pipeline
again would consume more of the free-tier's limited daily quota for no
benefit, since the tokens used would be identical. This script instead
reads the existing file, reapplies correct_pricing() to every recorded
step, and rewrites both each step's cost and every rolled-up total -
producing fully accurate cost figures with zero additional API calls.
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def recalculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Same formula as llm_client._calculate_cost(), using CURRENT config.py pricing."""
    input_cost = (input_tokens / 1_000_000) * config.PRICE_INPUT_PER_1M
    output_cost = (output_tokens / 1_000_000) * config.PRICE_OUTPUT_PER_1M
    return round(input_cost + output_cost, 8)


def main():
    results_path = config.RESULTS_DIR / "results.json"

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Recalculating costs in {results_path} using:")
    print(f"  PRICE_INPUT_PER_1M  = {config.PRICE_INPUT_PER_1M}")
    print(f"  PRICE_OUTPUT_PER_1M = {config.PRICE_OUTPUT_PER_1M}\n")

    grand_total_old = 0.0
    grand_total_new = 0.0

    for question in results:
        old_question_total = question["total_cost_usd"]
        new_question_total = 0.0

        for step in question["steps"]:
            if step["executed"]:
                new_cost = recalculate_cost(step["input_tokens"], step["output_tokens"])
                step["cost_usd"] = new_cost
                new_question_total += new_cost

        new_question_total = round(new_question_total, 8)
        question["total_cost_usd"] = new_question_total

        grand_total_old += old_question_total
        grand_total_new += new_question_total

        print(f"{question['question_id']}: ${old_question_total} -> ${new_question_total}")

    print(f"\nTOTAL (old, wrong pricing): ${round(grand_total_old, 6)}")
    print(f"TOTAL (corrected):          ${round(grand_total_new, 6)}")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nresults.json updated in place with corrected costs.")


if __name__ == "__main__":
    main()