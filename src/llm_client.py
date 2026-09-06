"""
Shared LLM client: every component in this project that needs to call
Gemini for TEXT GENERATION (not embeddings) goes through this single
wrapper - the router, the query-transform techniques, the final answer
generator, and the evaluation judge all use it.

Why we need it (centralizing this logic):
The task requires ACCURATE cost tracking - the actual number of tokens
used and actual cost of every LLM call made for a given question
(router call + rewrite call + generation call + judge call, etc.), not
a made-up flat number. If every component made its own raw API call,
we would have to duplicate the cost-calculation math everywhere, and it
would be easy for one component to forget to record its cost, silently
breaking the per-question cost total.

Why we read token counts from the API response instead of estimating:
Gemini's response includes an exact `usage_metadata` block with the
real prompt/output token counts as billed by Google. Using this instead
of a rough estimate (e.g. "1 token ~= 4 characters") gives us an honest,
defensible cost figure instead of an approximation.

Why every call uses a single model (gemini-3.6-flash), not a two-tier
strategy:
We initially designed a two-tier strategy (a lightweight model for
cheap tasks like routing/judging, the strong model reserved for final
answer generation only) to minimize cost. During testing, the
lightweight model (gemini-3.5-flash-lite) proved unreliable on the free
tier - repeated rate-limit errors caused 2-3 minute delays per call,
confirmed via the retry logging below. Given the project deadline, we
prioritized reliability over marginal cost savings and standardized on
gemini-3.6-flash for every call in the pipeline. The `model_name`
parameter below is kept for interface compatibility with existing
callers (router, query-transform techniques, etc.) that were written
expecting to be able to request a specific model - but it is currently
a NO-OP: whatever value is passed, generation and cost calculation both
use config.GENERATION_MODEL. This keeps calling code unchanged while
reflecting the actual, deliberate single-model decision. This trade-off
(reliability over marginal cost savings, given the deadline) is
documented in the final report.

If we skipped this centralized wrapper:
Each component would need its own retry logic, its own cost math, and
its own way of returning "the text plus how much this call cost." Any
missed or duplicated call here would make the final cost table wrong.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

genai.configure(api_key=config.GEMINI_API_KEY)


@dataclass
class LLMResponse:
    """Everything we need to know about one generation call."""
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Convert token counts into a USD cost using config.py's pricing.
    Since the project standardizes on a single model
    (config.GENERATION_MODEL) for every call, there is only one price
    table to apply - see the module docstring for why the two-tier
    pricing approach was dropped.
    """
    input_cost = (input_tokens / 1_000_000) * config.PRICE_INPUT_PER_1M
    output_cost = (output_tokens / 1_000_000) * config.PRICE_OUTPUT_PER_1M
    return round(input_cost + output_cost, 8)


def _log_retry_attempt(retry_state):
    """Print a visible message every time a call is retried, so slowness
    caused by rate-limit retries is obvious instead of silent."""
    exception = retry_state.outcome.exception()
    print(f"  [retry] Attempt {retry_state.attempt_number} failed with: {exception}. "
          f"Waiting before retrying...")


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(6),
    before_sleep=_log_retry_attempt,
)
def generate(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.3,
    model_name: str | None = None,
) -> LLMResponse:
    """
    Send one prompt to Gemini and return the generated text together
    with the exact token usage, cost, and latency for this single call.

    system_instruction: high-level behavioral instructions for the model
    (e.g. "You are a routing classifier, reply only in JSON").
    temperature: lower = more deterministic/focused (good for routing,
    judging); higher = more varied (rarely needed in this project).
    model_name: kept for interface compatibility with callers written
    for a two-tier model strategy. Currently a NO-OP - every call uses
    config.GENERATION_MODEL regardless of what is passed here. See the
    module docstring for why (free-tier reliability issues with the
    lighter model under the project deadline).
    """
    
    model = genai.GenerativeModel(
        config.GENERATION_MODEL,
        system_instruction=system_instruction,
    )

    start_time = time.time()
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature),
    )
    latency = time.time() - start_time

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    cost = _calculate_cost(input_tokens, output_tokens)

    return LLMResponse(
        text=response.text.strip(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_seconds=round(latency, 3),
    )


if __name__ == "__main__":
    # Quick manual test
    result = generate("Reply with exactly one word: Working")
    print(f"Text: {result.text}")
    print(f"Input tokens: {result.input_tokens}, Output tokens: {result.output_tokens}")
    print(f"Cost: ${result.cost_usd}")
    print(f"Latency: {result.latency_seconds}s")