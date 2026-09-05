"""
Component 8/8: Evaluation - Part 1: LLM-as-a-Judge core

Why we need a FIXED rubric (not just "rate this answer"):
If we asked the judge a vague question like "how good is this answer?"
on every call, its internal standard could drift between questions -
one call might be lenient, another strict, purely based on prompt
phrasing variance. A fixed rubric with explicit anchors for what a 1, a
3, and a 5 actually mean forces the same standard to be applied to
every question, which is what makes scores comparable across our ten
test questions in the final results table.

Why 1-5 instead of 0-10 or pass/fail:
A 5-point scale is granular enough to distinguish "clearly bad" (1),
"partially works" (3), and "clearly good" (5) without asking the judge
to make finer distinctions than an LLM can reliably and consistently
make (e.g., the difference between a 7 and an 8 out of 10 is rarely
meaningful or consistent).

Why we ask for a justification alongside the score:
A bare number gives us no way to sanity-check whether the judge is
scoring for the right reasons. The justification lets a human reviewer
(us, or the mentor) audit *why* a score was given, catching cases where
the judge's reasoning doesn't actually support its own score.

If we skipped a shared, fixed rubric like this:
Every metric evaluation would effectively be scored against a
different, implicit standard each time, making it meaningless to
compare "Context Relevance" scores across different questions in the
results table - defeating the whole purpose of a structured evaluation.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client

JUDGE_OUTPUT_INSTRUCTION = """
Return ONLY a JSON object, no markdown formatting, no extra text, in
exactly this shape:
{"score": <integer 1-5>, "justification": "<one concise sentence explaining the score>"}"""

RUBRICS = {
    "context_relevance": f"""You are an evaluation judge scoring RETRIEVAL QUALITY for a RAG system.
You will be given a QUESTION and the CONTEXT that was retrieved for it.
Score how relevant and useful the retrieved context is for answering
the question, using this fixed 1-5 scale:

5 - The context directly and fully contains the information needed to answer the question.
4 - The context contains most of what's needed, with minor gaps.
3 - The context is partially relevant but missing significant information.
2 - The context is only tangentially related to the question's topic.
1 - The context is irrelevant to the question.
{JUDGE_OUTPUT_INSTRUCTION}""",

    "faithfulness": f"""You are an evaluation judge scoring FAITHFULNESS (groundedness) for a RAG
system. You will be given the CONTEXT that was provided to the model and
the ANSWER it generated. Score whether the answer's claims are actually
supported by the context (i.e. it did not hallucinate beyond what the
context states), using this fixed 1-5 scale:

5 - Every claim in the answer is directly supported by the context (or the answer correctly states the context is insufficient, if that's the case).
4 - Nearly all claims are supported, with only very minor unsupported details.
3 - Some claims are supported, but the answer includes noticeable unsupported additions.
2 - Most of the answer is not supported by the context.
1 - The answer contradicts the context or is entirely fabricated relative to it.
{JUDGE_OUTPUT_INSTRUCTION}""",

    "answer_relevance": f"""You are an evaluation judge scoring ANSWER RELEVANCE for a RAG system.
You will be given a QUESTION and the ANSWER that was generated. Score
whether the answer actually addresses what was asked (regardless of
whether it's factually correct or grounded in any particular source),
using this fixed 1-5 scale:

5 - The answer directly and completely addresses the question asked.
4 - The answer addresses the question with minor omissions or tangents.
3 - The answer partially addresses the question, missing a notable part of it.
2 - The answer is mostly off-topic relative to the question.
1 - The answer does not address the question at all.
{JUDGE_OUTPUT_INSTRUCTION}""",

    "correctness": f"""You are an evaluation judge scoring FACTUAL CORRECTNESS for a RAG system.
You will be given a QUESTION, the ANSWER that was generated, and the
CONTEXT that was available. Using your own knowledge together with the
provided context, score how factually correct the answer is, using this
fixed 1-5 scale:

5 - The answer is fully accurate, with no factual errors.
4 - The answer is mostly accurate, with only minor inaccuracies that don't change the substance.
3 - The answer has a mix of accurate and inaccurate content.
2 - The answer has significant factual errors.
1 - The answer is factually wrong or fabricated.

Note: if the answer correctly states that the provided material does
not contain enough information to answer (rather than guessing), that
is the CORRECT behavior and should score 5, not be penalized.
{JUDGE_OUTPUT_INSTRUCTION}""",
}


@dataclass
class JudgeResult:
    metric: str
    score: int
    justification: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json(raw_text: str) -> dict:
    """Strip any markdown code-fence wrapping before parsing the JSON object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def run_judge(metric: str, prompt_content: str) -> JudgeResult:
    """
    Core judge call: applies the FIXED rubric for the given metric to
    the provided prompt content (which varies per metric - see
    metrics.py for how each metric assembles its specific content).
    """
    if metric not in RUBRICS:
        raise ValueError(f"Unknown metric: {metric}. Must be one of {list(RUBRICS.keys())}")

    llm_result = llm_client.generate(
        prompt=prompt_content,
        system_instruction=RUBRICS[metric],
        temperature=0.0,  # judging should be as consistent/deterministic as possible
    )

    try:
        parsed = _extract_json(llm_result.text)
        score = int(parsed.get("score", 3))
        score = max(1, min(5, score))  # clamp to valid range just in case
        justification = parsed.get("justification", "")
    except (json.JSONDecodeError, AttributeError, ValueError):
        score, justification = 3, "Fallback: judge output could not be parsed."

    return JudgeResult(
        metric=metric,
        score=score,
        justification=justification,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )

