"""
Component 8/8: Evaluation - Part 2: the four required metrics

Why these four specific metrics, and why each needs DIFFERENT inputs:
- Context Relevance only needs (question, context) - it purely measures
  whether retrieval did its job, independent of what the model then
  said.
- Faithfulness only needs (context, answer) - it purely measures
  whether the model stayed grounded in what it was given, independent
  of whether the question was even answered well.
- Answer Relevance only needs (question, answer) - it purely measures
  whether the model addressed what was asked, independent of whether
  that answer came from real evidence.
- Correctness needs (question, answer, context) - factual accuracy
  requires judging the claim itself, using the context as a reference
  and the judge's own knowledge as a backstop.

Why we deliberately give each metric ONLY the inputs it needs (not
everything every time):
Isolating each metric's inputs to exactly what it's meant to measure
prevents cross-contamination - e.g., if Faithfulness judging also saw
the original question, the judge might unconsciously reward the answer
for "sounding right" for the question rather than strictly checking
whether it's grounded in the given context. Keeping each metric's
inputs narrow keeps what it measures honest.

If we skipped separating these concerns:
A single vague "how good is this overall?" score would conflate very
different failure modes - a genuinely well-retrieved, well-grounded
answer that just doesn't quite address the question would get the same
low score as a completely hallucinated one, even though these are very
different problems requiring very different fixes.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.evaluation.judge import run_judge, JudgeResult


@dataclass
class EvaluationResults:
    context_relevance: JudgeResult
    faithfulness: JudgeResult
    answer_relevance: JudgeResult
    correctness: JudgeResult

    @property
    def total_judge_cost_usd(self) -> float:
        return round(
            self.context_relevance.cost_usd + self.faithfulness.cost_usd
            + self.answer_relevance.cost_usd + self.correctness.cost_usd,
            8,
        )

    @property
    def average_score(self) -> float:
        scores = [
            self.context_relevance.score, self.faithfulness.score,
            self.answer_relevance.score, self.correctness.score,
        ]
        return round(sum(scores) / len(scores), 2)


def evaluate_context_relevance(question: str, context: str) -> JudgeResult:
    prompt = f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}"
    return run_judge("context_relevance", prompt)


def evaluate_faithfulness(context: str, answer: str) -> JudgeResult:
    prompt = f"CONTEXT PROVIDED TO THE MODEL:\n{context}\n\nGENERATED ANSWER:\n{answer}"
    return run_judge("faithfulness", prompt)


def evaluate_answer_relevance(question: str, answer: str) -> JudgeResult:
    prompt = f"QUESTION:\n{question}\n\nGENERATED ANSWER:\n{answer}"
    return run_judge("answer_relevance", prompt)


def evaluate_correctness(question: str, answer: str, context: str) -> JudgeResult:
    prompt = f"QUESTION:\n{question}\n\nGENERATED ANSWER:\n{answer}\n\nAVAILABLE CONTEXT:\n{context}"
    return run_judge("correctness", prompt)


def evaluate_all(question: str, context: str, answer: str) -> EvaluationResults:
    """Run all four metrics for one question/context/answer triple."""
    return EvaluationResults(
        context_relevance=evaluate_context_relevance(question, context),
        faithfulness=evaluate_faithfulness(context, answer),
        answer_relevance=evaluate_answer_relevance(question, answer),
        correctness=evaluate_correctness(question, answer, context),
    )


if __name__ == "__main__":
    from src.retrieval.retriever import retrieve
    from src.generation.generator import generate_answer, format_context

    test_question = "What are the steps for the annual inventory count of stationery warehouses?"
    print(f"Question: {test_question}\n")

    chunks = retrieve(test_question, top_k=5)
    gen_result = generate_answer(test_question, chunks)
    context_text = format_context(chunks)

    print(f"Generated answer:\n{gen_result.answer}\n")

    eval_results = evaluate_all(test_question, context_text, gen_result.answer)

    for metric_result in [
        eval_results.context_relevance, eval_results.faithfulness,
        eval_results.answer_relevance, eval_results.correctness,
    ]:
        print(f"--- {metric_result.metric} ---")
        print(f"  Score: {metric_result.score}/5")
        print(f"  Justification: {metric_result.justification}")

    print(f"\nAverage score: {eval_results.average_score}/5")
    print(f"Total judge cost (4 calls): ${eval_results.total_judge_cost_usd}")
    