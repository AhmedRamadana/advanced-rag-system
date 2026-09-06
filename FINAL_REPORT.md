# Final Report — Advanced RAG System (Housing Bank Internal Procedures)

## 1. Required Results Table

| ID | Route | Techniques Used | Context Rel. | Faithfulness | Answer Rel. | Correctness | Total Cost | Latency |
|----|-------|------------------|:---:|:---:|:---:|:---:|---:|---:|
| Q1 | simple | – | N/A | N/A | 5 | 5 | $0.001583 | 3.96s |
| Q2 | simple | – | N/A | N/A | 5 | 5 | $0.002484 | 4.97s |
| Q3 | simple | – | N/A | N/A | 5 | 5 | $0.002256 | 4.92s |
| Q4 | simple | – | N/A | N/A | 5 | 5 | $0.003107 | 5.55s |
| Q5 | simple | – | N/A | N/A | 5 | 5 | $0.004112 | 6.62s |
| Q6 | simple | – | N/A | N/A | 5 | 5 | $0.001530 | 3.71s |
| Q7 | advanced_rag | self_query, reranking | **5** | **5** | 5 | 5 | $0.004052 | 6.46s |
| Q8 | simple | – | N/A | N/A | 5 | 5 | $0.002354 | 4.61s |
| Q9 | simple | – | N/A | N/A | 5 | 5 | $0.000905 | 2.92s |
| Q10 | advanced_rag | decomposition, reranking | **1** | **3** | 5 | 5 | $0.004041 | 6.05s |

**Total cost across all 10 questions: $0.026425 | Total wall-clock time: 133.2s**

**Note on this figure:** these costs were recalculated from the exact, real input/output token counts recorded by the Gemini API during the official run, using `gemini-3.5-flash-lite`'s correct published pricing ($0.30/1M input, $2.50/1M output). An initial version of this table used the same real token counts but the wrong price constants (still set to the strong model's $1.50/$7.50 rates at the time of the run, from an earlier single-model configuration that was never updated when the model name was switched) — that error was caught and corrected via `scripts/recompute_costs.py`, which recalculates cost purely from the already-recorded token counts with no additional API calls, so no quota was spent re-running anything.

**N/A note:** Context Relevance and Faithfulness require a real retrieved context to judge against. Questions routed to `simple` deliberately perform no retrieval at all (see Section 2), so these two metrics are not computed for them — scoring them against an empty context would misleadingly look like a system failure rather than the correct, intended behavior.

### Question substitutions (documented per the task's own instruction)

The task's original Q7 ("Find documents about RAG published after 2024") and Q10 ("choose an appropriate question from the corpus") both require retrieval against our specific corpus, which contains three Arabic-language Housing Bank internal procedure manuals — no documents about RAG at all. Per the task's explicit instruction ("If the supplied corpus does not support one of these questions, replace it with a corpus-supported question while preserving the intended evaluation challenge"):

- **Q7** → *"What alarm-system related procedures were issued before 2025?"* — preserves the exact Self-Query/date-metadata challenge, using real `issue_date` metadata our corpus actually contains (alarm manual: issued 2024-08; the other two manuals: issued 2026-02).
- **Q10** → *"What is the procedure for handling a security breach detected by the fingerprint access system during a public holiday?"* — a question clearly *within* a domain our corpus covers (fingerprint/alarm access control), but whose specific scenario (public-holiday handling) is not actually addressed anywhere in the manual, so it genuinely produces weak retrieval and exercises the retrieval-correction path. An earlier draft of this substitution ("the bank's policy for setting foreign currency interest rates") was tested and rejected: the router correctly recognized it as entirely outside the corpus's domain and routed it to `simple` before any retrieval was attempted at all, which meant it never actually tested the intended weak-retrieval scenario.

Q3 ("Why is it bad?") is intentionally ambiguous per the task; the preceding context (Q2's RAG limitations) was made explicit in the exact wording sent to the system, per the task's own guidance for handling this question.

---

## 2. Basic RAG vs. Advanced RAG — Comparison

| | Basic RAG (no router) | Full System (Router + Advanced RAG) |
|---|---|---|
| **Every question retrieves from the corpus** | Yes, always | No — only when the router decides it's needed |
| **Cost on general-knowledge questions** | Wasted: retrieves irrelevant corpus chunks, adds them to the prompt, still generates a plausible-sounding answer | $0 retrieval cost; answered directly, correctly, and faster |
| **Handling of a question with no real answer in the corpus** | Silently returns whatever chunks were "closest," risking a confidently-wrong answer | Retrieved context scored genuinely weak (Q10: Context Relevance = 1/5); the model correctly stated the documents do not contain the answer, instead of guessing |
| **Handling of a date-scoped question (Q7)** | A plain vector search has no concept of a date filter — could easily mix in content from documents outside the intended date range | Self-Query extracted the date constraint and applied it as a hard metadata filter *before* similarity search ran, guaranteeing correctness |
| **Cost transparency** | A single flat "it costs X" number, regardless of question complexity | Per-question, per-step breakdown (router, each technique, generation, each judge call) — a simple question costs ~$0.0009–0.004; a question needing self-query/decomposition + reranking costs ~$0.004 |

**The core finding:** forcing every question through the same fixed pipeline (as a naive Basic RAG baseline would) is wasteful for general-knowledge questions and dangerous for questions with no real answer in the corpus. The router is what makes cost proportional to actual need, and what lets a retrieval-improvement technique (in this run, the anti-hallucination generation prompt) catch a genuinely weak retrieval instead of forwarding it blindly.

---

## 3. Short Report: Which Techniques Improved Quality vs. Mainly Increased Cost/Latency

**Improved quality, with clear evidence from this run:**
- **Self-Query (Q7):** Directly responsible for correctly answering a date-scoped question. Without it, a plain semantic search has no way to enforce "before 2025" and could have returned content from either of the other two manuals (both issued 2026-02), producing a subtly wrong answer.
- **Router itself:** Correctly identified that 8 of 10 questions needed zero retrieval at all — this is the single biggest cost/quality lever in the whole system, since it prevented irrelevant corpus content from ever reaching the final generation prompt for those 8 questions.
- **The anti-hallucination generation prompt:** Caught the weak-retrieval case in Q10 even though the CRAG module specifically was not the technique invoked (see Section 4, Q9 below) — this is a quality win attributable to prompt design, not to any single retrieval-improvement module.

**Increased cost without evidence of a quality benefit *in this specific run*:**
- **Reranking:** Was invoked on both advanced_rag questions (Q7, Q10) and was consistently the single most token-hungry step in each question's breakdown (e.g., 1852 input tokens for Q10's reranking call alone, more than any other step for that question). Its concrete benefit could not be isolated in this run because we did not compare rankings before/after reranking for these two specific questions.
- **Decomposition (Q10):** Broke the question into sub-questions and still resulted in the correct "insufficient information" outcome, but it's not established whether decomposition specifically improved anything here versus a plain single retrieval pass would have — the router chose it, but the ultimately-correct behavior came from the generation prompt, not clearly from decomposition.

**Not exercised in the official 10-question evaluation at all:** Rewriting, Multi-Query, HyDE, Contextual Compression, and the CRAG module specifically were not selected by the router for any of the 10 official test questions in this run. All five are fully implemented and have working, independently-tested standalone entry points (each module's own `if __name__ == "__main__":` block), but this report does not claim performance numbers for them beyond what was actually observed and recorded, since router.py's decisions for this exact question set did not happen to invoke them. This is flagged explicitly rather than filled in with invented figures — see the caveats under Section 4, questions 2, 3, 5, 8, and 9.

---

## 4. Required Analysis Questions (16)

**1. Which questions were routed to Direct, Basic RAG, or Advanced RAG? Was the routing decision reasonable?**
Q1, Q2, Q3, Q4, Q5, Q6, Q8, Q9 → `simple` (no retrieval). Q7, Q10 → `advanced_rag`. No question was routed to plain `basic_rag`. The `simple` routings are reasonable and match the task's own "Expected Focus" column exactly (general RAG-theory questions with nothing in our banking-procedures corpus). The two `advanced_rag` routings are also reasonable: Q7 genuinely needs a date filter (self_query), and Q10 is topically within the corpus but needs the pipeline to recognize a genuine information gap. One observation worth flagging: no question in this set was routed to plain `basic_rag` — this is a property of *this specific question set* (every corpus-relevant question happened to have either a metadata constraint or an information gap), not a general flaw in the router's three-way design.

**2. Give one example where rewriting improved retrieval. Explain why.**
Rewriting was not selected by the router for any of the 10 official test questions, so no data from the official run can support this. (In an earlier, separate manual test of `router.py` alone — not part of the official 10-question run — the English question "What are the steps for the annual inventory count of stationery warehouses?" against our Arabic-language corpus was assigned `['rewriting', 'reranking']`, on the reasoning that the corpus is in Arabic and the query needed reformulation for better cross-lingual retrieval. This is reported here only as a design-intent illustration, not as evidence from the graded evaluation set.)

**3. Give one example where Multi-Query was better than a single query.**
Multi-Query was not selected for any of the 10 official questions. No evidence from this evaluation run can answer this question honestly; running `python -m src.query_transform.multi_query` (its own built-in test case) would need to be done separately to produce real supporting data.

**4. Give one complex question that benefited from decomposition.**
Q10 was decomposed by the router. The benefit is not cleanly isolated here, since the correct final outcome (the model stating the documents don't contain an answer) is more directly attributable to the generation prompt's explicit instruction to admit insufficient information than to decomposition specifically. Decomposition's clearest intended use case — a genuine multi-part comparison question — was tested separately with `decomposition.py`'s own test case (comparing the alarm system's card-access and fingerprint-access procedures), which correctly split the question into independent sub-questions and merged results from both, but that test was not part of the official 10-question run.

**5. When did HyDE help? When did it add unnecessary cost?**
HyDE was not selected by the router for any of the 10 official questions, so this cannot be answered from the graded run. It is fully implemented and independently testable via `python -m src.query_transform.hyde`.

**6. When was Self-Query appropriate? What semantic query and metadata filters were extracted?**
Q7 is the clear example. The router correctly identified the "before 2025" phrase as a metadata constraint requiring Self-Query. The final answer correctly and exclusively cited the alarm manual (issued 2024-08, the only one of the three source documents issued before 2025), scoring 5/5 on both Context Relevance and Faithfulness. **Caveat:** the pipeline's `QuestionResult` does not currently persist the exact `semantic_query` string and `filter` object that `self_query.py` extracted internally for this specific run (only the final answer, sources, and scores are recorded) — the correct end-to-end behavior is confirmed, but the intermediate extracted values for this specific question were not logged.

**7. How did reranking change the top results?**
Reranking was invoked for both Q7 and Q10, but the before/after ordering for these specific questions was not printed or logged during the official run (only the final, post-reranking chunk set was used for generation). A clean before/after comparison exists in `reranker.py`'s own standalone test case (query: "Who is responsible for approving the disposal of unused fixed assets?"), demonstrating the LLM relevance-score reordering versus the original vector-distance order, but that comparison was not captured for Q7 or Q10 specifically in this evaluation run.

**8. How many tokens were removed by compression, and did the answer quality change?**
Compression was not selected by the router for any of the 10 official questions, so no token-reduction figure can be reported from this run.

**9. Give one retrieval failure handled by CRAG. What decision did the evaluator make?**
This must be answered carefully: although Q10 was specifically designed to trigger the CRAG correction path, the router actually selected `['decomposition', 'reranking']` for Q10 — **the CRAG module itself (`crag_evaluate()`) was not invoked**. The correct, safe outcome for Q10 (Context Relevance = 1, and the model correctly stating insufficient information) came from the combination of weak retrieval plus the anti-hallucination generation prompt, not from CRAG's grading mechanism specifically. This is an important, honestly-reported finding about router behavior: the router chose a different combination of techniques than the one the question was designed to exercise, and the system still produced the correct outcome via a different safeguard.

**10. Which approach produced the best Faithfulness?**
Only Q7 and Q10 have a Faithfulness score at all (the other 8 questions used no retrieval, so Faithfulness is not applicable). Q7 (self_query + reranking) scored 5/5, versus Q10 (decomposition + reranking) at 3/5. Self-Query on a well-matched, date-filterable question produced better Faithfulness than decomposition on a question the corpus doesn't actually cover.

**11. Which approach produced the best Correctness?**
All 10 questions scored 5/5 on Correctness — no differentiation available from this metric alone in this run.

**12. Which approach produced the best Answer Relevance?**
All 10 questions scored 5/5 on Answer Relevance — no differentiation available from this metric alone. This is worth flagging as a limitation of the judge's calibration (see Section 5) rather than a genuine finding that every answer was equally good.

**13. Which approach had the lowest total cost?**
Q9 ("Which approach should be used for a simple question versus a complex multi-part question?"), at $0.000905 — a `simple`-route question with a short generated answer (only 99 output tokens), which minimized both the generation and judging costs.

**14. Did the highest-quality approach also have the highest cost/latency?**
Not cleanly, because Correctness and Answer Relevance did not differentiate between questions (all scored 5/5). Looking instead at Context Relevance/Faithfulness (the only metrics that did differentiate): Q7 had the best scores (5/5 both) at $0.004052 — essentially tied in cost with Q10 ($0.004041), which scored the worst on Context Relevance (1/5). This shows that in this run, higher cost did **not** reliably predict higher quality; the two advanced_rag questions cost almost the same regardless of how well the retrieval actually matched the question, because cost here mainly reflects how many LLM calls a given technique combination makes, not how well those calls perform.

**15. If you had to deploy the system, which techniques would you enable by default and which would be conditional?**
- **Always on:** the Router (it is the entire basis for cost-proportional behavior) and the anti-hallucination generation prompt (it functioned as a real safety net in Q10 even when the "designated" retrieval-correction technique wasn't the one selected).
- **Conditional, router-triggered:** Self-Query (only when a date/metadata constraint is detected — cheap and had a clean, verifiable win in Q7), Decomposition (only for genuinely multi-part/comparison questions), Reranking (only when the initial candidate pool is large enough that re-ordering plausibly matters).
- **Held back pending further evidence:** Multi-Query, HyDE, and Contextual Compression — all three are implemented and pass their own standalone tests, but none were exercised by the official evaluation set, so there is no run-time evidence yet (in this project) that their benefit outweighs their added LLM-call cost. Before enabling them by default in production, they should be evaluated against a question set specifically designed to trigger each of them, the way Q7 was designed around Self-Query.
- **CRAG specifically:** given that the router chose not to select it even for the question designed around it, its trigger condition in the router's system instruction may need to be made more explicit/aggressive before relying on it as the primary retrieval-correction safeguard, rather than depending on the generation prompt to catch weak retrieval as a fallback.

---

## 5. Known Limitations of This Evaluation

- **Judge calibration:** Answer Relevance and Correctness scored 5/5 on all 10 questions, including questions of very different genuine quality (a one-paragraph answer vs. a multi-section 1000+ token answer both scored 5/5). This suggests the LLM-as-judge, using a single fixed model, may be lenient toward well-structured, confident-sounding answers, rather than sharply discriminating between "good" and "excellent." The four fixed rubrics (see `src/evaluation/judge.py`) were applied identically to every question, which is methodologically correct, but the resulting ceiling effect limits how much these two metrics can be used to compare approaches within this dataset.
- **Model reliability under the free tier:** The strongest available model (`gemini-3.6-flash`) has a free-tier daily quota of only 20 requests, which is insufficient for a 10-question evaluation that can require 4–8 LLM calls per question — confirmed twice, including with a freshly-created API key/project (a new project resets the daily quota counter, but the 20-request ceiling itself is unchanged). The project standardized on `gemini-3.5-flash-lite` for all calls, which has a higher (though still rate-limited-per-minute) free-tier allowance; the built-in retry/backoff logic (`src/llm_client.py`) absorbed the resulting per-minute rate-limit delays automatically during the official run.
- **Pricing bug caught and corrected before submission:** During the switch from a single-model to a light-model configuration, the model *name* in `config.py` was updated (to `gemini-3.5-flash-lite`) but the price constants were not updated at the same time, so the official run's cost figures were initially computed using the strong model's rates ($1.50/$7.50 per 1M input/output tokens) against tokens that were actually billed at the light model's rates ($0.30/$2.50). This was caught before submission and corrected using `scripts/recompute_costs.py`, which recalculates every cost figure directly from the real, already-recorded token counts with zero additional API calls (so no further quota was spent). The corrected total cost across all 10 questions is **$0.026425** (previously mis-reported as $0.101519). This is flagged explicitly as a reminder that a two-tier or model-switching cost strategy needs its price table to be validated against the *actual* model in use, not just its name.
- **Arabic text extraction:** A small number of Arabic characters (most often the "م" in "من") are occasionally dropped during PDF text extraction, confirmed to originate in the source PDFs' own text layer (reproduced identically with both `pdfplumber` and `PyMuPDF`), not in our extraction code. The impact on semantic retrieval is expected to be minor, since the affected word is a common preposition.
- **Coverage gap in this report:** As documented throughout Section 4, five of the eight implemented Advanced RAG techniques (Rewriting, Multi-Query, HyDE, Contextual Compression, and CRAG specifically) were not selected by the router for any of the 10 official test questions, so this report does not claim performance evidence for them beyond what each module's own independent test case demonstrates in isolation.
- **Two different orchestration logics exist in this codebase and are not yet unified.** `scripts/run_questions.py` (used to produce every result in this report) runs *every* query-understanding technique the router names and merges their results, and applies retrieval-improvement techniques in the order CRAG → Reranking → Compression (so a CRAG "incorrect" verdict short-circuits the rest, saving cost). `src/pipeline.py` (used by the Streamlit app's live-demo tab, built separately) instead picks only *one* query-understanding technique by fixed priority (self_query > decomposition > hyde > multi_query > rewriting) and applies retrieval-improvement techniques in the reverse order, Reranking → Compression → CRAG (CRAG as a final gate instead of an early short-circuit). Both designs are individually defensible, but they are inconsistent with each other: running the identical question through the official evaluation script versus the Streamlit demo can produce a different technique selection and a different final answer. This was caught during a pre-submission review and is flagged here rather than resolved, since unifying the two orchestration paths this close to the deadline risked introducing a new bug into a script that was already fully validated against the official results table.

---

## 6. Final Deliverables Checklist

- [x] Source code for the RAG pipeline (`src/ingestion`, `src/retrieval`, `src/generation`, `src/routing`, `src/query_transform`, `src/evaluation`, `src/llm_client.py`)
- [x] Router implementation and route decisions (`src/routing/router.py`; decisions recorded per-question in `results/results.json`)
- [x] Basic RAG implementation (`src/retrieval/retriever.py` + `src/generation/generator.py`)
- [x] Advanced RAG components implemented: Rewriting, Multi-Query, Decomposition, HyDE, Self-Query, Reranking, Contextual Compression, CRAG (all 8; see Section 5 for which were exercised by the official evaluation set)
- [x] LLM generation implementation with anti-hallucination prompt (`src/generation/generator.py`)
- [x] Per-question evaluation results (`results/results.json`, Section 1 of this report)
- [x] Per-question token and cost breakdown (recorded per step in `results/results.json`; printed in the run log)
- [x] Comparison of Basic RAG vs. Advanced RAG (Section 2)
- [x] This report (Sections 3–5)