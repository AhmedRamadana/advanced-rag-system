"""
Streamlit Demo App

Why we need it:
Everything built so far runs from the terminal, which is fine for
development but not great for demonstrating the system to someone else
(like a mentor) in a live, visual way. This app gives two views:
1. A read-only table of the ten official test questions' results
   (loaded from results/results.json, produced by scripts/run_questions.py) -
   no API calls, safe to open anytime.
2. A live demo box where a NEW question can be typed in and run through
   the actual pipeline in real time - this DOES call the real API and
   consumes quota, so it's used deliberately, not automatically.

If we skipped this:
The project would only be demonstrable by reading terminal output or a
raw JSON file, which is much harder to present clearly to someone
reviewing the work.
"""

import json
import pandas as pd
import streamlit as st

import config
from src.pipeline import run_pipeline

st.set_page_config(page_title="Advanced RAG System - Housing Bank Procedures", layout="wide")
st.title("🏦 Advanced RAG System — Housing Bank Internal Procedures")

tab_results, tab_live = st.tabs(["📊 Results Table (10 Test Questions)", "💬 Live Demo"])

# =============================================================================
# TAB 1: Static results table (loaded from results.json - no API calls)
# =============================================================================
with tab_results:
    results_path = config.RESULTS_DIR / "results.json"

    if not results_path.exists():
        st.warning(
            "No results file found yet. Run `python scripts/run_questions.py` "
            "first to generate results/results.json."
        )
    else:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        total_cost = sum(r["total_cost_usd"] for r in results)
        total_latency = sum(r["total_latency_seconds"] for r in results)

        col1, col2, col3 = st.columns(3)
        col1.metric("Questions Evaluated", len(results))
        col2.metric("Total Cost (all questions)", f"${total_cost:.4f}")
        col3.metric("Total Latency", f"{total_latency:.1f}s")

        st.markdown("### Summary Table")
        summary_rows = []
        for r in results:
            summary_rows.append({
                "ID": r["question_id"],
                "Route": r["route"],
                "Techniques": ", ".join(r["techniques_named_by_router"]) or "-",
                "Context Rel.": r["context_relevance_score"] if r["context_relevance_score"] is not None else "N/A",
                "Faithfulness": r["faithfulness_score"] if r["faithfulness_score"] is not None else "N/A",
                "Answer Rel.": r["answer_relevance_score"],
                "Correctness": r["correctness_score"],
                "Cost ($)": round(r["total_cost_usd"], 5),
                "Latency (s)": round(r["total_latency_seconds"], 1),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("### Per-Question Details")
        for r in results:
            with st.expander(f"{r['question_id']}: {r['question']}"):
                if r.get("note"):
                    st.info(r["note"])
                st.markdown(f"**Route:** `{r['route']}`  \n**Reason:** {r['route_reason']}")
                st.markdown(f"**Answer:**\n\n{r['answer']}")
                if r["sources"]:
                    st.markdown("**Sources:**")
                    for s in r["sources"]:
                        st.markdown(f"- {s['source_file']}, page {s['page_number']}")

                step_rows = [
                    {"Step": s["step"], "In": s["input_tokens"], "Out": s["output_tokens"],
                     "Cost ($)": round(s["cost_usd"], 6), "Latency (s)": round(s["latency_seconds"], 2)}
                    for s in r["steps"] if s["executed"]
                ]
                st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: Live interactive demo - runs the REAL pipeline (uses real API quota)
# =============================================================================
with tab_live:
    st.warning("⚠️ This tab makes real API calls and uses your daily quota. Use it deliberately.")

    question = st.text_input("Type a question (English or Arabic):", "")
    run_clicked = st.button("Run through the pipeline", type="primary")

    if run_clicked and question.strip():
        with st.spinner("Running: routing → retrieval → generation..."):
            result = run_pipeline(question)

        st.markdown(f"**Route:** `{result.route}`  \n**Reason:** {result.route_reason}")
        if result.techniques_used:
            st.markdown(f"**Techniques used:** {', '.join(result.techniques_used)}")

        st.markdown("### Answer")
        st.markdown(result.answer)

        if result.sources:
            st.markdown("### Sources")
            for s in result.sources:
                st.markdown(f"- {s['source_file']}, page {s['page_number']}")

        st.markdown("### Cost & Latency Breakdown")
        call_rows = [
            {"Step": c.step, "In": c.input_tokens, "Out": c.output_tokens,
             "Cost ($)": round(c.cost_usd, 6), "Latency (s)": round(c.latency_seconds, 2)}
            for c in result.call_breakdown
        ]
        st.dataframe(pd.DataFrame(call_rows), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Cost", f"${result.total_cost_usd:.5f}")
        col2.metric("Total Latency", f"{result.total_latency_seconds:.1f}s")

    elif run_clicked:
        st.error("Please type a question first.")