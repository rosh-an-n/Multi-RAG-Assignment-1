"""
app.py
Streamlit UI for the agentic RAG system.
Upload a PDF, ask questions, see what each agent does.

Run with: streamlit run src/app.py
"""

import os

# fix segfault on Apple Silicon - FAISS and PyTorch fight over OpenMP threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.pdf_loader import load_pdf
from src.chunking import chunk_text
from src.embeddings import get_model, make_embeddings, build_index
from src.retrieval import find_top_chunks, build_context
from src.planner_agent import plan_query, check_complexity
from src.answer_agent import build_answer, combine_answers
from src.critic_agent import evaluate
from src.revision_agent import run_revision_loop
from src.memory import Memory


st.set_page_config(page_title="Agentic RAG", page_icon="📄", layout="wide")

st.title("📄 Multi-Agent RAG System")
st.markdown(
    "Upload a research paper and ask questions. "
    "The system uses a planner, retriever, answer generator, critic, "
    "and revision loop to build answers."
)

# session state init
if "pipe" not in st.session_state:
    st.session_state.pipe = None
if "mem" not in st.session_state:
    st.session_state.mem = Memory()
if "past_queries" not in st.session_state:
    st.session_state.past_queries = []


# sidebar - PDF upload
with st.sidebar:
    st.header("Setup")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded is not None and st.session_state.pipe is None:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                pdf_info = load_pdf(tmp_path)
                st.success(f"Loaded: {pdf_info['num_pages']} pages, {pdf_info['num_chars']} chars")

                chunks = chunk_text(pdf_info["text"], chunk_size=600, overlap=100)
                st.info(f"{len(chunks)} chunks created")

                emb_model = get_model()
                vecs = make_embeddings(chunks, emb_model)
                idx = build_index(vecs)

                st.session_state.pipe = {
                    "chunks": chunks,
                    "index": idx,
                    "model": emb_model,
                    "memory": st.session_state.mem,
                    "pdf_info": pdf_info,
                }
                st.success("Ready!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.pipe:
        st.success("✅ Document loaded")
        p = st.session_state.pipe
        st.caption(f"{p['pdf_info']['num_pages']} pages, {len(p['chunks'])} chunks")

    if st.session_state.past_queries:
        st.header("Previous Queries")
        for i, h in enumerate(st.session_state.past_queries):
            st.text(f"{i+1}. {h['query'][:50]}...")


# main area
if st.session_state.pipe is None:
    st.info("👈 Upload a PDF to get started")
else:
    query = st.text_input("Ask a question:", placeholder="e.g. What methodology was used?")

    if st.button("Ask", disabled=not query):
        pipe = st.session_state.pipe

        # complexity check + planning
        with st.expander("🗂 Step 1: Planning", expanded=True):
            complexity = check_complexity(query)
            st.markdown(f"**Complexity:** {'Complex' if complexity['is_complex'] else 'Simple'}")
            st.markdown(f"**Why:** {complexity['reason']}")
            
            plan = plan_query(query)
            st.markdown(f"**Compound query:** {plan['is_compound']}")
            st.markdown("**Subtasks:**")
            for t in plan["subtasks"]:
                st.markdown(f"- {t}")

        # retrieval
        with st.expander("🔍 Step 2: Retrieval", expanded=True):
            all_found = []
            all_ctx = []
            retrieval_info = None

            for subtask in plan["subtasks"]:
                sq = subtask.split(":", 1)[-1].strip()
                found, r_info = find_top_chunks(sq, pipe["index"], pipe["chunks"],
                                       pipe["model"], top_k=3)
                all_found.extend(found)
                all_ctx.append(build_context(found))
                if retrieval_info is None:
                    retrieval_info = r_info

            full_ctx = "\n\n".join(all_ctx)

            # show retrieval info
            if retrieval_info:
                st.markdown(f"**Query Intent:** {retrieval_info['intent']['intent']}")
                if retrieval_info.get('low_confidence'):
                    st.warning("⚠️ Low confidence retrieval - chunks may not be relevant")

            for r in all_found:
                w = f" ⚠️ {r['warning']}" if "warning" in r else ""
                ref = " 📚 [REF]" if r.get("is_reference_chunk") else ""
                st.markdown(f"**Rank {r['rank']}** (dist: {r['score']:.4f}){w}{ref}")
                st.text(r["chunk"]["text"][:200] + "...")
                st.divider()

        # answer
        with st.expander("💡 Step 3: Answer", expanded=True):
            with st.spinner("Generating..."):
                if len(plan["subtasks"]) > 1:
                    partial = []
                    for i, (subtask, ctx) in enumerate(zip(plan["subtasks"], all_ctx)):
                        sq = subtask.split(":", 1)[-1].strip()
                        ans = build_answer(sq, ctx)
                        partial.append(ans["answer"])
                        st.markdown(f"**Part {i+1}:** {ans['answer']}")
                    merged = combine_answers(query, partial)
                    first_answer = merged["answer"]
                else:
                    ans = build_answer(query, full_ctx)
                    first_answer = ans["answer"]

            st.markdown(f"**Answer:** {first_answer}")

        # critic
        with st.expander("🔎 Step 4: Critic", expanded=True):
            with st.spinner("Evaluating..."):
                ev = evaluate(first_answer, full_ctx, query, retrieval_confidence=retrieval_info)

            color = "green" if ev["score"] >= 7 else "orange" if ev["score"] >= 4 else "red"
            st.markdown(f"**Score:** :{color}[{ev['score']}/10]")
            st.markdown(f"**Relevance:** {ev.get('relevance', '?')}")
            st.markdown(f"**Grounding:** {ev.get('grounding', '?')}")
            st.markdown(f"**Needs revision:** {ev['needs_revision']}")

        # revision
        with st.expander("✏️ Step 5: Revision", expanded=True):
            final = first_answer
            rev = None
            
            if ev["needs_revision"]:
                with st.spinner("Revising..."):
                    rev = run_revision_loop(first_answer, full_ctx, query, evaluate, retrieval_info=retrieval_info)
                    final = rev["final_answer"]

                st.markdown(f"**Rounds:** {rev['rounds']}")
                st.markdown(f"**Score:** {rev['score_before']} ➡️ {rev['score_after']}")
                if rev['got_better']:
                    st.success("Score improved!")
                else:
                    st.info("Score didn't improve.")
                    
                st.text(final)
            else:
                st.info("Score is fine, no revision needed.")

        # cleanup author output if needed
        from src.utils import clean_author_output
        if retrieval_info and retrieval_info["intent"]["intent"] == "author":
            final = clean_author_output(final)

        # final answer
        st.divider()
        st.subheader("📋 Final Answer")
        st.markdown(final)

        # save
        st.session_state.mem.save(
            query=query,
            subtasks=plan["subtasks"],
            chunks_used=[r["chunk"]["text"] for r in all_found],
            answer=first_answer,
            critic_score=ev["score"],
            feedback=ev["feedback"],
            final_answer=final,
            revisions=rev["rounds"] if rev else 0,
        )

        st.session_state.past_queries.append({
            "query": query,
            "score": ev["score"],
            "answer": final,
        })
