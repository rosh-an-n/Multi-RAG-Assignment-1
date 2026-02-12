"""
main.py
Ties everything together - loads the PDF, builds the index, and runs
queries through all the agents.

The pipeline goes:
  PDF -> chunks -> embeddings -> index -> query -> plan -> retrieve -> answer -> critique -> revise

You can run this directly:
    python -m src.main                        # uses default PDF
    python -m src.main path/to/paper.pdf      # or specify one
"""

import os

# fix segfault on Apple Silicon - FAISS and PyTorch fight over OpenMP threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys

# make imports work when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_loader import load_pdf, show_pdf_info
from src.chunking import chunk_text, show_chunk_stats
from src.embeddings import get_model, make_embeddings, build_index
from src.retrieval import find_top_chunks, build_context, show_results
from src.planner_agent import plan_query, check_complexity, show_plan
from src.answer_agent import build_answer, combine_answers, show_answer
from src.critic_agent import evaluate, show_eval
from src.revision_agent import run_revision_loop, show_revision
from src.memory import Memory
from src.utils import clean_author_output


def setup_pipeline(pdf_path):
    """
    Does all the one-time setup: load PDF, chunk it, embed, build index.
    Returns everything we need to start answering questions.
    """
    print("=" * 50)
    print("SETTING UP PIPELINE")
    print("=" * 50)

    # step 1 - get text from PDF
    print("\n[1] Loading PDF...")
    pdf_info = load_pdf(pdf_path)
    show_pdf_info(pdf_info)

    # step 2 - split into chunks
    print("\n[2] Chunking...")
    chunks = chunk_text(pdf_info["text"], chunk_size=600, overlap=100)
    show_chunk_stats(chunks)

    if not chunks:
        print("ERROR: Got no chunks. PDF might be empty or scanned.")
        sys.exit(1)

    # step 3 - create embeddings
    print("\n[3] Creating embeddings...")
    emb_model = get_model()
    vecs = make_embeddings(chunks, emb_model)

    # step 4 - build search index
    print("\n[4] Building FAISS index...")
    idx = build_index(vecs)

    # step 5 - init memory
    mem = Memory()

    print("\n" + "=" * 50)
    print("READY TO GO")
    print("=" * 50)

    return {
        "chunks": chunks,
        "index": idx,
        "model": emb_model,
        "memory": mem,
        "pdf_info": pdf_info,
    }


def answer_query(query, pipe):
    """
    Runs a single query through the whole agent pipeline.
    
    The flow is:
    1. Check complexity and plan subtasks if needed
    2. Retrieve relevant chunks for each subtask
    3. Generate answer
    4. Critique it
    5. Revise if score is too low
    """
    chunks = pipe["chunks"]
    idx = pipe["index"]
    model = pipe["model"]
    mem = pipe["memory"]

    print("\n" + "=" * 50)
    print(f"QUERY: {query}")
    print("=" * 50)

    # -- step 1: check complexity and plan --
    print("\n[1] PLANNER")
    
    # first figure out if this is a simple or complex query
    complexity = check_complexity(query)
    print(f"  Complexity: {'Complex' if complexity['is_complex'] else 'Simple'}")
    print(f"  Reason: {complexity['reason']}")
    
    if complexity["is_complex"]:
        # complex query - use the full planner
        plan = plan_query(query)
        show_plan(plan)
    else:
        # simple query - skip the full planner, just do direct retrieval
        plan = plan_query(query)
        print(f"  -> Direct retrieval (no decomposition needed)")
        show_plan(plan)

    # -- step 2: retrieve --
    print("\n[2] RETRIEVAL")
    all_found = []
    all_contexts = []
    retrieval_info = None

    for subtask in plan["subtasks"]:
        # pull out the actual search text from "Subtask N: ..."
        search_q = subtask.split(":", 1)[-1].strip()
        print(f"\n  Searching: {search_q}")
        
        found, r_info = find_top_chunks(search_q, idx, chunks, model, top_k=3)
        show_results(found, r_info)
        
        # keep the retrieval info from the main query (first subtask)
        if retrieval_info is None:
            retrieval_info = r_info
        
        all_found.extend(found)
        ctx = build_context(found)
        all_contexts.append(ctx)

    full_context = "\n\n".join(all_contexts)

    # warn if retrieval confidence is low
    if retrieval_info and retrieval_info.get("low_confidence"):
        print("\n  ⚠ LOW CONFIDENCE RETRIEVAL - chunks may not be relevant")

    # -- step 3: generate answer --
    print("\n[3] ANSWER AGENT")

    if len(plan["subtasks"]) > 1:
        # multi-part: answer each subtask then combine
        partial = []
        for i, (subtask, ctx) in enumerate(zip(plan["subtasks"], all_contexts)):
            sq = subtask.split(":", 1)[-1].strip()
            ans = build_answer(sq, ctx)
            partial.append(ans["answer"])
            print(f"\n  Subtask {i+1}: {ans['answer']}")

        merged = combine_answers(query, partial)
        first_answer = merged["answer"]
    else:
        ans = build_answer(query, full_context)
        first_answer = ans["answer"]

    print(f"\n  Initial answer: {first_answer}")

    # -- step 4: critique --
    print("\n[4] CRITIC")
    ev = evaluate(first_answer, full_context, query, retrieval_confidence=retrieval_info)
    show_eval(ev)

    # -- step 5: revise if needed --
    final = first_answer
    rev_result = None

    if ev["needs_revision"]:
        print("\n[5] REVISION")
        rev_result = run_revision_loop(first_answer, full_context, query, evaluate, retrieval_info=retrieval_info)
        final = rev_result["final_answer"]
        show_revision(rev_result)
    else:
        print("\n[5] REVISION - Skipped (score is fine)")

    # save to memory
    chunk_texts = [r["chunk"]["text"] for r in all_found]
    mem.save(
        query=query,
        subtasks=plan["subtasks"],
        chunks_used=chunk_texts,
        answer=first_answer,
        critic_score=ev["score"],
        feedback=ev["feedback"],
        final_answer=final,
        revisions=rev_result["rounds"] if rev_result else 0,
    )

    # -- clean up author output if needed --
    if retrieval_info and retrieval_info["intent"]["intent"] == "author":
        final = clean_author_output(final)

    # -- final output --
    output = {
        "query": query,
        "complexity": complexity,
        "plan": plan,
        "found": all_found,
        "retrieval_info": retrieval_info,
        "first_answer": first_answer,
        "evaluation": ev,
        "revision": rev_result,
        "final_answer": final,
    }

    print_output(output)
    return output


def print_output(out):
    """Print everything in a structured way."""
    print("\n" + "=" * 50)
    print("FINAL OUTPUT")
    print("=" * 50)

    print(f"\nUser Query: {out['query']}")
    print(f"Subtasks: {', '.join(out['plan']['subtasks'])}")
    
    if out["found"]:
        top_score = out["found"][0]["score"]
        print(f"Top Similarity Score: {top_score:.4f}")
    
    print(f"\nInitial Answer:\n{out['first_answer']}")
    
    ev = out["evaluation"]
    print(f"\nCritic Score: {ev['score']}/10")
    print(f"  - Relevance: {ev.get('relevance', '?')}")
    print(f"  - Grounding: {ev.get('grounding', '?')}")
    print(f"  - Metadata Noise: {ev.get('has_noise', False)}")
    
    if out["revision"]:
        print(f"\nRevised Answer ({out['revision']['rounds']} rounds):\n{out['final_answer']}")
    else:
        print(f"\nFinal Answer:\n{out['final_answer']}")

    print("\n" + "=" * 50)


    print(f"\nRetrieved chunks:")
    for r in out["found"]:
        w = f" ⚠ {r['warning']}" if "warning" in r else ""
        print(f"  [{r['rank']}] dist={r['score']:.4f}{w}")
        print(f"      {r['chunk']['text'][:100]}...")

    print(f"\nInitial answer:\n  {out['first_answer']}")
    print(f"\nCritic score: {out['evaluation']['score']}/10")
    print(f"Feedback:\n{out['evaluation']['feedback']}")

    if out["revision"]:
        print(f"\nRevised answer ({out['revision']['rounds']} round(s)):")
        print(f"  {out['final_answer']}")
    else:
        print(f"\nFinal answer (no revision):")
        print(f"  {out['final_answer']}")


def interactive(pipe):
    """
    Interactive mode - keep asking questions until user quits.
    Type 'history' to see past interactions.
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE MODE")
    print("Type a question, 'history', or 'quit'")
    print("=" * 50)

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if q.lower() == "history":
            pipe["memory"].show()
            continue

        answer_query(q, pipe)


def main():
    # figure out which PDF to use
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "research_paper.pdf"
        )

    if not os.path.exists(pdf_path):
        print(f"Can't find PDF at: {pdf_path}")
        print("Usage: python -m src.main <path_to_pdf>")
        print("Or put a PDF at data/research_paper.pdf")
        sys.exit(1)

    pipe = setup_pipeline(pdf_path)

    # run some demo queries first
    demos = [
        "What is the problem statement of this paper?",
        "Explain the methodology and limitations.",
        "What are the key contributions compared to existing work?",
    ]

    print("\n" + "=" * 50)
    print("RUNNING DEMO QUERIES")
    print("=" * 50)

    for q in demos:
        answer_query(q, pipe)

    # then let user ask their own questions
    interactive(pipe)


if __name__ == "__main__":
    main()
