"""
revision_agent.py
Takes the critic's feedback and tries to make the answer better.

The loop goes: answer -> critique -> revise -> re-critique
We cap it at 2 rounds to avoid wasting time since small models
don't always improve much with multiple passes anyway.

Improvements:
  - revision prompt now explicitly includes critic feedback points
  - early stop if revised answer is identical to previous answer
  - more targeted instructions for the LLM
"""

from src.answer_agent import get_llm, LLM_NAME
from src.utils import clean_author_output

MAX_ROUNDS = 2


def revise(original_ans, feedback, context, question, model_name=LLM_NAME):
    """
    Generate a revised answer based on the critic's feedback.
    Still grounded in the same context - we don't retrieve new stuff here.
    
    The prompt now explicitly tells the model what to fix based
    on the critic's specific complaints.
    """
    llm = get_llm(model_name)

    prompt = (
        "Revise the answer below based on the following critique. "
        "Follow these rules strictly:\n"
        "1. Only use information from the provided context\n"
        "2. Improve relevance - make sure you answer the actual question\n"
        "3. Remove unnecessary metadata (emails, departments, addresses)\n"
        "4. Extract only the specific information requested\n"
        "5. Be concise and direct\n\n"
        f"Question: {question}\n\n"
        f"Context: {context[:500]}\n\n"
        f"Original answer: {original_ans}\n\n"
        f"Critique:\n{feedback}\n\n"
        "Improved answer:"
    )

    out = llm(prompt)
    new_ans = out[0]["generated_text"].strip()

    return {
        "revised": new_ans,
        "original": original_ans,
        "feedback_used": feedback,
    }


def run_revision_loop(answer, context, question, eval_fn, model_name=LLM_NAME,
                      max_rounds=MAX_ROUNDS, retrieval_info=None):
    """
    The full revision loop. Keeps trying until score >= 7 or we hit max rounds.
    
    eval_fn should be the critic's evaluate() function.
    
    Early stop conditions:
      - score >= 7 (good enough)
      - max rounds reached
      - revised answer is identical to previous (no improvement possible)
    """
    history = []
    current = answer

    # pass retrieval_info to initial eval
    ev = eval_fn(current, context, question, model_name, retrieval_confidence=retrieval_info)
    history.append({
        "round": 0,
        "answer": current,
        "score": ev["score"],
        "feedback": ev["feedback"],
    })

    rounds_done = 0
    query_intent = retrieval_info.get("intent", {}).get("intent") if retrieval_info else None

    while ev["needs_revision"] and rounds_done < max_rounds:
        rounds_done += 1

        # -- Deterministic Fix for Author Queries --
        # If this is an author query, the LLM often struggles to remove metadata purely by prompt.
        # So we force-clean it using our regex utility.
        if query_intent == "author":
            clean_ans = clean_author_output(current)
            if clean_ans != current:
                print(f"  Revision round {rounds_done}: applied deterministic author cleanup")
                current = clean_ans
                # RE-EVALUATE immediately after cleanup
                ev = eval_fn(current, context, question, model_name, retrieval_confidence=retrieval_info)
                history.append({
                    "round": rounds_done,
                    "answer": current,
                    "score": ev["score"],
                    "feedback": ev.get("feedback", "Cleanup applied"),
                    "method": "cleanup"
                })
                # if score is good now, stop
                if not ev["needs_revision"]:
                    break
        
        # -- LLM Revision --
        # try to fix it with LLM
        rev = revise(current, ev["feedback"], context, question, model_name)
        new_answer = rev["revised"]

        # early stop: if revision didn't change anything, no point continuing
        if new_answer.strip() == current.strip():
            print(f"  Revision round {rounds_done}: no change detected, stopping early")
            break

        current = new_answer

        # check if it got better
        ev = eval_fn(current, context, question, model_name, retrieval_confidence=retrieval_info)
        history.append({
            "round": rounds_done,
            "answer": current,
            "score": ev["score"],
            "feedback": ev["feedback"],
            "method": "llm"
        })

        print(f"  Revision round {rounds_done}: score = {ev['score']}/10")

    first_score = history[0]["score"]
    last_score = history[-1]["score"]

    return {
        "final_answer": current,
        "rounds": rounds_done,
        "history": history,
        "got_better": last_score > first_score,
        "score_before": first_score,
        "score_after": last_score,
    }


def show_revision(result):
    print(f"\n{'='*50}")
    print("REVISION AGENT")
    print(f"{'='*50}")
    print(f"Rounds: {result['rounds']}")
    print(f"Score: {result['score_before']}/10 -> {result['score_after']}/10")
    print(f"Improved: {result['got_better']}")
    print(f"\nFinal answer:\n{result['final_answer']}")


if __name__ == "__main__":
    from src.critic_agent import evaluate

    ctx = "The methodology uses transformers with attention mechanisms."
    q = "What is the methodology?"
    ans = "It uses something."  # intentionally bad

    result = run_revision_loop(ans, ctx, q, evaluate)
    show_revision(result)

