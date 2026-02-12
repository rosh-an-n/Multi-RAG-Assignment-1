"""
critic_agent.py
Evaluates the answer the LLM generated to see if its actually good.

This is what makes the system "self-reflective" - it doesn't just generate
an answer and call it done. It checks whether the answer is:
  - grounded in the actual retrieved context
  - complete enough to answer the question
  - not hallucinating stuff
  - free of metadata noise (emails, dept names, etc.)

Scoring formula:
  final_score = (0.5 * relevance + 0.3 * grounding + 0.2 * completeness) * 10

Hard caps:
  - relevance < 40% -> score capped at 5
  - grounding < 60% -> score reduced by 2
"""

import re
from src.answer_agent import get_llm, LLM_NAME
from src.utils import detect_metadata_noise


def heuristic_checks(answer, context, question, query_intent=None):
    """
    Rule-based evaluation using weighted scoring.
    
    Weights: relevance=0.5, grounding=0.3, completeness=0.2
    Relevance is the dominant factor - if the answer doesn't address
    the question, nothing else matters much.
    
    query_intent: dict, optional. If provided, we use it to adjust scoring.
                  e.g. if intent='author', we look for names, not just keywords.
    """
    notes = {}
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "it", "this", "that", "with", "by",
            "?", ".", ",", "!"}

    # --- completeness score (0 to 1) ---
    if len(answer) < 20:
        completeness = 0.2
        notes["length"] = "Very short answer (< 20 chars), probably incomplete"
    elif len(answer) < 50:
        completeness = 0.5
        notes["length"] = "Kinda short, might be missing detail"
    elif len(answer) < 100:
        completeness = 0.7
        notes["length"] = "Decent length"
    else:
        completeness = 1.0
        notes["length"] = "Length seems fine"

    # penalize if answer says "not found" or similar
    no_info = ["does not contain", "not found", "no information",
               "cannot answer", "not mentioned", "not available"]
    if any(p in answer.lower() for p in no_info):
        notes["no_info"] = "Answer says info is missing - might need different retrieval"
        completeness *= 0.4
    else:
        notes["no_info"] = "Doesn't flag missing info"

    # --- metadata noise check ---
    noise = detect_metadata_noise(answer)
    if noise["has_noise"]:
        noise_detail = ", ".join(noise["details"])
        notes["noise"] = f"Metadata noise detected: {noise_detail}"
        # reduce completeness based on how much noise
        completeness *= (1.0 - noise["noise_score"] * 0.5)
    else:
        notes["noise"] = "No metadata noise"

    # --- grounding score (0 to 1) ---
    ans_words = set(answer.lower().split()) - stop
    ctx_words = set(context.lower().split()) - stop
    
    if len(ans_words) > 0:
        grounding = len(ans_words & ctx_words) / len(ans_words)
    else:
        grounding = 0

    if grounding < 0.3:
        notes["grounding"] = f"Low grounding ({grounding:.0%}) - possible hallucination"
    elif grounding < 0.6:
        notes["grounding"] = f"Moderate grounding ({grounding:.0%})"
    else:
        notes["grounding"] = f"Good grounding ({grounding:.0%})"

    # --- relevance score (0 to 1) ---
    # Smart intent-aware relevance
    relevance = 0.0
    
    is_author_intent = query_intent and query_intent.get("intent") == "author"
    
    if is_author_intent:
        # Special logic for author queries: look for names
        # We roughly check for "Capitalized Name" patterns
        # or if the answer contains commas separating words
        name_pat = r"[A-Z][a-z]+ [A-Z][a-z]+"
        names_found = re.findall(name_pat, answer)
        
        if len(names_found) >= 1:
            relevance = 0.8  # Good start
            if len(names_found) >= 2:
                relevance = 1.0
            notes["relevance"] = f"Found {len(names_found)} potential names for author query"
        else:
            relevance = 0.2
            notes["relevance"] = "Author query but no names found"
    else:
        # Default keyword overlap
        q_words = set(question.lower().split()) - stop
        if len(q_words) > 0:
            relevance = len(q_words & set(answer.lower().split())) / len(q_words)
        
        if relevance < 0.3:
            notes["relevance"] = f"Low relevance ({relevance:.0%}) - might not be answering the right question"
        elif relevance < 0.5:
            notes["relevance"] = f"Moderate relevance ({relevance:.0%})"
        else:
            notes["relevance"] = f"Good relevance ({relevance:.0%})"

    # --- weighted scoring ---
    # weights: relevance=0.5, grounding=0.3, completeness=0.2
    # relevance is the most important - if it doesn't answer the question, score should be low
    relevance_contrib = 0.5 * relevance
    grounding_contrib = 0.3 * grounding
    completeness_contrib = 0.2 * completeness

    raw_score = (relevance_contrib + grounding_contrib + completeness_contrib) * 10
    score = round(max(1, min(10, raw_score)))

    # scoring breakdown for debugging
    notes["scoring"] = (
        f"Breakdown: relevance({relevance:.2f}×0.5={relevance_contrib:.2f}) "
        f"+ grounding({grounding:.2f}×0.3={grounding_contrib:.2f}) "
        f"+ completeness({completeness:.2f}×0.2={completeness_contrib:.2f}) "
        f"= {raw_score:.1f}/10"
    )

    # --- hard caps ---
    if relevance < 0.40:
        score = min(score, 5)
        notes["cap"] = f"Score capped at 5 due to low relevance ({relevance:.0%})"
    if grounding < 0.60:
        score = max(1, score - 2)
        if "cap" in notes:
            notes["cap"] += f" | Also reduced by 2 for low grounding ({grounding:.0%})"
        else:
            notes["cap"] = f"Score reduced by 2 due to low grounding ({grounding:.0%})"

    return {
        "notes": notes,
        "score": score,
        "relevance": round(relevance, 3),
        "grounding": round(grounding, 3),
        "completeness": round(completeness, 3),
        "noise": noise,
    }


def llm_eval(answer, context, question, model_name=LLM_NAME):
    """
    Ask the LLM to evaluate the answer.
    Take this with a grain of salt - small models aren't great at this.
    """
    llm = get_llm(model_name)

    prompt = (
        "Evaluate this answer for completeness and accuracy. "
        "Point out anything missing or wrong.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context[:500]}\n\n"
        f"Answer: {answer}\n\n"
        "Evaluation:"
    )

    out = llm(prompt)
    return {"llm_feedback": out[0]["generated_text"].strip()}


def evaluate(answer, context, question, model_name=LLM_NAME, retrieval_confidence=None):
    """
    Full evaluation combining heuristics and LLM feedback.
    Returns a score (1-10) and whether the answer needs revision.
    
    retrieval_confidence is an optional dict from the retrieval step
    that tells us if the retrieved chunks were actually relevant.
    """
    # extract intent if available
    query_intent = retrieval_confidence.get("intent") if retrieval_confidence else None
    
    # heuristics are the main signal
    h = heuristic_checks(answer, context, question, query_intent=query_intent)
    
    # llm adds some qualitative feedback
    l = llm_eval(answer, context, question, model_name)

    score = h["score"]

    # factor in retrieval confidence if available
    if retrieval_confidence and retrieval_confidence.get("low_confidence"):
        score = min(score, 4)

    # build a readable summary
    feedback_lines = []
    for name, note in h["notes"].items():
        feedback_lines.append(f"  - {name}: {note}")
    feedback_lines.append(f"  - llm says: {l['llm_feedback']}")
    
    if retrieval_confidence and retrieval_confidence.get("low_confidence"):
        feedback_lines.append(f"  - retrieval: LOW CONFIDENCE (avg dist: {retrieval_confidence.get('avg_distance', '?'):.2f})")

    summary = "\n".join(feedback_lines)

    # decide if revision is needed
    needs_revision = (
        score < 7
        or h["relevance"] < 0.50
        or h["grounding"] < 0.40
        or (retrieval_confidence and retrieval_confidence.get("low_confidence"))
        or h["noise"]["has_noise"]
    )

    return {
        "score": score,
        "feedback": summary,
        "notes": h["notes"],
        "llm_feedback": l["llm_feedback"],
        "needs_revision": needs_revision,
        "relevance": h["relevance"],
        "grounding": h["grounding"],
        "completeness": h["completeness"],
        "has_noise": h["noise"]["has_noise"],
    }


def show_eval(ev):
    print(f"\n{'='*50}")
    print("CRITIC AGENT")
    print(f"{'='*50}")
    print(f"Score: {ev['score']}/10")
    print(f"Relevance: {ev.get('relevance', '?')}")
    print(f"Grounding: {ev.get('grounding', '?')}")
    print(f"Completeness: {ev.get('completeness', '?')}")
    print(f"Metadata noise: {ev.get('has_noise', False)}")
    print(f"Needs revision: {ev['needs_revision']}")
    print(f"\nFeedback:\n{ev['feedback']}")


if __name__ == "__main__":
    test_ans = "The methodology uses transfer learning with a CNN."
    test_ctx = "The study employs transfer learning using a convolutional neural network."
    test_q = "What methodology was used?"

    result = evaluate(test_ans, test_ctx, test_q)
    show_eval(result)

