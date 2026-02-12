"""
retrieval.py
Searches the FAISS index to find chunks similar to the user's question.

This is the core of RAG - if we get bad chunks here, the answer will be bad too.

Updates:
  - query intent detection (e.g. author queries get special treatment)
  - section-aware scoring (reference chunks get penalized for non-citation queries)
  - proper similarity threshold enforcement with low-confidence warnings
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# thresholds for L2 distance
DISTANCE_THRESHOLD = 1.2       # above this = questionable match
LOW_CONFIDENCE_THRESHOLD = 1.4  # above this = almost certainly irrelevant

# penalty added to reference chunks for non-citation queries
REFERENCE_PENALTY = 0.5

# boost (subtracted from distance) for early-document chunks on author queries
AUTHOR_EARLY_BOOST = 0.3


# keywords that signal different query intents
_AUTHOR_PATTERNS = [
    r'\bauthor', r'\bwritten\s+by', r'\bwho\s+wrote', r'\bwho\s+are\s+the',
    r'\bname\s+of\s+author', r'\bresearcher', r'\bwho\s+published',
]
_AUTHOR_COMPILED = [re.compile(p, re.IGNORECASE) for p in _AUTHOR_PATTERNS]


def detect_query_intent(query):
    """
    Figures out what kind of info the user is looking for.
    Right now we just detect author-related queries, but this
    can be extended for other intents later.
    
    Returns a dict with intent type and any special flags.
    """
    q = query.lower().strip()

    # check for author-related intent
    for pat in _AUTHOR_COMPILED:
        if pat.search(q):
            return {
                "intent": "author",
                "boost_early_chunks": True,
                "penalize_references": True,
            }

    # check for citation/reference intent (where we WANT the references)
    citation_words = ["cite", "citation", "reference", "bibliography", "cited"]
    if any(w in q for w in citation_words):
        return {
            "intent": "citation",
            "boost_early_chunks": False,
            "penalize_references": False,
        }

    # default - no special treatment
    return {
        "intent": "general",
        "boost_early_chunks": False,
        "penalize_references": True,  # still deprioritize refs for general queries
    }


def find_top_chunks(query, index, chunk_list, model, top_k=3):
    """
    Find the top_k closest chunks to the query.
    
    Now includes:
      - query intent detection (author queries boost early chunks)
      - section-aware scoring (reference chunks get penalized)
      - low confidence detection when all matches are too distant
    
    Returns a list of dicts with the chunk, its adjusted score,
    its rank, and a low_confidence flag.
    """
    intent = detect_query_intent(query)

    # embed the query with same model we used for chunks
    q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # retrieve more than needed so we have room after re-ranking
    fetch_k = min(top_k * 3, len(chunk_list))
    dists, idxs = index.search(q_vec, fetch_k)

    candidates = []
    for dist, i in zip(dists[0], idxs[0]):
        if i == -1:
            continue  # faiss returns -1 when theres not enough vectors

        chunk = chunk_list[i]
        adjusted_score = float(dist)

        # --- section-aware scoring ---
        is_ref = chunk.get("is_reference", False)
        is_possible_ref = chunk.get("possible_reference", False)
        chunk_id = chunk.get("chunk_id", i)
        total_chunks = len(chunk_list)

        # penalize reference chunks for non-citation queries
        if intent["penalize_references"] and (is_ref or is_possible_ref):
            adjusted_score += REFERENCE_PENALTY
            if is_ref:
                # extra penalty for confirmed reference chunks
                adjusted_score += 0.2

        # boost early-document chunks for author queries
        if intent["boost_early_chunks"]:
            # chunks from first ~10% of document get a boost
            position = chunk.get("section_position", chunk_id / max(total_chunks, 1))
            if position < 0.10:
                adjusted_score -= AUTHOR_EARLY_BOOST
            elif position < 0.20:
                adjusted_score -= AUTHOR_EARLY_BOOST * 0.5

        candidates.append({
            "chunk": chunk,
            "raw_score": float(dist),
            "score": adjusted_score,
            "index": i,
        })

    # --- force-include first-page chunks for author queries ---
    # author names appear at the very start of papers, but FAISS can't
    # semantically match "name of authors?" to "John Smith, Jane Doe"
    # so we inject the first few chunks with a very low synthetic score
    if intent["boost_early_chunks"]:
        seen_indices = {c["index"] for c in candidates}
        num_inject = min(3, len(chunk_list))  # first 3 chunks
        for i in range(num_inject):
            if i not in seen_indices:
                candidates.append({
                    "chunk": chunk_list[i],
                    "raw_score": 0.0,  # not from FAISS
                    "score": 0.1 * (i + 1),  # very low = high priority
                    "index": i,
                })
            else:
                # already in candidates, just make sure it ranks high
                for c in candidates:
                    if c["index"] == i:
                        c["score"] = min(c["score"], 0.1 * (i + 1))
                        break


    # re-rank by adjusted score
    candidates.sort(key=lambda x: x["score"])

    # take top_k after re-ranking
    top = candidates[:top_k]

    # check if all results are low confidence
    all_low_confidence = all(c["score"] > LOW_CONFIDENCE_THRESHOLD for c in top) if top else True

    found = []
    for rank, c in enumerate(top):
        item = {
            "chunk": c["chunk"],
            "score": c["score"],
            "raw_score": c["raw_score"],
            "rank": rank + 1,
        }

        # flag quality issues
        if c["score"] > LOW_CONFIDENCE_THRESHOLD:
            item["warning"] = "Low confidence - match is likely irrelevant"
        elif c["score"] > DISTANCE_THRESHOLD:
            item["warning"] = "Moderate confidence - might not be fully relevant"

        if c["chunk"].get("is_reference"):
            item["is_reference_chunk"] = True

        found.append(item)

    return found, {
        "intent": intent,
        "low_confidence": all_low_confidence,
        "avg_distance": np.mean([c["score"] for c in top]) if top else 999,
    }


def build_context(found_chunks):
    """
    Combine the retrieved chunks into one string that we'll feed to the LLM.
    This is basically the 'knowledge' the model gets to work with.
    """
    parts = []
    for r in found_chunks:
        header = f"[Chunk {r['rank']}, dist={r['score']:.4f}]"
        parts.append(f"{header}\n{r['chunk']['text']}")
    return "\n\n".join(parts)


def show_results(found_chunks, retrieval_info=None):
    """Print what we found in a readable way."""
    print(f"\n{'='*50}")
    print(f"Found {len(found_chunks)} chunks:")
    if retrieval_info:
        print(f"  Intent: {retrieval_info['intent']['intent']}")
        print(f"  Low confidence: {retrieval_info['low_confidence']}")
    print(f"{'='*50}")

    for r in found_chunks:
        ref_tag = " [REF]" if r.get("is_reference_chunk") else ""
        print(f"\n--- Rank {r['rank']} (dist: {r['score']:.4f}, raw: {r.get('raw_score', r['score']):.4f}){ref_tag} ---")
        if "warning" in r:
            print(f"  ⚠ {r['warning']}")
        print(f"  {r['chunk']['text'][:200]}...")


if __name__ == "__main__":
    from embeddings import get_model, make_embeddings, build_index

    model = get_model()
    test_chunks = [
        {"text": "The methodology involves training a neural net on labeled data.", "chunk_id": 0, "section_position": 0.3, "is_reference": False, "possible_reference": False},
        {"text": "Limitations include high compute cost and data requirements.", "chunk_id": 1, "section_position": 0.6, "is_reference": False, "possible_reference": False},
        {"text": "Results show 95% accuracy on the test set.", "chunk_id": 2, "section_position": 0.5, "is_reference": False, "possible_reference": False},
        {"text": "Related work covers transformers and attention.", "chunk_id": 3, "section_position": 0.2, "is_reference": False, "possible_reference": False},
    ]
    vecs = make_embeddings(test_chunks, model)
    idx = build_index(vecs)

    results, info = find_top_chunks("What is the methodology?", idx, test_chunks, model, top_k=2)
    show_results(results, info)
