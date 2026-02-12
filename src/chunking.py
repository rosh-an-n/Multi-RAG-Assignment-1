"""
chunking.py
Splits the extracted text into smaller pieces (chunks) for embedding.

We use overlap between chunks to avoid breaking important sentences
right at the boundary. Without overlap you'd lose context at the edges
and retrieval quality drops. I tested with and without overlap and the
difference was noticeable.

Also includes section awareness — we tag chunks that look like they
come from the references/bibliography section so retrieval can
deprioritize them for non-citation queries.
"""

import re


# patterns that indicate a chunk is from references/bibliography
_REF_PATTERNS = [
    r'\[\d+\]',                        # [1], [23] etc - numbered citations
    r'proceedings\s+of',               # "Proceedings of ..."
    r'\b(ACL|EMNLP|NAACL|NeurIPS|ICML|ICLR|CVPR|AAAI|IJCAI)\b',  # conference names
    r'\bet\s+al\.\b',                  # "et al."
    r'(19|20)\d{2}\.\s',              # years like "2016. " in citation format
    r'pp\.\s*\d+',                     # "pp. 123" page numbers
    r'vol\.\s*\d+',                    # "vol. 5" volume numbers
    r'arXiv:\d+\.\d+',                # arXiv IDs
    r'\bReferences\b',                 # section header
    r'\bBibliography\b',              # section header
]

_REF_COMPILED = [re.compile(p, re.IGNORECASE) for p in _REF_PATTERNS]


def is_reference_chunk(text):
    """
    Checks if a chunk looks like its from the references section.
    
    We count how many reference patterns match. If 3+ patterns hit,
    its almost certainly a bibliography entry. This is a heuristic
    and won't catch every edge case, but works for typical papers.
    """
    hits = sum(1 for pat in _REF_COMPILED if pat.search(text))
    # 3+ pattern matches = very likely a reference entry
    return hits >= 3


def chunk_text(text, chunk_size=600, overlap=100):
    """
    Splits text into overlapping chunks.
    
    chunk_size = how many characters per chunk (600 works well for most papers)
    overlap = how many chars to repeat between consecutive chunks
    
    Returns a list of dicts with the chunk text and some position info.
    Each chunk is tagged with:
      - section_position: float 0.0 to 1.0 (where in the doc this chunk is)
      - is_reference: True if chunk looks like its from the bibliography
      - possible_reference: True if chunk is in the last 20% of the document
    """
    if not text or not text.strip():
        return []

    result = []
    pos = 0
    idx = 0
    total_len = len(text)

    while pos < total_len:
        end = pos + chunk_size

        # try to break at a sentence boundary instead of cutting mid-word
        if end < total_len:
            # look for a period, newline, or at least a space
            bp = text.rfind('. ', pos, end)
            if bp == -1:
                bp = text.rfind('\n', pos, end)
            if bp == -1:
                bp = text.rfind(' ', pos, end)
            if bp != -1 and bp > pos:
                end = bp + 1

        piece = text[pos:end].strip()

        if piece:
            # figure out where this chunk sits in the overall document
            section_pos = pos / total_len if total_len > 0 else 0.0

            result.append({
                "text": piece,
                "chunk_id": idx,
                "start": pos,
                "end": end,
                "section_position": round(section_pos, 3),
                "is_reference": is_reference_chunk(piece),
                "possible_reference": section_pos >= 0.80,
            })
            idx += 1

        # slide the window forward, but keep some overlap
        pos += chunk_size - overlap

    return result


def show_chunk_stats(chunk_list):
    """Print some basic stats about the chunks we made."""
    print(f"Total chunks: {len(chunk_list)}")
    if chunk_list:
        avg = sum(len(c['text']) for c in chunk_list) / len(chunk_list)
        ref_count = sum(1 for c in chunk_list if c.get('is_reference'))
        print(f"Avg chunk length: {avg:.0f} chars")
        print(f"Reference chunks detected: {ref_count}")
        print(f"\nFirst chunk:\n{chunk_list[0]['text'][:200]}...")


if __name__ == "__main__":
    test = "This is a test sentence. " * 100
    chunks = chunk_text(test)
    show_chunk_stats(chunks)
