"""
utils.py
Shared utility functions for cleanup and analysis.
Keeps the agent logic clean and avoids circular imports.
"""

import re

# patterns that indicate noisy metadata in an answer
_NOISE_PATTERNS = [
    (r'[\w.-]+@[\w.-]+\.\w+', "email address"),          # emails
    (r'\bDept\.?\b', "department reference"),               # Dept.
    (r'\bUniversity\b', "university name"),                 # University
    (r'\bInstitute\b', "institute name"),                   # Institute
    (r'\b\d{5,}\b', "long numeric string"),                # long numbers (zip, phone)
    (r'\bAddress\b', "address info"),                       # Address
    (r'\bPhone\b', "phone info"),                           # Phone
]

def detect_metadata_noise(answer):
    """
    Checks if the answer contains noisy metadata like emails,
    department names, university names, long numbers etc.
    Returns a noise score (0 to 1, where 0 = clean) and details.
    """
    found = []
    for pattern, label in _NOISE_PATTERNS:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            found.append(f"{label} ({len(matches)}x)")

    noise_score = min(1.0, len(found) * 0.2)  # each type adds 0.2
    return {
        "noise_score": noise_score,
        "details": found,
        "has_noise": len(found) > 0,
    }


def clean_author_output(text):
    """
    Heuristic cleanup for author queries.
    Extracts capitalized names and removes noise like emails, depts, etc.
    """
    # regex for Names (two or more capitalized words)
    # excludes things like "University of X", "Dept of Y" via heuristics later
    name_pat = r"([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+)"
    
    # common noise words in headers
    noise_words = {
        "University", "Institute", "Department", "Dept", "School", 
        "College", "Center", "Laboratory", "Systems", "Engineering",
        "Science", "Technology", "Research", "Abstract", "Introduction",
        "Conference", "Proceedings", "Vol", "Issue", "pp", "Email",
        "Address", "Received", "Accepted", "Copyright", "Key", "Words",
        "Keywords", "Table", "Figure", "Fig", "Algorithm"
    }

    candidates = re.findall(name_pat, text)
    
    cleaned_names = []
    seen = set()
    
    for name in candidates:
        name_clean = name.strip()
        # skip if contains noise words
        if any(w in name_clean for w in noise_words):
            continue
        # skip if it looks like an email or url
        if "@" in name_clean or "http" in name_clean:
            continue
        # check for duplicates
        if name_clean.lower() not in seen:
            cleaned_names.append(name_clean)
            seen.add(name_clean.lower())

    if not cleaned_names:
        return text  # fallback if we stripped everything

    return "The authors of the paper are:\n\n" + "\n".join(cleaned_names)
