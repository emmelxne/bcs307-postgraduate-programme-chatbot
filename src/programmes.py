"""Programme / study-mode detection and question-scope routing.

The handbook covers six programmes, and many questions ("how long is the
programme?") have a different answer for each. Every chunk is tagged with the
programme(s) named in its section headings. Questions are checked for a named
programme (tolerating spacing variants and typos such as "mastes in cyber
security"), for undergraduate programmes (out of scope), and for study load
(full / part time).
"""
import difflib
import re

PROGRAMMES = {
    "MDS": "Master of Data Science",
    "MAI_CONV": "Master in Artificial Intelligence (Conventional)",
    "MAI_ODL": "Master in Artificial Intelligence (ODL)",
    "MCS": "Master of Cybersecurity (ODL)",
    "PHD_COMP": "Doctor of Philosophy in Computing",
    "PHD_CS": "Doctor of Philosophy in Computer Science",
}
SHORT = {
    "MDS": "Data Science",
    "MAI_CONV": "Master in AI · Conventional",
    "MAI_ODL": "Master in AI · ODL",
    "MCS": "Cybersecurity · ODL",
    "PHD_COMP": "PhD in Computing",
    "PHD_CS": "PhD in Computer Science",
}

DEFAULT_PROGRAMME = "MAI_CONV"  # the programme this chatbot is built around

# Words used to repair typos in programme names before detection.
_KEY_WORDS = ["master", "masters", "artificial", "intelligence", "cybersecurity", "security",
              "science", "computing", "computer", "doctor", "philosophy", "conventional",
              "distance", "bachelor", "undergraduate", "diploma"]


def fix_typos(text: str) -> str:
    """Normalise spacing variants and correct near-miss spellings of key words."""
    t = text.lower()
    t = re.sub(r"cyber[\s-]+security|cyber[\s-]*sec\b", "cybersecurity", t)
    t = re.sub(r"data[\s-]*sci\b|datascience", "data science", t)
    t = re.sub(r"\bph\.?\s?d\b", "phd", t)
    out = []
    for tok in re.findall(r"[a-z0-9]+|[^a-z0-9]+", t):
        if tok.isalpha() and len(tok) >= 5 and tok not in _KEY_WORDS:
            m = difflib.get_close_matches(tok, _KEY_WORDS, n=1, cutoff=0.8)
            tok = m[0] if m else tok
        out.append(tok)
    return "".join(out)


def _detect(text: str, is_heading: bool) -> set[str]:
    t = text.lower() if is_heading else fix_typos(text)
    found: set[str] = set()
    if is_heading:
        ai = bool(re.search(r"master (in|of) artificial intelligence|\bmai\b", t))
    else:
        ai = bool(re.search(r"artificial intelligence|\bmai\b|\bai\b(?! tools?| based| driven| generated)", t))
    if ai:
        odl = bool(re.search(r"\bodl\b|distance|online", t))
        conv = bool(re.search(r"conventional|on[- ]campus|physical|face[- ]to[- ]face", t))
        if odl and not conv:
            found.add("MAI_ODL")
        elif conv and not odl:
            found.add("MAI_CONV")
        else:
            found |= {"MAI_CONV", "MAI_ODL"}
    if re.search(r"data science|\bmds\b|\bhumds\b", t) and not re.search(r"for data science|statistics for", t):
        found.add("MDS")
    if is_heading:
        if re.search(r"master of cybersecurity|\bhumcs\b", t):
            found.add("MCS")
    elif re.search(r"master(s)? (of|in) cybersecurity|cybersecurity (master|programme|program|degree|course)"
                   r"|\bmcs\b|about cybersecurity\b|cybersecurity\??$|for cybersecurity|cybersecurity (entry|minimum|cgpa|odl)", t):
        found.add("MCS")
    if re.search(r"\bphd\b|doctor|doctoral", t):
        if "computer science" in t:
            found.add("PHD_CS")
        elif re.search(r"in computing|phd computing", t):
            found.add("PHD_COMP")
        else:
            found |= {"PHD_COMP", "PHD_CS"}
    return found


def detect_in_heading(text: str) -> set[str]:
    return _detect(text, is_heading=True)


def detect_in_query(text: str) -> set[str]:
    return _detect(text, is_heading=False)


def detect_load(text: str) -> str | None:
    """Study load: 'full time' / 'part time' (separate from study mode Conventional/ODL)."""
    t = text.lower()
    if re.search(r"part[- ]?time|while working|working adult|working full[- ]?time", t):
        return "part time"
    if re.search(r"full[- ]?time", t):
        return "full time"
    return None


def is_undergraduate_request(text: str) -> bool:
    """A question about an undergraduate programme itself (not about holding a bachelor's)."""
    t = fix_typos(text)
    if not re.search(r"\bbachelor|\bundergrad|\bdiploma\b|\bfoundation\b|\bdegree in ai\b", t):
        return False
    if re.search(r"master|phd|doctor|\bapply\b|entry|admission|requirement|cgpa|qualif|i have|with a|"
                 r"holding|direct|straight|after (my|a|the)|first class|honours", t):
        return False
    return True
