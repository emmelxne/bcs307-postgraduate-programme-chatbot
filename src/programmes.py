"""Programme / study-mode detection.

The handbook covers six programmes, and many questions ("how long is the
programme?") have a different answer for each. We tag every chunk with the
programme(s) named in its section headings, and detect the programme named in
the user's question, so retrieval can prefer the right programme's sections.
"""
import re

PROGRAMMES = {
    "MDS": "Master of Data Science",
    "MAI_CONV": "Master in Artificial Intelligence (Conventional)",
    "MAI_ODL": "Master in Artificial Intelligence (ODL)",
    "MCS": "Master of Cybersecurity (ODL)",
    "PHD_COMP": "Doctor of Philosophy in Computing",
    "PHD_CS": "Doctor of Philosophy in Computer Science",
}

DEFAULT_PROGRAMME = "MAI_CONV"  # the programme this chatbot is built around


def _detect(text: str, is_heading: bool) -> set[str]:
    t = text.lower()
    found: set[str] = set()
    ai = bool(re.search(r"artificial intelligence|\bmai\b|\bai\b(?! tools?| based| driven)", t))
    if is_heading:
        # Headings name programmes explicitly; avoid tagging e.g. "Ethics and Cybersecurity".
        ai = bool(re.search(r"master (in|of) artificial intelligence|\bmai\b", t))
    if ai:
        odl = bool(re.search(r"\bodl\b|distance|online", t))
        conv = bool(re.search(r"conventional|on[- ]campus|physical", t))
        if odl and not conv:
            found.add("MAI_ODL")
        elif conv and not odl:
            found.add("MAI_CONV")
        else:
            found |= {"MAI_CONV", "MAI_ODL"}
    if re.search(r"data science|\bmds\b|\bhumds\b", t) and not re.search(r"for data science|statistics for", t):
        found.add("MDS")
    if re.search(r"master of cybersecurity|\bhumcs\b" if is_heading else r"master of cybersecurity|cyber ?security (master|programme|degree)|\bmcs\b|\bmaster.{0,15}cyber", t):
        found.add("MCS")
    if re.search(r"\bphd\b|doctor|doctoral", t):
        if "computer science" in t:
            found.add("PHD_CS")
        elif re.search(r"in computing", t):
            found.add("PHD_COMP")
        else:
            found |= {"PHD_COMP", "PHD_CS"}
    return found


def detect_in_heading(text: str) -> set[str]:
    return _detect(text, is_heading=True)


def detect_in_query(text: str) -> set[str]:
    return _detect(text, is_heading=False)


def detect_mode(text: str) -> str | None:
    t = text.lower()
    if re.search(r"part[- ]?time", t):
        return "part time"
    if re.search(r"full[- ]?time", t):
        return "full time"
    return None
