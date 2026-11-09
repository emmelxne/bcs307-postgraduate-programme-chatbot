"""Step 3 – Answer generation (extractive, grounded in the handbook).

1. Retrieve the top chunks (retriever.py).
2. Decide whether the handbook actually covers the question
   (confidence = TF-IDF cosine of the best chunk + query-term coverage).
   If not -> fixed "not found" message (no invented answers).
3. Detect the question intent (duration, credits, subjects, ...).
4. Select the most relevant sentences / table rows from the retrieved chunks
   and return them with the source section.

No generative model is used, so every answer sentence is copied from the
handbook and can be traced to its section.
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import expand_query  # noqa: E402
from programmes import DEFAULT_PROGRAMME, PROGRAMMES, detect_in_query  # noqa: E402
from retriever import Retriever, effective_programmes  # noqa: E402

NOT_FOUND = "The requested information was not found in the available programme documents."

# Confidence thresholds (tuned on the test set – see README / report)
MIN_TFIDF = 0.08
MIN_COVERAGE = 0.5

# Answer-type check: questions asking for an amount of money must be answered
# with a sentence that actually contains an amount (RM...).
MONEY_Q = r"\bfees?\b|\bcosts?\b|\bprice\b|how much (?:is|does|do|will)\b"
MONEY_A = r"RM ?\d"

# Intent -> regex that marks the sentence/table row that answers it
INTENTS = {
    "duration": (r"\bhow long\b|\bduration\b|\byears?\b|\bmonths?\b|\blength\b", r"Duration"),
    "credits": (r"how many credit|credits? (?:are |is )?(?:required|needed)|total credit|credit hours",
                r"(?i)total credit hours(?! for)|programme — total credit"),
    "subjects": (r"(what|which|list)( of)? (the )?(subjects|modules|courses)|(subjects|modules|courses) (will|do|are|in)\b"
                 r"|\belectives?\b|core (subjects|modules)|programme structure|study in",
                 r"Subject Name|Name of Course"),
}


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    coverage: float = 0.0
    intent: str = "general"
    hits: list = field(default_factory=list)


def short_section(path: str) -> str:
    parts = path.split(" > ")
    return parts[-1] if len(parts) == 1 else f"{parts[-2]} › {parts[-1]}"


def detect_intent(q: str) -> str:
    ql = q.lower()
    for name, (qpat, _) in INTENTS.items():
        if re.search(qpat, ql):
            return name
    return "general"


class Chatbot:
    def __init__(self, retriever: Retriever | None = None, method: str = "hybrid"):
        self.r = retriever or Retriever()
        self.method = method

    def ask(self, question: str, programme: str | None = DEFAULT_PROGRAMME) -> Answer:
        question = question.strip()
        if not question:
            return Answer("Please type a question.", False)
        progs = {programme} if programme else None
        hits = self.r.search(question, k=6, method=self.method, programmes=progs)
        top = hits[0]
        coverage = self.r.term_coverage(question, hits[:3])
        if top.tfidf < MIN_TFIDF or coverage < MIN_COVERAGE:
            return Answer(NOT_FOUND, False, confidence=top.tfidf, coverage=coverage, hits=hits)

        intent = detect_intent(question)
        lines, sources = [], []

        if intent == "subjects":
            ranked = self.r.search(question + " programme structure subject name",
                                   k=len(self.r.chunks), method=self.method, programmes=progs)
            lines, sources = self._subjects(ranked)
        elif intent in ("duration", "credits"):
            lines, sources = self._intent_rows(hits, INTENTS[intent][1],
                                               effective_programmes(question, progs))

        if not lines:  # general extractive answer
            intent = "general"
            lines, sources = self._extract(question, hits[:2])

        if not lines:
            return Answer(NOT_FOUND, False, confidence=top.tfidf, coverage=coverage, hits=hits)
        if re.search(MONEY_Q, question.lower()):
            money = [l for l in lines if re.search(MONEY_A, l)]
            if not money:
                return Answer(NOT_FOUND, False, confidence=top.tfidf, coverage=coverage,
                              intent="amount", hits=hits)
            lines = money

        # Tell the user which programme the answer refers to when they didn't say
        prefix = ""
        if not detect_in_query(question) and programme and set(hits[0].chunk["programmes"]) == {programme}:
            prefix = f"(Answer for the **{PROGRAMMES[programme]}**.)\n\n"
        text = prefix + "\n".join(f"- {l}" for l in lines)
        return Answer(text, True, sources, top.tfidf, coverage, intent, hits)

    # ------------------------------------------------------------------ #
    def _extract(self, question, hits, max_sentences=4):
        """Rank individual sentences/rows of the top chunks by similarity."""
        units, owners = [], []
        for h in hits:
            for u in h.chunk["units"]:
                units.append(u)
                owners.append(h.chunk["section"])
        vec = self.r.vectorizer
        sims = cosine_similarity(vec.transform([expand_query(question)]), vec.transform(units)).ravel()
        best = sims.max() if len(sims) else 0
        chosen = [i for i in sims.argsort()[::-1][:max_sentences] if sims[i] >= max(0.05, 0.35 * best)]
        chosen.sort()  # keep document order for readability
        # include a following list item if the chosen unit introduces a list
        out = []
        for i in chosen:
            out.append(units[i])
            if units[i].rstrip().endswith(":") and i + 1 < len(units) and i + 1 not in chosen:
                out.append(units[i + 1])
        sources = list(dict.fromkeys(owners[i] for i in chosen))
        return out, sources

    def _intent_rows(self, hits, pattern, progs):
        rows, sources = [], []
        for h in hits:
            cp = set(h.chunk["programmes"])
            if progs and cp and not cp & progs:
                continue  # row belongs to a different programme
            for u in h.chunk["units"]:
                if re.search(pattern, u):
                    label = short_section(h.chunk["section"]).split(" › ")[-1]
                    line = f"{label}: {u.rstrip('.')}"
                    if line not in rows:
                        rows.append(line)
                        sources.append(h.chunk["section"])
            if len(rows) >= 4:
                break
        return rows, list(dict.fromkeys(sources))

    def _subjects(self, hits):
        """List subjects from the best-matching programme-structure section."""
        best = next((h for h in hits if any(re.search(r"Subject Name|Name of Course", u)
                                             for u in h.chunk["units"])), None)
        if best is None:
            return [], []
        section = best.chunk["section"]
        rows = []
        for c in self.r.chunks:  # whole section, not just one chunk
            if c["section"] != section:
                continue
            for u in c["units"]:
                f = {k.strip(): v.strip() for k, v in re.findall(r"([A-Za-z /.]+?): ([^;]+?)(?:;|\.$)", u)}
                name = f.get("Subject Name") or f.get("Name of Course")
                if not name:
                    continue
                code = f.get("Subject Code") or f.get("Code", "")
                sem = f.get("Semester") or f.get("Semester / Year Offered", "")
                cls = f.get("Classification", "")
                cr = f.get("Credit Hours") or f.get("Credit") or ""
                detail = ", ".join(x for x in [cls, f"{cr} credits" if cr else ""] if x)
                rows.append(f"{(sem + ' — ') if sem else ''}{code} {name} ({detail})")
        return rows, [section]


if __name__ == "__main__":
    bot = Chatbot()
    print("Handbook chatbot – type a question (blank line to quit)")
    while True:
        q = input("\nYou: ")
        if not q.strip():
            break
        a = bot.ask(q)
        print("\nBot:\n" + a.text)
        if a.sources:
            print("Source: " + " | ".join(a.sources))
