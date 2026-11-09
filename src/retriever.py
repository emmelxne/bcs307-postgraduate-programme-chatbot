"""Step 2 – Information retrieval over handbook chunks.

Three interchangeable rankers so they can be compared in the evaluation:
  * tfidf  – TF-IDF (stemmed unigrams + bigrams) with cosine similarity
  * bm25   – Okapi BM25 over the same stemmed tokens
  * hybrid – weighted combination of the two (default)

Scores are then re-weighted using the programme / study mode detected in the
question, so "How long is the AI programme?" prefers the AI sections.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import analyzer, expand_query, normalise, query_concepts  # noqa: E402
from programmes import detect_in_query, detect_mode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.json"


@dataclass
class Hit:
    chunk: dict
    score: float       # final ranking score (after boosts)
    tfidf: float       # raw cosine similarity – used as the confidence signal


class Retriever:
    def __init__(self, chunks_path: Path = CHUNKS):
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.bm25 = BM25Okapi([normalise(t) for t in texts])
        self.vocab = set(self.vectorizer.vocabulary_)

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 5, method: str = "hybrid",
               programmes: set[str] | None = None) -> list[Hit]:
        expanded = expand_query(query)
        tfidf = cosine_similarity(self.vectorizer.transform([expanded]), self.matrix).ravel()
        bm25 = np.asarray(self.bm25.get_scores(normalise(expanded)), dtype=float)

        def norm(x):
            return x / x.max() if x.max() > 0 else x

        if method == "tfidf":
            base = tfidf.copy()
        elif method == "bm25":
            base = norm(bm25)
        else:
            base = 0.6 * norm(tfidf) + 0.4 * norm(bm25)

        # Programme / study-mode re-weighting
        progs = effective_programmes(query, programmes)
        mode = detect_mode(query)
        boost = np.ones(len(self.chunks))
        for i, c in enumerate(self.chunks):
            cp = set(c["programmes"])
            if progs and cp:
                boost[i] *= 1.15 if cp & progs else 0.5
            if mode and c["mode"]:
                boost[i] *= 1.2 if c["mode"] == mode else 0.8
        final = base * boost

        order = np.argsort(-final)[:k]
        return [Hit(self.chunks[i], float(final[i]), float(tfidf[i])) for i in order]

    # ------------------------------------------------------------------ #
    def term_coverage(self, query: str, hits: list[Hit]) -> float:
        """Share of the question's concepts found together in ONE retrieved chunk.

        Low coverage means the question asks about something (e.g. 'hostel fee',
        'programme leader') that no retrieved passage actually mentions.
        """
        concepts = [c for c in query_concepts(query) if not c <= GENERIC_TERMS]
        if not concepts:
            return 1.0
        best = 0.0
        for h in hits:
            toks = set(normalise(h.chunk["text"]))
            best = max(best, sum(bool(c & toks) for c in concepts) / len(concepts))
        return best


def effective_programmes(query: str, selected: set[str] | None) -> set[str]:
    """Question names a programme -> use it (narrowed by the UI selection when
    compatible, e.g. "AI" + selected MAI_CONV); otherwise use the UI selection."""
    qp, ui = detect_in_query(query), (selected or set())
    return (qp & ui or qp) if qp else ui


# Words that are in nearly every question and should not count towards coverage.
GENERIC_TERMS = set(normalise(
    "programme program master masters degree course ai artificial intelligence "
    "data science cybersecurity phd doctor computing computer science odl conventional "
    "full part time student candidate hello hi"))
