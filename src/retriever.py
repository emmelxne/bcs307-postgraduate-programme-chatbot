"""Step 2 – Information retrieval over handbook chunks.

Three interchangeable rankers so they can be compared in the evaluation:
  * tfidf  – TF-IDF (stemmed unigrams + bigrams) with cosine similarity
  * bm25   – Okapi BM25 over the same stemmed tokens
  * hybrid – weighted combination of the two (default)

Programme scope is applied as a FILTER before ranking: only chunks tagged with
the active programme, plus shared (untagged) policy chunks, can be returned.
Chunks of other programmes are never candidates, so they cannot leak into an
answer. Study load (full / part time) is a softer re-weighting.
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
from programmes import detect_load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.json"


@dataclass
class Hit:
    chunk: dict
    score: float       # final ranking score (after re-weighting)
    tfidf: float       # raw cosine similarity – used as the confidence signal


class Retriever:
    def __init__(self, chunks_path: Path = CHUNKS):
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.bm25 = BM25Okapi([normalise(t) for t in texts])
        self.tokens = [set(normalise(t)) for t in texts]

    # ------------------------------------------------------------------ #
    def candidates(self, scope: set[str] | None, topics: set[str] | None = None,
                   tagged_only: bool = False) -> np.ndarray:
        """Boolean mask of chunks allowed for this programme scope / topic."""
        mask = np.ones(len(self.chunks), dtype=bool)
        for i, c in enumerate(self.chunks):
            cp = set(c["programmes"])
            if scope and cp and not cp & scope:
                mask[i] = False            # belongs to another programme
            if tagged_only and scope and not cp:
                mask[i] = False            # shared text not allowed for programme-only facts
            if topics and c["topic"] not in topics:
                mask[i] = False
        return mask

    def search(self, query: str, k: int = 5, method: str = "hybrid",
               scope: set[str] | None = None, topics: set[str] | None = None,
               tagged_only: bool = False) -> list[Hit]:
        mask = self.candidates(scope, topics, tagged_only)
        if not mask.any():
            return []
        expanded = expand_query(query)
        tfidf = cosine_similarity(self.vectorizer.transform([expanded]), self.matrix).ravel()
        bm25 = np.asarray(self.bm25.get_scores(normalise(expanded)), dtype=float)

        def norm(x):
            m = x[mask].max() if mask.any() else 0
            return x / m if m > 0 else x

        if method == "tfidf":
            base = tfidf.copy()
        elif method == "bm25":
            base = norm(bm25)
        else:
            base = 0.6 * norm(tfidf) + 0.4 * norm(bm25)

        load = detect_load(query)
        boost = np.ones(len(self.chunks))
        for i, c in enumerate(self.chunks):
            if scope and set(c["programmes"]) & scope:
                boost[i] *= 1.15           # programme-specific text beats shared text
            if load and c.get("load"):
                boost[i] *= 1.2 if c["load"] == load else 0.8
        final = np.where(mask, base * boost, -1.0)

        order = [i for i in np.argsort(-final)[:k] if mask[i]]
        return [Hit(self.chunks[i], float(final[i]), float(tfidf[i])) for i in order]

    # ------------------------------------------------------------------ #
    def term_coverage(self, query: str, hits: list[Hit]) -> float:
        """Share of the question's concepts found together in ONE retrieved chunk."""
        concepts = [c for c in query_concepts(query) if not c <= GENERIC_TERMS]
        if not concepts:
            return 1.0
        best = 0.0
        for h in hits:
            toks = self.tokens[h.chunk["id"]]
            best = max(best, sum(bool(c & toks) for c in concepts) / len(concepts))
        return best


# Words that are in nearly every question and should not count towards coverage.
GENERIC_TERMS = set(normalise(
    "programme program master masters degree course ai artificial intelligence "
    "data science cybersecurity phd doctor computing computer science odl conventional "
    "full part time student candidate hello hi"))
