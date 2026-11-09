"""Step 4 – Evaluation.

Runs every question in tests/test_questions.csv through the chatbot with each
retrieval method and labels the outcome:

  Correct        – answerable question, all expected keywords present
  Incomplete     – answerable, some (not all) expected keywords present
  Irrelevant     – answerable, answer given but none of the keywords present
  Missed         – answerable, but the bot said "not found"
  Correct (NF)   – unanswerable question, bot correctly said "not found"
  Unsupported    – unanswerable question, bot still produced an answer

Two question sets are used:
  tests/test_questions.csv    – development set (used while tuning thresholds,
                                synonyms and intents)
  tests/heldout_questions.csv – held-out set written separately and NOT used for
                                tuning, so it gives an honest accuracy estimate

Outputs results/<set>_<method>.csv and results/summary.md

Run:  python src/evaluate.py
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chatbot import Chatbot  # noqa: E402
from retriever import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SETS = {"dev": ROOT / "tests" / "test_questions.csv",
        "heldout": ROOT / "tests" / "heldout_questions.csv"}
OUT = ROOT / "results"
METHODS = ["tfidf", "bm25", "hybrid"]


def label(row, ans) -> str:
    if row["answerable"] == "no":
        return "Correct (NF)" if not ans.found else "Unsupported"
    if not ans.found:
        return "Missed"
    kws = [k for k in row["keywords"].split("|") if k]
    hit = sum(k.lower() in ans.text.lower() for k in kws)
    if hit == len(kws):
        return "Correct"
    return "Incomplete" if hit else "Irrelevant"


def run_set(name, path, retriever, summary):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    summary += [f"\n## {name} set ({len(rows)} questions)\n",
                "| Method | Correct | Accuracy | Outcomes |", "|---|---|---|---|"]
    cat_lines = []
    for method in METHODS:
        bot = Chatbot(retriever, method=method)
        out_rows, counts, by_cat = [], Counter(), defaultdict(lambda: [0, 0])
        for r in rows:
            a = bot.ask(r["question"])
            lab = label(r, a)
            ok = lab in ("Correct", "Correct (NF)")
            counts[lab] += 1
            by_cat[r["category"]][0] += ok
            by_cat[r["category"]][1] += 1
            out_rows.append({
                "id": r["id"], "category": r["category"], "question": r["question"],
                "expected": r["expected"], "chatbot_answer": a.text.replace("\n", " "),
                "source": " | ".join(a.sources), "confidence": f"{a.confidence:.3f}",
                "coverage": f"{a.coverage:.2f}", "result": lab, "correct": "Yes" if ok else "No",
            })
        with (OUT / f"{name}_{method}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        n_ok = counts["Correct"] + counts["Correct (NF)"]
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        summary.append(f"| {method} | {n_ok}/{len(rows)} | {n_ok / len(rows):.0%} | {outcomes} |")
        cat_lines.append(f"- **{method}** by category: " +
                         ", ".join(f"{c} {ok}/{n}" for c, (ok, n) in by_cat.items()))
        print(f"[{name}] {method:7s} {n_ok}/{len(rows)} = {n_ok / len(rows):.0%}  {dict(counts)}")
    summary += [""] + cat_lines


def main():
    OUT.mkdir(exist_ok=True)
    retriever = Retriever()
    summary = ["# Evaluation summary"]
    for name, path in SETS.items():
        run_set(name, path, retriever, summary)
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("Saved -> results/")


if __name__ == "__main__":
    main()
