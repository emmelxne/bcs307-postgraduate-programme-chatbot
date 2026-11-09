"""Step 4 – Evaluation.

Runs each question set through the chatbot with each retrieval method and
labels the outcome. Question sets:

  tests/test_questions.csv    – development set (used while building the rules)
  tests/heldout_questions.csv – first held-out set (now seen; kept for comparison)
  tests/audit_questions.csv   – 31 audit cases from the team's browser testing,
                                including clarification / switch / scope cases
  tests/final_questions.csv   – (optional) fresh final set written by the team
                                and frozen BEFORE running; this is the score to report

CSV columns: id, category, question, expected, keywords (| separated, all
required), and either
  * answerable = yes/no   (older sets), or
  * expected_status = answer | related | not_found | clarify | switch | scope | meta
  * programme = selected programme key (blank = MAI_CONV, ANY = no programme)
  * forbidden = text that must NOT appear (e.g. another programme's CGPA)

Outcome labels:
  Correct        – expected type of reply, all keywords, nothing forbidden
  Incomplete     – right type, some keywords missing
  Irrelevant     – right type, none of the keywords
  Wrong scope    – forbidden text present (e.g. another programme's rule)
  Missed         – should answer, but abstained / asked instead
  Unsupported    – should abstain, but produced an answer
  Wrong reply    – any other mismatch (e.g. answered instead of asking)

Keyword checks are a quick regression signal. For the final reported score,
also judge each full answer and its sources manually (see README).

Run:  python src/evaluate.py
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chatbot import Chatbot  # noqa: E402
from programmes import DEFAULT_PROGRAMME  # noqa: E402
from retriever import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SETS = {"dev": "test_questions.csv", "heldout": "heldout_questions.csv",
        "audit": "audit_questions.csv", "final": "final_questions.csv"}
METHODS = ["tfidf", "bm25", "hybrid"]
ANSWERED = {"answer", "related"}
ABSTAIN = {"not_found", "scope"}


def expected_statuses(row) -> set[str]:
    if row.get("expected_status"):
        s = row["expected_status"]
        return ANSWERED if s in ANSWERED else ABSTAIN if s in ABSTAIN else {s}
    return ANSWERED if row["answerable"] == "yes" else ABSTAIN


def label(row, ans) -> str:
    exp = expected_statuses(row)
    full = " ".join([ans.text, *ans.sources, *(o["label"] for o in ans.options)])
    forbidden = [f for f in (row.get("forbidden") or "").split("|") if f]
    if any(re.search(re.escape(f).replace("RM", r"RM ?\d"), full) for f in forbidden):
        return "Wrong scope"
    if ans.status not in exp:
        if exp == ANSWERED and ans.status in ABSTAIN | {"clarify"}:
            return "Missed"
        if exp == ABSTAIN and ans.status in ANSWERED:
            return "Unsupported"
        return "Wrong reply"
    kws = [k for k in row["keywords"].split("|") if k]
    hit = sum(k.lower() in full.lower() for k in kws)
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
            prog = (r.get("programme") or DEFAULT_PROGRAMME).strip()
            a = bot.ask(r["question"], programme=None if prog == "ANY" else prog)
            switched = ""
            if a.status == "switch" and "switch" not in expected_statuses(r):
                # Older sets name the programme in the question: act like a user who
                # clicks the first "Switch to ..." button, then re-ask.
                target = a.options[0]["value"]
                switched = f"(switched to {target}) "
                a = bot.ask(r["question"], programme=target)
            lab = label(r, a)
            ok = lab == "Correct"
            counts[lab] += 1
            by_cat[r["category"]][0] += ok
            by_cat[r["category"]][1] += 1
            out_rows.append({
                "id": r["id"], "category": r["category"], "programme": prog, "question": r["question"],
                "expected": r["expected"], "status": switched + a.status, "chatbot_answer": a.text.replace("\n", " "),
                "options": " | ".join(o["label"] for o in a.options), "sources": " | ".join(a.sources),
                "confidence": f"{a.confidence:.3f}", "coverage": f"{a.coverage:.2f}",
                "result": lab, "correct": "Yes" if ok else "No", "manual_verdict": "",
            })
        with (OUT / f"{name}_{method}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        n_ok = counts["Correct"]
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        summary.append(f"| {method} | {n_ok}/{len(rows)} | {n_ok / len(rows):.0%} | {outcomes} |")
        cat_lines.append(f"- **{method}** by category: " +
                         ", ".join(f"{c} {ok}/{n}" for c, (ok, n) in by_cat.items()))
        print(f"[{name}] {method:7s} {n_ok}/{len(rows)} = {n_ok / len(rows):.0%}  {dict(counts)}")
    summary += [""] + cat_lines


def main():
    OUT.mkdir(exist_ok=True)
    retriever = Retriever()
    summary = ["# Evaluation summary",
               "", "Automatic keyword/status checks. Fill the `manual_verdict` column in the CSVs "
               "for the final reported score."]
    for name, fname in SETS.items():
        path = ROOT / "tests" / fname
        if path.exists():
            run_set(name, path, retriever, summary)
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("Saved -> results/")


if __name__ == "__main__":
    main()
