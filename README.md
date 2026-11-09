# BCS307 NLP Project – Postgraduate Programme Chatbot

A question-answering chatbot for the **HELP University FCDT Postgraduate Handbook 2026/2027**
(primary focus: *Master in Artificial Intelligence – Conventional*; all six programmes are indexed).

Pipeline: **Handbook (.docx) → text preparation → TF-IDF / BM25 retrieval → extractive answer → evaluation**

No generative model is used: every answer sentence is copied from the handbook and shown with its source section.
If the handbook does not cover a question, the bot replies:
*"The requested information was not found in the available programme documents."*

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/ingest.py               # 1. build data/processed/chunks.json from the handbook
streamlit run app.py               # 2. open the chatbot at http://localhost:8501
python src/evaluate.py             # 3. run the test questions -> results/
python src/chatbot.py              # (optional) command-line chat
```

## Project structure

```
data/raw/            original handbook (.docx)
data/processed/      chunks.json – prepared text used by the chatbot
src/preprocess.py    tokenising, stop words, Porter stemming, synonym query expansion
src/programmes.py    detects which programme / study mode a heading or question refers to
src/ingest.py        docx -> sections -> chunks (tables turned into sentences)
src/retriever.py     TF-IDF, BM25 and hybrid ranking + programme boosting + coverage check
src/chatbot.py       intent detection, "not found" logic, extractive answer generation
src/evaluate.py      runs test sets, labels outcomes, writes results/
app.py               Streamlit interface
tests/               test_questions.csv (dev, 24) and heldout_questions.csv (held-out, 15)
results/             per-question CSVs + summary.md
docs/TEAM_PLAN.md    task split and GitHub workflow
```

## NLP methods (summary)

| Stage | Technique |
|---|---|
| Text preparation | Reads paragraphs **and tables** in order, keeps the heading path, converts each table row to a sentence (`Subject Code: MAI503; Subject Name: Natural Language Processing; …`), chunks sections to ≤160 words, tags each chunk with programme and study mode |
| Normalisation | Lower-casing, regex tokenisation, stop-word removal, Porter stemming, unigrams + bigrams |
| Query understanding | Synonym/paraphrase expansion (e.g. *how long → duration*, *subjects → modules*, *hostel → accommodation*), programme detection, intent detection (duration / credits / subjects / general) |
| Retrieval | TF-IDF cosine similarity, Okapi BM25, or hybrid (0.6 TF-IDF + 0.4 BM25), re-weighted by programme and study mode |
| Answerability | Not found if best TF-IDF cosine < 0.08, or < 50 % of question concepts appear in any one retrieved chunk, or a fee/cost question gets no RM amount |
| Answer generation | Extractive: intent-specific rows (durations, totals, subject list) or top-ranked sentences by TF-IDF similarity, with source section |

## Results (from `results/summary.md`)

| Set | TF-IDF | BM25 | Hybrid |
|---|---|---|---|
| Development (24 Qs, used for tuning) | 100 % | 96 % | 100 % |
| Held-out (15 Qs, not used for tuning) | 67 % | 60 % | 67 % |

The gap between the two sets shows over-fitting of the synonym list and thresholds to the development questions;
see the per-question CSVs for the failures (vocabulary mismatch, people/role questions, one unsupported answer).

## Limitations
- Keyword-based: fails when the student's words never appear in the handbook (e.g. *switch* vs *change of status*).
- Answers are extracted sentences, so they can be long or include neighbouring, less relevant lines.
- The handbook itself has inconsistencies (e.g. code MAI502 used for two modules; MDS vs MAI codes in different tables), which the bot repeats.
- Covers only one document; no fees, intakes or career statistics are available.
