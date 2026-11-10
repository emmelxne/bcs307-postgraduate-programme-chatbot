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
python -m streamlit run app.py     # 2. open the chatbot at http://localhost:8501
python src/evaluate.py             # 3. run the test questions -> results/
python src/chatbot.py              # (optional) command-line chat
```

Python 3.10 or newer is required.

### Chat history
The sidebar lists previous chats (titled by their first question) with **New chat** and **✕** to delete.
Chats are kept in `st.session_state` within the current browser session. New Chat, opening chats,
and deleting chats operate only on that session. A new independent session starts empty;
chat history is not loaded from or written to `.chat_history.json`.

### Interface
The interface uses Streamlit’s light theme with a white background, charcoal text, a neutral sidebar
and HELP-red controls. The centred header uses separate local HELP and FCDT logo cutouts from the
supplied screenshot, without redrawing or recolouring them. It adapts to desktop and phone widths.
Programme selection, the active scope and voice input appear above the chat; typed input stays at the bottom.

### Voice input (optional extension)
Open **🎤 Ask by voice**, record, check/edit the transcript, then **Send question**. The transcript goes
through exactly the same pipeline as a typed question.

- Default: `SpeechRecognition` (in `requirements.txt`) – uses Google's free Web Speech service, needs internet.
- Offline: `pip install -r requirements-voice-offline.txt` (faster-whisper). It is picked automatically when
  installed; the first recording downloads a ~75 MB model, so do this before the demo.
- Force a backend with `STT_BACKEND=whisper` or `STT_BACKEND=google`.
- If the microphone is blocked, the clip is empty or transcription fails, the app says why and typing still works.

## Project structure

```
data/raw/            original handbook (.docx)
data/processed/      chunks.json – prepared text used by the chatbot
src/preprocess.py    tokenising, stop words, Porter stemming, synonym query expansion
src/programmes.py    detects which programme / study mode a heading or question refers to
src/ingest.py        docx -> sections -> chunks (tables turned into sentences)
src/retriever.py     TF-IDF, BM25 and hybrid ranking + programme boosting + coverage check
src/chatbot.py       response states, programme routing, topic handlers, extractive answers
src/speech.py        speech-to-text adapter for voice questions
src/evaluate.py      runs test sets, labels outcomes, writes results/
app.py               Streamlit interface
ui.py                local branding and presentation helpers
.streamlit/config.toml native light theme
assets/              local HELP/FCDT logos and minimal responsive CSS
tests/               test_questions.csv (dev, 24), heldout_questions.csv (15),
                     audit_questions.csv (31), correctness_regressions.csv (25),
                     test_correctness.py, final_manual_questions.csv (18 fresh, unrun)
results/             per-question CSVs + summary.md
```

## NLP methods (summary)

| Stage | Technique |
|---|---|
| Text preparation | Reads paragraphs **and tables** in order, keeps the heading path, converts each table row to a sentence (`Subject Code: MAI503; Subject Name: Natural Language Processing; …`), chunks sections to ≤160 words, tags each chunk with programme and study mode |
| Normalisation | Lower-casing, regex tokenisation, stop-word removal, Porter stemming, unigrams + bigrams |
| Query understanding | Synonym/paraphrase expansion (e.g. *how long → duration*, *subjects → modules*, *hostel → accommodation*), programme detection, intent detection (duration / credits / subjects / general) |
| Retrieval | **TF-IDF and cosine similarity** (stemmed unigrams + bigrams, sublinear TF). Programme scope is a filter before ranking. BM25 and a TF-IDF/BM25 hybrid are kept in `evaluate.py` only for comparison |
| Answerability | Answer only if the best TF-IDF cosine ≥ 0.08 and ≥ 50 % of the question's concepts appear together in one retrieved passage; otherwise "not found" |
| Answer generation | Extractive: intent-specific rows (durations, totals, subject list) or top-ranked sentences by TF-IDF similarity, with source section |

## Response types

| Status | When | Example |
|---|---|---|
| Answer | Handbook states the fact | "takes 1–3 years full time or 2–5 years part time" |
| Related | Not stated, but a related passage exists | Internship → industry-linked Project Report |
| Not in handbook | Fact missing | Tuition, hostel price, intake dates (with contact route) |
| Need more detail | Unfinished question, unclear cost, programme needed | "Which programme do you mean?" |
| Different programme | Question names another programme | "Switch to Cybersecurity?" |
| Outside scope | Undergraduate, personal or general-knowledge questions | "What is AI?", "How old am I?" |

## Results (`results/summary.md`)

| Method | Dev (24) | Held-out (15) | Audit (31) |
|---|---|---|---|
| **TF-IDF (used)** | **100 %** | **80 %** | **100 %** |
| BM25 | 96 % | 73 % | 100 % |
| Hybrid (TF-IDF + BM25) | 100 % | 80 % | 100 % |

TF-IDF matched or beat every other method on every test set, so the simplest method is used in the app.
The table above records the pre-trial results. The local correctness trial retains 24/24 development
and 31/31 audit with TF-IDF, and reaches 13/15 on the older held-out set by abstaining on unsupported
employment statistics. These are automated keyword/status results, not a fresh manual assessment score.
The two remaining older held-out misses concern study-load switching and the counselling phone number.
Dev was used to build the rules and audit cases drove the fixes, so only the held-out column is close to an unbiased estimate.
A semantic-search variant (all-MiniLM-L6-v2 sentence embeddings, alone and combined with TF-IDF/BM25) was also tested:
87 % held-out (one extra question: "phone number for counselling"), but semantic-only missed an exact-term question,
and it needs PyTorch plus a model download. The gain (one question out of 15) did not justify the added complexity,
so it is listed as a possible improvement.

**For the report, write a fresh `tests/final_questions.csv` (≥15 questions) before running it, and judge full
answers manually (`manual_verdict` column).** Keyword checks alone can pass an answer that also contains wrong text.

## Limitations
- Keyword-based: misses paraphrases whose words don't appear in the handbook
  (held-out: "switch from full time to part time", "phone number for counselling").
- Rule-based routing covers the question types seen in testing; new phrasings may fall back to general retrieval.
- Measured employment rates, travel times, and module teaching assignments are not stated in the handbook;
  similarity to programme aims, duration, or a timetable is not evidence for those facts.
- The handbook has inconsistencies (MAI502 used for two modules; MDS/MAI codes differ between tables;
  MAI503 credits blank in the semester table but 4 in the summary). The bot shows both rather than guessing.
- PhD direct entry: the handbook says no direct entry from Bachelor's; confirm with the lecturer if told otherwise.
- Voice accuracy depends on the microphone, accent and noise; the live microphone path must be tested on the demo laptop.

### Local correctness regressions

Run `python -B -m unittest discover -s tests -p test_correctness.py -v` for the focused CSV
regressions and source/programme invariants. The original evaluator remains `python src/evaluate.py`.
`tests/final_manual_questions.csv` contains 18 fresh, unrun questions reserved for final manual
assessment; they are deliberately separate from the evaluator's development and regression sets.
