"""Streamlit user interface:  Enter question -> Receive answer.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chatbot import Chatbot  # noqa: E402
from programmes import DEFAULT_PROGRAMME, PROGRAMMES  # noqa: E402
from retriever import Retriever  # noqa: E402

st.set_page_config(page_title="HELP FCDT Postgraduate Chatbot", page_icon="🎓", layout="centered")


@st.cache_resource
def load_retriever():
    return Retriever()


retriever = load_retriever()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    labels = {**{k: v for k, v in PROGRAMMES.items()}, None: "Any programme (auto-detect)"}
    keys = list(PROGRAMMES) + [None]
    programme = st.selectbox("Programme", keys, index=keys.index(DEFAULT_PROGRAMME),
                             format_func=lambda k: labels[k])
    method = st.radio("Retrieval method", ["hybrid", "tfidf", "bm25"], horizontal=True,
                      help="hybrid = 0.6 × TF-IDF cosine + 0.4 × BM25 (both normalised)")
    show_debug = st.checkbox("Show retrieval details", value=False)
    st.divider()
    st.caption("Try asking:")
    examples = [
        "How long is the Master in AI programme?",
        "What subjects will I study?",
        "What are the entry requirements?",
        "Is there an internship?",
        "How many credits are required?",
        "What happens if I fail a module?",
        "What is the hostel fee?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    st.caption(f"Source: FCDT Postgraduate Handbook 2026/2027 · {len(retriever.chunks)} indexed chunks")
    if st.button("Clear chat"):
        st.session_state.history = []

bot = Chatbot(retriever, method=method)

# ---------------- Main ----------------
st.title("🎓 Postgraduate Programme Chatbot")
st.caption("Answers come only from the HELP University FCDT Postgraduate Handbook 2026/2027. "
           "If the handbook does not contain the answer, the bot says so.")

st.session_state.setdefault("history", [])

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask about programmes, entry requirements, subjects, credits…")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    ans = bot.ask(question, programme=programme)
    body = ans.text
    if ans.sources:
        body += "\n\n" + "  \n".join(f"📄 *Source: {s}*" for s in ans.sources)
    with st.chat_message("assistant"):
        st.markdown(body)
        if show_debug:
            with st.expander("Retrieval details"):
                st.write(f"Intent: **{ans.intent}** · Confidence (TF-IDF cosine): **{ans.confidence:.3f}** "
                         f"· Concept coverage: **{ans.coverage:.2f}**")
                for i, h in enumerate(ans.hits[:5], 1):
                    st.markdown(f"**{i}. score {h.score:.3f}** (tf-idf {h.tfidf:.3f}) — {h.chunk['section']}")
                    st.caption(h.chunk["text"][:300] + "…")
    st.session_state.history.append({"role": "assistant", "content": body})
