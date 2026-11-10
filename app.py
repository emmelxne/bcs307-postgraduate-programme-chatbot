"""Streamlit user interface:  type or speak a question -> receive an answer.

Run:  python -m streamlit run app.py
"""
import hashlib
import sys
import time
import uuid
from pathlib import Path

import streamlit as st
from ui import apply_style, render_header

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chatbot import Answer, Chatbot, md_escape  # noqa: E402
from programmes import DEFAULT_PROGRAMME, PROGRAMMES, SHORT  # noqa: E402
from retriever import Retriever  # noqa: E402
import speech  # noqa: E402

st.set_page_config(page_title="Postgraduate Programme Assistant", page_icon="🎓",
                   layout="centered", initial_sidebar_state="auto")  # auto = closed on phones

apply_style()

STATUS = {  # text labels, not colour alone
    "answer": "✅ Answer",
    "related": "ℹ️ Related information",
    "not_found": "❌ Information not available",
    "clarify": "❓ Need a bit more detail",
    "switch": "🔁 Different programme",
    "scope": "🚫 Outside scope",
    "meta": "👋",
}
PROG_KEYS = list(PROGRAMMES) + ["ANY"]
PROG_LABEL = {**PROGRAMMES, "ANY": "Any programme (I'll ask if needed)"}


@st.cache_resource
def load_retriever():
    return Retriever()


try:
    retriever = load_retriever()
except Exception as e:  # noqa: BLE001
    print(f"[app] failed to load index: {e!r}")
    st.error("Programme information couldn't be loaded. Please try again. "
             "(If you are the developer: run `python src/ingest.py` first.)")
    st.stop()

ss = st.session_state
if "chats" not in ss:
    ss.chats = {}                    # isolated to this browser's Streamlit session


def new_chat():
    cid = uuid.uuid4().hex[:8]
    ss.chats[cid] = {"title": "New chat", "updated": time.time(), "messages": []}
    ss.current = cid


def open_chat(cid):
    ss.current = cid


def delete_chat(cid):
    ss.chats.pop(cid, None)
    if ss.get("current") == cid:
        ss.current = None


if not ss.get("current") or ss.current not in ss.chats:
    empty = [c for c, v in ss.chats.items() if not v["messages"]]
    if empty:
        ss.current = empty[0]
    else:
        new_chat()
ss.history = ss.chats[ss.current]["messages"]   # same list object -> edits go into the chat
ss.setdefault("programme", DEFAULT_PROGRAMME)
ss.setdefault("programme_selector", ss.programme)
METHOD = "tfidf"   # TF-IDF + cosine similarity (chosen after comparing TF-IDF, BM25, hybrid and semantic; see README)
ss.setdefault("debug", False)
ss.setdefault("voice_key", 0)
ss.setdefault("voice_text", "")
ss.setdefault("voice_error", "")
ss.setdefault("voice_digest", "")


# ------------------------------------------------------------------ #
def submit(question: str):
    question = question.strip()
    if not question:
        return
    prog = None if ss.programme == "ANY" else ss.programme
    bot = Chatbot(retriever, method=METHOD)
    ans = bot.ask(question, programme=prog)
    record_answer(question, ans)


def record_answer(question: str, ans: Answer, add_user: bool = True):
    chat = ss.chats[ss.current]
    if not chat["messages"]:
        chat["title"] = question[:40] + ("…" if len(question) > 40 else "")
    chat["updated"] = time.time()
    ss.history = chat["messages"]
    if add_user:
        ss.history.append({"role": "user", "content": question})
    debug = None
    if ans.hits:
        debug = {"intent": ans.intent, "confidence": ans.confidence, "coverage": ans.coverage,
                 "hits": [(h.score, h.tfidf, h.chunk["section"], h.chunk["text"][:280]) for h in ans.hits[:5]]}
    ss.history.append({"role": "assistant", "status": ans.status, "lead": ans.lead, "details": ans.details,
                       "sources": ans.sources, "options": ans.options, "origin": question,
                       "programme": ans.programme or ss.programme, "intent": ans.intent, "debug": debug, "collapse": ans.collapse,
                       "text": ans.text})


def select_programme():
    ss.programme = ss.programme_selector


def on_option(opt: dict, origin: str, index: int):
    # Consume the prompt before acting, including double clicks / stale callbacks.
    if index >= len(ss.history) or not ss.history[index].get("options"):
        return
    ss.history[index]["options"] = []
    action = opt["action"]
    if action == "ask":
        submit(opt["value"])
    elif action in ("switch", "set_programme"):
        ss.programme = opt["value"]
        ss.programme_selector = ss.programme
        submit(opt.get("question", origin))
    elif action == "keep":
        ans = Answer("meta", f"Keeping **{PROGRAMMES[ss.programme]}** selected. Please ask a new question about this programme.",
                     programme=ss.programme)
        record_answer(origin, ans, add_user=False)


def send_voice():
    text = ss.get("voice_edit", "").strip()
    ss.voice_text, ss.voice_error = "", ""
    ss.voice_key += 1          # resets the recorder so the same clip isn't re-sent
    submit(text)


def discard_voice():
    ss.voice_text, ss.voice_error = "", ""
    ss.voice_key += 1


# Handle a new typed question BEFORE drawing the sidebar, so the chat list is up to date
question = st.chat_input("Ask about entry requirements, modules, credits, duration…")
if question:
    submit(question)

# ---------------- Sidebar: chat history + settings ---------------- #
with st.sidebar:
    st.button("New chat", icon=":material/add:", use_container_width=True, on_click=new_chat)
    st.subheader("Chat history")
    chats = sorted(((c, v) for c, v in ss.chats.items() if v["messages"]),
                   key=lambda kv: kv[1]["updated"], reverse=True)
    if not chats:
        st.caption("No previous chats yet.")
    for cid, chat in chats:
        c1, c2 = st.columns([0.8, 0.2], gap="small", vertical_alignment="center")
        c1.button(("💬 " if cid != ss.current else "▶ ") + chat["title"], key=f"open-{cid}",
                  use_container_width=True, type="primary" if cid == ss.current else "secondary",
                  on_click=open_chat, args=(cid,))
        c2.button("✕", key=f"del-{cid}", use_container_width=True, help="Delete this chat", on_click=delete_chat, args=(cid,))
    st.divider()
    with st.expander("About"):
        st.caption("Answers are based on the HELP University FCDT Postgraduate Handbook 2026/2027 "
                   f"({len(retriever.chunks)} indexed passages). Undergraduate programmes are not covered. "
                   "Chat history is kept only in this browser session.")
    with st.expander("Show how it works"):
        st.caption("Retrieval: TF-IDF vectors ranked by cosine similarity.")
        ss.debug = st.checkbox("Show retrieval details under each answer", value=ss.debug)
        st.caption(f"Voice backend: {speech.available_backend() or 'not installed'}")

# ---------------- Header + programme choice ---------------- #
render_header()
st.selectbox("Programme", PROG_KEYS, key="programme_selector", format_func=lambda k: PROG_LABEL[k],
             on_change=select_programme)


# ---------------- Voice input ---------------- #
answering = PROG_LABEL[ss.programme] if ss.programme != "ANY" else "any programme"
st.caption(f"Answering for: **{answering}**")
with st.expander("🎤 Ask by voice"):
    audio = st.audio_input("Record a question", key=f"voice-{ss.voice_key}")
    if audio is not None:
        wav = audio.getvalue()
        digest = hashlib.md5(wav).hexdigest()
        if digest != ss.voice_digest:            # transcribe each new clip once
            ss.voice_digest = digest
            with st.spinner("Transcribing…"):
                ss.voice_text, ss.voice_error = speech.transcribe(wav)
    if ss.voice_error:
        st.warning(ss.voice_error)
    if ss.voice_text:
        st.text_input("Transcript — check and edit before sending", value=ss.voice_text, key="voice_edit")
        c1, c2 = st.columns(2)
        c1.button("Send question", type="primary", use_container_width=True, on_click=send_voice)
        c2.button("Discard", use_container_width=True, on_click=discard_voice)


# ---------------- Chat history ---------------- #
if not ss.history:
    st.markdown("#### How can I help?")
    st.caption("Choose a programme, then type or record your question.")
    cols = st.columns(3)
    for col, q in zip(cols, ["What are the entry requirements?", "How long is the programme?",
                             "What subjects will I study?"]):
        col.button(q, type="primary" if q == "What are the entry requirements?" else "secondary",
                   use_container_width=True, on_click=submit, args=(q,))

last_assistant = max((i for i, m in enumerate(ss.history) if m["role"] == "assistant"), default=-1)
for i, m in enumerate(ss.history):
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/smart_toy:"):
        if m["role"] == "user":
            st.markdown(m["content"])
            continue
        label = STATUS.get(m["status"], "")
        prog_name = "Any programme" if m["programme"] == "ANY" else SHORT[m["programme"]]
        st.caption(f"{label}  ·  {prog_name}" if label != "👋" else prog_name)
        st.markdown(md_escape(m["lead"]))
        if m["details"]:
            bullets = "\n".join(f"- {md_escape(d)}" for d in m["details"])
            if m.get("collapse"):
                with st.expander(f"All routes and conditions ({len(m['details'])})"):
                    st.markdown(bullets)
            else:
                st.markdown(bullets)
        if m["sources"]:
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"📄 {s}")
        if m["options"] and i == last_assistant:
            cols = st.columns(min(len(m["options"]), 3))
            for j, opt in enumerate(m["options"]):
                cols[j % len(cols)].button(opt["label"], key=f"opt-{i}-{j}", use_container_width=True,
                                           on_click=on_option, args=(opt, m["origin"], i))
        if ss.debug and m.get("debug"):
            d = m["debug"]
            with st.expander("Retrieval details"):
                st.write(f"Intent: **{d['intent']}** · Confidence (TF-IDF cosine): **{d['confidence']:.3f}** "
                         f"· Concept coverage: **{d['coverage']:.2f}**")
                for k, (score, tf, sec, text) in enumerate(d["hits"], 1):
                    st.markdown(f"**{k}. score {score:.3f}** (tf-idf {tf:.3f}) — {sec}")
                    st.caption("Text: " + text + "…")
        elif ss.debug and m["role"] == "assistant":
            st.caption(f"Intent: {m['intent']} (rule-based handler, no ranking shown)")
