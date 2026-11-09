"""Speech-to-text adapter for voice questions.

The app records audio with st.audio_input (WAV bytes), this module turns it into
text, and the student reviews / edits the transcript before it is sent to the
SAME chatbot pipeline as typed questions.

Backends (picked automatically, or set STT_BACKEND=whisper|google):
  * whisper – faster-whisper running locally (offline after a one-time model
              download of ~75 MB for 'base.en'). Install: pip install faster-whisper
  * google  – SpeechRecognition's free Google Web Speech API (needs internet,
              no model download). Install: pip install SpeechRecognition
"""
import io
import os
import re
import tempfile

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
_whisper = None


def available_backend() -> str | None:
    forced = os.environ.get("STT_BACKEND")
    if forced:
        return forced
    try:
        import faster_whisper  # noqa: F401
        return "whisper"
    except ImportError:
        pass
    try:
        import speech_recognition  # noqa: F401
        return "google"
    except ImportError:
        return None


def _whisper_transcribe(wav: bytes) -> str:
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        path = f.name
    try:
        segments, _ = _whisper.transcribe(path, language="en", vad_filter=True,
                                          initial_prompt="HELP University postgraduate handbook: Master in AI, "
                                                         "MAI503, CGPA, MUET, ODL, Cybersecurity, PhD.")
        return " ".join(s.text.strip() for s in segments)
    finally:
        os.unlink(path)


def _google_transcribe(wav: bytes) -> str:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav)) as source:
        audio = r.record(source)
    return r.recognize_google(audio, language=os.environ.get("STT_LANGUAGE", "en-US"))


def clean_transcript(text: str) -> str:
    """Fix common speech-to-text forms of handbook terms."""
    t = text.strip()
    t = re.sub(r"\b([mM])\.?\s?([aAdDcC])\.?\s?([iIsS])\.?\s?(\d)\s?(\d)\s?(\d)\b",
               lambda m: f"{m[1]}{m[2]}{m[3]}{m[4]}{m[5]}{m[6]}".upper(), t)      # "M A I 5 0 3" -> MAI503
    t = re.sub(r"\bc\.?\s?g\.?\s?p\.?\s?a\b", "CGPA", t, flags=re.I)
    t = re.sub(r"\bp\.?\s?h\.?\s?d\b", "PhD", t, flags=re.I)
    t = re.sub(r"\bo\.?\s?d\.?\s?l\b", "ODL", t, flags=re.I)
    t = re.sub(r"\bmuet\b|\bmu et\b|\bmew it\b", "MUET", t, flags=re.I)
    return t


def transcribe(wav: bytes) -> tuple[str, str | None]:
    """Return (transcript, error message). Never raises."""
    if not wav or len(wav) < 2000:
        return "", "The recording was empty or too short. Please try again or type your question."
    backend = available_backend()
    if backend is None:
        return "", ("Voice input isn't set up on this computer. Install it with "
                    "`pip install faster-whisper` (offline) or `pip install SpeechRecognition` (online), "
                    "or type your question.")
    try:
        if backend == "mock":   # used only for automated UI tests
            text = os.environ.get("STT_MOCK_TEXT", "")
        else:
            text = _whisper_transcribe(wav) if backend == "whisper" else _google_transcribe(wav)
    except Exception as e:  # noqa: BLE001 – report any backend failure to the user
        kind = type(e).__name__
        if kind == "UnknownValueError" or kind == "ValueError":
            return "", "I couldn't make out any speech. Please try again in a quieter place, or type your question."
        if kind == "RequestError":
            return "", "The online speech service couldn't be reached. Check your internet connection or type your question."
        print(f"[speech] {backend} failed: {e!r}")
        return "", "Transcription failed. Please try again or type your question."
    text = clean_transcript(text)
    if not text:
        return "", "I couldn't make out any speech. Please try again in a quieter place, or type your question."
    return text, None
