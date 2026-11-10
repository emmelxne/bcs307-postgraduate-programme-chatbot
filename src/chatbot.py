"""Step 3 – Answer generation (extractive, grounded in the handbook).

Pipeline for every question (typed or transcribed from voice):

1. Conversation checks   – greeting / bot name / capability, unfinished
                           question, unspecified "how much does it cost?",
                           undergraduate programme, personal or general-
                           knowledge questions (out of scope).
2. Programme resolution  – detect a named programme (typo tolerant). If it
                           differs from the selected one -> offer a switch.
                           If the fact is programme-specific and no programme
                           is known -> ask which programme.
3. Topic handlers        – duration, credits, subjects, electives, module
                           semester, entry requirements, fees, refunds,
                           intake, internship, teaching hours, failing a
                           module. Each reads only the passages that can
                           support that specific fact.
4. General retrieval     – TF-IDF + cosine search restricted to the programme,
                           answerability gate, extractive sentence selection.

Every answer line is copied from (or directly computed from) the handbook and
cited by section number. No generative model is used.
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import expand_query  # noqa: E402
from programmes import (DEFAULT_PROGRAMME, PROGRAMMES, SHORT, detect_in_query,  # noqa: E402
                        detect_load, fix_typos, is_comparison, is_undergraduate_request)
from retriever import Retriever  # noqa: E402

NOT_FOUND = "The requested information was not found in the available programme documents."
CONTACT = "HELP University (Tel: 03-7849 3000 | Email: marketing@help.edu.my)"
EXAMPLES = "entry requirements, modules, credits, study duration, grading or refunds"

MIN_TFIDF = 0.08      # answerability gate (tuned on the development set)
MIN_COVERAGE = 0.5
MONEY_A = r"RM ?\d"

MODULE_ALIASES = {
    "nlp": "natural language processing", "computer vision": "deep learning for computer vision",
    "deep learning": "deep learning for computer vision", "machine learning": "machine learning",
    "ml": "machine learning", "statistics": "statistics", "stats": "statistics",
    "cryptography": "network security and cryptography", "network security": "network security and cryptography",
    "ethics": "ethics and cybersecurity", "programming": "programming", "research method": "research method",
    "data management": "data management", "project report": "project report", "fyp": "project report",
    "dissertation": "dissertation", "thesis": "thesis", "forensics": "digital forensics",
    "ethical hacking": "ethical hacking", "penetration": "ethical hacking",
}
# Words that make a short "what is X?" a handbook question rather than general knowledge
HANDBOOK_TERMS = (r"programme|program|module|subject|course|credit|fee|grade|gpa|cgpa|muet|ielts|cefr|odl|"
                  r"mypride|lrc|issd|apel|helplive|dissertation|project report|thesis|viva|supervisor|"
                  r"semester|intake|refund|deposit|elective|faculty|dean|handbook|help university|hep|mqa|"
                  r"\b(mds|mai|mcs|phd)\d{3}\b|plo|peo|ueo")


# ====================================================================== #
@dataclass
class Answer:
    status: str                         # answer | related | not_found | clarify | switch | scope | meta
    lead: str
    details: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    options: list[dict] = field(default_factory=list)   # buttons: {label, action, value}
    programme: str | None = None
    intent: str = "general"
    confidence: float = 0.0
    coverage: float = 0.0
    hits: list = field(default_factory=list)
    collapse: bool = False              # UI: show details in an expander (long lists)

    @property
    def found(self) -> bool:
        return self.status in ("answer", "related")

    @property
    def text(self) -> str:
        lines = [self.lead] + [f"- {md_escape(d)}" for d in self.details]
        return "\n".join(lines)


def md_escape(line: str) -> str:
    """Stop handbook numbering ('1.', 'iii.') turning into Markdown lists."""
    return re.sub(r"^((?:\d+|[ivx]+))\. ", r"\1\\. ", line)


def cite(section_path: str) -> str:
    """'5. X > 5.3 Master in AI — Conventional (N/..)(..) > Programme Structure' -> '§5.3 … › …'."""
    parts = section_path.split(" > ")
    numbered = [p for p in parts if re.match(r"\d+(\.\d+)*\.?\s", p)]
    head = numbered[-1] if numbered else parts[0]
    head = re.sub(r"\s*\([^)]*/[^)]*\)", "", head)          # drop MQA reference codes
    m = re.match(r"(\d+(?:\.\d+)*)\.?\s+(.*)", head)
    label = f"§{m.group(1)} {m.group(2)}" if m else head
    if parts[-1] not in numbered and parts[-1] != head:
        label += f" › {parts[-1]}"
    return label


def fields(unit: str) -> dict:
    return {k.strip(): v.strip() for k, v in re.findall(r"([A-Za-z /.]+?): ([^;]+?)(?:;|\.$|$)", unit)}


def years_range(text: str) -> tuple[str, str] | None:
    mn = re.search(r"Minimum Duration: (\d+) Years?", text)
    mx = re.search(r"Maximum Duration: (\d+) Years?", text)
    return (mn.group(1), mx.group(1)) if mn and mx else None


# ====================================================================== #
class Chatbot:
    def __init__(self, retriever: Retriever | None = None):
        self.r = retriever or Retriever()
        self.modules = self._module_index()

    # ------------------------------------------------------------------ #
    def ask(self, question: str, programme: str | None = DEFAULT_PROGRAMME,
            keep_programme: bool = False) -> Answer:
        """programme: selected programme key, or None for 'Any programme'.
        keep_programme: user chose to stay on the selected programme after a switch prompt."""
        q = question.strip()
        ql = fix_typos(q)

        early = self._conversation_checks(q, ql)
        if early:
            return early

        # ---------------- programme resolution ----------------
        named = detect_in_query(q)
        intent = self._intent(ql)
        # Unqualified AI and PhD names leave the variant ambiguous. Explicit
        # comparisons must never silently inherit the selected programme.
        ambiguous_family = named in ({"MAI_CONV", "MAI_ODL"}, {"PHD_COMP", "PHD_CS"})
        comparison = is_comparison(ql)
        if len(named) > 1 and (comparison or not ambiguous_family):
            followup = {"admission": "What are the entry requirements?", "duration": "How long is the programme?",
                        "credits": "How many credits are required?", "subjects": "What subjects will I study?",
                        "electives": "What are the electives?"}.get(intent, "Tell me about the programme")
            return Answer("clarify", "I can show one programme at a time. Which would you like to see first?",
                          options=[{"label": SHORT[p], "action": "set_programme", "value": p,
                                    "question": followup} for p in sorted(named)], intent=intent)
        if named and programme and programme not in named and not keep_programme:
            targets = sorted(named)
            names = " / ".join(PROGRAMMES[t] for t in targets)
            opts = [{"label": f"Switch to {SHORT[t]}", "action": "switch", "value": t} for t in targets]
            opts.append({"label": f"Keep {SHORT[programme]}", "action": "keep", "value": programme})
            return Answer("switch", f"You're viewing **{PROGRAMMES[programme]}**. Your question is about "
                                    f"**{names}**. Switch programme?", options=opts, programme=programme)
        if keep_programme and programme:
            return Answer("meta", f"Keeping **{PROGRAMMES[programme]}** selected. Please ask a new question about this programme.",
                          programme=programme)
        elif programme:
            scope = {programme}
        else:
            scope = named

        prog = next(iter(scope)) if len(scope) == 1 else None
        fact = self._requested_fact(ql)
        if fact:
            ans = self._answer_fact(q, scope, fact)
            ans.programme = prog
            return ans
        programme_specific = intent in {"overview", "duration", "credits", "module_credits", "module_lookup", "subjects", "electives", "module_semester",
                                        "admission", "internship", "hours"}
        if programme_specific and len(scope) != 1:
            choices = sorted(scope) if scope else list(PROGRAMMES)
            return Answer("clarify", "Which programme do you mean?", intent=intent,
                          options=[{"label": SHORT[c], "action": "set_programme", "value": c} for c in choices])

        handler = getattr(self, f"_h_{intent}", None)
        ans = handler(q, ql, prog, scope) if handler else None
        if ans is None:
            ans = self._general(q, scope, prog)
        ans.intent = ans.intent if ans.intent != "general" else intent
        ans.programme = prog
        return ans

    # ================= 1. conversation checks ========================= #
    def _conversation_checks(self, q: str, ql: str) -> Answer | None:
        words = re.findall(r"[a-z0-9']+", ql)
        if not words:
            return Answer("meta", "Please type a question about the FCDT postgraduate programmes.")
        if re.search(r"your name|who are (you|u)\b|what are (you|u)\b", ql):
            return Answer("meta", "I'm the HELP FCDT Postgraduate Programme Assistant. I can answer questions about "
                                  f"FCDT postgraduate programmes, including {EXAMPLES}.",
                          options=self._starters())
        if re.fullmatch(r"(hi|hello|hey|hai|good (morning|afternoon|evening)|thanks?|thank (you|u))[ !.]*", ql):
            return Answer("meta", f"Hello! Ask me about the FCDT postgraduate programmes — for example {EXAMPLES}.",
                          options=self._starters())
        dangling = re.search(r"\b(for|the|a|an|of|to|in|and|or|on|with|my|is|what|how|which)\s*[?.!]*$", ql)
        if dangling or ql.rstrip().endswith("...") or len(words) < 2:
            return Answer("clarify", "Could you finish your question? I can help with FCDT postgraduate "
                                     f"programmes — for example {EXAMPLES}.", options=self._starters())
        if re.search(r"what (can|do) (you|u) do|how (can|do) (you|u) help|what is this (bot|app|chatbot)", ql):
            return Answer("meta", "I provide information on HELP University's FCDT postgraduate programmes, including "
                                  f"{EXAMPLES}. Please select a programme, then ask your question.",
                          options=self._starters())
        cost_words = r"tuition|hostel|accommodation|continuation|extension|status|refund|deposit|application|test|muet|ielts|fee"
        if re.search(r"how much (does|do|will|is) (it|this|that)\b|^(and |so )?how much( is it| does it cost)?\s*\??$|"
                     r"what does it cost|^(and |so )?(the )?(cost|price)\??$", ql) and not re.search(cost_words, ql):
            return Answer("clarify", "Could you specify which cost you are referring to?",
                          options=[{"label": l, "action": "ask", "value": v} for l, v in [
                              ("Tuition fee", "What is the tuition fee?"),
                              ("Studying beyond the standard duration", "What is the continuation fee?"),
                              ("Changing full/part-time status", "What is the change of status fee?"),
                              ("Refund if I withdraw", "Do I get a refund if I withdraw from the programme?")]])
        if is_undergraduate_request(q):
            return Answer("scope", "I provide information on FCDT **postgraduate** programmes (Master's and PhD) only. "
                                   f"For Bachelor's or Diploma programmes, please contact {CONTACT}.")
        if re.search(r"how old am i|who am i\b|\bam i (a|an|old|young)\b|my (age|birthday)|weather|joke", ql):
            return Answer("scope", f"I can only assist with questions about FCDT postgraduate programmes. You may ask "
                                   f"about {EXAMPLES}.", options=self._starters())
        m = re.fullmatch(r"(?:what|who) (?:is|are|r) (?:an? )?([a-z0-9 .\-']{1,30}?)\s*\??|"
                         r"(?:define|explain|meaning of) (?:an? )?([a-z0-9 .\-']{1,30}?)\s*\??", ql)
        if m:
            topic = (m.group(1) or m.group(2) or "").strip()
            if not topic.startswith("the ") and not re.search(HANDBOOK_TERMS, topic):
                related = self._related_definition(topic)
                if related:
                    return related
                return Answer("scope", "General definitions are outside the scope of this assistant. I can only assist "
                                       f"with questions about FCDT postgraduate programmes, such as {EXAMPLES}.", options=self._starters())
        return None

    def _related_definition(self, topic: str) -> Answer | None:
        """'What is AI / NLP / data science?' -> the handbook has no definition, but it does
        describe a matching programme or module, so show that passage labelled as related."""
        progs = detect_in_query(topic)
        mod = self._find_module(topic)
        topic = topic.upper() if len(topic) <= 4 else topic
        if mod:
            for c in self.r.chunks:
                if c["topic"] == "subject_descriptions" and any(u.startswith(f"{mod[0]} {mod[1]}") for u in c["units"]):
                    idx = next(i for i, u in enumerate(c["units"]) if u.startswith(f"{mod[0]} {mod[1]}"))
                    if idx + 1 < len(c["units"]):
                        return Answer("related", f"A general definition of \"{topic}\" is outside the scope of this "
                                      f"assistant. Here is the related module information for **{mod[0]} {mod[1]}**:",
                                      [c["units"][idx + 1]], [cite(c["section"])], intent="definition")
        if progs:
            p = "MAI_CONV" if {"MAI_CONV", "MAI_ODL"} <= progs else sorted(progs)[0]
            overview = next((c for c in self.r.chunks if c["programmes"] == [p] and c["topic"] == "programme"), None)
            if overview:
                return Answer("related", f"A general definition of \"{topic}\" is outside the scope of this "
                              f"assistant. Here is the related programme information for the **{PROGRAMMES[p]}**:", [overview["units"][0]], [cite(overview["section"])],
                              intent="definition", options=[{"label": l, "action": "ask", "value": l} for l in
                                                            ["What subjects will I study?", "What are the entry requirements?"]])
        return None

    def _starters(self):
        return [{"label": s, "action": "ask", "value": s} for s in
                ["What are the entry requirements?", "How long is the programme?", "What subjects will I study?"]]

    # ================= 2. intent ====================================== #
    def _intent(self, ql: str) -> str:
        m = re.match(r"^(what is|what's|what are|tell me (more )?about|describe|overview of|about)( the)? "
                     r"(.{2,60}?)( programme| program)?( about| like)?\??$", ql)
        if m:
            # Only when the subject IS a programme name ("the Master in AI"), not
            # "entry requirements for the Master in AI".
            leftover = re.sub(r"\b(masters?|in|of|artificial|intelligence|ai|data|science|cybersecurity|phd|doctor|"
                              r"philosophy|computing|computer|odl|conventional|programme|program|degree|course|"
                              r"the|this|mai|mds|mcs)\b", " ", m.group(4)).strip()
            if not leftover:
                return "overview"
        if re.search(r"^tell me (more )?about\b", ql):
            return "overview"
        if re.search(r"withdraw|refund|money back|deposit.*back|drop out|\bquit\b|discontinue|\bleave\b|\bdrop\b.*\b(subject|module|course|class)\b", ql):
            return "refund"
        if re.search(r"\bfees?\b|\bcosts?\b|\bprice\b|tuition|how much (is|does|do|will)|\bpay\b", ql):
            return "fee"
        if re.search(r"\bintake|start date|commencement date|when (does|do|will|can) .*(start|begin|commence)", ql):
            return "intake"
        if re.search(r"internship|industrial training|placement|industrial attachment", ql):
            return "internship"
        if re.search(r"\bhours?\b", ql) and not re.search(r"credit hours?", ql) and \
                re.search(r"class|teaching|lecture|contact|study|per week|weekly|each (course|subject|module)|a week", ql):
            return "hours"
        if re.search(r"\bcredits?\b", ql) and (self._find_module(ql) or
                re.search(r"\b[a-z]{2,5}\d{3}\b|\b(module|subject)\b", ql) or
                (re.search(r"credits?(?: hours?)?\s+(?:is|does|for)\b|\b(?:carry|worth)\b", ql)
                 and not re.search(r"programme|program|degree|master|graduate|graduation|\btotal\b", ql)
                 and not detect_in_query(ql))):
            return "module_credits"
        if re.search(r"how long|duration|how many (years|months)|\byears? (to|does|will|is)|length of|finish the", ql):
            return "duration"
        if re.search(r"how many credit|credits? (are |is )?(required|needed)|total credit|credit hours (required|needed)|"
                     r"credits? to graduate", ql):
            return "credits"
        if re.search(r"(which|what) semester|when (do|will|can|should) (i|we) (take|study|do)", ql) or \
                (self._module_codes(ql) and re.search(r"\bsemesters?\b", ql)):
            return "module_semester"
        if re.search(r"\belectives?\b|optional (module|subject)", ql):
            return "electives"
        if re.search(r"(what|which|list)( of)? (the )?(subjects|modules|courses)|(subjects|modules|courses) (will|do|are|in)\b|"
                     r"core (subjects|modules)|programme structure|study in|what will i (study|learn)", ql):
            return "subjects"
        if re.search(r"\bfail|retake|resit|re-?sit|\bredo\b|repeat (a|the) (module|subject)", ql):
            return "fail"
        if re.search(r"entry|\benter\b|directly|admission|requirement|eligib|qualif|\bapply\b|cgpa|get in|accepted|muet|ielts|cefr|"
                     r"english (test|requirement|result|score|proficiency)|non[- ]?computing|not (?:in |a )?computing|"
                     r"degree.*(?:different field|unrelated)|\bjoin\b", ql):
            return "admission"
        if self._module_codes(ql):
            return "module_lookup"
        return "general"

    def _requested_fact(self, ql):
        """Field constraints take precedence over broad duration / general retrieval.

        Similarity to a programme or module is insufficient evidence for a
        journey duration, teaching assignment, or measured employment outcome.
        """
        if re.search(r"(?:how long|duration|time).*\b(?:get to|travel|commut|journey|reach)|travel time|commut.*time", ql):
            return "travel time"
        if re.search(r"who.*\b(?:teach|teaches|teaching|lectur|instruct)|(?:module|subject|course).*\b(?:lecturer|instructor)|"
                     r"(?:lecturer|instructor).*(?:for|of|mai|mds|mcs)", ql):
            return "module teaching assignment"
        if re.search(r"(?:employment|employed|employability|placement|graduates?.*(?:job|work)).*(?:rate|percent|%|proportion)|"
                     r"(?:rate|percent|%|proportion).*graduates?|employment rate", ql):
            return "graduate employment rate"
        return None

    def _answer_fact(self, q, scope, fact):
        hits = self.r.search(q, k=6, scope=scope or None)
        # Require the requested relationship in a single source unit. These are
        # evidence constraints, not a change to the global similarity threshold.
        patterns = {
            "travel time": r"(?:travel|journey|commut|from .+ to).*(?:\d+|one|two|three)\s*(?:minutes?|hours?)",
            "module teaching assignment": r"(?:taught|lectured|instructed) by|(?:lecturer|instructor)\s*:",
            "graduate employment rate": r"(?:\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*percent).*(?:employ|job)|(?:employ|job).*\d+(?:\.\d+)?\s*(?:%|percent)",
        }
        supported = []
        for h in hits:
            units = [u for u in h.chunk["units"] if re.search(patterns[fact], u, re.I)]
            # A lecturer named elsewhere is not a module assignment. A journey
            # must match its origin; an outcome must match a requested timeframe.
            code = re.search(r"\b[a-z]{2,5}\d{3}\b", q, re.I)
            if fact == "module teaching assignment" and code:
                units = [u for u in units if code[0].lower() in u.lower()]
            if fact == "travel time":
                origin = re.search(r"\bfrom (.+?)(?:\?|$| to )", q, re.I)
                units = [u for u in units if origin and origin[1].strip().lower() in u.lower()]
            if fact == "graduate employment rate":
                period = re.search(r"(?:within|after) (.+? (?:months?|years?))", q, re.I)
                if period:
                    units = [u for u in units if period[1].lower() in u.lower()]
            if units and h.tfidf >= MIN_TFIDF and self.r.term_coverage(q, [h]) >= MIN_COVERAGE:
                supported.extend((u, cite(h.chunk["section"])) for u in units)
        if supported:
            return Answer("answer", supported[0][0], [u for u, _ in supported[1:]],
                          list(dict.fromkeys(s for _, s in supported)), intent="fact")
        return Answer("not_found", f"Information not available: the handbook does not state the requested {fact}.",
                      intent="fact", hits=hits)

    # ================= 3. topic handlers ============================== #
    def _chunks(self, prog, topic=None, load=None):
        return [c for c in self.r.chunks if prog in c["programmes"]
                and (topic is None or c["topic"] == topic) and (load is None or c.get("load") == load)]

    def _h_overview(self, q, ql, prog, scope):
        c = next((c for c in self.r.chunks if c["programmes"] == [prog] and c["topic"] == "programme"), None)
        if c is None:
            return None
        return Answer("answer", f"About the **{PROGRAMMES[prog]}**:", [c["units"][0]], [cite(c["section"])],
                      intent="overview", options=[{"label": l, "action": "ask", "value": l} for l in
                                                  ["How long is the programme?", "What subjects will I study?",
                                                   "What are the entry requirements?"]])

    def _h_duration(self, q, ql, prog, scope):
        programme_context = re.search(r"programme|program|master|phd|degree|stud(?:y|ying|ies)|working|"
                                      r"full[- ]?time|part[- ]?time|finish|complete", ql)
        bare_duration = re.fullmatch(r"(?:how long (?:is it|does it take|will it take)|"
                                     r"what is the duration|how many (?:years|months)(?: does it take)?)\??", ql)
        if (self._find_module(ql) and not re.search(r"programme|program|master|degree", ql)) or not (programme_context or bare_duration):
            return Answer("not_found", "Information not available: the handbook does not state the requested duration. "
                          "The programme's study duration does not establish that fact.", intent="duration")
        rows = {}
        for c in self._chunks(prog, "structure"):
            yr = years_range(" ".join(c["units"]))
            if yr and c["load"] not in rows:
                rows[c["load"]] = (yr, c["section"])
        if not rows:
            return None
        load = detect_load(q)
        loads = [load] if load in rows else [l for l in ("full time", "part time") if l in rows]
        parts = [f"{rows[l][0][0]}–{rows[l][0][1]} years {l}" for l in loads]
        details = [f"{l.capitalize()}: minimum {rows[l][0][0]} years, maximum {rows[l][0][1]} years"
                   .replace("minimum 1 years", "minimum 1 year") for l in loads]
        return Answer("answer", f"The **{PROGRAMMES[prog]}** takes " + " or ".join(parts) + ".",
                      details, [cite(rows[l][1]) for l in loads], intent="duration")

    @staticmethod
    def _module_codes(text):
        return list(dict.fromkeys(re.findall(r"\b[a-z]{2,5}\d{3}\b", text.upper(), re.I)))

    @staticmethod
    def _parse_module_row(unit, section, load=None, table="structure"):
        f = fields(unit)
        name = f.get("Subject Name") or f.get("Name of Course")
        if not name:
            return None
        return {"semester": f.get("Semester") or f.get("Semester / Year Offered", ""),
                "code": f.get("Subject Code") or f.get("Code", ""), "name": name,
                "cls": f.get("Classification", ""), "credits": f.get("Credit Hours") or f.get("Credit", ""),
                "section": section, "load": load, "table": table}

    def _module_rows(self, prog, load=None):
        return [row for c in self._chunks(prog) if c["topic"] in {"structure", "structure_summary"}
                and (not load or not c.get("load") or c["load"] == load)
                for u in c["units"]
                if (row := self._parse_module_row(u, c["section"], c.get("load"), c["topic"]))]

    def _match_module_rows(self, text, rows):
        """Resolve explicit codes first, then longest names / aliases. Keep all
        matches so duplicated codes in the source produce a clarification.
        """
        t = text.lower()
        codes = self._module_codes(text)
        names = [r["name"].lower() for r in rows if r["name"].lower() in t]
        name = max(names, key=len) if names else None
        if codes:
            return [r for r in rows if r["code"].upper() == codes[0] and
                    (not name or r["name"].lower() == name)]
        if name:
            return [r for r in rows if r["name"].lower() == name]
        for alias, target in sorted(MODULE_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(alias)}\b", t):
                return [r for r in rows if target in r["name"].lower()]
        return []

    def _resolve_module(self, q, prog):
        codes = self._module_codes(q)
        if len(codes) > 1:
            return [], Answer("clarify", "Which single module code do you mean? Please ask about one module at a time.")
        available = self._module_rows(prog, detect_load(q))
        rows = self._match_module_rows(q, available)
        if not rows:
            choices = [p for p in PROGRAMMES if p != prog and
                       self._match_module_rows(q, self._module_rows(p, detect_load(q)))]
            requested = codes[0] if codes else "The requested module"
            guidance = "Please check the module code or name." + (" You can also choose a listed programme below." if choices else "")
            return [], Answer("not_found", f"{requested} is not listed in the applicable {PROGRAMMES[prog]} structure or summary. {guidance}",
                              sources=list(dict.fromkeys(cite(r["section"]) for r in available)),
                              options=[{"label": SHORT[p], "action": "set_programme", "value": p} for p in choices])
        names = sorted({r["name"] for r in rows})
        if len(names) > 1:
            code = rows[0]["code"]
            return [], Answer("clarify", f"The handbook uses {code} for more than one module in {PROGRAMMES[prog]}. Which module do you mean?",
                              sources=list(dict.fromkeys(cite(r["section"]) for r in rows)),
                              options=[{"label": n, "action": "ask", "value": q + " — " + n} for n in names])
        return rows, None

    def _h_module_lookup(self, q, ql, prog, scope):
        rows, issue = self._resolve_module(q, prog)
        if issue:
            return issue
        module = f"{rows[0]['code']} {rows[0]['name']}"
        return Answer("clarify", f"What would you like to know about **{module}** in the {PROGRAMMES[prog]}?",
                      sources=list(dict.fromkeys(cite(r["section"]) for r in rows)),
                      options=[{"label": "Semester", "action": "ask", "value": f"Which semester is {module}?"},
                               {"label": "Credits", "action": "ask", "value": f"How many credits is {module}?"}])

    def _h_module_credits(self, q, ql, prog, scope):
        rows, issue = self._resolve_module(q, prog)
        if issue:
            return issue
        details = []
        for r in rows:
            table = "Summary table" if r["table"] == "structure_summary" else f"{r['load'].capitalize()} semester table"
            value = f"{r['credits']} credit hours" if r["credits"] else "the credit cell is blank (not zero credits)"
            detail = f"{table} ({cite(r['section']).split(' ')[0]}): {r['code']} {r['name']} — {value}."
            if detail not in details:
                details.append(detail)
        status = "answer" if any(r["credits"] for r in rows) else "not_found"
        return Answer(status, f"Module credit information for **{rows[0]['name']}** in the **{PROGRAMMES[prog]}**:",
                      details, list(dict.fromkeys(cite(r["section"]) for r in rows)), intent="module_credits")

    def _h_credits(self, q, ql, prog, scope):
        for c in self._chunks(prog, "structure_summary"):
            for u in c["units"]:
                m = re.match(r"Total Credit Hours[^:]*: (\d+)", u, re.I)
                if m:
                    return Answer("answer", f"The **{PROGRAMMES[prog]}** requires **{m.group(1)} credit hours** in total.",
                                  [], [cite(c["section"])], intent="credits")
        return None

    def _structure_rows(self, prog, load):
        """Parsed module rows from the semester structure table (falls back to the summary table)."""
        chunks = self._chunks(prog, "structure", load) or self._chunks(prog, "structure_summary")
        rows, section = [], chunks[0]["section"] if chunks else ""
        for c in chunks:
            for u in c["units"]:
                row = self._parse_module_row(u, c["section"], c.get("load"), c["topic"])
                if row:
                    rows.append(row)
        return rows, section

    def _summary_credits(self, prog, code, name):
        for c in self._chunks(prog, "structure_summary"):
            for u in c["units"]:
                f = fields(u)
                if f.get("Subject Code") == code and f.get("Subject Name") == name and f.get("Credit Hours"):
                    return f["Credit Hours"], cite(c["section"])
        return None, None

    def _fmt_row(self, prog, r, with_sem=True):
        if r["credits"]:
            cr = f"{r['credits']} credits"
        else:
            sc, src = self._summary_credits(prog, r["code"], r["name"])
            cr = f"credits blank in this table; {sc} in {src.split(' ')[0]} summary" if sc else "credits not stated"
        sem = f"{r['semester']} — " if with_sem and r["semester"] else ""
        return f"{sem}{r['code']} {r['name']} ({r['cls']}, {cr})"

    def _h_subjects(self, q, ql, prog, scope):
        load = detect_load(q) or "full time"
        rows, section = self._structure_rows(prog, load)
        if not rows:
            return None
        note = "" if detect_load(q) else " (ask about part-time study to see that sequence)"
        lead = f"Modules in the **{PROGRAMMES[prog]}**, {load} structure{note}:"
        return Answer("answer", lead, [self._fmt_row(prog, r) for r in rows], [cite(section)], intent="subjects")

    def _h_electives(self, q, ql, prog, scope):
        load = detect_load(q) or "full time"
        rows, section = self._structure_rows(prog, load)
        el = [r for r in rows if r["cls"].lower().startswith("elective")]
        if not el:
            return Answer("not_found", f"The {load} structure of the {PROGRAMMES[prog]} lists no electives.",
                          sources=[cite(section)] if section else [], intent="electives")
        groups = {}
        for r in el:
            groups.setdefault(r["semester"], []).append(r)
        details = [f"{sem or 'Elective group'} — choose 1: " + " OR ".join(self._fmt_row(prog, r, False) for r in rs)
                   for sem, rs in groups.items()]
        return Answer("answer", f"Electives in the **{PROGRAMMES[prog]}** ({load}) — you pick one from each group:",
                      details, [cite(section)], intent="electives")

    def _module_index(self):
        mods = {}
        for c in self.r.chunks:
            if c["topic"] in ("structure", "structure_summary"):
                for u in c["units"]:
                    f = fields(u)
                    name = f.get("Subject Name") or f.get("Name of Course")
                    code = f.get("Subject Code") or f.get("Code")
                    if name and code:
                        mods[name.lower()] = (code, name)
        return mods

    def _find_module(self, text):
        t = text.lower()
        code = re.search(r"\b((?:mds|mai|mcs|phd)\d{3})\b", t)
        for lname, (c, name) in self.modules.items():
            if lname in t:
                return c, name
        for alias, target in sorted(MODULE_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(alias)}\b", t):
                for lname, (c, name) in self.modules.items():
                    if target in lname:
                        return c, name
        if code:
            for lname, (c, name) in self.modules.items():
                if c.lower() == code.group(1):
                    return c, name
        return None

    def _h_module_semester(self, q, ql, prog, scope):
        rows, issue = self._resolve_module(q, prog)
        if issue:
            return issue
        details, sources = [], []
        for r in rows:
            if r["table"] == "structure" and r["semester"]:
                details.append(f"{r['load'].capitalize()}: {r['semester']} ({r['cls']})")
                sources.append(cite(r["section"]))
        if not details:
            return Answer("not_found", f"The semester for {rows[0]['name']} is not listed in the {PROGRAMMES[prog]} structure.",
                          intent="module_semester")
        return Answer("answer", f"**{rows[0]['code']} {rows[0]['name']}** in the {PROGRAMMES[prog]}:", list(dict.fromkeys(details)),
                      list(dict.fromkeys(sources)), intent="module_semester")

    def _h_admission(self, q, ql, prog, scope):
        chunks = self._chunks(prog, "admission")
        if not chunks:
            return None
        units = [u for c in chunks for u in c["units"]]
        src = [cite(chunks[0]["section"])]
        name = PROGRAMMES[prog]
        direct = [u for u in units if "no direct entry" in u.lower()]
        if direct and re.search(r"direct|straight|bachelor|first class|lecturer", ql):
            details = [u for u in units if "convert their candidacy" in u]
            if re.search(r"lecturer|told|advis|guidance", ql):
                details.append("Any differing lecturer guidance needs faculty confirmation for this programme and intake. "
                               "An exception is not documented in the available handbook.")
            return Answer("answer", f"For the **{name}**, the handbook states: {direct[0]}", details, src, intent="admission")
        if re.search(r"english|muet|ielts|cefr|toefl|international", ql) and not re.search(r"cgpa|degree", ql):
            sel = [u for u in units if re.search(r"MUET|CEFR|IELTS|English", u) and "Requirement (" not in u]
            return Answer("answer", f"English requirement for international applicants to the **{name}**:",
                          sel, src, intent="admission")
        if re.search(r"non[- ]?computing|not (?:in |a )?computing|different field|unrelated|business|arts|accounting", ql):
            sel = [u for u in units if re.search(r"non-Computing|without a Computing|prerequisite|re-assessed|^Rigorous assessment", u)]
            if not sel:
                return Answer("not_found", f"A non-Computing entry route is not stated for the {name}.", sources=src, intent="admission")
            return Answer("answer", f"Conditional routes for applicants **without a Computing degree** — {name}. "
                          "Admission is subject to the applicable route and conditions below:",
                          sel, src, intent="admission")
        routes = [u for u in units if re.match(r"^([ivx]+\.|-|A Master|Candidates must)", u) or "Bachelor" in u
                  or "APEL" in u]
        notes = [u for u in units if u not in routes and not u.startswith("Admission to")
                 and "Requirement (" not in u and u != "Notes" and not u.startswith("Additional Notes")]
        cgpas = [float(x) for u in routes for x in re.findall(r"minimum CGPA of (\d\.\d+)", u)]
        if cgpas and re.match(r"i\.", routes[0]):
            std = re.search(r"minimum CGPA of (\d\.\d+)", routes[0])
            lead = (f"Standard entry to the **{name}**: a Bachelor's degree (MQF Level 6) in a related field with "
                    f"a minimum CGPA of **{std.group(1)}**. Lower CGPAs (from **{min(cgpas):.2f}**) can be accepted "
                    f"subject to a rigorous assessment or prerequisite courses. You need to meet **any one** route.")
        else:
            lead = f"Entry requirements for the **{name}**:"
        return Answer("answer", lead, routes + notes, src, intent="admission", collapse=len(routes + notes) > 4)

    def _h_fail(self, q, ql, prog, scope):
        grading = [c for c in self.r.chunks if c["topic"] == "grading" and (not c["programmes"] or prog in c["programmes"])]
        units = [(u, c["section"]) for c in grading for u in c["units"]]
        pick = [x for x in units if re.search(r"RETAKE|REDO|repeat or retake|Additional Assignment|counselling", x[0])]
        prereq = [(u, c["section"]) for c in self.r.chunks if c["section_no"] == "10.2" for u in c["units"]
                  if "failed the first of a sequence" in u]
        if not pick:
            return None
        lead_u = next((u for u, _ in pick if "repeat or retake" in u), pick[0][0])
        details = [u for u, _ in pick if u != lead_u] + [f"Prerequisite rule: {u}" for u, _ in prereq]
        srcs = list(dict.fromkeys(cite(s) for _, s in pick + prereq))
        return Answer("answer", re.sub(r"^\d+\. ", "", lead_u), details, srcs, intent="fail")

    def _unit(self, section_no, pattern):
        for c in self.r.chunks:
            if c["section_no"] == section_no:
                for u in c["units"]:
                    if re.search(pattern, u):
                        return u, cite(c["section"])
        return None, None

    def _h_refund(self, q, ql, prog, scope):
        subject = re.search(r"(?:drop|withdraw).*\b(subject|module|class)\b|\b(subject|module)\b.*withdraw|drop (?:one|a|the|my) course", ql)
        if subject:
            u1, s = self._unit("10.5", r"No refund")
            u2, _ = self._unit("10.5", r"write officially")
            u3, _ = self._unit("10.5", r"approval of Management")
            return Answer("answer", f"Withdrawing from a **subject**: {u1}", [u2, u3], [s], intent="refund")
        if not re.search(r"programme|program|university|degree|master|\bquit\b|\bleave\b|drop out|discontinue", ql):
            return Answer("clarify", "Are you withdrawing from one subject or from the whole programme?", intent="refund",
                          options=[{"label": "One subject", "action": "ask", "value": q + " — withdrawing from one subject"},
                                   {"label": "Whole programme", "action": "ask", "value": q + " — withdrawing from the programme"}])
        rules = {k: self._unit("10.6", p) for k, p in {
            "before": r"before the commencement date", "within": r"within two weeks",
            "after": r"more than two weeks", "noshow": r"does not attend class for one month"}.items()}
        src = [rules["after"][1]]
        days = None
        number = r"\d+|one|two|three|four|five|six|a|an"
        m = re.search(rf"({number})\s*(day|week|month)s?\b", ql)
        week = re.search(rf"\b(?:in |during )?week\s+({number})\b", ql)
        if re.search(r"(?:after|since|from).*(?:application|applying|offer|payment|deposit paid)", ql) and not re.search(r"commencement|enrolment|enrollment|start", ql):
            return Answer("clarify", "How far is the withdrawal from the programme's commencement or enrolment date? "
                          "The refund timing is measured from that date, rather than an application, offer or payment.",
                          sources=src, intent="refund")
        if re.search(r"before (it |the programme |classes |the semester )?(start|begin|commenc)", ql):
            days = -1
        elif m or week:
            match = m or week
            n = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}.get(match.group(1))
            n = int(match.group(1)) if n is None else n
            # In week 3 is beyond the first two weeks; do not treat it as
            # an unspecified time or apply the within-two-weeks rule.
            days = n * {"day": 1, "week": 7, "month": 30}[m.group(2)] if m else (n - 1) * 7 + 1
        if days is None:
            return Answer("clarify", "When would you withdraw from the programme, relative to its commencement or enrolment date? "
                          "Please specify before it starts, a week number, or the number of days or weeks after it starts.",
                          sources=src, intent="refund")
        if days < 0:
            lead, extra = rules["before"][0], []
        elif days <= 14:
            lead, extra = rules["within"][0], []
        else:
            lead, extra = rules["after"][0], []
        if re.search(r"not attend|no.show|stop attending", ql) and days >= 30:
            lead, extra = rules["noshow"][0], []
        deposit_note, _ = self._unit("10.6", r"failure to return the HELP Student Identification Card")
        extra.append(deposit_note)
        when = "before the programme starts" if days < 0 else (f"in week {week.group(1)}" if week else f"after {m.group(0)}")
        extra.append("Timing above is relative to programme commencement or enrolment.")
        return Answer("answer", f"Withdrawing from the programme {when}: {lead}", extra, src, intent="refund")

    def _h_fee(self, q, ql, prog, scope):
        progname = PROGRAMMES[prog] if prog else "this programme"
        if re.search(r"continu|extension|extend|beyond|exceed|extra year|longer than|overstay", ql):
            u, s = self._unit("5.1", r"continuation fee")
            lead = (f"Tuition fee information is not currently available. If you are referring to studying beyond the standard duration: {u}"
                    if "tuition" in ql else u)
            return Answer("answer", lead, [], [s], intent="fee")
        if re.search(r"change of (programme )?status|switch(ing)? (to|from) (full|part)|change (to|from) (full|part)", ql):
            u, s = self._unit("10.4", r"Change of Status Fee")
            u0, _ = self._unit("10.4", r"NOT be charged")
            return Answer("answer", u, [u0, "This applies to the Master of Data Science only (§10.4)."], [s], intent="fee")
        if re.search(r"hostel|accommodation|housing|room|dorm", ql):
            u, s = self._unit("18.3", r"accommodation")
            return Answer("not_found", "Hostel and accommodation fees are not included in the available programme information.", [u], [s], intent="fee")
        if re.search(r"muet|ielts|test fee|exam fee", ql):
            return Answer("not_found", "English test fees are not included in the available programme information.",
                          [f"For further assistance, please contact {CONTACT}."], intent="fee")
        if re.search(r"tuition|course fee|programme fee|program fee|total (fee|cost)|how much (is|does) (the )?(master|phd|programme|program|course|degree)|"
                     r"\bfees?\b(?!.*(late|penalty|status|continuation))", ql):
            return Answer("not_found", f"Tuition fee information for the {progname} is not currently available.",
                          [f"For the latest fee details, please contact {CONTACT}."], intent="fee")
        return None

    def _h_intake(self, q, ql, prog, scope):
        return Answer("not_found", "Intake dates are not included in the available programme information.",
                      [f"For upcoming intake dates, please contact {CONTACT}."], intent="intake")

    def _h_internship(self, q, ql, prog, scope):
        lead = (f"There is no stated internship or industrial training requirement for the "
                f"**{PROGRAMMES[prog]}**.")
        rel = []
        for c in self._chunks(prog, "subject_descriptions"):
            for u in c["units"]:
                if re.search(r"industry", u) and re.search(r"project|dissertation", u, re.I):
                    rel.append((u, c["section"]))
        if not rel:
            return Answer("not_found", lead, [], intent="internship")
        u, s = rel[0]
        return Answer("related", lead + " Related information: the project module may involve industry partners.",
                      [u], [cite(s)], intent="internship")

    def _h_hours(self, q, ql, prog, scope):
        u, s = self._unit("5.2", r"hours of teaching")
        if prog == "MDS":
            return Answer("answer", f"Master of Data Science: {u.lstrip('3. ')}",
                          ["Self-study hours are not specified."], [s], intent="hours")
        return Answer("not_found", f"Weekly class hours and self-study time are not specified for the "
                                   f"{PROGRAMMES[prog]}.",
                      ["Weekly teaching hours (five hours per subject) are specified only for the Master of Data Science (§5.2)."],
                      intent="hours")

    # ================= 4. general retrieval =========================== #
    def _general(self, q, scope, prog) -> Answer:
        hits = self.r.search(q, k=6, scope=scope or None)
        if not hits:
            return Answer("not_found", NOT_FOUND)
        top = hits[0]
        coverage = self.r.term_coverage(q, hits[:3])
        if top.tfidf < MIN_TFIDF or coverage < MIN_COVERAGE:
            return Answer("not_found", NOT_FOUND, [f"Please try rephrasing your question, or ask about {EXAMPLES}."],
                          confidence=top.tfidf, coverage=coverage, hits=hits)
        lines = self._extract(q, hits[:2])
        if not lines:
            return Answer("not_found", NOT_FOUND, confidence=top.tfidf, coverage=coverage, hits=hits)
        (lead, lead_src), rest = lines[0], lines[1:]
        srcs = list(dict.fromkeys(cite(s) for _, s in lines))
        return Answer("answer", lead, [u for u, _ in rest], srcs,
                      confidence=top.tfidf, coverage=coverage, hits=hits)

    def _extract(self, question, hits, max_lines=4):
        """Rank individual sentences / rows of the top chunks by TF-IDF similarity."""
        units, owners = [], []
        for h in hits:
            for u in h.chunk["units"]:
                if len(u.split()) <= 4 and not u.endswith("."):
                    continue                       # skip heading-like fragments ('Dean', 'Notes')
                units.append(u)
                owners.append(h.chunk["section"])
        if not units:
            return []
        vec = self.r.vectorizer
        sims = cosine_similarity(vec.transform([expand_query(question)]), vec.transform(units)).ravel()
        best = sims.max()
        ranked = [i for i in sims.argsort()[::-1][:max_lines] if sims[i] >= max(0.05, 0.45 * best)]
        lead = ranked[0]
        rest = sorted(i for i in ranked[1:])
        out = [(units[lead], owners[lead])]
        for i in rest:
            out.append((units[i], owners[i]))
            if units[i].rstrip().endswith(":") and i + 1 < len(units):
                out.append((units[i + 1], owners[i + 1]))
        return list(dict.fromkeys(out))


if __name__ == "__main__":
    bot = Chatbot()
    print("Handbook chatbot – type a question (blank line to quit)")
    while True:
        q = input("\nYou: ")
        if not q.strip():
            break
        a = bot.ask(q)
        print(f"\n[{a.status}] " + a.text)
        if a.sources:
            print("Sources: " + " | ".join(a.sources))
        for o in a.options:
            print(f"  [option] {o['label']}")
