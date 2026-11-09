"""Text preprocessing shared by indexing and querying.

Pipeline: lowercase -> tokenise -> remove stop words -> Porter stemming.
Queries are additionally expanded with a small domain synonym dictionary so that
student wording ("how long", "subjects", "entry requirements") matches the
handbook's wording ("duration", "modules", "admission").
"""
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()

# Generic stop words plus words that appear everywhere in the handbook and
# therefore carry no retrieval signal.
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "help", "university", "please", "tell", "know", "want", "does", "do",
    "i", "me", "my", "can", "could", "would", "will", "shall", "is", "are",
    "what", "which", "who", "how", "when", "where", "there", "any",
    # question filler verbs that rarely occur in the handbook
    "take", "takes", "finish", "need", "needs", "get", "allowed", "happens",
    "happen", "use", "using", "much", "many", "available", "offer", "offered",
}
# Keep words that matter for intent detection.
STOP_WORDS -= {"full", "part", "first", "second", "third"}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Query-side synonym / paraphrase expansion (keys are matched on the raw,
# lower-cased query; values are appended as extra query text).
SYNONYMS = {
    r"\bhow long\b|\blength\b|\byears?\b(?! of)|\bmonths?\b": "duration minimum maximum years",
    r"\bsubjects?\b|\bcourses?\b|\bmodules?\b|\bstudy\b|\blearn\b": "subject module course programme structure",
    r"\bentry\b|\badmission\b|\bqualif|\beligib|\bapply\b|\bget in\b|\bentrance\b|\bcgpa\b": "admission requirements bachelor degree cgpa",
    r"\benglish\b|\bielts\b|\bmuet\b|\btoefl\b": "english competency muet cefr ielts",
    r"\bcredits?\b": "credit hours",
    r"\binternship\b|\bindustrial training\b|\bplacement\b|\bindustry\b": "industry supervisor project dissertation partner",
    r"\bcareers?\b|\bjobs?\b|\bemploy|\bopportunit": "career graduates industry professionals",
    r"\bfinal year project\b|\bfyp\b|\bthesis\b|\bdissertation\b": "project report dissertation thesis",
    r"\bgrades?\b|\bmarks?\b|\bgpa\b|\bpass\b|\bfail|\bdistinction\b": "grade grading mark pass",
    r"\bfail(ed)?\b|\bretake\b|\bresit\b|\brepeat\b": "retake redo referred fail attempt",
    r"\bonline\b|\bdistance\b|\bodl\b|\bremote\b": "odl open distance learning",
    r"\bpart[- ]?time\b": "part time",
    r"\bfull[- ]?time\b": "full time",
    r"\bwithdraw|\bquit\b|\bdrop out\b|\brefund\b": "withdrawal withdraw refund",
    r"\bhostel\b|\bdorm|\bhousing\b|\blive\b": "accommodation",
    r"\bvisa\b|\bstudent pass\b|\binternational\b": "international student pass visa issd",
    r"\blibrary\b": "learning resource centre lrc library",
    r"\bportal\b|\bresults?\b": "mypride portal results",
    r"\bexempt|\btransfer\b": "credit exemption transfer",
    r"\bassess|\bexam|\bweight": "assessment continuous final weightage",
    r"\bplagiar|\bcheat|\bchatgpt\b|\bgenerative\b|\bai tools?\b": "academic integrity plagiarism generative ai",
    r"\bsupervis": "supervisor appointment principal",
    r"\blecturers?\b|\bstaff\b|\bresearchers?\b": "supervisor researcher lecturer staff",
    r"\battendance\b|\battend\b": "attendance",
    r"\bopening hours\b|\boffice\b|\bopen\b": "faculty office open hours",
    r"\bcounsel": "counselling",
    r"\bdean\b": "dean faculty",
    r"\bassignments?\b|\bhomework\b|\bcoursework\b": "assignment assessment work",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalise(text: str) -> list[str]:
    """Tokenise, drop stop words and stem. Used for both documents and queries."""
    return [_stemmer.stem(t) for t in tokenize(text) if t not in STOP_WORDS and len(t) > 1]


def expand_query(query: str) -> str:
    q = query.lower()
    extra = [exp for pattern, exp in SYNONYMS.items() if re.search(pattern, q)]
    return query + (" " + " ".join(extra) if extra else "")


def query_concepts(query: str) -> list[set[str]]:
    """Split a question into 'concepts' for the coverage check.

    Each synonym rule that fires is one concept (any of its words counts as a
    match); every remaining content word is its own concept.
    """
    q = query.lower()
    concepts = []
    for pattern, exp in SYNONYMS.items():
        m = list(re.finditer(pattern, q))
        if m:
            concepts.append(set(normalise(" ".join(x.group(0) for x in m) + " " + exp)))
            q = re.sub(pattern, " ", q)
    concepts += [{t} for t in normalise(q)]
    return concepts


def analyzer(text: str) -> list[str]:
    """Analyzer for scikit-learn vectorisers: stemmed unigrams + bigrams."""
    toks = normalise(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
