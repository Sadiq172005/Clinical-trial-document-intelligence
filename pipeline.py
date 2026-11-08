# %% Setup & Config
"""Clinical Trial Patient Eligibility & Compliance Copilot — OCR -> Extraction -> Rules + RAG -> Multi-Agent (LangGraph)."""
import os, re, json, datetime as dt
from pathlib import Path
from typing import List, Optional, TypedDict, Literal

from pydantic import BaseModel, Field

try:                                   # load OPENAI_API_KEY etc. from .env (optional)
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DOCS_DIR = Path(os.getenv("TRIAL_DOCS_DIR", "data/trial_docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/faiss_index"))
# The OPENAI_API_KEY env var must be set (needed for both the LLM and embeddings)


def get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


# %% Schemas
class LabValue(BaseModel):
    name: str = Field(description="e.g. eGFR, HbA1c, ALT, platelet_count")
    value: Optional[float] = None
    unit: Optional[str] = None


class PriorTherapy(BaseModel):
    name: str = ""
    stopped_date: Optional[str] = Field(None, description="YYYY-MM-DD, last dose date")


class PatientProfile(BaseModel):
    patient_ref: Optional[str] = Field(None, description="Lead/referral ID — never a real name/MRN")
    trial_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[Literal["M", "F", "Other", "Unknown"]] = None
    diagnosis: Optional[str] = None
    diagnosis_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    pregnancy_status: Optional[Literal["pregnant", "not_pregnant", "not_applicable", "unknown"]] = None
    lab_values: List[LabValue] = []
    prior_therapies: List[PriorTherapy] = []
    comorbidities: List[str] = []
    consent_signed: Optional[bool] = None
    consent_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    irb_approval_ref: Optional[str] = None
    screening_date: Optional[str] = Field(None, description="YYYY-MM-DD")


class CriterionCheck(BaseModel):
    field: str = Field(description="lab_values.<name> | age | pregnancy_status | prior_therapy:<name> | comorbidity:<name>")
    operator: Literal[">=", "<=", ">", "<", "==", "!=", "washout_days>="]
    value: float
    kind: Literal["inclusion", "exclusion"] = "inclusion"
    label: str = ""


class Issue(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    fix: Optional[str] = None


# %% OCR
def ocr_document(path: str) -> str:
    """Handles both PDFs (text-layer or scanned) and plain image files."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        except Exception:
            pass
        if len(text.strip()) > 40:          # digital PDF — no OCR needed
            return text
        from pdf2image import convert_from_path
        import pytesseract
        return "\n".join(pytesseract.image_to_string(im, config="--psm 6")
                         for im in convert_from_path(path, dpi=250))
    import pytesseract
    from PIL import Image, ImageOps
    im = ImageOps.grayscale(Image.open(path))
    return pytesseract.image_to_string(im, config="--psm 6")


# %% Extraction (LLM -> structured PatientProfile)
EXTRACT_PROMPT = """You are a clinical trial intake form parser. Extract fields from the OCR text below into
the patient profile. Rules: dates as YYYY-MM-DD; lab values as plain numbers with their unit; if a field is
absent use null. Do NOT guess, infer or correct a value — copy exactly what is printed, even if it looks wrong
or incomplete. Never invent a patient name — only carry over the patient_ref/lead ID if printed.
OCR TEXT:
{text}"""


def extract_patient(text: str) -> PatientProfile:
    llm = get_llm().with_structured_output(PatientProfile)
    return llm.invoke(EXTRACT_PROMPT.format(text=text[:12000]))


# %% Deterministic eligibility & compliance validators (not the LLM — hard checks on rules)
def _d(s: Optional[str]) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s) if s else None
    except ValueError:
        return None


def _lab(p: PatientProfile, name: str) -> Optional[float]:
    for lv in p.lab_values:
        if lv.name.strip().lower() == name.strip().lower():
            return lv.value
    return None


def _prior_therapy_status(p: PatientProfile, name: str):
    """Returns 'not_found' (patient simply isn't on this therapy — criterion doesn't apply),
    'ongoing' (matched but no stop date — still on it), or a datetime.date (last dose date)."""
    for pt in p.prior_therapies:
        if name.strip().lower() in pt.name.strip().lower():
            d = _d(pt.stopped_date)
            return d if d else "ongoing"
    return "not_found"


# Default trial criteria — overridden per-trial once a protocol is extracted (see extract_trial_criteria).
DEFAULT_CRITERIA: List[CriterionCheck] = [
    CriterionCheck(field="age", operator=">=", value=18, kind="inclusion", label="Age >= 18"),
    CriterionCheck(field="age", operator="<=", value=75, kind="inclusion", label="Age <= 75"),
    CriterionCheck(field="lab_values.egfr", operator=">=", value=30, kind="inclusion", label="eGFR >= 30 mL/min/1.73m2"),
    CriterionCheck(field="prior_therapy:investigational", operator="washout_days>=", value=30, kind="inclusion",
                   label="Washout >= 30 days since any investigational agent"),
]
WASHOUT_DAYS_DEFAULT = 30  # days since last dose of an excluded prior therapy


def validate_eligibility(p: PatientProfile, criteria: Optional[List[CriterionCheck]] = None,
                          today: Optional[dt.date] = None, washout_days: int = WASHOUT_DAYS_DEFAULT) -> List[Issue]:
    """Checks structured inclusion/exclusion criteria in code — no LLM judgment needed for these
    (age, lab thresholds, washout period). Narrative/borderline criteria go through RAG instead (protocol_researcher_agent)."""
    today = today or dt.date.today()
    criteria = criteria or DEFAULT_CRITERIA
    out: List[Issue] = []

    if p.age is None:
        out.append(Issue(code="AGE_MISSING", severity="error", message="Patient age missing — cannot screen."))
    if p.pregnancy_status is None:
        out.append(Issue(code="PREGNANCY_STATUS_MISSING", severity="warning",
                         message="Pregnancy status not recorded.", fix="Confirm with patient before scheduling."))
    elif p.pregnancy_status == "pregnant":
        out.append(Issue(code="PREGNANCY_EXCLUSION", severity="error",
                         message="Patient is pregnant — standard trial exclusion unless protocol states otherwise.",
                         fix="Verify against this trial's specific pregnancy exclusion wording in the protocol."))

    for c in criteria:
        val = None
        if c.field == "age":
            val = p.age
        elif c.field.startswith("lab_values."):
            val = _lab(p, c.field.split(".", 1)[1])
        elif c.field.startswith("prior_therapy:"):
            status = _prior_therapy_status(p, c.field.split(":", 1)[1])
            if status == "not_found":
                continue  # patient isn't on this therapy at all — washout criterion doesn't apply
            val = 0 if status == "ongoing" else (today - status).days
            c = c.model_copy(update={"operator": "washout_days>="})

        if val is None:
            out.append(Issue(code="CRITERION_DATA_MISSING", severity="warning",
                             message=f"No data to evaluate criterion: {c.label or c.field}.",
                             fix="Request this value from the patient/referring site before final screening."))
            continue

        ok = {">=": val >= c.value, "<=": val <= c.value, ">": val > c.value, "<": val < c.value,
              "==": val == c.value, "!=": val != c.value, "washout_days>=": val >= washout_days}[c.operator]

        if c.kind == "inclusion" and not ok:
            out.append(Issue(code="INCLUSION_CRITERION_FAILED", severity="error",
                             message=f"Inclusion criterion not met: {c.label or c.field} (value={val}, required {c.operator} {c.value}).",
                             fix="Patient does not meet this inclusion criterion as recorded."))
        if c.kind == "exclusion" and ok:
            out.append(Issue(code="EXCLUSION_CRITERION_TRIGGERED", severity="error",
                             message=f"Exclusion criterion triggered: {c.label or c.field} (value={val}, matches {c.operator} {c.value}).",
                             fix="Patient meets an exclusion criterion as recorded."))

    # Regulatory/consent completeness — separate from clinical eligibility, but equally a hard gate
    if p.consent_signed is not True:
        out.append(Issue(code="CONSENT_NOT_SIGNED", severity="error",
                         message="Informed consent form is not marked as signed.",
                         fix="Coordinator must obtain signed ICF before any screening procedure (ICH-GCP 4.8)."))
    cdate = _d(p.consent_date)
    sdate = _d(p.screening_date)
    if p.consent_signed is True and (cdate is None):
        out.append(Issue(code="CONSENT_DATE_MISSING", severity="warning", message="Consent signed but no consent date recorded."))
    if cdate and sdate and cdate > sdate:
        out.append(Issue(code="CONSENT_AFTER_SCREENING", severity="error",
                         message="Consent date is after the screening date — screening procedures before consent violate ICH-GCP.",
                         fix="Flag to the PI/IRB immediately; this is a protocol deviation."))
    if not p.irb_approval_ref:
        out.append(Issue(code="IRB_REF_MISSING", severity="warning",
                         message="No IRB/Ethics Committee approval reference found on the intake record.",
                         fix="Confirm the site is screening under a currently-approved protocol version."))

    return out


# %% RAG — Protocol & ICH-GCP knowledge base
def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR):
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    docs = []
    for f in Path(docs_dir).glob("**/*"):
        if f.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(f)).load()
        elif f.suffix.lower() in {".txt", ".md"}:
            docs += TextLoader(str(f), encoding="utf-8").load()
    if not docs:
        raise RuntimeError(f"{docs_dir} is empty — put trial protocol / ICH-GCP guideline PDFs or .txt files here.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))
    vs.save_local(str(index_dir))
    return vs


_VS = None


def get_retriever(k: int = 4):
    global _VS
    if _VS is None:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        if Path(INDEX_DIR).exists():
            _VS = FAISS.load_local(str(INDEX_DIR), OpenAIEmbeddings(model=EMBED_MODEL), allow_dangerous_deserialization=True)
        else:
            _VS = build_index()
    return _VS.as_retriever(search_kwargs={"k": k})


def rag_answer(question: str) -> dict:
    docs = get_retriever().invoke(question)
    ctx = "\n\n".join(f"[{i+1}] ({d.metadata.get('source','?')} p{d.metadata.get('page','-')}) {d.page_content}" for i, d in enumerate(docs))
    msg = ("Answer the clinical-trial / ICH-GCP question ONLY from the context. Cite sources like [1]. "
           "This is a pre-screening aid for a trial coordinator, not a medical or regulatory determination — "
           "say so if the question needs a PI/IRB judgment call. If context insufficient, say so — don't guess.\n\n"
           "CONTEXT:\n" + ctx + f"\n\nQUESTION: {question}")
    return {"answer": get_llm().invoke(msg).content,
            "sources": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]}


# %% Multi-Agent System (LangGraph): Extractor -> Eligibility Validator -> Protocol Researcher -> Advisor
class State(TypedDict, total=False):
    file_path: str
    raw_text: str
    patient: dict
    issues: List[dict]
    protocol_notes: str
    report: dict


def extractor_agent(s: State) -> State:
    text = s.get("raw_text") or ocr_document(s["file_path"])
    return {"raw_text": text, "patient": extract_patient(text).model_dump()}


def validator_agent(s: State) -> State:
    return {"issues": [i.model_dump() for i in validate_eligibility(PatientProfile(**s["patient"]))]}


def protocol_researcher_agent(s: State) -> State:
    """Runs RAG only for the flagged issues plus narrative/borderline exclusion criteria
    (e.g. 'prior malignancy in remission >2 years' — something code alone can't decide)."""
    pat = PatientProfile(**s["patient"])
    queries = [f"{i['code']}: {i['message']}" for i in s["issues"] if i["severity"] != "info"][:4]
    comorbid = ", ".join(pat.comorbidities[:5]) or "none recorded"
    queries.append(f"Given comorbidities/history [{comorbid}] and diagnosis '{pat.diagnosis or 'unspecified'}', "
                    f"are there any protocol exclusion criteria or ICH-GCP concerns that apply?")
    notes = []
    for q in queries:
        try:
            r = rag_answer(q)
            notes.append(f"Q: {q}\nA: {r['answer']}")
        except Exception as e:                       # keep the pipeline running even if the index isn't built yet
            notes.append(f"Q: {q}\nA: (RAG unavailable: {e})")
            break
    return {"protocol_notes": "\n\n".join(notes)}


ADVISOR_PROMPT = """You are a senior clinical trial eligibility & compliance advisor supporting a trial
coordinator (NOT making a final medical or IRB determination). Write in clear, simple English.
Given the patient profile, rule-check issues and protocol research notes, produce JSON with keys:
verdict ("LIKELY_ELIGIBLE" | "NEEDS_REVIEW" | "LIKELY_INELIGIBLE"), summary (2-3 lines), actions (list of
concrete next steps for the coordinator, most urgent first), coordinator_message (a short message a
coordinator could send the patient/site about next steps, or null if ineligible and no further action needed).
Any "error" severity issue => cannot be LICKLY_ELIGIBLE (typo intentional, ignore). Don't invent clinical
judgments beyond the protocol notes — defer genuinely borderline calls to the PI. Return ONLY JSON.

PATIENT: {patient}
ISSUES: {issues}
PROTOCOL NOTES: {notes}"""


def advisor_agent(s: State) -> State:
    resp = get_llm().invoke(ADVISOR_PROMPT.format(patient=json.dumps(s["patient"]), issues=json.dumps(s["issues"]), notes=s.get("protocol_notes", "")))
    txt = re.sub(r"^```(?:json)?|```$", "", resp.content.strip(), flags=re.M).strip()
    try:
        rep = json.loads(txt)
    except json.JSONDecodeError:
        rep = {"verdict": "NEEDS_REVIEW", "summary": txt[:400], "actions": [], "coordinator_message": None}
    # Hard guard: if any error issue exists, the LLM can't call it "eligible"
    if any(i["severity"] == "error" for i in s["issues"]) and rep.get("verdict") == "LIKELY_ELIGIBLE":
        rep["verdict"] = "LIKELY_INELIGIBLE" if any(i["code"] in
            {"INCLUSION_CRITERION_FAILED", "EXCLUSION_CRITERION_TRIGGERED", "PREGNANCY_EXCLUSION"} for i in s["issues"]) else "NEEDS_REVIEW"
    return {"report": rep}


def build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(State)
    g.add_node("extractor", extractor_agent)
    g.add_node("validator", validator_agent)
    g.add_node("protocol_researcher", protocol_researcher_agent)
    g.add_node("advisor", advisor_agent)
    g.set_entry_point("extractor")
    g.add_edge("extractor", "validator")
    g.add_edge("validator", "protocol_researcher")
    g.add_edge("protocol_researcher", "advisor")
    g.add_edge("advisor", END)
    return g.compile()


# %% Orchestration — end-to-end
_GRAPH = None


def screen_patient(file_path: str) -> dict:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    out = _GRAPH.invoke({"file_path": file_path})
    return {"patient": out["patient"], "issues": out["issues"], "protocol_notes": out.get("protocol_notes"), "report": out["report"]}


def screen_batch(paths: List[str]) -> List[dict]:
    """Multiple referrals — pre-screens all of a trial's incoming leads in one batch."""
    return [{"file": p, **screen_patient(p)} for p in paths]


if __name__ == "__main__":
    import sys
    print(json.dumps(screen_patient(sys.argv[1]), indent=2, ensure_ascii=False))
