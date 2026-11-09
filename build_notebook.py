"""pipeline.py + extra sections -> clinical_trial_eligibility_copilot.ipynb  (run: python build_notebook.py)"""
import re, nbformat as nbf
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
src = open("pipeline.py").read()
sections = dict(zip(*[iter(re.split(r"^# %% (.+)$", src, flags=re.M)[1:])] * 2))

nb = nbf.v4.new_notebook(); c = nb.cells
c += [M("""# Clinical Trial Patient Eligibility & Compliance Copilot — Capstone
**Problem:** Screening incoming patient referrals/leads for eligibility by hand is slow, and consent/IRB
documentation gaps are easy to miss — both lead to enrollment delays or protocol deviations.
**Solution:** OCR → LLM extraction → deterministic eligibility + compliance rule engine → RAG over protocol/ICH-GCP → 4-agent LangGraph → pre-screening verdict + actions.

**Contents:** 0 Setup · 1 Architecture · 2 Data · 3 Schemas · 4 OCR · 5 Extraction · 6 Rule engine · 7 RAG · 8 Agents · 9 Orchestration · 10 Demo · 11 Evaluation · 12 API + UI · 13 Deployment · 14 Limitations & next steps"""),
      M("## 0. Setup\nSystem packages (outside the notebook): `sudo apt install tesseract-ocr poppler-utils` (Mac: `brew install tesseract poppler`).\nPython: `pip install -e \".[all]\"` (from pyproject.toml) · set `OPENAI_API_KEY` in `.env` (copy `.env.example`)."),
      C("import shutil, os\nfrom dotenv import load_dotenv; load_dotenv()\nfor b in ['tesseract', 'pdftoppm']:\n    print(b, '✅' if shutil.which(b) else '❌ needs installing')\nprint('OPENAI_API_KEY', '✅' if os.getenv('OPENAI_API_KEY') else '❌ set it in .env')"),
      M("""## 1. Architecture
```
Patient intake form (PDF/scan) -> OCR -> Extractor Agent -> Eligibility Validator Agent (code rules) -> Protocol Researcher Agent (RAG) -> Advisor Agent -> verdict/actions/coordinator message
```
**Design decisions:** (1) age/lab/washout cutoffs are checked in **code** — so the LLM can't hallucinate them;
(2) the LLM is used only for extraction, narrative protocol interpretation, and advice; (3) a hard guard stops the
advisor from saying "ELIGIBLE" whenever an error is present; (4) RAG answers come with sources, and explicitly call
out a "PI/IRB judgment call" when one applies; (5) only synthetic/lead-ref data is ever used — no real patient
name/MRN is stored."""),
      M("## 2. Data\n`data/trial_docs/` holds the protocol's relevant sections + ICH-GCP guideline text (full checklist: `data/README.md`). Test intake forms: `python generate_samples.py`."),
      C("from pathlib import Path\ndocs = list(Path('data/trial_docs').glob('**/*'))\nprint(len(docs), 'docs:', [d.name for d in docs][:10])\nprint(len(list(Path('samples').glob('*.pdf'))), 'sample PDFs')")]
for title in ["Setup & Config", "Schemas", "OCR", "Extraction (LLM -> structured PatientProfile)",
              "Deterministic eligibility & compliance validators (not the LLM — hard checks on rules)",
              "RAG — Protocol & ICH-GCP knowledge base", "Multi-Agent System (LangGraph): Extractor -> Eligibility Validator -> Protocol Researcher -> Advisor",
              "Orchestration — end-to-end"]:
    code = sections[title].strip().replace('if __name__ == "__main__":', "if False:  # CLI — skipped in the notebook")
    hdr = {"Setup & Config": "### Config & LLM factory", "Schemas": "## 3. Schemas (Pydantic)", "OCR": "## 4. OCR\nDigital PDF -> text layer; scanned -> pdf2image + Tesseract.",
           "Extraction (LLM -> structured PatientProfile)": "## 5. Structured extraction",
           "Deterministic eligibility & compliance validators (not the LLM — hard checks on rules)": "## 6. Rule engine (deterministic)",
           "RAG — Protocol & ICH-GCP knowledge base": "## 7. RAG over protocol + ICH-GCP",
           "Multi-Agent System (LangGraph): Extractor -> Eligibility Validator -> Protocol Researcher -> Advisor": "## 8. Multi-agent system (LangGraph)",
           "Orchestration — end-to-end": "## 9. Orchestration"}[title]
    c += [M(hdr), C(code)]
    if title.startswith("Deterministic"):
        c += [M("**Offline demo — no API key needed:**"), C("""p = PatientProfile(age=79, pregnancy_status='not_applicable', consent_signed=True, consent_date='2026-10-01',
  screening_date='2026-10-05', irb_approval_ref='IRB-2026-0091',
  lab_values=[LabValue(name='eGFR', value=58, unit='mL/min/1.73m2')])
for i in validate_eligibility(p): print(i.severity.upper().ljust(8), i.code.ljust(28), i.message)""")]
    if title.startswith("RAG"):
        c += [M("Build the index the first time (and again whenever the docs change):"), C("# build_index()   # uncomment — uses OpenAI embeddings (small, one-time cost)\n# rag_answer('Does a 2-year cancer remission satisfy the malignancy exclusion window?')")]
c += [M("## 10. End-to-end demo"),
      C("import json\nres = screen_patient('samples/02_age_out_of_range.pdf')\nprint(json.dumps(res['report'], indent=2, ensure_ascii=False))\nfor i in res['issues']: print(i['severity'], i['code'], '-', i['message'])"),
      M("## 11. Evaluation\nMatches each verdict against `samples/expected.json` (`python evaluate.py`). Rule-engine unit tests: `python test_validators.py`."),
      C("import json\nexp = json.load(open('samples/expected.json')); rows = []\nfor f, e in exp.items():\n    r = screen_patient(f'samples/{f}'); got = r['report']['verdict']\n    rows.append((f, e['expected_verdict'], got, 'PASS' if got == e['expected_verdict'] else 'FAIL'))\nimport pandas as pd; pd.DataFrame(rows, columns=['file', 'expected', 'got', 'result'])"),
      M("## 12. API + UI\n`uvicorn app:app --reload` -> `/docs` · `streamlit run streamlit_app.py`. Endpoints: `/screen`, `/screen-batch`, `/ask`, `/reindex`, `/health`."),
      M("## 13. Deployment (AWS)\nDocker -> ECR -> App Runner (port 8000, `OPENAI_API_KEY` via Secrets Manager, health check `/health`). Keep the FAISS index on S3, and use a VPC-only/Cognito-gated endpoint since this handles patient data. Full commands: `README.md`."),
      M("""## 14. Limitations & next steps
- `DEFAULT_CRITERIA` is a demo trial's numeric cutoffs — real use needs per-trial criteria extracted from that
  trial's actual protocol (an `extract_trial_criteria`-style LLM step, or a maintained `criteria_overrides.json`).
- Narrative/borderline exclusion judgment (prior malignancy remission windows, ambiguous comorbidity histories) is
  explicitly routed to RAG + human PI review — this tool never makes that call on its own.
- OCR accuracy is lower on handwritten/low-quality scans; a vision-LLM fallback could be added.
- Live protocol amendments aren't tracked yet — reindex the docs whenever the protocol version changes.
- This is a pre-screening *assistant*, not a substitute for the PI/IRB. Follow PHI handling/consent/HIPAA-DPDP
  requirements for any real patient data — never commit real PHI to this repo or the vector index.""")]
nbf.write(nb, "clinical_trial_eligibility_copilot.ipynb"); print("cells:", len(c))
