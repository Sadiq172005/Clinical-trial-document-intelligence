# Clinical Trial Document Intelligence

Upload a patient referral/intake document (PDF/scan) → OCR → structured extraction → eligibility rule checks → protocol/GCP-grounded RAG → **pre-screening verdict + action list + coordinator message**.

## Architecture
```
Upload → OCR (pdfplumber / Tesseract) → Extractor Agent (LLM → Pydantic PatientProfile)
       → Eligibility Validator Agent (deterministic: age range, lab thresholds, washout period,
         pregnancy exclusion, consent-signed/dated, IRB approval ref)
       → Protocol Researcher Agent (RAG over protocol + ICH-GCP, borderline/narrative exclusion judgment calls)
       → Advisor Agent (verdict, actions, coordinator message)      [LangGraph StateGraph]
FastAPI: /screen  /screen-batch  /ask  /reindex  /health   →  Docker → AWS
```
Design choice: **numeric thresholds are checked in code, not by the LLM** — the LLM is only used for extraction,
narrative protocol interpretation, and advice. The advisor can only return "LIKELY_ELIGIBLE" when there is no hard
error (hard guard). Built to plug straight into a lead-recruitment dashboard: `/screen` is the exact call made right
after a lead submits a pre-screening intake form.

## Quick start (pyproject)
```bash
cp .env.example .env            # add your OPENAI_API_KEY
make install samples test index eval
make api   # or: make ui
```
Data checklist: `data/README.md`. Regenerate the notebook with: `python build_notebook.py`.

## Run locally
```bash
pip install -r requirements.txt          # + apt: tesseract-ocr poppler-utils
export OPENAI_API_KEY=sk-...
python test_validators.py                # offline tests (no key needed)
# add protocol sections + ICH-GCP guideline text to data/trial_docs/
uvicorn app:app --reload                 # http://localhost:8000/docs
curl -F file=@samples/02_age_out_of_range.pdf localhost:8000/screen
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"Does a 2-year cancer remission satisfy the malignancy exclusion window?"}'
```

## Samples + Streamlit UI
```bash
pip install -r requirements-dev.txt
python generate_samples.py               # writes 9 PDFs + 1 scanned PNG + expected.json to samples/
streamlit run streamlit_app.py           # http://localhost:8501
```
| Sample | Expected |
|---|---|
| 01_clean_eligible | LIKELY_ELIGIBLE |
| 02_age_out_of_range | INCLUSION_CRITERION_FAILED (age) |
| 03_low_egfr | INCLUSION_CRITERION_FAILED (eGFR) |
| 04_consent_not_signed | CONSENT_NOT_SIGNED |
| 05_consent_after_screening | CONSENT_AFTER_SCREENING (protocol deviation) |
| 06_pregnant_exclusion | PREGNANCY_EXCLUSION |
| 07_recent_excluded_therapy | INCLUSION_CRITERION_FAILED (washout <30 days) |
| 08_missing_irb_ref | IRB_REF_MISSING (warning only) |
| 09_borderline_remission_history | NEEDS_REVIEW via RAG (remission-window judgment call) |
| 10_scanned_clean.png | OCR path (noisy scan — tests the extractor's real-world robustness) |

The UI has 3 tabs: single patient (upload or pick a sample), batch (CSV download), and Ask protocol/GCP. Enter a
deployed API URL in the sidebar to have the same UI hit the AWS-deployed service instead of the local pipeline.

## AWS deployment (simplest: ECR + App Runner)
```bash
aws ecr create-repository --repository-name clinical-trial-copilot --region ap-south-1
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com
docker build -t clinical-trial-copilot . && docker tag clinical-trial-copilot:latest <ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com/clinical-trial-copilot:latest
docker push <ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com/clinical-trial-copilot:latest
```
Then in the Console → **App Runner → Create service** → source: ECR image, port `8000`, env var `OPENAI_API_KEY`
(better via Secrets Manager), health check path `/health`, 1 vCPU / 2 GB. Since patient data flows through this
service: put it behind a VPC-only endpoint or Cognito auth, enable S3 SSE-KMS + versioning if raw docs are retained,
and turn on CloudTrail/access logging for an audit trail.
Alternative: EC2 (t3.small) with `docker run -p 80:8000 -e OPENAI_API_KEY=... <image>`.
Tip: keep `data/faiss_index` on S3 and download it on startup so you don't re-embed on every deploy.

## Important notes
- `DEFAULT_CRITERIA` is a demo trial's (`TRL-METAB-07`) numeric cutoffs for this capstone. Real use needs the actual
  trial's inclusion/exclusion criteria, passed in via `validate_eligibility(patient, criteria=...)`.
- This is a **pre-screening assistant for a trial coordinator** — it never makes a final eligibility, medical, or IRB
  determination. Borderline/narrative judgment calls are explicitly routed to RAG and flagged for PI review.
- Use only synthetic data for demos/tests. Real patient referrals carry PHI — follow your site's HIPAA/DPDP consent
  and data-handling SOPs, and never commit real PHI to this repo or the FAISS index.
- Protocol versions, IRB approvals, and consent requirements change — keep `data/trial_docs/` in sync with the live,
  IRB-approved protocol version before relying on `/ask` answers.

## Demo script (capstone presentation)
1. Clean intake → LIKELY_ELIGIBLE. 2. Age out of range → hard-fail flag. 3. Consent not signed → hard-fail flag.
4. Pregnancy exclusion → hard-fail flag. 5. Borderline prior-malignancy history → NEEDS_REVIEW via RAG, routed to PI.
6. `/screen-batch` with 10 referrals → summary counts — this is the call a lead-recruitment dashboard would make.
