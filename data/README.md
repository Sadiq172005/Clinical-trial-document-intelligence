# data/ — what's needed here

## 1. `data/trial_docs/` (RAG knowledge base) — REQUIRED
Put the trial protocol's relevant sections + ICH-GCP / regulatory guideline docs here. PDFs or .txt/.md both work.
Use text-layer PDFs (not scanned) — otherwise retrieval quality suffers.

| Document | Why it's needed |
|---|---|
| This trial's protocol (Section 4: Inclusion/Exclusion Criteria, Section on prohibited therapies) | For narrative/borderline exclusion criteria — numeric cutoffs are checked in code via `DEFAULT_CRITERIA`/extracted criteria, but judgment calls like "prior malignancy in remission >2 years" go through RAG |
| ICH-GCP E6(R2) guideline (full text or key sections: 4.8 Informed Consent, 3 IRB/IEC) | Consent and IRB-process questions |
| Informed Consent Form (ICF) template for this trial | `/ask` queries about what the ICF actually says |
| Sponsor safety/exclusion addenda, protocol amendments | Keeps RAG answers current with the live protocol version |
| Country-specific research ethics guidance (e.g. India: ICMR National Ethical Guidelines, New Drugs & Clinical Trials Rules 2019) if the site is in India | DPDP Act / local-consent-law grounding |

## 2. Optional (improves accuracy)
- `data/criteria_overrides.json` — per-trial `CriterionCheck` list (age/lab/washout cutoffs) if this isn't the demo trial `TRL-METAB-07`. Pass it into `validate_eligibility(patient, criteria=...)`.
- `data/icd_codes.csv` — diagnosis code reference, if you want to validate `diagnosis` against a structured code list instead of free text.

## 3. Generated (don't commit)
- `data/faiss_index/` — built by `make index`; rebuild it (`/reindex`) whenever the docs change.

## 4. Test data
- `samples/` — `python generate_samples.py` (100% synthetic intake forms; no real patient data).
- **If you use real patient/referral data**: follow PHI masking, HIPAA/DPDP consent, and your IRB-approved data-handling SOP. Never commit real PHI to this repo or to the FAISS index.
