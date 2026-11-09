"""Offline tests — run without any LLM/API key: python test_validators.py"""
import datetime as dt
from pipeline import validate_eligibility, PatientProfile, LabValue, PriorTherapy, CriterionCheck, DEFAULT_CRITERIA

TODAY = dt.date(2026, 10, 19)


def codes(p, criteria=None, today=TODAY):
    return {i.code for i in validate_eligibility(p, criteria=criteria, today=today)}


def good(**kw):
    base = dict(patient_ref="LEAD-001", trial_id="TRL-METAB-07", age=52, sex="F", pregnancy_status="not_applicable",
                diagnosis="Type 2 Diabetes Mellitus", diagnosis_date="2018-03-01",
                lab_values=[LabValue(name="eGFR", value=58, unit="mL/min/1.73m2"), LabValue(name="HbA1c", value=7.8, unit="%")],
                prior_therapies=[PriorTherapy(name="Metformin", stopped_date=None)],
                comorbidities=["Hypertension"], consent_signed=True, consent_date="2026-10-01",
                irb_approval_ref="IRB-2026-0091", screening_date="2026-10-05")
    base.update(kw); return PatientProfile(**base)


# Clean patient -> no issues
assert codes(good()) == set(), codes(good())

# Age out of range
assert "INCLUSION_CRITERION_FAILED" in codes(good(age=79))
assert "INCLUSION_CRITERION_FAILED" in codes(good(age=10))

# Low eGFR
assert "INCLUSION_CRITERION_FAILED" in codes(good(lab_values=[LabValue(name="eGFR", value=22, unit="mL/min/1.73m2")]))

# Missing lab data entirely -> warning, not a hard fail
assert "CRITERION_DATA_MISSING" in codes(good(lab_values=[]))

# Consent not signed is a hard error
assert "CONSENT_NOT_SIGNED" in codes(good(consent_signed=False))

# Consent after screening is a hard error (protocol deviation)
assert "CONSENT_AFTER_SCREENING" in codes(good(consent_date="2026-10-08", screening_date="2026-10-05"))

# Pregnancy exclusion
assert "PREGNANCY_EXCLUSION" in codes(good(pregnancy_status="pregnant"))

# Missing IRB ref -> warning only, not a hard fail by itself
irb_codes = codes(good(irb_approval_ref=None))
assert "IRB_REF_MISSING" in irb_codes and "CONSENT_NOT_SIGNED" not in irb_codes

# Washout violation: investigational agent stopped 5 days ago, needs >=30
assert "INCLUSION_CRITERION_FAILED" in codes(good(prior_therapies=[PriorTherapy(name="Investigational Agent X", stopped_date="2026-10-14")]))
# Washout satisfied: stopped 60 days ago
assert "INCLUSION_CRITERION_FAILED" not in codes(good(prior_therapies=[PriorTherapy(name="Investigational Agent X", stopped_date="2026-08-20")]))

print("All validator tests passed ✅")
