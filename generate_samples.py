"""Sample patient intake forms (PDFs + one scanned PNG) — each one covers a different eligibility/compliance scenario.
100% synthetic data — no real patient, no real trial. Run: python generate_samples.py -> samples/*.pdf + expected.json
"""
import json
import datetime as dt
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

OUT = Path("samples"); OUT.mkdir(exist_ok=True)
TODAY = dt.date(2026, 10, 19)


def make(fname, *, ref="LEAD-2026-0142", trial="TRL-METAB-07", age=52, sex="F", diagnosis="Type 2 Diabetes Mellitus",
         diag_date="2018-03-01", pregnancy="not_applicable", labs=None, prior=None, comorbid=None,
         consent_signed=True, consent_date="2026-10-01", screening_date="2026-10-05", irb="IRB-2026-0091"):
    labs = labs if labs is not None else [("eGFR", 58, "mL/min/1.73m2"), ("HbA1c", 7.8, "%")]
    prior = prior if prior is not None else [("Metformin", None)]
    comorbid = comorbid if comorbid is not None else ["Hypertension"]
    ss = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / fname), pagesize=A4, topMargin=36, bottomMargin=36)
    el = [Paragraph("CLINICAL TRIAL — PATIENT PRE-SCREENING INTAKE FORM", ss["Title"]),
          Paragraph(f"<b>Trial ID:</b> {trial} &nbsp;&nbsp; <b>Lead/Referral Ref:</b> {ref}", ss["Normal"]), Spacer(1, 8),
          Paragraph(f"Age: {age} &nbsp;&nbsp; Sex: {sex} &nbsp;&nbsp; Pregnancy status: {pregnancy}", ss["Normal"]),
          Paragraph(f"Diagnosis: {diagnosis} &nbsp;&nbsp; Diagnosis date: {diag_date}", ss["Normal"]), Spacer(1, 8)]

    rows = [["Lab", "Value", "Unit"]] + [[n, str(v), u] for n, v, u in labs]
    t = Table(rows, colWidths=[150, 80, 150])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    el += [Paragraph("<b>Lab values</b>", ss["Normal"]), t, Spacer(1, 10)]

    prow = [["Prior therapy", "Last dose date"]] + [[n, d or "ongoing"] for n, d in prior]
    pt = Table(prow, colWidths=[200, 180])
    pt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    el += [Paragraph("<b>Prior / concomitant therapy</b>", ss["Normal"]), pt, Spacer(1, 10)]

    el += [Paragraph(f"<b>Comorbidities:</b> {', '.join(comorbid) if comorbid else 'None reported'}", ss["Normal"]), Spacer(1, 10),
           Paragraph(f"<b>Informed consent signed:</b> {'Yes' if consent_signed else 'No'} "
                     f"&nbsp;&nbsp; <b>Consent date:</b> {consent_date or '—'}", ss["Normal"]),
           Paragraph(f"<b>IRB/Ethics Committee approval ref:</b> {irb or '—'}", ss["Normal"]),
           Paragraph(f"<b>Screening date:</b> {screening_date}", ss["Normal"])]
    doc.build(el)


SCENARIOS = {
    "01_clean_eligible.pdf": ({}, "LIKELY_ELIGIBLE", "No issues"),
    "02_age_out_of_range.pdf": (dict(age=79), "LIKELY_INELIGIBLE", "INCLUSION_CRITERION_FAILED (age <= 75)"),
    "03_low_egfr.pdf": (dict(labs=[("eGFR", 22, "mL/min/1.73m2"), ("HbA1c", 8.1, "%")]), "LIKELY_INELIGIBLE", "INCLUSION_CRITERION_FAILED (eGFR >= 30)"),
    "04_consent_not_signed.pdf": (dict(consent_signed=False), "LIKELY_INELIGIBLE", "CONSENT_NOT_SIGNED"),
    "05_consent_after_screening.pdf": (dict(consent_date="2026-10-08", screening_date="2026-10-05"), "LIKELY_INELIGIBLE", "CONSENT_AFTER_SCREENING"),
    "06_pregnant_exclusion.pdf": (dict(sex="F", pregnancy="pregnant"), "LIKELY_INELIGIBLE", "PREGNANCY_EXCLUSION"),
    "07_recent_excluded_therapy.pdf": (dict(prior=[("Investigational Agent X", (TODAY - dt.timedelta(days=5)).isoformat())]),
                                        "LIKELY_INELIGIBLE", "INCLUSION_CRITERION_FAILED (washout >= 30 days not met — stopped 5 days ago)"),
    "08_missing_irb_ref.pdf": (dict(irb=None), "NEEDS_REVIEW", "IRB_REF_MISSING (warning only)"),
    "09_borderline_remission_history.pdf": (dict(comorbid=["Breast cancer — remission since 2023-11-01"],
                                                  diagnosis="Type 2 Diabetes Mellitus; history of malignancy"),
                                             "NEEDS_REVIEW", "Borderline via RAG (prior malignancy remission window)"),
}

if __name__ == "__main__":
    exp = {}
    for fname, (kw, verdict, why) in SCENARIOS.items():
        make(fname, **kw); exp[fname] = {"expected_verdict": verdict, "expected_flags": why}
    # Scanned-style PNG (OCR test): render the clean intake as an image with slight noise/rotation
    try:
        from pdf2image import convert_from_path
        from PIL import ImageFilter
        im = convert_from_path(str(OUT / "01_clean_eligible.pdf"), dpi=200)[0].convert("L").rotate(0.8, expand=True, fillcolor=255).filter(ImageFilter.GaussianBlur(0.6))
        im.save(OUT / "10_scanned_clean.png"); exp["10_scanned_clean.png"] = {"expected_verdict": "LIKELY_ELIGIBLE", "expected_flags": "OCR path test"}
    except Exception as e:
        print("PNG skip:", e)
    (OUT / "expected.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False))
    print(f"{len(exp)} samples -> {OUT}/")
