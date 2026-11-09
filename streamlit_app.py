"""Streamlit UI — run: streamlit run streamlit_app.py
Default: direct import of the pipeline. Set an API URL in the sidebar to hit the deployed FastAPI (AWS) service instead."""
import os, tempfile, json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clinical Trial Eligibility Copilot", page_icon="🧪", layout="wide")

VERDICT_UI = {"LIKELY_ELIGIBLE": ("✅ Likely Eligible", "success"), "NEEDS_REVIEW": ("⚠️ Needs Review", "warning"),
              "LIKELY_INELIGIBLE": ("⛔ Likely Ineligible", "error")}
SEV_ICON = {"error": "🔴 error", "warning": "🟠 warning", "info": "🔵 info"}

with st.sidebar:
    st.title("🧪 Trial Copilot")
    api_url = st.text_input("API URL (optional)", placeholder="http://<app-runner-url>", help="Leave blank to use the local pipeline")
    if not api_url and not os.getenv("OPENAI_API_KEY"):
        key = st.text_input("OPENAI_API_KEY", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    st.caption("Sample intake forms are in the `samples/` folder (python generate_samples.py).")
    st.caption("⚠️ Pre-screening aid only — the final eligibility call always belongs to the PI/IRB. Use synthetic data only, never real PHI.")


def run_screen(name: str, data: bytes) -> dict:
    if api_url:
        import requests
        r = requests.post(api_url.rstrip("/") + "/screen", files={"file": (name, data)}, timeout=180)
        r.raise_for_status()
        return r.json()
    from pipeline import screen_patient
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as f:
        f.write(data)
    try:
        return screen_patient(f.name)
    finally:
        os.unlink(f.name)


def render(res: dict):
    rep, issues, pat = res["report"], res["issues"], res["patient"]
    label, kind = VERDICT_UI.get(rep.get("verdict"), (rep.get("verdict", "?"), "info"))
    getattr(st, kind)(f"**{label}** — {rep.get('summary', '')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Errors", sum(i["severity"] == "error" for i in issues))
    c2.metric("Warnings", sum(i["severity"] == "warning" for i in issues))
    c3.metric("Age", pat.get("age") or "—")
    if issues:
        st.subheader("Eligibility & compliance issues")
        st.dataframe(pd.DataFrame([{"Severity": SEV_ICON[i["severity"]], "Code": i["code"], "Issue": i["message"], "Fix": i.get("fix") or ""} for i in issues]),
                     width="stretch", hide_index=True)
    else:
        st.success("No issues found by the rule checks.")
    if rep.get("actions"):
        st.subheader("Action items")
        for a in rep["actions"]:
            st.markdown(f"- {a}")
    if rep.get("coordinator_message"):
        st.subheader("Message to send to patient/site")
        st.code(rep["coordinator_message"], language=None)
    with st.expander("Extracted patient profile"):
        st.json(pat)
    if res.get("protocol_notes"):
        with st.expander("Protocol research (RAG)"):
            st.text(res["protocol_notes"])


tab1, tab2, tab3 = st.tabs(["🧾 Single patient", "📚 Batch", "💬 Ask protocol / GCP"])

with tab1:
    up = st.file_uploader("Upload an intake form (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"], key="one")
    samples = sorted(Path("samples").glob("*.[pP][dD][fF]")) + sorted(Path("samples").glob("*.png"))
    pick = st.selectbox("...or pick a sample", ["—"] + [p.name for p in samples])
    if st.button("Screen", type="primary"):
        if up:
            name, data = up.name, up.getvalue()
        elif pick != "—":
            name, data = pick, (Path("samples") / pick).read_bytes()
        else:
            st.stop()
        with st.spinner("OCR → extraction → eligibility rules → protocol research → advice..."):
            try:
                render(run_screen(name, data))
            except Exception as e:
                st.error(f"Failed: {e}")

with tab2:
    files = st.file_uploader("Multiple intake forms", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="many")
    if st.button("Screen all") and files:
        rows, results = [], {}
        bar = st.progress(0.0)
        for n, f in enumerate(files, 1):
            try:
                res = run_screen(f.name, f.getvalue()); results[f.name] = res
                rows.append({"File": f.name, "Verdict": res["report"].get("verdict"), "Errors": sum(i["severity"] == "error" for i in res["issues"]),
                             "Warnings": sum(i["severity"] == "warning" for i in res["issues"]), "Age": res["patient"].get("age")})
            except Exception as e:
                rows.append({"File": f.name, "Verdict": f"ERROR: {e}", "Errors": None, "Warnings": None, "Age": None})
            bar.progress(n / len(files))
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "trial_screening_batch.csv", "text/csv")
        for name, res in results.items():
            with st.expander(name):
                render(res)

with tab3:
    q = st.text_input("Protocol / ICH-GCP question", placeholder="Prior malignancy in remission >2 years — exclusion applies?")
    if st.button("Ask") and q:
        with st.spinner("Searching protocol & GCP docs..."):
            try:
                if api_url:
                    import requests
                    r = requests.post(api_url.rstrip("/") + "/ask", json={"question": q}, timeout=120); r.raise_for_status(); out = r.json()
                else:
                    from pipeline import rag_answer
                    out = rag_answer(q)
                st.markdown(out["answer"])
                st.caption("Sources: " + ", ".join(f"{s['source']} (p{s['page']})" for s in out["sources"]))
            except Exception as e:
                st.error(f"Failed: {e} — have you added docs to data/trial_docs/?")
