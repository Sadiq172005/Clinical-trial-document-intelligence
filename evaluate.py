"""End-to-end eval on the samples: compares each verdict against expected.json. Needs OPENAI_API_KEY + indexed docs.
Run: python evaluate.py"""
import json, sys
from pathlib import Path
from pipeline import screen_patient

exp = json.loads(Path("samples/expected.json").read_text())
ok = 0
for fname, e in exp.items():
    try:
        res = screen_patient(str(Path("samples") / fname))
        got = res["report"]["verdict"]; codes = sorted({i["code"] for i in res["issues"]})
    except Exception as ex:
        got, codes = f"ERROR {ex}", []
    hit = got == e["expected_verdict"]; ok += hit
    print(f"{'PASS' if hit else 'FAIL'}  {fname:34s} expected={e['expected_verdict']:17s} got={got:17s} codes={codes}")
print(f"\n{ok}/{len(exp)} verdicts match")
sys.exit(0 if ok == len(exp) else 1)
