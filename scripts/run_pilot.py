"""Пилотный прогон одного сценария: python scripts/run_pilot.py <номер> <повторов>"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json, os, sys, time
from ab_sensitivity.simulate import SimParams, repeat

SCEN = {
    1: ("A/A, эффекта нет",             SimParams(n=20000, effect=0.0)),
    2: ("A/B, эффект 3%",               SimParams(n=20000, effect=0.03)),
    3: ("A/B, эффект 3%, новичков 50%", SimParams(n=20000, effect=0.03, share_new=0.5)),
    4: ("A/B, эффект 3%, без новичков", SimParams(n=20000, effect=0.03, share_new=0.0)),
}
idx, reps = int(sys.argv[1]), int(sys.argv[2])
title, prm = SCEN[idx]
t0 = time.time()
r = repeat(prm, reps, seed=100 + idx)
r["title"], r["secs"], r["params"] = title, round(time.time() - t0, 1), vars(prm)
old = json.load(open("results/pilot.json")) if os.path.exists("results/pilot.json") else {}
old[str(idx)] = r
json.dump(old, open("results/pilot.json", "w"), ensure_ascii=False, indent=1)
print(title, "|", r["secs"], "c | rho", round(r["rho"], 3))
for k, v in r["by"].items():
    print(f"  {k:<10}{v['reject']:>8.1%}{v['effect']:>9.2%}{v['var_ratio']:>8.3f}")
