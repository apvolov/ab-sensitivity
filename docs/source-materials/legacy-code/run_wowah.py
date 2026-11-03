"""Полусинтетический эксперимент на данных WoWAH.

    python run_wowah.py <эффект> <повторов> [дата] [окно]

Данные реальные, эффект вносится нами, поэтому доля отвержений нулевой
гипотезы измерима: при эффекте 0 это фактический уровень значимости,
при эффекте больше 0 — мощность. Повторы различаются только случайным
разбиением на группы, сама выборка персонажей одна и та же.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

from ab.wowah import build_daily, build_experiment, describe, to_sample
from ab.methods import run_all

CSV = os.environ.get("WOWAH_CSV", "data/wowah_data.csv")
BOOST = dict(n_estimators=100, n_jobs=4)
FOLDS = 3

effect = float(sys.argv[1])
reps = int(sys.argv[2])
t0 = sys.argv[3] if len(sys.argv) > 3 else "2008-03-01"
window = int(sys.argv[4]) if len(sys.argv) > 4 else 14

daily = build_daily(CSV)
df = build_experiment(daily, t0=t0, window_days=window)
info = describe(df)

keys = ("plain", "cuped", "linreg", "cupac")
hit = {k: 0 for k in keys}
est = {k: [] for k in keys}
var = {k: [] for k in keys}
t_start = time.time()
for i in range(reps):
    s = to_sample(df, effect=effect, seed=1000 + i)
    res = run_all(s, keys, seed=i, n_folds=FOLDS, params=BOOST)
    for k, r in res.items():
        hit[k] += r.p < 0.05
        est[k].append(r.rel)
        var[k].append(r.var_adj)

base = np.mean(var["plain"])
out = {"t0": t0, "window": window, "effect": effect, "reps": reps,
       "secs": round(time.time() - t_start, 1), "data": info,
       "by": {k: {"reject": hit[k] / reps,
                  "effect_mean": float(np.mean(est[k])),
                  "effect_sd": float(np.std(est[k], ddof=1)),
                  "var_ratio": float(np.mean(var[k]) / base)} for k in keys}}

path = "wowah_results.json"
old = json.load(open(path)) if os.path.exists(path) else {}
old[f"{t0}_{window}_{effect}"] = out
json.dump(old, open(path, "w"), ensure_ascii=False, indent=1)

print(f"{t0}, окно {window} д, эффект {effect:.0%}, повторов {reps}, {out['secs']} c")
print(f"  персонажей {info['персонажей']}, новичков {info['доля новичков']:.1%}, "
      f"корреляция {info['корреляция предыстории и метрики']:.3f}")
print(f"  {'процедура':<10}{'отвержений':>12}{'оценка':>10}{'ст.откл.':>10}{'дисперсия':>11}")
for k in keys:
    v = out["by"][k]
    print(f"  {k:<10}{v['reject']:>11.1%}{v['effect_mean']:>10.2%}{v['effect_sd']:>10.2%}{v['var_ratio']:>11.3f}")
