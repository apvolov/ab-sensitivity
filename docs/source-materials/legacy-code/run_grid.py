"""Сетка по датам и размерам эффекта на данных WoWAH.

    python run_grid.py <повторов> <процедуры через запятую> [даты через запятую]

Пример:
    python run_grid.py 500 plain,cuped,linreg
    python run_grid.py 100 cupac 2008-02-01,2008-06-01,2008-10-01

Каждое окно дает свою выборку персонажей, внутри окна повторы различаются
случайным разбиением на группы. Разброс между окнами показывает, насколько
результат устойчив, разброс внутри окна — точность оценки доли отвержений.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

from ab.wowah import build_daily, build_experiment, prepare, assign, describe
from ab.methods import run_all

EFFECTS = [float(x) for x in os.environ.get("EFFECTS", "0.0,0.05,0.10").split(",")]
DATES = ["2008-02-01", "2008-03-01", "2008-04-01", "2008-05-01", "2008-06-01",
         "2008-07-01", "2008-08-01", "2008-09-01", "2008-10-01"]
BOOST = dict(n_estimators=300, num_leaves=15, learning_rate=0.03,
             min_child_samples=50, n_jobs=4)   # подобрано на окне 2008-06-01
WINDOW = 14
PATH = "grid_results.json"

reps = int(sys.argv[1])
procs = tuple(sys.argv[2].split(","))
dates = sys.argv[3].split(",") if len(sys.argv) > 3 else DATES

daily = build_daily(os.environ.get("WOWAH_CSV", "data/wowah_data.csv"))
out = json.load(open(PATH)) if os.path.exists(PATH) else {}

for t0 in dates:
    df = build_experiment(daily, t0=t0, window_days=WINDOW)
    info = describe(df)
    pr = prepare(df)
    for eff in EFFECTS:
        t_start = time.time()
        hit = {k: 0 for k in procs}
        var = {k: [] for k in procs}
        est = {k: [] for k in procs}
        for i in range(reps):
            s = assign(pr, eff, seed=hash((t0, eff, i)) % 2**31)
            res = run_all(s, procs, seed=i, n_folds=3, params=BOOST)
            for k, r in res.items():
                hit[k] += r.p < 0.05
                var[k].append(r.var_adj)
                est[k].append(r.rel)
        base = np.mean(var["plain"]) if "plain" in procs else np.nan
        key = f"{t0}|{eff}"
        row = out.get(key, {"t0": t0, "effect": eff, "window": WINDOW, "data": info, "by": {}})
        for k in procs:
            row["by"][k] = {"reject": hit[k] / reps, "reps": reps,
                            "effect_mean": float(np.mean(est[k])),
                            "var_ratio": float(np.mean(var[k]) / base) if base == base else None}
        out[key] = row
        json.dump(out, open(PATH, "w"), ensure_ascii=False, indent=1)
        print(f"{t0} эффект {eff:>5.0%} n={info['персонажей']:>5} "
              f"rho={info['корреляция предыстории и метрики']:.3f} "
              + "  ".join(f"{k} {hit[k]/reps:>5.1%}" for k in procs)
              + f"   ({time.time()-t_start:.0f} c)", flush=True)
