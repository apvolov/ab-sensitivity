"""Графики для статьи по результатам сетки."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

d = json.load(open("grid_results.json"))
GREY, BLUE, ORANGE, GREEN = "#555555", "#2E5A88", "#E07B39", "#2E7D4F"

# ----- рисунок 1: мощность на окне 2008-03-01
win = {v["effect"]: v for v in d.values() if v["t0"] == "2008-03-01"}
eff = [0.0, 0.05, 0.10]
series = [("без коррекции", "plain", GREY, "o-"), ("CUPED", "cuped", BLUE, "s-"),
          ("линейная коррекция", "linreg", GREEN, "^-"), ("CUPAC", "cupac", ORANGE, "d--")]
fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=200)
for name, key, c, st in series:
    y = [win[e]["by"][key]["reject"] * 100 for e in eff if key in win[e]["by"]]
    x = [e * 100 for e in eff if key in win[e]["by"]]
    ax.plot(x, y, st, color=c, label=name, lw=2, ms=5)
ax.axhline(5, color="#999999", ls=":", lw=1)
ax.text(9.9, 7.5, "номинальный уровень 5%", color="#777777", fontsize=8, ha="right")
ax.axhline(80, color="#999999", ls=":", lw=1)
ax.text(0.15, 83, "мощность 80%", color="#777777", fontsize=8)
ax.set_xlabel("истинный эффект, %"); ax.set_ylabel("доля отвержений нулевой гипотезы, %")
ax.set_ylim(0, 100); ax.set_xticks([0, 5, 10]); ax.grid(alpha=0.3)
ax.legend(loc="center right", fontsize=9, frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("fig1_power.png")

# ----- рисунок 2: измеренное и расчетное снижение дисперсии по окнам
rows = sorted([v for v in d.values() if v["effect"] == 0.0], key=lambda r: r["data"]["корреляция предыстории и метрики"])
rho = np.array([r["data"]["корреляция предыстории и метрики"] for r in rows])
hsh = np.array([1 - r["data"]["доля новичков"] for r in rows])
cuped = np.array([r["by"]["cuped"]["var_ratio"] for r in rows])
linreg = np.array([r["by"]["linreg"]["var_ratio"] for r in rows])
pred = hsh * (1 - rho ** 2) + (1 - hsh)

fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
import pandas as pd
from ab.wowah import build_daily, build_experiment
daily = build_daily("data/wowah_data.csv")
wvar = []
for r in rows:
    df = build_experiment(daily, r["t0"], 14)
    y = df.post_minutes.to_numpy(float); pre = df.pre_minutes.to_numpy(float)
    h = ~np.isnan(pre)
    ss = (y - y.mean()) ** 2
    wvar.append(ss[h].sum() / ss.sum())
wvar = np.array(wvar)
pred_var = wvar * (1 - rho ** 2) + (1 - wvar)

o = np.argsort(rho)
ax.plot(rho[o], pred[o], color=GREY, ls="--", lw=1.6, marker="x", ms=5,
        label="расчет по долям пользователей")
ax.plot(rho[o], pred_var[o], color=ORANGE, ls="-", lw=1.6, marker="x", ms=5,
        label="расчет по вкладу в дисперсию")
ax.scatter(rho, cuped, color=BLUE, s=46, label="CUPED, измерено", zorder=3)
ax.scatter(rho, linreg, color=GREEN, s=46, marker="^", label="линейная коррекция, измерено", zorder=3)
ax.set_xlabel("корреляция метрики с предысторией")
ax.set_ylabel("доля исходной дисперсии")
ax.set_ylim(0.25, 0.80); ax.grid(alpha=0.3)
ax.legend(fontsize=9, frameon=False, loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("fig2_variance.png")

print("измерено CUPED:    ", np.round(cuped, 3))
print("по долям:          ", np.round(pred, 3))
print("по вкладу в дисп.: ", np.round(pred_var, 3))
print("макс. отклонение формулы по вкладу:", round(float(np.max(np.abs(pred_var - cuped))), 3))
