"""Генератор синтетических пользователей и каркас повторных прогонов.

Модель пользователя:
  скрытая активность L = lognormal(0, s_level) * k(платформа) * k(страна)
  метрика периода    = L * lognormal(0, s_noise)
Общий множитель L задает связь предыстории с метрикой эксперимента,
независимые множители отвечают за то, что периоды не повторяют друг друга.
Корреляция не задается напрямую: она измеряется по сгенерированным данным
и сообщается вместе с результатом.
"""

from dataclasses import dataclass
import numpy as np

from .methods import Sample, run_all

PLATFORM = {"iOS": 1.35, "Android": 1.0}
COUNTRY = {"RU": 1.0, "KZ": 0.8, "BY": 0.9, "other": 1.2}


@dataclass
class SimParams:
    n: int = 30000
    share_new: float = 0.2      # доля пользователей без предыстории
    effect: float = 0.03        # истинный относительный эффект в группе B
    s_level: float = 1.1        # разброс активности между пользователями
    s_noise: float = 0.33       # разброс одного периода относительно активности
    heavy: bool = False         # режим метрики монетизации (нули и тяжелый хвост)
    pay_rate: float = 0.035


def make_sample(prm: SimParams, rng: np.random.Generator) -> tuple[Sample, dict]:
    n = prm.n
    plat = rng.integers(0, 2, n)
    ctry = rng.choice(4, n, p=[0.6, 0.15, 0.1, 0.15])
    kp = np.array(list(PLATFORM.values()))[plat]
    kc = np.array(list(COUNTRY.values()))[ctry]
    level = rng.lognormal(0, prm.s_level, n) * kp * kc

    if prm.heavy:
        p = np.clip(prm.pay_rate * level, 0, 0.9)
        pre = (rng.random(n) < p) * level * rng.lognormal(4.5, 1.1, n)
        post = (rng.random(n) < p) * level * rng.lognormal(4.5, 1.1, n)
    else:
        pre = level * rng.lognormal(4.6, prm.s_noise, n)
        post = level * rng.lognormal(4.6, prm.s_noise, n)

    t = (rng.random(n) < 0.5).astype(int)
    post = post * (1 + prm.effect * t)

    new = rng.random(n) < prm.share_new
    pre = np.where(new, np.nan, pre)

    feats = np.column_stack([plat == 1,
                             ctry[:, None] == np.arange(1, 4)]).astype(float)
    s = Sample(y=post, t=t, pre=pre, feats=feats)
    h = s.has_hist
    info = {"rho": float(np.corrcoef(s.pre[h], s.y[h])[0, 1]) if h.sum() > 10 else np.nan,
            "share_new": float((~h).mean())}
    return s, info


def repeat(prm: SimParams, reps: int, seed: int = 0, which=("plain", "cuped", "linreg", "cupac"),
           alpha: float = 0.05) -> dict:
    """Повторные прогоны одного сценария.

    Возвращает по каждой процедуре: долю отвержений нулевой гипотезы
    (уровень значимости при effect = 0, мощность иначе), среднюю оценку
    эффекта и среднее снижение дисперсии.
    """
    rng = np.random.default_rng(seed)
    hit = {k: 0 for k in which}
    est = {k: [] for k in which}
    var = {k: [] for k in which}
    rho, shn = [], []
    for i in range(reps):
        s, info = make_sample(prm, rng)
        rho.append(info["rho"]); shn.append(info["share_new"])
        res = run_all(s, which, seed=i, n_folds=3, params=dict(n_estimators=80, n_jobs=4))
        for k, r in res.items():
            hit[k] += r.p < alpha
            est[k].append(r.rel)
            var[k].append(r.var_adj)
    base = np.mean(var["plain"])
    out = {"params": prm, "reps": reps,
           "rho": float(np.mean(rho)), "share_new": float(np.mean(shn)),
           "by": {}}
    for k in which:
        out["by"][k] = {"reject": hit[k] / reps,
                        "effect": float(np.mean(est[k])),
                        "var_ratio": float(np.mean(var[k]) / base)}
    return out
