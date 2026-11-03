"""Подбор параметров градиентного бустинга на отдельном окне данных.

Качество ковариаты измеряется долей объясненной дисперсии метрики:
снижение дисперсии после коррекции равно квадрату корреляции между
предсказанием и метрикой. Предсказания получены по схеме перекрестной
проверки, поэтому оценка не завышена переобучением.

Окно для подбора не совпадает с окном, на котором считаются мощность и
уровень значимости: иначе выбор параметров подстраивался бы под итоговую
таблицу результатов.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from ab_sensitivity import DATA
from ab_sensitivity.wowah import build_daily, build_experiment, to_sample
from ab_sensitivity.methods import _design

t0 = sys.argv[1] if len(sys.argv) > 1 else "2008-06-01"
window = int(sys.argv[2]) if len(sys.argv) > 2 else 14

daily = build_daily(DATA)
df = build_experiment(daily, t0=t0, window_days=window)
s = to_sample(df, effect=0.0, seed=1)
X, y = _design(s), s.y
print(f"окно подбора {t0}, персонажей {len(y)}, признаков {X.shape[1]}")


def oof_r2(params, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.permutation(len(y)) % folds
    pred = np.empty(len(y))
    for k in range(folds):
        tr, te = f != k, f == k
        m = lgb.LGBMRegressor(**params).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    rho = np.corrcoef(pred, y)[0, 1]
    return rho ** 2, rho


# линейная модель как ориентир
beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
lin_pred = np.column_stack([np.ones(len(X)), X]) @ beta
print(f"{'линейная регрессия':<46}{np.corrcoef(lin_pred, y)[0,1]**2:>8.3f}")

GRID = [
    dict(n_estimators=100, num_leaves=31, learning_rate=0.05, min_child_samples=20),
    dict(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=20),
    dict(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=50),
    dict(n_estimators=600, num_leaves=7, learning_rate=0.03, min_child_samples=100),
    dict(n_estimators=400, num_leaves=7, learning_rate=0.05, min_child_samples=200),
    dict(n_estimators=800, num_leaves=3, learning_rate=0.03, min_child_samples=200),
    dict(n_estimators=400, num_leaves=15, learning_rate=0.03, min_child_samples=100,
         reg_lambda=10.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
]
best = None
for p in GRID:
    p = dict(objective="regression", verbose=-1, n_jobs=4, **p)
    r2, rho = oof_r2(p)
    tag = (f"leaves={p['num_leaves']} trees={p['n_estimators']} lr={p['learning_rate']} "
           f"min_child={p['min_child_samples']}" + (" + регуляризация" if "reg_lambda" in p else ""))
    print(f"{tag:<46}{r2:>8.3f}")
    if best is None or r2 > best[0]:
        best = (r2, p)
print("\nлучшая конфигурация:", {k: v for k, v in best[1].items()
                                 if k not in ("objective", "verbose", "n_jobs")})
print(f"объясненная дисперсия {best[0]:.3f}, ожидаемая доля остаточной дисперсии {1-best[0]:.3f}")
