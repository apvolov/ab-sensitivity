"""Сравнение вариантов ковариаты на отдельном окне, все оценки вне обучения.

Для каждой модели предсказание получается по схеме перекрестной проверки,
поэтому доли объясненной дисперсии сопоставимы между собой.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sys
import numpy as np
import lightgbm as lgb

from ab_sensitivity import DATA
from ab_sensitivity.wowah import build_daily, build_experiment, to_sample
from ab_sensitivity.methods import _design

t0 = sys.argv[1] if len(sys.argv) > 1 else "2008-06-01"
daily = build_daily(DATA)
df = build_experiment(daily, t0=t0, window_days=14)
s = to_sample(df, effect=0.0, seed=1)
X, y = _design(s), s.y
pre0 = np.where(np.isnan(s.pre), 0.0, s.pre)
Xlog = np.column_stack([X, np.log1p(pre0)])
FOLDS, SEED = 5, 0
folds = np.random.default_rng(SEED).permutation(len(y)) % FOLDS


def oof(fit_predict, XX):
    pred = np.empty(len(y))
    for k in range(FOLDS):
        tr, te = folds != k, folds == k
        pred[te] = fit_predict(XX[tr], y[tr], XX[te])
    return np.corrcoef(pred, y)[0, 1] ** 2


def lin(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ b


def boost(params):
    def f(Xtr, ytr, Xte):
        m = lgb.LGBMRegressor(objective="regression", verbose=-1, n_jobs=4, **params)
        return m.fit(Xtr, ytr).predict(Xte)
    return f


def hybrid(params):
    """Линейная модель плюс бустинг на ее остатках."""
    def f(Xtr, ytr, Xte):
        base_tr = lin(Xtr, ytr, Xtr)
        base_te = lin(Xtr, ytr, Xte)
        m = lgb.LGBMRegressor(objective="regression", verbose=-1, n_jobs=4, **params)
        m.fit(Xtr, ytr - base_tr)
        return base_te + m.predict(Xte)
    return f


P = dict(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=50)
PS = dict(n_estimators=400, num_leaves=7, learning_rate=0.03, min_child_samples=100,
          reg_lambda=10.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)

rows = [
    ("линейная регрессия", oof(lin, X)),
    ("линейная + log(предыстория)", oof(lin, Xlog)),
    ("бустинг", oof(boost(P), X)),
    ("бустинг + log(предыстория)", oof(boost(P), Xlog)),
    ("бустинг, сильная регуляризация", oof(boost(PS), X)),
    ("линейная + бустинг на остатках", oof(hybrid(P), X)),
    ("линейная + бустинг на остатках, регуляризация", oof(hybrid(PS), X)),
]
print(f"окно {t0}, персонажей {len(y)}\n")
print(f"{'модель ковариаты':<48}{'R2':>7}{'остаточная дисперсия':>22}")
for name, r2 in rows:
    print(f"{name:<48}{r2:>7.3f}{1-r2:>22.3f}")
