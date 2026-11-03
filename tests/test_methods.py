"""Проверки корректности процедур коррекции.

Запуск: pytest -q

Проверяется то, на чем держатся выводы работы:
  1) коррекция не смещает оценку эффекта;
  2) при отсутствии эффекта доля отвержений близка к номинальному уровню;
  3) измеренное снижение дисперсии согласуется с расчетом по вкладу
     пользователей с историей в дисперсию метрики;
  4) ковариата, измеренная после начала воздействия, ломает оценку —
     это обоснование требования использовать только предэкспериментальные
     данные.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ab_sensitivity.methods import Sample, plain, cuped, linreg, run_all
from ab_sensitivity.simulate import SimParams, make_sample

ALPHA = 0.05


def samples(n_rep, **kw):
    rng = np.random.default_rng(kw.pop("seed", 0))
    prm = SimParams(n=kw.pop("n", 8000), **kw)
    for _ in range(n_rep):
        yield make_sample(prm, rng)[0]


def test_correction_is_unbiased():
    """Средняя оценка эффекта совпадает с заложенной."""
    true = 0.05
    est = {"plain": [], "cuped": [], "linreg": []}
    for s in samples(60, effect=true, seed=1):
        for k, r in run_all(s, ("plain", "cuped", "linreg")).items():
            est[k].append(r.rel)
    for k, v in est.items():
        err = abs(np.mean(v) - true)
        assert err < 0.01, f"{k}: смещение {err:.3f}"


def test_type_one_error_near_nominal():
    """При отсутствии эффекта доля отвержений близка к 5%."""
    reps = 400
    hit = {"plain": 0, "cuped": 0, "linreg": 0}
    for s in samples(reps, effect=0.0, n=4000, seed=2):
        for k, r in run_all(s, tuple(hit)).items():
            hit[k] += r.p < ALPHA
    for k, c in hit.items():
        rate = c / reps
        # допуск три стандартных отклонения доли
        tol = 3 * np.sqrt(ALPHA * (1 - ALPHA) / reps)
        assert abs(rate - ALPHA) < tol, f"{k}: уровень {rate:.3f}"


def test_variance_reduction_matches_formula():
    """Снижение дисперсии согласуется с расчетом по вкладу в дисперсию."""
    s = next(iter(samples(1, effect=0.0, n=20000, seed=3)))
    h = s.has_hist
    rho = np.corrcoef(s.pre[h], s.y[h])[0, 1]
    ss = (s.y - s.y.mean()) ** 2
    w = ss[h].sum() / ss.sum()
    predicted = w * (1 - rho ** 2) + (1 - w)
    measured = cuped(s).var_adj / plain(s).var_adj
    assert abs(measured - predicted) < 0.05, f"измерено {measured:.3f}, расчет {predicted:.3f}"
    # оценка по долям пользователей систематически выше
    by_count = h.mean() * (1 - rho ** 2) + (1 - h.mean())
    assert by_count > predicted


def test_post_treatment_covariate_biases_estimate():
    """Ковариата, затронутая воздействием, смещает оценку эффекта."""
    true, biased = 0.10, []
    for s in samples(40, effect=true, seed=4):
        # имитируем ошибку сбора: в предысторию попал период эксперимента
        bad = Sample(y=s.y, t=s.t, pre=s.y.copy(), feats=s.feats)
        biased.append(cuped(bad).rel)
    assert np.mean(biased) < true / 2, "смещение не обнаружено"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightgbm") is None,
    reason="lightgbm не установлен")
def test_cupac_runs_and_reduces_variance():
    from ab_sensitivity.methods import cupac
    s = next(iter(samples(1, effect=0.05, n=6000, seed=5)))
    r = cupac(s, n_folds=3, params=dict(n_estimators=60, n_jobs=2))
    assert r.var_adj < plain(s).var_adj
