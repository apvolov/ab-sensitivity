"""Процедуры анализа A/B эксперимента.

Все процедуры принимают один и тот же набор данных и возвращают оценку
эффекта, ее стандартную ошибку и p-value. Различаются только тем, какая
величина подставляется в двухвыборочный критерий:

    plain     исходная метрика
    cuped     CUPED по предыстории, новички остаются без коррекции
    linreg    линейная регрессия: предыстория + индикатор ее отсутствия + признаки
    cupac     предсказание градиентного бустинга как единственная ковариата

Общая схема коррекции:  Y_adj = Y - (pred - mean(pred)),
где pred строится только по величинам, известным до начала эксперимента.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class Sample:
    """Данные одного эксперимента.

    y        метрика за период эксперимента
    t        0 контроль, 1 воздействие
    pre      метрика за предэкспериментальный период, np.nan если истории нет
    feats    матрица признаков, известных до эксперимента (уже числовая)
    """
    y: np.ndarray
    t: np.ndarray
    pre: np.ndarray
    feats: np.ndarray | None = None

    @property
    def has_hist(self) -> np.ndarray:
        return ~np.isnan(self.pre)


@dataclass
class Result:
    effect: float      # разность средних, B минус A
    rel: float         # то же в долях от среднего контрольной группы
    se: float
    p: float
    var_adj: float     # дисперсия скорректированной метрики
    name: str = ""


def welch(y: np.ndarray, t: np.ndarray, name: str = "", base: float | None = None) -> Result:
    a, b = y[t == 0], y[t == 1]
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    diff = mb - ma
    # число степеней свободы по Уэлчу; при n в тысячах практически совпадает с нормальным
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * stats.t.sf(abs(diff / se), df)
    denom = base if base is not None else ma
    return Result(diff, diff / denom, se, p, float(np.var(y, ddof=1)), name)


def _adjust(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Вычесть предсказание, сохранив общее среднее."""
    return y - (pred - pred.mean())


def _theta(x: np.ndarray, y: np.ndarray) -> float:
    vx = np.var(x, ddof=1)
    return float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else 0.0


# ---------------------------------------------------------------- процедуры

def plain(s: Sample) -> Result:
    return welch(s.y, s.t, "без коррекции")


def cuped(s: Sample) -> Result:
    """CUPED по предыстории. Новички остаются с исходным значением метрики."""
    h = s.has_hist
    y = s.y.astype(float).copy()
    if h.sum() > 10:
        th = _theta(s.pre[h], s.y[h])
        y[h] = s.y[h] - th * (s.pre[h] - s.pre[h].mean())
    return welch(y, s.t, "CUPED", base=s.y[s.t == 0].mean())


def _design(s: Sample) -> np.ndarray:
    """Матрица признаков: предыстория с нулем вместо пропуска, индикатор
    отсутствия истории и прочие признаки."""
    h = s.has_hist
    cols = [np.where(h, np.nan_to_num(s.pre), 0.0)]
    if (~h).any():
        cols.append((~h).astype(float))
    if s.feats is not None and s.feats.size:
        cols.append(s.feats)
    return np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])


def linreg(s: Sample) -> Result:
    """Линейная коррекция по предыстории, индикатору и признакам."""
    X = _design(s)
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, s.y, rcond=None)
    pred = X1 @ beta
    return welch(_adjust(s.y, pred), s.t, "линейная коррекция",
                 base=s.y[s.t == 0].mean())


def cupac(s: Sample, n_folds: int = 5, seed: int = 0, params: dict | None = None) -> Result:
    """CUPAC: предсказание градиентного бустинга как ковариата.

    Предсказания строятся по схеме перекрестной проверки, поэтому значение
    для каждого наблюдения получено моделью, не видевшей это наблюдение.
    Без этого коррекция частично подстраивается под собственный шум.
    """
    import lightgbm as lgb

    X = _design(s)
    y = s.y
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(y)) % n_folds
    pred = np.empty(len(y), float)
    p = dict(objective="regression", num_leaves=31, learning_rate=0.05,
             n_estimators=200, min_child_samples=50, verbose=-1)
    if params:
        p.update(params)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        model = lgb.LGBMRegressor(**p).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    # ковариатой служит предсказание, коэффициент как в CUPED
    th = _theta(pred, y)
    return welch(y - th * (pred - pred.mean()), s.t, "CUPAC",
                 base=s.y[s.t == 0].mean())


PROCEDURES = {"plain": plain, "cuped": cuped, "linreg": linreg, "cupac": cupac}


def run_all(s: Sample, which=("plain", "cuped", "linreg", "cupac"), **kw) -> dict[str, Result]:
    return {k: PROCEDURES[k](s, **kw) if k == "cupac" else PROCEDURES[k](s) for k in which}
