"""Подготовка данных WoWAH (World of Warcraft Avatar History).

Исходный файл — журнал опросов игрового мира: одна строка это факт того,
что персонаж был онлайн в момент опроса. Опросы идут примерно раз в
10 минут, поэтому время в игре за период оценивается как число записей,
умноженное на интервал опроса.

Схема работы:
    1) build_daily  — свертка журнала в таблицу «персонаж, дата, минуты»
       (один раз, результат кэшируется);
    2) build_experiment — выбор окон до и во время эксперимента и сборка
       таблицы признаков;
    3) to_sample — случайное разбиение на группы и внесение известного
       эффекта, после чего применимы процедуры из ab.methods.

ВНИМАНИЕ: имена столбцов и формат даты заданы по описанию набора данных
и проверяются функцией peek() на реальном файле. При расхождении
достаточно поправить CFG, остальной код не меняется.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .methods import Sample


@dataclass
class Cfg:
    col_char: str = "char"
    col_time: str = "timestamp"
    col_level: str = "level"
    col_race: str = "race"
    col_class: str = "charclass"
    col_guild: str = "guild"
    time_format: str | None = "%m/%d/%y %H:%M:%S"
    poll_minutes: int = 10
    chunk: int = 2_000_000


CFG = Cfg()


def peek(path: str, n: int = 5) -> pd.DataFrame:
    """Прочитать первые строки и показать, как выглядят столбцы."""
    df = pd.read_csv(path, nrows=n)
    df.columns = [c.strip() for c in df.columns]
    print("столбцы:", list(df.columns))
    print(df.head(n).to_string())
    return df


def build_daily(path: str, out: str = "data/daily.parquet", cfg: Cfg = CFG) -> pd.DataFrame:
    """Свернуть журнал в таблицу: персонаж, дата, минуты в игре, уровень.

    Читается по частям, поэтому требует немного памяти. Результат
    сохраняется и при повторном запуске просто читается с диска.
    """
    import os
    if os.path.exists(out):
        return pd.read_parquet(out)

    usecols = [cfg.col_char, cfg.col_time, cfg.col_level, cfg.col_race,
               cfg.col_class, cfg.col_guild]
    parts = []
    for chunk in pd.read_csv(path, chunksize=cfg.chunk, skipinitialspace=True):
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk = chunk[[c for c in usecols if c in chunk.columns]]
        ts = pd.to_datetime(chunk[cfg.col_time], format=cfg.time_format, errors="coerce")
        chunk = chunk.assign(date=ts.dt.date).dropna(subset=["date"])
        g = chunk.groupby([cfg.col_char, "date"]).agg(
            polls=(cfg.col_time, "size"),
            level=(cfg.col_level, "max"),
            race=(cfg.col_race, "last"),
            charclass=(cfg.col_class, "last"),
            guild=(cfg.col_guild, "last"),
        ).reset_index()
        parts.append(g)
    daily = pd.concat(parts, ignore_index=True)
    # записи одного дня могли попасть в разные части файла
    daily = daily.groupby([cfg.col_char, "date"], as_index=False).agg(
        polls=("polls", "sum"), level=("level", "max"),
        race=("race", "last"), charclass=("charclass", "last"), guild=("guild", "last"))
    daily["minutes"] = daily["polls"] * CFG.poll_minutes
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_parquet(out, index=False)
    return daily


def build_experiment(daily: pd.DataFrame, t0: str, window_days: int = 14,
                     cfg: Cfg = CFG) -> pd.DataFrame:
    """Собрать таблицу эксперимента вокруг даты t0.

    Период до эксперимента: [t0 - window, t0)
    Период эксперимента:    [t0, t0 + window)

    В выборку попадают персонажи, активные хотя бы раз в любом из периодов.
    Предыстория равна NaN, если до t0 персонаж не встречался ни разу — это
    и есть новички, для которых ковариата не определена.
    """
    t0 = pd.Timestamp(t0)
    w = pd.Timedelta(days=window_days)
    pre = daily[(daily.date >= t0 - w) & (daily.date < t0)]
    post = daily[(daily.date >= t0) & (daily.date < t0 + w)]

    g_pre = pre.groupby(cfg.col_char).agg(
        pre_minutes=("minutes", "sum"),
        pre_days=("minutes", "size"),
        level=("level", "max"),
        race=("race", "last"),
        charclass=("charclass", "last"),
        guild=("guild", "last"))
    g_post = post.groupby(cfg.col_char).agg(
        post_minutes=("minutes", "sum"),
        level_post=("level", "max"),
        race_post=("race", "last"),
        charclass_post=("charclass", "last"),
        guild_post=("guild", "last"))

    df = g_pre.join(g_post, how="outer")
    df["post_minutes"] = df["post_minutes"].fillna(0.0)
    df["is_new"] = df["pre_minutes"].isna()
    # признаки новичков берутся из первого наблюдения в периоде эксперимента:
    # они известны в момент включения в эксперимент
    for a, b in [("level", "level_post"), ("race", "race_post"),
                 ("charclass", "charclass_post"), ("guild", "guild_post")]:
        df[a] = df[a].where(~df["is_new"], df[b])
    df["pre_days"] = df["pre_days"].fillna(0)
    df = df.drop(columns=["level_post", "race_post", "charclass_post", "guild_post"])
    return df.reset_index()


def prepare(df: pd.DataFrame, use_feats: bool = True) -> dict:
    """Один раз собрать метрики и матрицу признаков.

    Используется, когда по одной и той же выборке делается много случайных
    разбиений: пересобирать признаки на каждом повторе незачем.
    """
    feats = None
    if use_feats:
        cols = [df["level"].fillna(df["level"].median()).to_numpy(float),
                df["pre_days"].to_numpy(float),
                df["guild"].notna().to_numpy(float)]
        for name in ("race", "charclass"):
            if name in df:
                d = pd.get_dummies(df[name].astype("category"), prefix=name, drop_first=True)
                cols.append(d.to_numpy(float))
        feats = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])
    return {"post": df["post_minutes"].to_numpy(float),
            "pre": df["pre_minutes"].to_numpy(float),
            "feats": feats}


def assign(prep: dict, effect: float, seed: int) -> Sample:
    """Случайное разбиение на группы и внесение известного эффекта."""
    rng = np.random.default_rng(seed)
    t = (rng.random(len(prep["post"])) < 0.5).astype(int)
    return Sample(y=prep["post"] * (1 + effect * t), t=t,
                  pre=prep["pre"], feats=prep["feats"])


def to_sample(df: pd.DataFrame, effect: float, seed: int,
              use_feats: bool = True) -> Sample:
    """Полусинтетический эксперимент: распределения и связь периодов
    настоящие, а истинный эффект задан нами, поэтому мощность и уровень
    значимости измеримы."""
    return assign(prepare(df, use_feats), effect, seed)


def describe(df: pd.DataFrame) -> dict:
    """Сводка по таблице эксперимента: что писать в статью про данные."""
    pre, post = df["pre_minutes"], df["post_minutes"]
    h = pre.notna()
    rho = float(np.corrcoef(pre[h], post[h])[0, 1]) if h.sum() > 10 else np.nan
    srt = np.sort(post.to_numpy())
    k = max(1, len(srt) // 100)
    return {"персонажей": len(df),
            "доля новичков": float((~h).mean()),
            "корреляция предыстории и метрики": rho,
            "среднее минут за период": float(post.mean()),
            "медиана минут": float(post.median()),
            "доля нулей": float((post == 0).mean()),
            "доля метрики у 1% самых активных": float(srt[-k:].sum() / srt.sum())}
