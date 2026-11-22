"""Modelos de vendas, comparação em holdout temporal e plano de produção (newsvendor)."""
from __future__ import annotations

import csv
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from . import data as D

FEATURES = {
    "so_temperatura": ["temperatura"],
    "multivariado": ["temperatura", "chuva", "fim_de_semana", "feriado"],
}
EXPERIMENTS = D.ROOT / "reports" / "experimentos.csv"


def split(df: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * frac)  # holdout temporal: os últimos 20% dos dias
    return df.iloc[:cut], df.iloc[cut:]


@lru_cache(maxsize=4)
def compare() -> list[dict]:
    df = D.load()
    train, test = split(df)
    rows = []
    for name, cols in FEATURES.items():
        m = LinearRegression().fit(train[cols], train["vendas"])
        p = np.maximum(m.predict(test[cols]), 0)
        rows.append({"modelo": name, "r2": round(float(r2_score(test["vendas"], p)), 3),
                     "mae": round(float(mean_absolute_error(test["vendas"], p)), 2), "n_teste": len(test)})
    return rows


@lru_cache(maxsize=1)
def final_model() -> tuple[LinearRegression, np.ndarray]:
    """Modelo multivariado treinado em toda a série e resíduos do holdout (para intervalos)."""
    df = D.load()
    cols = FEATURES["multivariado"]
    train, test = split(df)
    m_hold = LinearRegression().fit(train[cols], train["vendas"])
    resid = test["vendas"].to_numpy() - m_hold.predict(test[cols])
    return LinearRegression().fit(df[cols], df["vendas"]), resid


def predict(temperatura: float, chuva: bool = False, fim_de_semana: bool = False, feriado: bool = False) -> dict:
    if not -5 <= float(temperatura) <= 45:
        raise ValueError("temperatura fora da faixa plausível (-5 a 45 °C)")
    m, resid = final_model()
    x = pd.DataFrame([[float(temperatura), int(chuva), int(fim_de_semana), int(feriado)]], columns=FEATURES["multivariado"])
    p = float(m.predict(x)[0])
    lo, hi = np.quantile(resid, [0.1, 0.9])
    return {"previsao": max(p, 0.0), "p10": max(p + lo, 0.0), "p90": max(p + hi, 0.0)}


def coefficients() -> dict:
    m, _ = final_model()
    c = dict(zip(FEATURES["multivariado"], m.coef_))
    return {"intercepto": float(m.intercept_), "por_grau": float(c["temperatura"]), "chuva": float(c["chuva"]),
            "fim_de_semana": float(c["fim_de_semana"]), "feriado": float(c["feriado"])}


def plan(dias: list[dict], custo_sobra: float, custo_falta: float) -> list[dict]:
    """Newsvendor: produz no quantil crítico custo_falta/(custo_falta+custo_sobra) da demanda prevista."""
    if custo_sobra <= 0 or custo_falta <= 0:
        raise ValueError("custos precisam ser positivos")
    _, resid = final_model()
    q = custo_falta / (custo_falta + custo_sobra)
    ajuste = float(np.quantile(resid, q))
    out = []
    for d in dias:
        p = predict(d["temperatura"], d.get("chuva", False), d.get("fim_de_semana", False), d.get("feriado", False))
        out.append({**d, "previsao": round(p["previsao"], 1), "producao": int(max(0, round(p["previsao"] + ajuste)))})
    return out


def log_experiment(nota: str = "") -> dict:
    """Registra a comparação em reports/experimentos.csv e, se o MLflow estiver instalado, também nele."""
    rows = compare()
    EXPERIMENTS.parent.mkdir(exist_ok=True)
    new = not EXPERIMENTS.exists()
    quando = datetime.now(UTC).isoformat(timespec="seconds")
    with EXPERIMENTS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quando", "modelo", "r2", "mae", "n_teste", "nota"])
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"quando": quando, **r, "nota": nota})
    mlflow_ok = False
    try:
        import mlflow

        mlflow.set_tracking_uri(f"file:{D.ROOT / 'mlruns'}")
        for r in rows:
            with mlflow.start_run(run_name=r["modelo"]):
                mlflow.log_params({"modelo": r["modelo"], "nota": nota})
                mlflow.log_metrics({"r2": r["r2"], "mae": r["mae"]})
        mlflow_ok = True
    except ImportError:
        pass
    return {"quando": quando, "linhas": rows, "mlflow": mlflow_ok}
