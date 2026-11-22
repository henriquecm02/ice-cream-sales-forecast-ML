"""Gerador do histórico diário de vendas de sorvete.

O projeto original não trazia dados. A série é gerada por uma fórmula conhecida
(gravada em data/parametros_reais.json), para que a avaliação confira se o modelo
recupera o efeito real da temperatura, da chuva, do fim de semana e dos feriados.
"""
from __future__ import annotations

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CSV = DATA_DIR / "vendas_sorvete.csv"
PARAMS = DATA_DIR / "parametros_reais.json"

TRUE_PARAMS = {
    "inicio": "2023-01-01",
    "fim": "2024-12-31",
    "intercepto": -80.0,
    "por_grau": 9.0,
    "fim_de_semana": 40.0,
    "feriado": 25.0,
    "chuva": -35.0,
    "ruido_desvio": 15.0,
    "semente": 7,
}


def generate(p: dict = TRUE_PARAMS) -> pd.DataFrame:
    rng = np.random.default_rng(p["semente"])
    dates = pd.date_range(p["inicio"], p["fim"], freq="D")
    n = len(dates)
    # clima do Sudeste: verão em janeiro (~30 °C), inverno em julho (~19 °C)
    temp = 24.5 + 5.5 * np.cos(2 * np.pi * (dates.dayofyear - 15) / 365.25) + rng.normal(0, 2.2, n)
    p_chuva = np.clip(0.18 + 0.17 * np.cos(2 * np.pi * (dates.dayofyear - 15) / 365.25), 0.03, 0.6)
    chuva = (rng.random(n) < p_chuva).astype(int)
    temp = temp - 2.0 * chuva
    fds = (dates.dayofweek >= 5).astype(int)
    br = holidays.country_holidays("BR", years=sorted(set(dates.year)))
    feriado = np.array([d.date() in br for d in dates], dtype=int)
    media = p["intercepto"] + p["por_grau"] * temp + p["fim_de_semana"] * fds + p["feriado"] * feriado + p["chuva"] * chuva
    vendas = np.maximum(0, np.round(media + rng.normal(0, p["ruido_desvio"], n))).astype(int)
    return pd.DataFrame({"data": dates, "temperatura": temp.round(1), "chuva": chuva, "fim_de_semana": fds,
                         "feriado": feriado, "vendas": vendas})


def write() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(CSV, index=False)
    PARAMS.write_text(json.dumps(TRUE_PARAMS, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(CSV, parse_dates=["data"]) if CSV.exists() else write()


if __name__ == "__main__":
    print(f"{len(write())} dias gravados em {CSV}")
