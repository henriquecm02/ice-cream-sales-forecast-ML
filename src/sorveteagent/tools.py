"""Ferramentas do agente de vendas de sorvete. Cada uma devolve um dict com 'resumo'."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from . import data as D
from . import models as M
from .agentcore import Agent, Tool, make_llm

REPORTS = D.ROOT / "reports"


def t_prever(temperatura: float, chuva: bool = False, fim_de_semana: bool = False, feriado: bool = False) -> dict:
    p = M.predict(temperatura, chuva, fim_de_semana, feriado)
    ctx = ", ".join(x for x, on in (("com chuva", chuva), ("fim de semana", fim_de_semana), ("feriado", feriado)) if on)
    return {**{k: round(v, 1) for k, v in p.items()}, "temperatura": float(temperatura),
            "resumo": (f"A {float(temperatura):.1f} °C{(' (' + ctx + ')') if ctx else ''}, a previsão é de "
                       f"{p['previsao']:.1f} sorvetes (faixa p10–p90 de {p['p10']:.1f} a {p['p90']:.1f}).")}


def _semana(temperaturas: list[float], chuvas: list[int] | None) -> list[dict]:
    if not temperaturas or len(temperaturas) > 14:
        raise ValueError("informe de 1 a 14 temperaturas")
    chuvas = chuvas or [0] * len(temperaturas)
    inicio = D.load()["data"].iloc[-1] + pd.Timedelta(days=1)
    dias = []
    for i, t in enumerate(temperaturas):
        d = inicio + pd.Timedelta(days=i)
        dias.append({"data": str(d.date()), "temperatura": float(t), "chuva": bool(chuvas[i] if i < len(chuvas) else 0),
                     "fim_de_semana": d.dayofweek >= 5, "feriado": False})
    return dias


def t_planejar(temperaturas: list[float], chuvas: list[int] | None = None, custo_sobra: float = 1.5,
               custo_falta: float = 4.0) -> dict:
    plano = M.plan(_semana(temperaturas, chuvas), float(custo_sobra), float(custo_falta))
    total = sum(d["producao"] for d in plano)
    q = float(custo_falta) / (float(custo_falta) + float(custo_sobra))
    return {"dias": plano, "producao_total": total, "quantil_critico": round(q, 3),
            "resumo": (f"Plano de {len(plano)} dias: produzir {total} sorvetes no total. Com custo de sobra "
                       f"{float(custo_sobra):.2f} e de falta {float(custo_falta):.2f}, produzo no quantil {q:.3f} da demanda.")}


def t_explicar() -> dict:
    c = M.coefficients()
    return {**{k: round(v, 2) for k, v in c.items()},
            "resumo": (f"Cada grau a mais vende {c['por_grau']:.2f} sorvetes. Chuva muda as vendas em {c['chuva']:.2f}, "
                       f"fim de semana em {c['fim_de_semana']:.2f} e feriado em {c['feriado']:.2f} unidades por dia.")}


def t_comparar() -> dict:
    rows = M.compare()
    txt = "; ".join(f"{r['modelo']}: R² {r['r2']:.3f}, MAE {r['mae']:.2f}" for r in rows)
    melhor = max(rows, key=lambda r: r["r2"])
    return {"tabela": rows, "melhor_modelo": melhor["modelo"],
            "resumo": f"Holdout com os últimos {rows[0]['n_teste']} dias. {txt}. Melhor: {melhor['modelo']}."}


def t_registrar(nota: str = "") -> dict:
    r = M.log_experiment(nota)
    return {**r, "resumo": f"Experimento registrado em reports/experimentos.csv ({len(r['linhas'])} modelos)"
            + (" e no MLflow." if r["mlflow"] else ".")}


def t_confirmar(temperaturas: list[float], chuvas: list[int] | None = None, custo_sobra: float = 1.5,
                custo_falta: float = 4.0) -> dict:
    """Ação sensível: confirma a ordem de produção para a fábrica."""
    plano = t_planejar(temperaturas, chuvas, custo_sobra, custo_falta)
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(plano["dias"]).to_csv(REPORTS / "ordem_producao.csv", index=False)
    return {"producao_total": plano["producao_total"],
            "resumo": f"Ordem de produção confirmada: {plano['producao_total']} sorvetes (reports/ordem_producao.csv)."}


SEMANA = "lista de 1 a 14 temperaturas em °C, uma por dia a partir de amanhã"
TOOLS = [
    Tool("prever_vendas", "Prevê as vendas de um dia.",
         {"temperatura": "número em °C", "chuva": "bool", "fim_de_semana": "bool", "feriado": "bool"}, t_prever),
    Tool("planejar_producao", "Quantidade a produzir por dia minimizando sobra e falta (newsvendor).",
         {"temperaturas": SEMANA, "chuvas": "lista de 0/1", "custo_sobra": "número, padrão 1,5",
          "custo_falta": "número, padrão 4,0"}, t_planejar),
    Tool("explicar_modelo", "Efeito de temperatura, chuva, fim de semana e feriado nas vendas.", {}, t_explicar),
    Tool("comparar_modelos", "Compara o modelo só com temperatura e o multivariado (R² e MAE em holdout).", {}, t_comparar),
    Tool("registrar_experimento", "Registra a comparação de modelos (CSV e MLflow).", {"nota": "texto"}, t_registrar),
    Tool("confirmar_producao", "Envia a ordem de produção para a fábrica.",
         {"temperaturas": SEMANA, "chuvas": "lista de 0/1", "custo_sobra": "número", "custo_falta": "número"},
         t_confirmar, sensitive=True),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()


def offline_router(question: str) -> list[dict]:
    q = _norm(question)
    temps = [float(x.replace(",", ".")) for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:graus|°|oc|c\b)", q)]
    flags = {"chuva": "chuv" in q, "fim_de_semana": any(w in q for w in ("sabado", "domingo", "fim de semana")),
             "feriado": "feriado" in q}
    semana = temps or [28.0, 29.0, 30.0, 27.0, 25.0, 31.0, 32.0]
    if "confirm" in q or "envi" in q and "fabrica" in q:
        return [{"tool": "confirmar_producao", "args": {"temperaturas": semana}}]
    if any(w in q for w in ("produz", "produc", "fabricar", "plano")):
        return [{"tool": "planejar_producao", "args": {"temperaturas": semana}}]
    if any(w in q for w in ("registr", "mlflow", "experiment")):
        return [{"tool": "registrar_experimento", "args": {"nota": question}}]
    if any(w in q for w in ("compar", "r2", "r²", "melhor modelo", "erro")):
        return [{"tool": "comparar_modelos", "args": {}}]
    if any(w in q for w in ("efeito", "por grau", "impacto", "explic", "quanto a chuva")):
        return [{"tool": "explicar_modelo", "args": {}}]
    if temps or "vend" in q or "prev" in q:
        return [{"tool": "prever_vendas", "args": {"temperatura": temps[0] if temps else 28.0, **flags}}]
    return []


def build_agent(llm=None, use_env: bool = True) -> Agent:
    if llm is None and use_env:
        llm = make_llm()
    return Agent(TOOLS, llm=llm, offline_router=offline_router)
