"""Avaliação contra gabarito. Roda sem GEMINI_API_KEY.

Uso: python eval/run_eval.py [--gemini]  (grava eval/resultados[_gemini].json e .md)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sorveteagent import data as D  # noqa: E402
from sorveteagent import models as M  # noqa: E402
from sorveteagent.agentcore import ScriptedLLM  # noqa: E402
from sorveteagent.tools import build_agent, t_planejar  # noqa: E402

ROTEAMENTO = [
    ("Quantos sorvetes vou vender amanhã com 31 graus?", "prever_vendas"),
    ("Previsão de vendas para sábado de 27 °C com chuva", "prever_vendas"),
    ("Quanto devo produzir na próxima semana?", "planejar_producao"),
    ("Monte o plano de produção para 30, 31, 29, 26, 24, 28 e 33 graus", "planejar_producao"),
    ("Qual o efeito de cada grau a mais nas vendas?", "explicar_modelo"),
    ("Quanto a chuva derruba as vendas?", "explicar_modelo"),
    ("Compare os modelos: qual tem o melhor R²?", "comparar_modelos"),
    ("Registre o experimento no MLflow", "registrar_experimento"),
    ("Confirme a ordem de produção para a fábrica", "confirmar_producao"),
]


def check(nome, ok, obtido, esperado):
    return {"caso": nome, "ok": bool(ok), "obtido": obtido, "esperado": esperado}


def main() -> int:
    real = json.loads(D.PARAMS.read_text(encoding="utf-8")) if D.PARAMS.exists() else D.TRUE_PARAMS
    out = []
    c = M.coefficients()
    out.append(check("efeito por grau recuperado (±0,5)", abs(c["por_grau"] - real["por_grau"]) <= 0.5,
                     round(c["por_grau"], 2), real["por_grau"]))
    out.append(check("efeito da chuva recuperado (±8)", abs(c["chuva"] - real["chuva"]) <= 8, round(c["chuva"], 2), real["chuva"]))
    out.append(check("efeito do fim de semana recuperado (±6)", abs(c["fim_de_semana"] - real["fim_de_semana"]) <= 6,
                     round(c["fim_de_semana"], 2), real["fim_de_semana"]))
    tab = {r["modelo"]: r for r in M.compare()}
    out.append(check("multivariado com R² >= 0,85 no holdout", tab["multivariado"]["r2"] >= 0.85, tab["multivariado"]["r2"], ">= 0.85"))
    out.append(check("multivariado erra menos que só temperatura", tab["multivariado"]["mae"] < tab["so_temperatura"]["mae"],
                     tab["multivariado"]["mae"], f"< {tab['so_temperatura']['mae']}"))
    p_frio = M.predict(10)["previsao"]
    out.append(check("previsão nunca negativa (10 °C)", p_frio >= 0, round(p_frio, 1), ">= 0"))
    barato = t_planejar([30] * 7, custo_sobra=4.0, custo_falta=1.0)["producao_total"]
    caro = t_planejar([30] * 7, custo_sobra=1.0, custo_falta=4.0)["producao_total"]
    out.append(check("faltar caro => produzir mais", caro > barato, caro, f"> {barato}"))

    usar_gemini = "--gemini" in sys.argv
    agent = build_agent(use_env=usar_gemini)
    if usar_gemini and agent.llm is None:
        print("--gemini pedido, mas GEMINI_API_KEY não está definida")
        return 2
    for pergunta, esperado in ROTEAMENTO:
        r = agent.run(pergunta)
        tools = [x["tool"] for x in r.calls]
        ok = tools[:1] == [esperado]
        ok = ok and (bool(r.pending) and not r.results if esperado == "confirmar_producao" else r.status == "ok")
        out.append(check(f"roteamento: {pergunta}", ok, tools, esperado))

    llm = ScriptedLLM([{"calls": [{"tool": "explicar_modelo", "args_json": "{}"}]},
                       {"answer": "Cada grau vende 15 sorvetes a mais.", "numbers_cited": [15]}])
    r = build_agent(llm=llm, use_env=False).run("Qual o efeito do grau?")
    out.append(check("número inventado é barrado", r.status == "revisar", r.status, "revisar"))

    acertos = sum(x["ok"] for x in out)
    suf = "_gemini" if usar_gemini else ""
    res = {"acertos": acertos, "total": len(out), "modelos": M.compare(), "coeficientes": c, "casos": out}
    (ROOT / "eval" / f"resultados{suf}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = [f"# Resultados da avaliação\n\n**{acertos} de {len(out)} casos corretos.**\n", "| Caso | Resultado | Obtido | Esperado |",
          "|---|---|---|---|"] + [f"| {x['caso']} | {'ok' if x['ok'] else 'FALHOU'} | {x['obtido']} | {x['esperado']} |" for x in out]
    md += ["\n## Modelos (holdout com os últimos 20% dos dias)\n", "| Modelo | R² | MAE |", "|---|---|---|"]
    md += [f"| {r['modelo']} | {r['r2']:.3f} | {r['mae']:.2f} |" for r in M.compare()]
    (ROOT / "eval" / f"resultados{suf}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{acertos}/{len(out)} casos corretos")
    for x in out:
        if not x["ok"]:
            print("FALHOU:", x)
    return 0 if acertos == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
