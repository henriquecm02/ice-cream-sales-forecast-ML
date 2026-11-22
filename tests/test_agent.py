"""Testes sem rede."""
import pytest

from sorveteagent import models as M
from sorveteagent.agentcore import ScriptedLLM
from sorveteagent.tools import build_agent, t_explicar, t_planejar


def test_slope_close_to_truth():
    assert abs(M.coefficients()["por_grau"] - 9.0) <= 0.5


def test_prediction_non_negative_and_interval_ordered():
    p = M.predict(12, chuva=True)
    assert 0 <= p["p10"] <= p["previsao"] <= p["p90"] or p["previsao"] == 0


def test_implausible_temperature_rejected():
    with pytest.raises(ValueError):
        M.predict(80)


def test_newsvendor_more_production_when_stockout_costs_more():
    assert t_planejar([30] * 7, custo_sobra=1, custo_falta=4)["producao_total"] > t_planejar([30] * 7, custo_sobra=4, custo_falta=1)["producao_total"]


def test_offline_router_extracts_temperature():
    r = build_agent(use_env=False).run("Quantos sorvetes vou vender com 33 graus?")
    assert r.calls[0]["args"]["temperatura"] == 33.0 and r.status == "ok"


def test_confirm_needs_approval():
    r = build_agent(use_env=False).run("Confirme a ordem de produção para a fábrica")
    assert r.pending and not r.results


def test_grounding_flags_and_passes():
    bad = build_agent(llm=ScriptedLLM([{"calls": [{"tool": "explicar_modelo", "args_json": "{}"}]},
                                       {"answer": "Cada grau vende 15.", "numbers_cited": [15]}]), use_env=False).run("x")
    assert bad.status == "revisar"
    g = t_explicar()["por_grau"]
    ok = build_agent(llm=ScriptedLLM([{"calls": [{"tool": "explicar_modelo", "args_json": "{}"}]},
                                      {"answer": f"Cada grau vende {g:.2f}.", "numbers_cited": [g]}]), use_env=False).run("x")
    assert ok.status == "ok", ok.ungrounded
