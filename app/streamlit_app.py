"""Dashboard e agente de vendas de sorvete. Rodar: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sorveteagent import data as D  # noqa: E402
from sorveteagent import models as M  # noqa: E402
from sorveteagent.tools import build_agent, t_planejar  # noqa: E402

st.set_page_config(page_title="Sorvete · previsão e produção", page_icon="🍦", layout="wide")


@st.cache_resource
def get_agent():
    return build_agent()


df = D.load()
agent = get_agent()
st.session_state.setdefault("pending", [])
st.session_state.setdefault("log", [])

tema.aplicar_tema(
    "Vendas de sorvete: previsão e plano de produção",
    "Agente de IA que prevê vendas pelo clima e decide quanto produzir equilibrando sobra e falta. "
    "Os números vêm de scikit-learn; o modelo de linguagem só interpreta.",
    kicker="Agente de IA · produção",
    chips=["Gemini", "Regressão multivariada", "Dados sintéticos com fórmula conhecida"])
if agent.llm is None:
    st.info("Modo offline: sem GEMINI_API_KEY, o agente escolhe ferramentas por palavras-chave.", icon="🔌")

tab_dash, tab_agent, tab_eval, tab_aprov = st.tabs(["Dashboard", "Agente", "Avaliação", "Aprovações"])

with tab_dash:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Temperatura × vendas")
        coef = M.coefficients()
        xs = np.linspace(df["temperatura"].min(), df["temperatura"].max(), 50)
        fig = go.Figure()
        for chuva, cor, nome in ((0, tema.neutro(), "Dias secos"), (1, tema.cor(1), "Dias com chuva")):
            sub = df[df["chuva"] == chuva]
            fig.add_trace(go.Scatter(x=sub["temperatura"], y=sub["vendas"], mode="markers", name=nome,
                                     marker={"color": cor, "size": 5, "opacity": 0.55}))
        fig.add_trace(go.Scatter(x=xs, y=coef["intercepto"] + coef["por_grau"] * xs, name="Modelo (dia útil seco)",
                                 line={"color": tema.cor(0), "width": 3}))
        fig.update_layout(height=400, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title="°C", yaxis_title="sorvetes/dia",
                          legend={"orientation": "h", "y": -0.18})
        st.plotly_chart(tema.estilo_plotly(fig), use_container_width=True)
    with c2:
        st.subheader("Plano de produção da semana")
        temps = [st.slider(f"Dia {i + 1} (°C)", 10, 40, v, key=f"t{i}") for i, v in enumerate([28, 29, 30, 27, 25, 31, 32])]
        cs, cf = st.columns(2)
        custo_sobra = cs.number_input("Custo de sobra (R$)", 0.1, 20.0, 1.5, 0.5)
        custo_falta = cf.number_input("Custo de falta (R$)", 0.1, 20.0, 4.0, 0.5)
    plano = t_planejar(temps, None, custo_sobra, custo_falta)
    k1, k2, k3, k4 = st.columns(4)
    tab = {r["modelo"]: r for r in M.compare()}
    k1.metric("Produção da semana", f"{plano['producao_total']} un")
    k2.metric("Vendas por grau", f"{coef['por_grau']:.2f}")
    k3.metric("R² (holdout)", f"{tab['multivariado']['r2']:.3f}", delta=f"{tab['multivariado']['r2'] - tab['so_temperatura']['r2']:+.3f} vs. só temperatura")
    k4.metric("Efeito da chuva", f"{coef['chuva']:.1f} un/dia")
    pl = pd.DataFrame(plano["dias"])[["data", "temperatura", "fim_de_semana", "previsao", "producao"]]
    fig2 = go.Figure([go.Bar(x=pl["data"], y=pl["previsao"], name="Demanda prevista", marker_color=tema.neutro()),
                      go.Bar(x=pl["data"], y=pl["producao"], name="Produzir", marker_color=tema.cor(0))])
    fig2.update_layout(barmode="group", height=300, margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(tema.estilo_plotly(fig2), use_container_width=True)
    st.caption(f"Produção no quantil {plano['quantil_critico']:.3f} da demanda: quanto mais cara a falta, mais se produz.")

with tab_agent:
    st.subheader("Pergunte ao agente")
    ex = st.selectbox("Exemplos", ["", "Quantos sorvetes vou vender amanhã com 31 graus?", "Quanto a chuva derruba as vendas?",
                                   "Quanto devo produzir na próxima semana?", "Compare os modelos: qual tem o melhor R²?",
                                   "Confirme a ordem de produção para a fábrica"], placeholder="Escolha um exemplo")
    q = st.text_input("Pergunta", value=ex)
    if st.button("Perguntar", type="primary") and q.strip():
        r = agent.run(q)
        st.session_state["log"].append(r)
        st.session_state["pending"].extend(r.pending)
    for r in reversed(st.session_state["log"][-5:]):
        with st.container(border=True):
            st.markdown(f"**{r.question}**")
            st.write(r.answer)
            badge = {"ok": "✅ números conferidos", "revisar": "⚠️ revisar: número sem fonte", "erro": "❌ erro"}[r.status]
            st.caption(f"{badge} · modo {r.mode} · ferramentas: {', '.join(c['tool'] for c in r.calls) or 'nenhuma'}")
            if r.pending:
                st.info("Ação sensível aguardando aprovação na aba Aprovações.")
            with st.expander("Rastro"):
                st.json({"calls": r.calls, "results": r.results, "errors": r.errors}, expanded=False)

with tab_eval:
    p = ROOT / "eval" / "resultados.json"
    if p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        st.metric("Casos corretos", f"{res['acertos']} de {res['total']}")
        st.dataframe(pd.DataFrame(res["casos"]).astype(str), hide_index=True, use_container_width=True)
    else:
        st.write("Rode `python eval/run_eval.py`.")

with tab_aprov:
    if not st.session_state["pending"]:
        st.write("Nenhuma ação pendente. Peça ao agente para confirmar a ordem de produção.")
    for i, pa in enumerate(list(st.session_state["pending"])):
        with st.container(border=True):
            st.markdown(f"**{pa.tool}** · `{pa.args}`")
            a, b = st.columns(2)
            if a.button("Aprovar e executar", key=f"ok{i}", type="primary"):
                st.success(agent.approve(pa)["resumo"])
                st.session_state["pending"].remove(pa)
            if b.button("Recusar", key=f"no{i}"):
                st.session_state["pending"].remove(pa)
                st.rerun()
