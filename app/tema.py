"""Identidade visual compartilhada pelos dashboards: a mesma do portfólio (henriquecm02.github.io).

Cores, fontes e raios ficam em `.streamlit/config.toml` (modo claro e escuro). Este módulo só acrescenta o que o tema
do Streamlit não cobre: o cabeçalho, os cards de KPI, as abas em pílula e cores para traços com cor explícita.
Copiado de forma idêntica em cada repositório, como o agentcore.py.
"""
from __future__ import annotations

import html

import streamlit as st

# Mesma ordem das paletas do config.toml (validadas para daltonismo). Cor segue a entidade, nunca a posição.
PALETA = {"claro": ["#E3263B", "#2a78d6", "#eda100", "#1baf7a", "#6250d6"],
          "escuro": ["#E8455A", "#3987e5", "#c98500", "#199e70", "#9085e9"]}
NEUTRO = {"claro": "#9A8A8D", "escuro": "#8C7E80"}  # série de referência: histórico, linha de base
# Escala sequencial (magnitude): um só tom, do claro ao escuro (no modo escuro, do escuro ao claro).
SEQUENCIAL = {"claro": ["#FDECEE", "#F6B3BC", "#EC6A7B", "#E3263B", "#B3132A", "#7A0A1B"],
              "escuro": ["#3A1218", "#7A0A1B", "#B3132A", "#E3263B", "#EC6A7B", "#F6B3BC"]}
STATUS = {"ok": ("#157347", "✅"), "atencao": ("#8A5200", "⚠️"), "critico": ("#C8102E", "⛔")}


def modo() -> str:
    try:
        return "escuro" if st.context.theme.type == "dark" else "claro"
    except Exception:  # noqa: BLE001 - fora de uma sessão (testes, scripts) não há tema
        return "claro"


def cor(i: int) -> str:
    """Cor categórica do slot i (0 = vermelho da marca)."""
    return PALETA[modo()][i]


def cores(n: int) -> list[str]:
    """As n primeiras cores categóricas, na ordem fixa. Mais de 5 séries: agrupar em "Outros" ou separar."""
    if n > len(PALETA["claro"]):
        raise ValueError("Mais de 5 séries: agrupe em 'Outros' ou use pequenos múltiplos.")
    return PALETA[modo()][:n]


def sequencial() -> list[str]:
    return SEQUENCIAL[modo()]


def neutro() -> str:
    return NEUTRO[modo()]


def texto_secundario() -> str:
    """Cor de rótulos de valor dentro de gráficos: texto, nunca a cor da série."""
    return "#B9ACAE" if modo() == "escuro" else "#6B5C5F"


def rgba(hex_: str, alpha: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def selo(nivel: str, texto: str) -> str:
    """Selo de status em HTML: sempre ícone + texto, nunca só a cor."""
    c, icone = STATUS[nivel]
    return (f'<span class="hc-selo" style="--c:{c}">{icone} {html.escape(texto)}</span>')


_CSS = """
<style>
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.2rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { letter-spacing: -0.01em; }
.hc-hero { position: relative; overflow: hidden; border-radius: 28px; padding: 30px 34px 28px;
  margin: 0 0 18px;
  color: #fff; background: radial-gradient(120% 140% at 100% 0%, #F0566A 0%, rgba(240,86,106,0) 55%),
  linear-gradient(135deg, #E3263B 0%, #B3132A 62%, #7A0A1B 100%);
  box-shadow: 0 20px 60px rgba(122,10,27,.22); }
.hc-hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
  border-radius: 50%;
  border: 38px solid rgba(255,255,255,.07); }
.hc-hero .k { font: 700 12px/1 var(--hc-mono, "JetBrains Mono", monospace); letter-spacing: .08em;
  text-transform: uppercase;
  opacity: .85; margin-bottom: 10px; }
.hc-hero h1 { color: #fff !important; font-size: clamp(1.7rem, 3.2vw, 2.5rem); line-height: 1.1;
  margin: 0 0 10px; padding: 0; }
.hc-hero p { color: rgba(255,255,255,.9); max-width: 780px; margin: 0; font-size: 1rem; line-height: 1.55; }
.hc-hero .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.hc-hero .chips span { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
  border-radius: 999px;
  padding: 5px 12px; font-size: .82rem; font-weight: 600; }
[data-testid="stMetric"] { background: var(--hc-surface); border: 1px solid var(--hc-line);
  border-radius: 20px;
  padding: 16px 18px; min-height: 118px;
  box-shadow: 0 1px 2px rgba(60,10,20,.05), 0 6px 18px rgba(60,10,20,.06); }
[data-testid="stMetricLabel"] p { font-weight: 600; color: var(--hc-muted); }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; }
[data-testid="stMetricDelta"] { max-width: 100%; border-radius: 10px !important; height: auto !important; }
[data-testid="stMetricDelta"] p { white-space: normal !important; overflow: visible !important; }
[role="tablist"] { gap: 8px; flex-wrap: wrap; border: 0 !important; box-shadow: none !important;
  padding-bottom: 4px; }
[role="tablist"]::after, [role="tablist"]::before { display: none !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .react-aria-SelectionIndicator { display: none
  !important; }
[role="tab"] { border-radius: 999px !important; padding: 8px 18px !important;
  border: 1px solid var(--hc-line) !important;
  background: var(--hc-surface) !important; height: auto !important;
  transition: background .15s, border-color .15s; }
[role="tab"]:hover { border-color: var(--hc-red) !important; }
[role="tab"] p { font-weight: 600; margin: 0; }
[role="tab"][aria-selected="true"] { background: var(--hc-soft) !important;
  border-color: transparent !important; }
[role="tab"][aria-selected="true"] p { color: var(--hc-red); }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius:
  20px; }
[data-testid="stAlert"] { border-radius: 16px; }
.hc-selo { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-weight: 600;
  font-size: .85rem; color: var(--c); background: color-mix(in srgb, var(--c) 12%, transparent); }
.hc-nota { color: var(--hc-muted); font-size: .9rem; }
</style>
"""
_TOKENS = {
    "claro": "--hc-surface:#FFFFFF;--hc-line:#EEE4E5;--hc-muted:#6B5C5F;--hc-soft:#FDECEE;--hc-red:#C8102E;",
    "escuro": "--hc-surface:#1F191A;--hc-line:#3A2F31;--hc-muted:#B9ACAE;--hc-soft:#3A1218;--hc-red:#F08E9B;",
}


def aplicar_tema(titulo: str, subtitulo: str, kicker: str = "", chips: list[str] | None = None) -> None:
    """Injeta o CSS do portfólio e desenha o cabeçalho. Chamar logo depois de st.set_page_config."""
    st.html(f"<style>:root{{{_TOKENS[modo()]}}}</style>" + _CSS)
    chips_html = "".join(f"<span>{html.escape(c)}</span>" for c in chips or [])
    st.html(f'<div class="hc-hero">{f"<div class=k>{html.escape(kicker)}</div>" if kicker else ""}'
            f"<h1>{html.escape(titulo)}</h1><p>{html.escape(subtitulo)}</p>"
            f'{f"<div class=chips>{chips_html}</div>" if chips_html else ""}</div>')


def estilo_plotly(fig, altura: int | None = None, **layout):
    """Acabamento comum dos gráficos Plotly: legenda no topo, hover unificado, margens enxutas."""
    fig.update_layout(margin={"l": 8, "r": 8, "t": 36, "b": 8},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title_text": ""},
                      hoverlabel={"font_family": "Figtree"}, **({"height": altura} if altura else {}), **layout)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(zeroline=False)
    return fig
