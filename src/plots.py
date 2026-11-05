"""Figuras G1-G4 (Plotly) e tabela de dados reais. Cor fixa por teste."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import config, simulate

C = config.COLORS
GRP = {"com_bot": "#2ca02c", "sem_bot": "#7f7f7f"}


def g1(real, med, sig, seed=config.SEED):
    rng = simulate.make_rng(seed)
    a, b = simulate.generate(rng, "lognormal", 5000, 0.0, 1, med, sig)
    sim = {"com_bot": a[0], "sem_bot": b[0]}
    fig = make_subplots(1, 2, subplot_titles=("Dados reais (n=30 por grupo)", "Dados simulados (lognormal, n=5000 por grupo)"))
    for g in ("com_bot", "sem_bot"):
        fig.add_histogram(x=real[real.grupo == g].horas_ate_merge, name=g, marker_color=GRP[g], opacity=.65,
                          xbins=dict(size=25), legendgroup=g, row=1, col=1)
        fig.add_histogram(x=np.clip(sim[g], 0, 500), name=g, marker_color=GRP[g], opacity=.65,
                          xbins=dict(size=25), legendgroup=g, showlegend=False, row=1, col=2)
    fig.update_layout(barmode="overlay", title="G1 — Tempo até o merge: real × simulado (simulado truncado em 500 h)")
    fig.update_xaxes(title_text="Tempo até o merge (horas)")
    fig.update_yaxes(title_text="Nº de PRs")
    return fig


def g2_example(med, sig, n=config.N_PER_GROUP, seed=7):
    rng = simulate.make_rng(seed)
    a, b = simulate.generate(rng, "lognormal", n, 0.0, 1, med, sig)
    a, b = a[0], b[0]
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return a, b, t, u


def g2(med, sig, n=config.N_PER_GROUP):
    rng = simulate.make_rng(config.SEED)
    A, B = simulate.generate(rng, "lognormal", n, 0.0, config.N_SIM, med, sig)
    ts = stats.ttest_ind(A, B, equal_var=False, axis=-1).statistic
    a, b, t, u = g2_example(med, sig, n)
    df = 2 * n - 2
    crit = stats.t.ppf(1 - config.ALPHA / 2, df)
    x = np.linspace(-5, 5, 600)
    pdf = stats.t.pdf(x, df)
    fig = go.Figure()
    fig.add_histogram(x=ts, histnorm="probability density", nbinsx=80, name="Estatística t simulada (H0)",
                      marker_color="rgba(31,119,180,.35)", xbins=dict(start=-5, end=5, size=.125))
    fig.add_scatter(x=x, y=pdf, name=f"t de Student teórica (gl≈{df})", line=dict(color=C["Welch"], width=2))
    for sgn, lab in ((-1, "Região de rejeição (α=0,05)"), (1, None)):
        m = (x <= -crit) if sgn < 0 else (x >= crit)
        fig.add_scatter(x=x[m], y=pdf[m], fill="tozeroy", mode="none", fillcolor="rgba(214,39,40,.45)",
                        name=lab or "", showlegend=lab is not None)
    tt = float(t.statistic)
    for sgn in (-1, 1):
        m = (x <= -abs(tt)) if sgn < 0 else (x >= abs(tt))
        fig.add_scatter(x=x[m], y=pdf[m], fill="tozeroy", mode="none", fillcolor="rgba(0,0,0,.6)",
                        name=f"Área do p-valor do exemplo = {t.pvalue:.3f}", showlegend=sgn > 0)
    fig.add_vline(x=tt, line_dash="dash", annotation_text=f"t observado = {tt:.2f}")
    fig.update_layout(title="G2 — Distribuição da estatística t de Welch sob H0 (dados lognormais, n=30 por grupo)",
                      xaxis_title="Estatística t (adimensional)", yaxis_title="Densidade")
    return fig, t, u


def g3(res):
    r = res[res.d == 0]
    fig = go.Figure()
    for teste in C:
        s = r[r.teste == teste]
        fig.add_bar(x=s.situacao, y=s.taxa_rejeicao * 100, name=teste, marker_color=C[teste])
    fig.add_hline(y=5, line_dash="dash", line_color="black", annotation_text="α = 5%")
    fig.update_layout(barmode="group", title="G3 — Erro tipo I por situação simulada (10.000 simulações, H0 verdadeira)",
                      xaxis_title="Situação simulada", yaxis_title="Rejeições de H0 (%)")
    return fig


def g4(res):
    sits = list(res.situacao.unique())
    fig = make_subplots(2, 2, subplot_titles=sits, shared_yaxes=True)
    for i, s in enumerate(sits):
        for teste in C:
            d = res[(res.situacao == s) & (res.teste == teste)]
            fig.add_scatter(x=d.d, y=d.taxa_rejeicao * 100, mode="lines+markers", name=teste,
                            line=dict(color=C[teste]), legendgroup=teste, showlegend=i == 0,
                            row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(title_text="Tamanho do efeito d (desvios-padrão)")
    fig.update_yaxes(title_text="Poder (% de rejeições)")
    fig.update_layout(title="G4 — Curvas de poder: Welch × Mann-Whitney", height=650)
    return fig


def real_table(real):
    a = real[real.grupo == "com_bot"].horas_ate_merge.values
    b = real[real.grupo == "sem_bot"].horas_ate_merge.values
    tw = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / sp
    rbc = 2 * mw.statistic / (len(a) * len(b)) - 1  # correlação bisserial por postos
    f = lambda p: "rejeita H0" if p < config.ALPHA else "não rejeita H0"
    return pd.DataFrame([
        dict(Teste="Welch", Estatística=f"t = {tw.statistic:.2f}", **{"p-valor": round(float(tw.pvalue), 4)},
             **{"Tamanho do efeito": f"d de Cohen = {d:.2f}"}, Conclusão=f(tw.pvalue)),
        dict(Teste="Mann-Whitney", Estatística=f"U = {mw.statistic:.0f}", **{"p-valor": round(float(mw.pvalue), 4)},
             **{"Tamanho do efeito": f"correlação bisserial = {rbc:.2f}"}, Conclusão=f(mw.pvalue)),
    ])
