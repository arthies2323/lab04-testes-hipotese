"""Figuras G3-G4 (Plotly) e tabela de dados reais. Cor fixa por teste.
G1 e G2 ficam em sprint01_plots, usados pelos dois dashboards."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import config

C = config.COLORS


def type1_summary(res):
    """Estimativas sob H0 e intervalos pontuais de Monte Carlo."""
    null = res.loc[res.d.eq(0)].copy()
    intervals = [
        stats.binomtest(int(row.rejeicoes), int(row.repeticoes))
        .proportion_ci(confidence_level=0.95, method="wilson")
        for row in null.itertuples()
    ]
    return null.assign(ic_inf=[ci.low for ci in intervals],
                       ic_sup=[ci.high for ci in intervals])


def g3(res):
    null = type1_summary(res)
    alpha = null.alpha.iloc[0]
    fig = go.Figure()
    for teste, color in C.items():
        s = null.loc[null.teste.eq(teste)]
        fig.add_bar(
            x=s.situacao, y=100 * s.taxa_rejeicao, name=teste, marker_color=color,
            error_y=dict(type="data", symmetric=False,
                         array=100 * (s.ic_sup - s.taxa_rejeicao),
                         arrayminus=100 * (s.taxa_rejeicao - s.ic_inf)),
            customdata=s[["rejeicoes", "repeticoes", "ic_inf", "ic_sup"]].to_numpy(),
            hovertemplate=("%{x}<br>Taxa: %{y:.2f}%<br>"
                           "Rejeições: %{customdata[0]:.0f}/%{customdata[1]:.0f}<br>"
                           "IC 95%: %{customdata[2]:.2%}–%{customdata[3]:.2%}<extra>%{fullData.name}</extra>"),
        )
    fig.add_hline(y=100 * alpha, line_dash="dash", line_color="black",
                  annotation_text=f"Nível nominal ({alpha:.0%})")
    fig.update_layout(barmode="group", title="G3 — Erro tipo I e incerteza de Monte Carlo",
                      xaxis_title="Situação simulada", yaxis_title="Rejeições de H0 (%)")
    return fig


def g4(res):
    """Curvas de poder por situação. Em d=0 a curva é o erro tipo I (referência em α)."""
    sits = list(res.situacao.unique())
    alpha = 100 * res.alpha.iloc[0] if "alpha" in res else 100 * config.ALPHA
    fig = make_subplots(2, 2, subplot_titles=sits, shared_yaxes=True, vertical_spacing=0.16)
    for i, s in enumerate(sits):
        r, c = i // 2 + 1, i % 2 + 1
        for teste in C:
            d = res[(res.situacao == s) & (res.teste == teste)].sort_values("d")
            fig.add_scatter(x=d.d, y=d.taxa_rejeicao * 100, mode="lines+markers", name=teste,
                            line=dict(color=C[teste]), legendgroup=teste, showlegend=i == 0,
                            hovertemplate="d = %{x}<br>Rejeições: %{y:.1f}%<extra>" + teste + "</extra>",
                            row=r, col=c)
        fig.add_hline(y=alpha, line_dash="dash", line_color="black", row=r, col=c,
                      annotation_text=f"α = {alpha:.0f}% (d = 0)" if i == 0 else None)
        fig.add_hline(y=80, line_dash="dot", line_color="gray", row=r, col=c,
                      annotation_text="80% de poder" if i == 0 else None)
    fig.update_xaxes(title_text="Tamanho do efeito d (desvios-padrão)")
    fig.update_yaxes(title_text="Rejeições de H0 (%)", range=[0, 100])
    fig.update_layout(title="G4 — Curvas de poder: Welch × Mann-Whitney (Q2 e Q3)", autosize=True,
                      height=700, legend=dict(orientation="h", y=-0.12))
    return fig


def g4_reading(res, min_gap=0.02):
    """Leitura derivada dos resultados: quem lidera em cada situação e onde a vantagem muda."""
    out = []
    for s, g in res[res.d > 0].groupby("situacao", sort=False):
        w = g[g.teste == "Welch"].set_index("d").taxa_rejeicao
        m = g[g.teste == "Mann-Whitney"].set_index("d").taxa_rejeicao
        diff = (m - w)
        lead = diff.map(lambda x: "Mann-Whitney" if x > min_gap else "Welch" if x < -min_gap else "empate")
        d_max = diff.abs().idxmax()
        txt = f"{s}: dif. máxima de {abs(diff[d_max])*100:.0f} p.p. a favor do "               f"{'Mann-Whitney' if diff[d_max] > 0 else 'Welch'} (d = {d_max:g})"
        leaders = set(lead) - {"empate"}
        if len(leaders) > 1:
            txt += "; a vantagem muda de lado (" + ", ".join(f"d={d:g}: {l}" for d, l in lead.items()) + ")"
        elif leaders:
            txt += f"; {leaders.pop()} lidera, sem troca de lado"
        else:
            txt += "; empate prático (diferença abaixo de 2 p.p.)"
        out.append(txt)
    return out


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
