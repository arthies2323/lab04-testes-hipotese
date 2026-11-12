"""Figuras G3-G4 e de sensibilidade de Q3 (Plotly), leituras dos resultados e tabela de dados reais.
Cor fixa por teste.
G1 e G2 ficam em sprint01_plots, usados pelos dois dashboards."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import config, simulate

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


def q3_comparison(res):
    """Q3: variação (p.p.) da taxa de rejeição de cada situação em relação à referência, por teste
    e na mesma grade de d (d = 0 é o erro tipo I; d > 0, o poder)."""
    piv = 100 * res.pivot_table(index=["situacao", "teste"], columns="d", values="taxa_rejeicao", sort=False)
    ref = piv.xs(simulate.REFERENCE, level="situacao")
    return piv.drop(index=simulate.REFERENCE, level="situacao").sub(ref, level="teste")


def q3_reading(res):
    """Leitura de Q3 derivada de q3_comparison: mudança no erro tipo I e a maior mudança de poder."""
    out = []
    for sit, g in q3_comparison(res).groupby(level="situacao", sort=False):
        parts = []
        for (_, test), row in g.iterrows():
            power = row.drop(0.0)
            d = power.abs().idxmax()
            parts.append(f"{test} {row[0.0]:+.1f} p.p. no erro tipo I e {power[d]:+.1f} p.p. de poder em d = {d:g}")
        out.append(f"{sit}: " + "; ".join(parts))
    return out


def sample_size_figure(sens, d_ref):
    """Q3: erro tipo I (d = 0, tracejado) e poder (d = d_ref, contínuo) × n por grupo, por forma."""
    shapes = list(sens.forma.unique())
    alpha = 100 * sens.alpha.iloc[0]
    fig = make_subplots(1, len(shapes), subplot_titles=shapes, shared_yaxes=True)
    for i, shape in enumerate(shapes, 1):
        for teste, color in C.items():
            for d, dash, what in ((0.0, "dash", "erro tipo I (d = 0)"), (d_ref, "solid", f"poder (d = {d_ref:g})")):
                s = sens[sens.forma.eq(shape) & sens.teste.eq(teste) & np.isclose(sens.d, d)].sort_values("n")
                fig.add_scatter(x=s.n, y=100 * s.taxa_rejeicao, mode="lines+markers", name=f"{teste}: {what}",
                                line=dict(color=color, dash=dash), legendgroup=f"{teste}-{d}", showlegend=i == 1,
                                hovertemplate="n = %{x}<br>Rejeições: %{y:.1f}%<extra>" + teste + "</extra>",
                                row=1, col=i)
        fig.add_hline(y=alpha, line_dash="dash", line_color="black", row=1, col=i,
                      annotation_text=f"α = {alpha:.0f}%" if i == 1 else None)
        fig.add_hline(y=80, line_dash="dot", line_color="gray", row=1, col=i,
                      annotation_text="80% de poder" if i == 1 else None)
    fig.update_xaxes(title_text="Tamanho da amostra (PRs por grupo)", type="log", tickvals=sorted(sens.n.unique()))
    fig.update_yaxes(title_text="Rejeições de H0 (%)", range=[0, 100])
    fig.update_layout(title=f"Q3 — Sensibilidade ao tamanho da amostra: erro tipo I e poder (d = {d_ref:g})",
                      height=480, legend=dict(orientation="h", y=-0.3))
    return fig


def sample_size_reading(sens, d_ref, target=0.8):
    """Por forma: erro tipo I no menor e no maior n, e o menor n da grade com poder ≥ 80% em d_ref."""
    out = []
    for shape, g in sens.groupby("forma", sort=False):
        parts = []
        for test in C:
            t = g[g.teste.eq(test)]
            null = t[t.d.eq(0)].sort_values("n")
            power = t[np.isclose(t.d, d_ref)].sort_values("n")
            reach = power.loc[power.taxa_rejeicao.ge(target), "n"]
            n80 = (f"{target:.0%} de poder a partir de n = {reach.iloc[0]}" if len(reach) else
                   f"não chega a {target:.0%} até n = {power.n.iloc[-1]} ({100 * power.taxa_rejeicao.iloc[-1]:.0f}%)")
            parts.append(f"{test}: erro tipo I de {100 * null.taxa_rejeicao.iloc[0]:.1f}% (n = {null.n.iloc[0]}) "
                         f"a {100 * null.taxa_rejeicao.iloc[-1]:.1f}% (n = {null.n.iloc[-1]}); {n80}")
        out.append(f"**{shape}**: " + "; ".join(parts))
    return out


def real_effect_g4_scale(real):
    """d observado na escala do G4: diferença das médias do log, em desvios-padrão do log (combinado)."""
    a, b = (np.log(real.loc[real.grupo.eq(g), "horas_ate_merge"].to_numpy()) for g in ("com_bot", "sem_bot"))
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


def power_at(curve, d):
    """Taxa de rejeição de uma curva (colunas d e taxa_rejeicao) interpolada linearmente em |d|."""
    curve = curve.sort_values("d")
    return float(np.interp(abs(d), curve.d, curve.taxa_rejeicao))


def n_for_power(d, power=0.8, alpha=config.ALPHA):
    """n por grupo para o poder desejado pela aproximação normal do teste t no log (bicaudal)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z / d) ** 2))


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
