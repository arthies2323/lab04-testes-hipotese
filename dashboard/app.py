"""Dashboard Lab04 (tela única, sem rolagem). Rodar: streamlit run dashboard/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from src import config, plots, simulate, sprint01, sprint01_plots

st.set_page_config(page_title="Lab04 — Welch × Mann-Whitney", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- layout fixo (sem rolagem)
st.html("""
<style>
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] {display: none !important;}
html, body {background: #f3f4f6;}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {height: 100vh; overflow: hidden !important;}
.block-container {padding: .55rem 1.2rem 0 1.2rem !important; max-width: 100% !important;}
[data-testid="stVerticalBlock"] {gap: .45rem;}
.stApp {background: #f3f4f6;}
.topbar {display: flex; align-items: center; gap: .6rem;}
.topbar h1 {font-size: 1.15rem; margin: 0; padding: 0; color: #111827; white-space: nowrap;}
.topbar span {font-size: .78rem; color: #6b7280;}
.kpis {display: grid; grid-template-columns: repeat(6, 1fr); gap: .5rem;}
.kpi {background: #fff; border-radius: 8px; padding: .35rem .7rem; border-left: 4px solid #1f77b4;
      box-shadow: 0 1px 2px rgba(0,0,0,.06);}
.kpi .l {font-size: .7rem; color: #6b7280; text-transform: uppercase; letter-spacing: .03em;}
.kpi .v {font-size: 1.25rem; font-weight: 700; color: #111827; line-height: 1.25;}
.kpi .s {font-size: .7rem; color: #6b7280;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stTable"] {
    background: #fff; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,.06);}
.card-title {font-size: .8rem; font-weight: 600; color: #374151; line-height: 1.5; padding: .1rem 0 .15rem .1rem;}
.kpis {margin-bottom: .55rem;}
/* os gráficos ocupam a altura que sobra da janela (sem rolagem em qualquer tela) */
[class*="st-key-full-"], [class*="st-key-g1-"], [class*="st-key-q3-"], [class*="st-key-box-"] {flex: 0 0 auto !important;}
[class*="st-key-full-"] > div, [class*="st-key-g1-"] > div, [class*="st-key-q3-"] > div, [class*="st-key-box-"] > div {
    height: 100% !important;}
[class*="st-key-full-"] {height: calc(100vh - 215px) !important; min-height: 330px;}
[class*="st-key-g1-"]   {height: calc(100vh - 345px) !important; min-height: 260px;}
[class*="st-key-q3-"]   {height: calc(100vh - 500px) !important; min-height: 200px;}
[class*="st-key-box-"]  {height: calc(100vh - 345px) !important; min-height: 260px;}
</style>
""")


# ---------------------------------------------------------------- dados (cacheados)
@st.cache_data
def load_simulation_results(path, revision):
    path = Path(path)
    if path.exists():
        saved = pd.read_csv(path)
        if {"rejeicoes", "repeticoes", "alpha"}.issubset(saved.columns):
            return saved
    return simulate.run_all()


@st.cache_data
def load_sensitivity(path, revision):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else simulate.run_sample_sizes()


@st.cache_data
def load_real(revision):
    return simulate.load_real()


@st.cache_data
def checks_table(med, sig):
    return simulate.scenario_checks(med, sig, size=4_000)


def mtime(path):
    path = Path(path)
    return path.stat().st_mtime_ns if path.exists() else None


real = load_real(mtime(config.DATA_CSV))
cal = simulate.calibrate(real)
med, sig = simulate.pooled_params(real)
res = load_simulation_results(config.RESULTS_CSV, mtime(config.RESULTS_CSV))
sens = load_sensitivity(config.SENSITIVITY_CSV, mtime(config.SENSITIVITY_CSV))
piv = res.pivot_table(index=["situacao", "d"], columns="teste", values="taxa_rejeicao")
desc = sprint01.describe(real)
sample = sprint01.example(real)
tab = plots.real_table(real)
d_obs = plots.real_effect_g4_scale(real)
ref_curve = res[res.situacao.eq(simulate.REFERENCE)]
big = sens[sens.forma.eq(simulate.SHAPES["lognormal"]) & sens.n.eq(sens.n.max())]
pw = {t: plots.power_at(ref_curve[ref_curve.teste.eq(t)], d_obs) for t in config.COLORS}
pw_big = {t: plots.power_at(big[big.teste.eq(t)], d_obs) for t in config.COLORS}
n80 = plots.n_for_power(d_obs)

def fit(fig, **layout):
    """Só ajusta o enquadramento (altura livre e margens) para caber na tela; os dados do gráfico não mudam."""
    fig.update_layout(height=None, autosize=True, margin=layout.pop("margin", dict(t=55, b=40, l=55, r=20)),
                      paper_bgcolor="white", **layout)
    return fig


_slot = iter(range(1000))


def chart(fig, area="full"):
    """Gráfico que estica até a altura da área (definida no CSS em função da altura da janela)."""
    with st.container(key=f"{area}-{next(_slot)}"):
        st.plotly_chart(fig, width="stretch", height="stretch", config={"displaylogo": False, "responsive": True})


def title(text):
    st.html(f'<div class="card-title">{text}</div>')


# ---------------------------------------------------------------- cabeçalho + KPIs
head, links = st.columns([12, 1.4], vertical_alignment="center")
head.markdown('<div class="topbar"><h1>Testes de hipótese por dentro: Welch × Mann-Whitney</h1>'
              '<span>Tempo até o merge de PRs do dotnet/runtime · com × sem review do bot Copilot</span></div>',
              unsafe_allow_html=True)
with links.popover("ℹ Cenário", width="stretch"):
    st.markdown(f"""
**Pergunta:** PRs que recebem review do bot Copilot demoram mais ou menos para ser mergeados?

- **H₀:** a distribuição do tempo até o merge é a mesma com e sem bot (bicaudal).
- **H₁:** os tempos diferem (bicaudal, pois não sabemos se o bot acelera ou atrasa).

**Dados reais:** PRs mergeados do repositório público `dotnet/runtime`, coletados pelo `gh` CLI
(`src/collect.py`); "com bot" = review/comentário de `copilot-pull-request-reviewer`.
Amostra aleatória (semente 42) de **{cal['com_bot']['n']} PRs com bot e {cal['sem_bot']['n']} sem bot**
dentre os 600 mais recentes.

**Repositório / Projects:** https://github.com/arthies2323/lab04-testes-hipotese
""")

p_w, p_m = tab.iloc[0]["p-valor"], tab.iloc[1]["p-valor"]
kpis = [
    ("PRs com bot", f"{cal['com_bot']['n']}", f"mediana {desc.loc['com_bot', 'mediana']:.1f} h", "#2ca02c"),
    ("PRs sem bot", f"{cal['sem_bot']['n']}", f"mediana {desc.loc['sem_bot', 'mediana']:.1f} h", "#7f7f7f"),
    ("p-valor Welch", f"{p_w:.3f}", tab.iloc[0]["Conclusão"], config.COLORS["Welch"]),
    ("p-valor Mann-Whitney", f"{p_m:.3f}", tab.iloc[1]["Conclusão"], config.COLORS["Mann-Whitney"]),
    ("Efeito observado (d, log)", f"{d_obs:.2f}",
     f"poder n=30: {pw['Welch']:.0%} W · {pw['Mann-Whitney']:.0%} MW", "#6b7280"),
    ("n por grupo p/ 80% de poder", f"{n80}", f"α = {config.ALPHA:.0%}, bicaudal", "#d62728"),
]
st.markdown('<div class="kpis">' + "".join(
    f'<div class="kpi" style="border-left-color:{c}"><div class="l">{l}</div><div class="v">{v}</div>'
    f'<div class="s">{s}</div></div>' for l, v, s, c in kpis) + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- navegação entre páginas
PAGES = ["Dados", "Mecânica dos testes", "Erro tipo I", "Poder", "Q3 — outliers e n", "Dados reais e conclusão"]
nav, slicer, info = st.columns([7, 3, 1.4], vertical_alignment="center")
page = nav.segmented_control("Página", PAGES, default=PAGES[0], label_visibility="collapsed") or PAGES[0]
reading = info.popover("ℹ Leitura", width="stretch")

# ---------------------------------------------------------------- páginas
if page == "Dados":
    chart(fit(sprint01_plots.g1(real, sample)), "g1")
    title("Estatísticas descritivas dos dados reais")
    st.dataframe(desc.round({c: 3 for c in desc.columns if not c.startswith("shapiro")}).rename(columns=sprint01_plots.DESC_LABELS), width="stretch", height=110,
                 column_config={c: st.column_config.NumberColumn(c, format="%.2e")
                                for c in ("Shapiro-Wilk p (horas)", "Shapiro-Wilk p (log das horas)")})
    reading.markdown(sprint01.shape_text(desc) + f"""

Nas simulações Q1–Q3, H₀ verdadeira exige a mesma distribuição nos dois grupos: usamos a **lognormal comum com
mediana {med:.1f} h e σ = {sig:.2f}** e aplicamos o efeito como deslocamento no log, medido em desvios-padrão (d).
""")

elif page == "Mecânica dos testes":
    table, t, u, z = sprint01.analyze(sample)
    left, right = st.columns([2.3, 1])
    with left:
        chart(fit(sprint01_plots.g2(real, sample), margin=dict(t=55, b=45, l=55, r=10),
                  legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10))))
    with right:
        title("Teste aplicado ao conjunto simulado do G1")
        ex = (table.drop(columns=["interpretacao"]).set_index("teste")
              .rename(columns={"estatistica": "Estatística", "p_valor": "p-valor", "efeito": "Tamanho do efeito",
                               "medida": "Medida", "gl": "gl (Satterthwaite)", "decisao": "Decisão"}))
        ex = ex.T.map(lambda v: "—" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v))
        st.dataframe(ex, width="stretch", height=250)
        title("Calibração da lognormal por grupo")
        st.dataframe(pd.DataFrame(cal).T.rename(columns={"median": "Mediana (h)", "sigma": "σ do log", "n": "n"}).round(3),
                     width="stretch", height=110)
    crit_t, crit_z = stats.t.ppf(1 - config.ALPHA / 2, t.df), stats.norm.ppf(1 - config.ALPHA / 2)
    reading.markdown(f"""
**Welch** compara **médias** (estatística t, sem supor variâncias iguais); **Mann-Whitney** compara **postos**
(probabilidade de um valor de um grupo superar o do outro). Cada painel mostra a distribuição da estatística quando H₀ é
verdadeira; a região vermelha (|t| > {crit_t:.2f} com {t.df:.1f} gl de Satterthwaite, |z| > {crit_z:.2f} para U) é onde
rejeitamos H₀ com α = 5%, nas duas caudas por ser bicaudal. No conjunto simulado do G1, t = {t.statistic:.2f} com
**p = {t.pvalue:.3f}** e U = {u.statistic:.0f} (|z| = {z:.2f}) com **p = {u.pvalue:.3f}**: a área cinza é a probabilidade,
sob H₀, de uma estatística tão ou mais extrema que a observada. Não é a probabilidade de H₀ ser verdadeira.
""")

elif page == "Erro tipo I":
    left, right = st.columns([1.15, 1])
    with left:
        chart(fit(plots.g3(res)))
    with right:
        title("Situações simuladas (Q1–Q3)")
        st.dataframe(simulate.scenario_table(med, sig).rename(columns={
            "situacao": "Situação", "violacao": "Pressuposto do Welch violado",
            "alterado": "Parâmetro alterado (vs. referência)", "fixos": "Parâmetros fixos"}),
            hide_index=True, width="stretch", height=180)
        title("Conferência sob H₀ (4.000 réplicas)")
        checks = checks_table(med, sig)
        st.dataframe(checks.rename(columns={
            "situacao": "Situação", "n": "n por grupo", "assimetria_tipica": "Assimetria típica",
            "razao_medias_h0": "Média b ÷ média a (H₀)", "correlacao_grupos": "Correlação entre grupos"}).round(3),
            hide_index=True, width="stretch", height=180)
    null = plots.type1_summary(res)
    alpha = null.alpha.iloc[0]
    summary = []
    for test, rows in null.groupby("teste", sort=False):
        below = rows.loc[rows.ic_sup.lt(alpha), "situacao"].tolist()
        above = rows.loc[rows.ic_inf.gt(alpha), "situacao"].tolist()
        departures = [f"abaixo em {', '.join(below)}"] if below else []
        if above:
            departures.append(f"acima em {', '.join(above)}")
        summary.append(f"**{test}**: " + (" e ".join(departures) if departures else
                                        "todos os intervalos incluem o nível nominal"))
    ref_skew = checks.set_index("situacao").loc[simulate.REFERENCE, "assimetria_tipica"]
    reading.markdown(
        f"Cada situação muda um único aspecto da referência **{simulate.REFERENCE}**; em todas, a grade de d, "
        f"α = {config.ALPHA}, {f'{config.N_SIM:_}'.replace('_', '.')} réplicas e a semente {config.SEED} ficam fixas. "
        "A conferência gera 4.000 réplicas sob H₀: a contaminação atinge os dois grupos igualmente (razão das "
        f"médias entre {checks.razao_medias_h0.min():.2f} e {checks.razao_medias_h0.max():.2f}) e os grupos são "
        f"independentes (|correlação| ≤ {checks.correlacao_grupos.abs().max():.3f}). A assimetria típica de uma amostra "
        f"simulada da referência ({ref_skew:.2f}) é comparável à dos dados reais "
        f"({desc.assimetria.min():.2f} e {desc.assimetria.max():.2f}).\n\n"
        f"No G3, a linha em {alpha:.0%} é o nível nominal e as barras de erro mostram intervalos pontuais de Wilson "
        "de 95% da simulação. " + ". ".join(summary) + ". "
        "Os intervalos quantificam a variação entre réplicas do modelo simulado.")

elif page == "Poder":
    left, right = st.columns([1.8, 1.2])
    with left:
        chart(fit(plots.g4(res), margin=dict(t=60, b=40, l=55, r=20)))
    with right:
        title("Escala do efeito d")
        scale = simulate.effect_scale_table(med, sig).rename(columns={
            "d": "d", "normal_deslocamento_h": "Normal: deslocamento (h)",
            "lognormal_deslocamento_log": "Lognormal: deslocamento no log",
            "lognormal_razao_medianas": "Lognormal: razão das medianas"})
        st.dataframe(scale.round(2), hide_index=True, width="stretch", height=215, row_height=30)
        title("Poder em d = 0,8 por situação (%)")
        st.dataframe((100 * piv.xs(0.8, level="d")).round(1), width="stretch", height=155, row_height=30,
                     column_config={"situacao": st.column_config.TextColumn("Situação", width="large")})
    p = piv.xs(0.8, level="d")
    reading.markdown(
        "**Escala do efeito (Q2).** O poder é a taxa de rejeição de H₀ quando há efeito (d > 0); em d = 0 a mesma "
        "curva é o erro tipo I. O d é um deslocamento padronizado: na normal, o grupo b fica d desvios-padrão acima "
        "(em horas); na lognormal, o deslocamento é d·σ na escala log, ou seja, a mediana de b é exp(d·σ) vezes a de a."
        "\n\n**Leitura das curvas (derivada dos resultados):**\n"
        + "".join(f"\n- {_t}" for _t in plots.g4_reading(res)) + f"""

Com dados assimétricos o Mann-Whitney detecta muito mais: com d = 0,8 e n=30 ele rejeita
{p.loc['Lognormal assimétrica (n=30)','Mann-Whitney']*100:.0f}% das vezes contra
{p.loc['Lognormal assimétrica (n=30)','Welch']*100:.0f}% do Welch, e com outliers
{p.loc['Lognormal com outliers (n=30)','Mann-Whitney']*100:.0f}% contra {p.loc['Lognormal com outliers (n=30)','Welch']*100:.0f}%.
A vantagem **muda de lado** com dados normais, onde o Welch ganha por pouco
({p.loc['Normal, bem comportada (n=30)','Welch']*100:.0f}% contra {p.loc['Normal, bem comportada (n=30)','Mann-Whitney']*100:.0f}%).
Com n=8 os dois têm pouco poder, mas o Mann-Whitney segue à frente.
""")

elif page == "Q3 — outliers e n":
    d_ref = slicer.select_slider("Efeito d", options=config.EFFECTS[1:], value=0.6, label_visibility="collapsed",
                                 format_func=lambda d: f"d = {d:g}")
    chart(fit(plots.sample_size_figure(sens, d_ref), margin=dict(t=55, b=40, l=55, r=20),
              legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10))), "q3")
    title("Variação da taxa de rejeição vs. a referência (p.p.)")
    delta = plots.q3_comparison(res)
    delta.columns = [f"d = {d:g}" + (" (erro tipo I)" if d == 0 else "") for d in delta.columns]
    st.dataframe(delta.round(1), width="stretch", height=215, row_height=30)
    mu, sd = simulate.normal_moments(med, sig)
    reading.markdown(
        f"**Tabela:** cada célula é a taxa de rejeição da situação menos a da referência **{simulate.REFERENCE}**, "
        "em pontos percentuais, para o mesmo teste e o mesmo d. Valores negativos em d > 0 indicam perda de poder."
        + "".join(f"\n- {_t}" for _t in plots.q3_reading(res))
        + f"\n\n**Gráfico:** cada forma foi simulada com n = {', '.join(map(str, config.SAMPLE_SIZES))} PRs por grupo, "
        "a mesma grade de d e 10.000 réplicas por ponto. Tracejado: erro tipo I (d = 0); contínuo: poder no d escolhido."
        + "".join(f"\n- {_t}" for _t in plots.sample_size_reading(sens, d_ref)) + f"""

**Limites da comparação**
- **O mesmo d não é o mesmo efeito em horas.** Na lognormal, d desloca o log em d·σ (σ = {sig:.2f}); na normal, o
  deslocamento é de d desvios-padrão em horas. Com outliers, a contaminação aumenta a dispersão sem mudar o
  deslocamento, então o mesmo d vira um efeito padronizado menor.
- **A normal de controle não é realista:** com média {mu:.1f} h e DP {sd:.1f} h, {stats.norm.cdf(0, mu, sd):.0%} dos
  tempos gerados são negativos.
- **Uma só intensidade de contaminação** ({config.CONTAM_FRAC:.0%} × {config.CONTAM_MULT:g}, nos dois grupos) e grupos
  balanceados com o mesmo σ.
- **Mann-Whitney com n ≤ 8** usa p-valor exato (discreto) no SciPy e, acima disso, a aproximação normal.
- **Erro de Monte Carlo:** com 10.000 réplicas, ≈ ±0,4 p.p. perto de 5% e ≈ ±1 p.p. perto de 50%.
- **Independência:** as réplicas supõem PRs independentes.
""")

else:  # Dados reais e conclusão
    title("Testes aplicados aos dados reais")
    st.dataframe(tab, hide_index=True, width="stretch", height=110)
    left, right = st.columns([1.25, 1])
    with right:
        box = px.box(real, x="grupo", y="horas_ate_merge", color="grupo", points="all", log_y=True,
                     color_discrete_map=sprint01_plots.COLORS,
                     labels={"grupo": "Grupo", "horas_ate_merge": "Tempo até o merge (h, escala log)"},
                     title="Tempo até o merge por grupo (dados reais)")
        chart(fit(box, showlegend=False), "box")
    with left:
        title("Poder para o efeito observado")
        power = pd.DataFrame({
            "Teste": list(config.COLORS),
            "d observado (log)": [round(d_obs, 2)] * 2,
            "Poder com n = 30": [f"{pw[t]:.0%}" for t in config.COLORS],
            f"Poder com n = {big.n.max()}": [f"{pw_big[t]:.0%}" for t in config.COLORS],
            "n por grupo p/ 80%": [n80] * 2})
        st.dataframe(power, hide_index=True, width="stretch", height=110)
        title("Recomendação por cenário")
        st.dataframe(pd.DataFrame({
            "Cenário": ["Tempo até o merge (assimétrico, com outliers)", "Dados ~simétricos (ex.: log do tempo)",
                        "Interesse na diferença de médias"],
            "Teste recomendado": ["Mann-Whitney", "Welch", "Welch"],
            "Motivo": ["Welch fica conservador e perde poder", "Welch tem poder ligeiramente maior",
                       "Mann-Whitney não testa médias"]}), hide_index=True, width="stretch", height=145)
    med_com, med_sem = desc.loc["com_bot", "mediana"], desc.loc["sem_bot", "mediana"]
    decisions = set(tab["Conclusão"])
    agree = (f"Os dois testes chegam à mesma decisão (**{decisions.pop()}**)" if len(decisions) == 1
             else "Os testes chegam a decisões **diferentes**")
    reading.markdown(f"""
{agree} (Welch p = {p_w}, Mann-Whitney p = {p_m}). Na prática, a mediana com bot é
{med_com / med_sem:.2f} vezes a sem bot ({med_com - med_sem:+.1f} h), pouco diante da dispersão dos tempos. O d de Cohen
da tabela está em horas e não se lê no eixo do G4: na escala do G4 o efeito observado é **d = {d_obs:.2f}**. A aproximação
normal indica ≈ {n80} PRs por grupo para 80%: o resultado **não prova ausência de efeito**, só mostra que a amostra é
pequena demais para detectar um efeito desse tamanho.

**Esperado × encontrado:** esperávamos que o Mann-Whitney fosse melhor com dados assimétricos e com outliers, e que o Welch
fosse equivalente ou melhor com dados normais. A simulação confirmou isso, e mostrou também que o Welch se torna
**conservador** (erro tipo I abaixo de 5%) e perde poder, em vez de ficar liberal como se poderia temer.

**Limitações:** um único repositório, 30 PRs por grupo, o grupo "com bot" não é aleatório (confusão com tamanho e tipo
de PR) e a simulação supõe lognormal com mesmo σ nos grupos.
""")
