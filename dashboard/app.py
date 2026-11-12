"""Dashboard Lab04. Rodar: streamlit run dashboard/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import pandas as pd
import streamlit as st
from scipy import stats

from src import config, plots, simulate, sprint01, sprint01_plots

st.set_page_config(page_title="Lab04 — Testes de hipótese", layout="wide")
real = simulate.load_real()
cal = simulate.calibrate(real)
med, sig = simulate.pooled_params(real)
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


results_path = Path(config.RESULTS_CSV)
revision = results_path.stat().st_mtime_ns if results_path.exists() else None
res = load_simulation_results(str(results_path), revision)
sens_path = Path(config.SENSITIVITY_CSV)
sens = load_sensitivity(str(sens_path), sens_path.stat().st_mtime_ns if sens_path.exists() else None)
piv = res.pivot_table(index=["situacao", "d"], columns="teste", values="taxa_rejeicao")

st.title("Testes de hipótese por dentro: Welch × Mann-Whitney")
st.markdown("**Repositório / Projects:** https://github.com/arthies2323/lab04-testes-hipotese  ·  "
            "**Dashboard:** `streamlit run dashboard/app.py`")

st.header("1. Cenário")
st.markdown(f"""
**Pergunta:** PRs que recebem review do bot Copilot demoram mais ou menos para ser mergeados?

- **H₀:** a distribuição do tempo até o merge é a mesma com e sem bot (bicaudal).
- **H₁:** os tempos diferem (bicaudal, pois não sabemos se o bot acelera ou atrasa).

**Dados reais:** PRs mergeados do repositório público `dotnet/runtime`, coletados pelo `gh` CLI
(`src/collect.py`); "com bot" = review/comentário de `copilot-pull-request-reviewer`.
Amostra aleatória (semente 42) de **{cal['com_bot']['n']} PRs com bot e {cal['sem_bot']['n']} sem bot** dentre os 600 mais recentes.
""")

st.header("2. Os dados")
desc = sprint01.describe(real)
sample = sprint01.example(real)
st.dataframe(desc.rename(columns=sprint01_plots.DESC_LABELS), width="stretch")
st.plotly_chart(sprint01_plots.g1(real, sample), width="stretch")
st.markdown(sprint01.shape_text(desc) + f"""
Nas simulações Q1–Q3, H₀ verdadeira exige a mesma distribuição nos dois grupos: usamos a **lognormal comum com
mediana {med:.1f} h e σ = {sig:.2f}** e aplicamos o efeito como deslocamento no log, medido em desvios-padrão (d).
""")

st.header("3. Como os testes funcionam")
_, t, u, z = sprint01.analyze(sample)
st.plotly_chart(sprint01_plots.g2(real, sample), width="stretch")
crit_t, crit_z = stats.t.ppf(1 - config.ALPHA / 2, t.df), stats.norm.ppf(1 - config.ALPHA / 2)
st.markdown(f"""
**Welch** compara **médias** (estatística t, sem supor variâncias iguais); **Mann-Whitney** compara **postos**
(probabilidade de um valor de um grupo superar o do outro). Cada painel mostra a distribuição da estatística quando H₀ é
verdadeira; a região vermelha (|t| > {crit_t:.2f} com {t.df:.1f} gl de Satterthwaite, |z| > {crit_z:.2f} para U) é onde
rejeitamos H₀ com α = 5%, nas duas caudas por ser bicaudal. No conjunto simulado do G1, t = {t.statistic:.2f} com
**p = {t.pvalue:.3f}** e U = {u.statistic:.0f} (|z| = {z:.2f}) com **p = {u.pvalue:.3f}**: a área cinza é a probabilidade,
sob H₀, de uma estatística tão ou mais extrema que a observada. Não é a probabilidade de H₀ ser verdadeira.
""")

st.header("4. Erro tipo I")
st.subheader("Situações simuladas (Q1–Q3)")
st.table(simulate.scenario_table(med, sig).rename(columns={
    "situacao": "Situação", "violacao": "Pressuposto do Welch violado",
    "alterado": "Parâmetro alterado (vs. referência)", "fixos": "Parâmetros fixos"}).set_index("Situação"))
checks = simulate.scenario_checks(med, sig, size=4_000)
ref_skew = checks.set_index("situacao").loc[simulate.REFERENCE, "assimetria_tipica"]
st.dataframe(checks.rename(columns={
    "situacao": "Situação", "n": "n por grupo", "assimetria_tipica": "Assimetria típica de uma amostra",
    "razao_medias_h0": "Média b ÷ média a (H₀)", "correlacao_grupos": "Correlação entre grupos"}).round(3),
    hide_index=True, width="stretch")
st.markdown(
    f"Cada situação muda um único aspecto da referência **{simulate.REFERENCE}**; em todas, a grade de d, "
    f"α = {config.ALPHA}, {f'{config.N_SIM:_}'.replace('_', '.')} réplicas e a semente {config.SEED} ficam fixas. "
    "A conferência acima gera 4.000 réplicas sob H₀: a contaminação atinge os dois grupos igualmente (razão das "
    f"médias entre {checks.razao_medias_h0.min():.2f} e {checks.razao_medias_h0.max():.2f}) e os grupos são "
    f"independentes (|correlação| ≤ {checks.correlacao_grupos.abs().max():.3f}). A assimetria típica de uma amostra "
    f"simulada da referência ({ref_skew:.2f}) é comparável à dos dados reais "
    f"({desc.assimetria.min():.2f} e {desc.assimetria.max():.2f}).")
st.subheader("G3 — Erro tipo I por situação")
st.plotly_chart(plots.g3(res), width="stretch")
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
st.markdown(f"Sob H₀, a linha em {alpha:.0%} é o nível nominal e as barras de erro "
            "mostram intervalos pontuais de Wilson de 95% da simulação. "
            + ". ".join(summary) + ". "
            "Os intervalos quantificam a variação entre réplicas do modelo simulado.")

st.header("5. Poder")
st.plotly_chart(plots.g4(res), width="stretch")
st.markdown("**Escala do efeito (Q2).** O poder é a taxa de rejeição de H₀ quando há efeito (d > 0); em d = 0 a mesma "
            "curva é o erro tipo I. O d é um deslocamento padronizado: na normal, o grupo b fica d desvios-padrão acima "
            "(em horas); na lognormal, o deslocamento é d·σ na escala log, ou seja, a mediana de b é exp(d·σ) vezes a de a.")
_scale = simulate.effect_scale_table(med, sig).rename(columns={
    "d": "d", "normal_deslocamento_h": "Normal: deslocamento (h)",
    "lognormal_deslocamento_log": "Lognormal: deslocamento no log",
    "lognormal_razao_medianas": "Lognormal: razão das medianas"})
st.dataframe(_scale.round(2), hide_index=True, width="stretch")
st.markdown("**Leitura das curvas (derivada dos resultados):**")
for _t in plots.g4_reading(res):
    st.markdown(f"- {_t}")
p = piv.xs(0.8, level="d")
st.markdown(f"""
Com dados assimétricos o Mann-Whitney detecta muito mais: com d = 0,8 e n=30 ele rejeita
{p.loc['Lognormal assimétrica (n=30)','Mann-Whitney']*100:.0f}% das vezes contra
{p.loc['Lognormal assimétrica (n=30)','Welch']*100:.0f}% do Welch, e com outliers
{p.loc['Lognormal com outliers (n=30)','Mann-Whitney']*100:.0f}% contra {p.loc['Lognormal com outliers (n=30)','Welch']*100:.0f}%.
A vantagem **muda de lado** com dados normais, onde o Welch ganha por pouco
({p.loc['Normal, bem comportada (n=30)','Welch']*100:.0f}% contra {p.loc['Normal, bem comportada (n=30)','Mann-Whitney']*100:.0f}%).
Com n=8 os dois têm pouco poder, mas o Mann-Whitney segue à frente.
""")

st.subheader("Q3: efeito de outliers e de n pequeno (mesma grade de d)")
delta = plots.q3_comparison(res)
delta.columns = [f"d = {d:g}" + (" (erro tipo I)" if d == 0 else "") for d in delta.columns]
st.dataframe(delta.round(1), width="stretch")
st.markdown(f"Cada célula é a taxa de rejeição da situação menos a da referência **{simulate.REFERENCE}**, em pontos "
            "percentuais, para o mesmo teste e o mesmo d. Valores negativos em d > 0 indicam perda de poder.")
for _t in plots.q3_reading(res):
    st.markdown(f"- {_t}")

st.subheader("Q3: sensibilidade ao tamanho da amostra")
d_ref = st.select_slider("Efeito usado nas curvas de poder (d)", options=config.EFFECTS[1:], value=0.6)
st.plotly_chart(plots.sample_size_figure(sens, d_ref), width="stretch")
st.markdown(f"Cada forma foi simulada com n = {', '.join(map(str, config.SAMPLE_SIZES))} PRs por grupo, a mesma grade "
            "de d e 10.000 réplicas por ponto. As linhas tracejadas são o erro tipo I (d = 0) e as contínuas, o poder no "
            "d escolhido acima.")
for _t in plots.sample_size_reading(sens, d_ref):
    st.markdown(f"- {_t}")
with st.expander("Limites da comparação de Q3"):
    mu, sd = simulate.normal_moments(med, sig)
    st.markdown(f"""
- **O mesmo d não é o mesmo efeito em horas.** Na lognormal, d desloca o log em d·σ (σ = {sig:.2f}), ou seja, a
  mediana do grupo b fica exp(d·σ) vezes a do grupo a; na normal, o deslocamento é de d desvios-padrão em horas. Com
  outliers, a contaminação aumenta a dispersão sem mudar o deslocamento, então o mesmo d vira um efeito padronizado menor.
- **A normal de controle não é realista:** com média {mu:.1f} h e DP {sd:.1f} h, {stats.norm.cdf(0, mu, sd):.0%} dos
  tempos gerados são negativos. Ela isola o efeito da forma, mas não descreve PRs reais.
- **Uma só intensidade de contaminação** ({config.CONTAM_FRAC:.0%} × {config.CONTAM_MULT:g}, nos dois grupos) e grupos
  balanceados com o mesmo σ: não foram simulados outliers em um só grupo, variâncias diferentes nem n desiguais.
- **Mann-Whitney com n ≤ 8** usa p-valor exato (discreto) no SciPy e, acima disso, a aproximação normal: a comparação
  entre tamanhos de amostra mistura o efeito do n e o do método.
- **Erro de Monte Carlo:** com 10.000 réplicas, a margem de 95% é de ≈ ±0,4 p.p. perto de 5% e ≈ ±1 p.p. perto de 50%.
  A grade de n usa outro fluxo aleatório, então os pontos em comum com G3/G4 (ex.: lognormal com n = 30) diferem
  dentro dessa margem.
- **Independência:** as réplicas supõem PRs independentes; dependência entre PRs do mesmo autor não foi simulada.
""")

st.header("6. Dados reais")
tab = plots.real_table(real)
st.dataframe(tab, hide_index=True, width="stretch")
d_obs = plots.real_effect_g4_scale(real)
ref_curve = res[res.situacao.eq(simulate.REFERENCE)]
big = sens[sens.forma.eq(simulate.SHAPES["lognormal"]) & sens.n.eq(sens.n.max())]
pw = {t: plots.power_at(ref_curve[ref_curve.teste.eq(t)], d_obs) for t in config.COLORS}
pw_big = {t: plots.power_at(big[big.teste.eq(t)], d_obs) for t in config.COLORS}
decisions = set(tab["Conclusão"])
agree = (f"Os dois testes chegam à mesma decisão (**{decisions.pop()}**)" if len(decisions) == 1
         else "Os testes chegam a decisões **diferentes**")
med_com, med_sem = desc.loc["com_bot", "mediana"], desc.loc["sem_bot", "mediana"]
st.markdown(f"""
{agree} (Welch p = {tab.iloc[0]['p-valor']}, Mann-Whitney p = {tab.iloc[1]['p-valor']}). Na prática, a mediana com bot é
{med_com / med_sem:.2f} vezes a sem bot ({med_com - med_sem:+.1f} h), pouco diante da dispersão dos tempos (DP de
{desc.dp.min():.0f} a {desc.dp.max():.0f} h). O d de Cohen da tabela está em horas e não se lê no eixo do G4: na escala do G4
(diferença das médias do log, em desvios-padrão do log) o efeito observado é **d = {d_obs:.2f}**, para o qual a curva da
referência com n = 30 dá poder de ≈ {pw['Welch']:.0%} (Welch) e ≈ {pw['Mann-Whitney']:.0%} (Mann-Whitney). Mesmo com
{big.n.max()} PRs por grupo o poder ficaria em ≈ {pw_big['Welch']:.0%} e ≈ {pw_big['Mann-Whitney']:.0%}, e a aproximação
normal indica ≈ {plots.n_for_power(d_obs)} PRs por grupo para 80%: o resultado **não prova ausência de efeito**, só mostra que
a amostra é pequena demais para detectar um efeito desse tamanho.
""")

st.header("7. Conclusão")
st.markdown("""
**Esperado × encontrado:** esperávamos que o Mann-Whitney fosse melhor com dados assimétricos e com outliers, e que o Welch
fosse equivalente ou melhor com dados normais. A simulação confirmou isso, e mostrou também que o Welch se torna
**conservador** (erro tipo I abaixo de 5%) e perde poder, em vez de ficar liberal como se poderia temer.

**Recomendação:** para tempo até o merge de PRs, que é assimétrico e tem outliers, usar o **Mann-Whitney**. A escolha mudaria
para o Welch se os dados fossem aproximadamente simétricos (ex.: log do tempo), ou se o interesse fosse especificamente a
diferença de **médias**, que o Mann-Whitney não testa.

**Limitações:** um único repositório, 30 PRs por grupo, o grupo "com bot" não é aleatório (confusão com tamanho e tipo
de PR) e a simulação supõe lognormal com mesmo σ nos grupos.
""")
