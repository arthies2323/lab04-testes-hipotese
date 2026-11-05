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
if not Path(config.RESULTS_CSV).exists():
    simulate.run_all().to_csv(config.RESULTS_CSV, index=False)
res = pd.read_csv(config.RESULTS_CSV)
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
st.plotly_chart(plots.g3(res), width="stretch")
r0 = piv.xs(0.0, level="d")
st.markdown(f"""
Com H₀ verdadeira o esperado é ≈ 5% (±0,4 p.p. com 10.000 simulações). O **Mann-Whitney ficou em 4,7–5,1%** em todas as
situações. O **Welch ficou conservador** nos dados assimétricos: {r0['Welch'].min()*100:.1f}% com n=8,
{r0.loc['Lognormal com outliers (n=30)','Welch']*100:.1f}% com outliers e
{r0.loc['Lognormal assimétrica (n=30)','Welch']*100:.1f}% só pela assimetria. Só com dados normais ele fica em
{r0.loc['Normal, bem comportada (n=30)','Welch']*100:.1f}%.
""")

st.header("5. Poder")
st.plotly_chart(plots.g4(res), width="stretch")
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

st.header("6. Dados reais")
tab = plots.real_table(real)
st.dataframe(tab, hide_index=True, width="stretch")
st.markdown(f"""
Os dois testes concordam: **não rejeitam H₀** (Welch p = {tab.iloc[0]['p-valor']}, Mann-Whitney p = {tab.iloc[1]['p-valor']}).
O efeito observado é pequeno (d ≈ 0,24; bisserial ≈ 0,08). Com n=30 por grupo o poder para um efeito desse tamanho é baixo
(≈ 7–11% no G4, d = 0,2), então o resultado **não prova ausência de efeito**: a amostra é pequena demais para detectá-lo.
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
