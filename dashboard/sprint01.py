"""Entrada exclusiva LAB04S01: streamlit run dashboard/sprint01.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
from src import sprint01, sprint01_plots

st.set_page_config(page_title='LAB04S01 — Welch × Mann-Whitney', layout='wide')
st.title('LAB04S01 — Testes de hipótese por dentro')
st.markdown('[Repositório](https://github.com/arthies2323/lab04-testes-hipotese) · '
            '[Issues da S01](https://github.com/arthies2323/lab04-testes-hipotese/milestone/1) · '
            'Dashboard local: `streamlit run dashboard/sprint01.py`')
try:
    real = sprint01.load_real()
except (ValueError, OSError) as exc:
    st.error(f'Não foi possível carregar a amostra real: {exc}')
    st.stop()
cal = sprint01.calibrate(real)
sample = sprint01.example(real)
table, t, u, z = sprint01.analyze(sample)
st.header('1. Cenário')
st.markdown((ROOT / 'docs' / 'cenario_sprint01.md').read_text(encoding='utf-8'))
st.dataframe(cal.rename(columns={'median':'Mediana (h)', 'sigma':'Desvio dos logaritmos'}))
st.header('2. Os dados — G1')
st.plotly_chart(sprint01_plots.g1(real, sample), width='stretch')
st.markdown('A simulação usa uma lognormal por grupo, com a mediana e o desvio dos logaritmos '
            'calibrados na amostra real acima. G1 apresenta todas as observações, sem cortar ou '
            'acumular artificialmente a cauda; os mesmos dados simulados alimentam os dois testes. '
            'As diferenças visuais entre amostras pequenas são esperadas e não demonstram causalidade.')
st.download_button('Baixar conjunto simulado', sample.to_csv(index=False), 'exemplo_simulado.csv', 'text/csv')
st.header('3. Como os testes funcionam — G2')
@st.cache_data
def null_figure(real, sample):
    return sprint01_plots.g2(real, sample)
st.plotly_chart(null_figure(real, sample), width='stretch')
st.markdown('Os histogramas mostram estatísticas geradas sob uma distribuição comum aos grupos, '
            'enquanto as curvas são referências aproximadas: t com os graus de liberdade de Welch '
            'do exemplo e normal para U padronizado. Vermelho indica a rejeição a 5%; cinza indica '
            'as duas caudas tão extremas quanto o exemplo, cuja área é o p-valor reportado. '
            'O p-valor supõe H0 e não representa a probabilidade de H0 ser verdadeira; '
            'a referência de Welch pode diferir do histograma com forte assimetria.')
st.subheader('Aplicação a um conjunto simulado e interpretação')
st.dataframe(table.drop(columns=['interpretacao']), hide_index=True)
for interpretation in table.interpretacao:
    st.write(interpretation)
st.caption('Welch compara médias; Mann-Whitney compara distribuições por postos. '
           'Interpretar Mann-Whitney como diferença de medianas exige formas semelhantes. '
           'Os tamanhos de efeito têm escalas diferentes e descrevem o exemplo, não o efeito real do bot.')
st.download_button('Baixar resultados do exemplo', table.to_csv(index=False), 'testes_exemplo.csv', 'text/csv')
