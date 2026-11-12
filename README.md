# Lab04 — Testes de hipótese por dentro

Par escolhido: **P1 — t de Welch × Mann-Whitney** (tempo até o merge de PRs do `dotnet/runtime`, com × sem review do bot Copilot).

## LAB04S01 — execução desta entrega (#3, #5, #7, #17)

Após instalar `pip install -r requirements.txt`, abrir a Sprint 01:

```bash
streamlit run dashboard/sprint01.py
```

Gerar o exemplo, calibração, resultados interpretados e gráficos HTML independentes:

```bash
python -m src.sprint01
```

Arquivos em `output/sprint01/`: `exemplo_simulado.csv`, `testes_exemplo.csv`,
`calibracao.csv`, `descritivas.csv`, `G1.html` e `G2.html`. Semente 42; o mesmo exemplo é usado
nos testes e nos gráficos. A distribuição nula ilustrativa do G2 usa semente 43.
A entrada S01 não lê `data/results.csv` nem executa Q1–Q3.
O cenário e a expectativa anterior à execução estão em `docs/cenario_sprint01.md`.
Teste: `python -m pytest -q`.

Foram selecionadas #3 (cenário), #5 (exemplo interpretado) e #7 (correção do G2,
reaberta pois usava gl=2n−2 no Welch e não mostrava Mann-Whitney).
A issue adicional #17 cobre a entrada isolada da S01 e a integração de G1/G2.
Os módulos e resultados já existentes das outras sprints foram preservados;
os comandos abaixo descrevem a versão anterior completa e não são necessários à S01.

A issue #20 justifica a lognormal com estatísticas descritivas dos dados reais
(assimetria, outliers e Shapiro-Wilk nas horas e no log das horas, em `descritivas.csv`
e na seção 2) e faz o dashboard completo (`dashboard/app.py`) usar o mesmo G1/G2 e o
mesmo exemplo da S01.

## LAB04S02 — Q1, Q2, Q3, G3 e G4 (#23–#28)

`python -m src.simulate` gera `data/results.csv` (Q1–Q3: 4 situações × 6 valores de d × 2 testes,
10.000 réplicas, semente 42, com as contagens de rejeição) e `data/sensitivity.csv` (Q3: normal,
lognormal e lognormal com outliers com n = 8, 15, 30 e 60 por grupo). No dashboard completo, a
seção 4 descreve e confere as situações simuladas e mostra o G3; a seção 5 mostra o G4, a comparação
de Q3 com a referência e a sensibilidade ao tamanho da amostra, com os limites da comparação.

## Entrega final — dashboard

`streamlit run dashboard/app.py` abre as 7 seções na ordem pedida. Se os CSVs de `data/` não existirem,
o dashboard os regenera com a mesma semente. A seção 6 lê o efeito real na escala do G4 (log, em σ)
para estimar o poder com a amostra coletada e o n necessário para 80% de poder.

## Como rodar (comando único após instalar)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```
Reproduzir tudo do zero (semente fixa 42, parâmetros em `src/config.py`):
```
python -m src.collect     # coleta data/prs.csv (requer gh autenticado)
python -m src.simulate    # Q1-Q3 -> data/results.csv e sensibilidade de Q3 ao n -> data/sensitivity.csv (~15 s)
python -m pytest -p no:asyncio
```

## Estrutura
- `src/collect.py` coleta dos dados reais · `src/simulate.py` simulação e testes · `src/sprint01_plots.py` G1 e G2 · `src/plots.py` G3, G4 e Q3 · `dashboard/app.py` dashboard completo · `dashboard/sprint01.py` só a S01
- `data/prs.csv` dados reais · `data/results.csv` taxas de rejeição simuladas · `data/sensitivity.csv` Q3 por forma e n por grupo (8, 15, 30, 60)

## Links
- Repositório / Projects: https://github.com/arthies2323/lab04-testes-hipotese
- Dashboard: <preencher após publicar>
