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
- `src/collect.py` coleta dos dados reais · `src/simulate.py` simulação e testes · `src/plots.py` gráficos G1–G4 · `dashboard/app.py` dashboard
- `data/prs.csv` dados reais · `data/results.csv` taxas de rejeição simuladas · `data/sensitivity.csv` Q3 por forma e n por grupo (8, 15, 30, 60)

## Links
- Repositório / Projects: https://github.com/arthies2323/lab04-testes-hipotese
- Dashboard: <preencher após publicar>
