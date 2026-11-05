# Lab04 — Testes de hipótese por dentro

Par escolhido: **P1 — t de Welch × Mann-Whitney** (tempo até o merge de PRs do `dotnet/runtime`, com × sem review do bot Copilot).

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
python -m src.simulate    # simulações Q1-Q3 -> data/results.csv (~30 s)
python -m pytest -p no:asyncio
```

## Estrutura
- `src/collect.py` coleta dos dados reais · `src/simulate.py` simulação e testes · `src/plots.py` gráficos G1–G4 · `dashboard/app.py` dashboard
- `data/prs.csv` dados reais · `data/results.csv` taxas de rejeição simuladas

## Links
- Repositório / Projects: https://github.com/arthies2323/lab04-testes-hipotese
- Dashboard: <preencher após publicar>
