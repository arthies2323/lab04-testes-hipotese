# Lab04 — Testes de hipótese por dentro

Par escolhido: **P1 — t de Welch × Mann-Whitney** (tempo até o merge de PRs, com × sem bot de revisão).

## Como rodar
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -p no:asyncio
```
(O comando único do dashboard será documentado aqui: `streamlit run dashboard/app.py`.)

Semente fixa: `numpy.random.default_rng(42)`. Parâmetros em `src/config.py`.

## Links
- Repositório / Projects: <preencher>
- Dashboard: <preencher>
