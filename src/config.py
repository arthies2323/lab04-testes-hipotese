"""Configuração central do Lab04 (par P1: t de Welch x Mann-Whitney)."""
SEED = 42
N_SIM = 10_000
ALPHA = 0.05
N_PER_GROUP = 30
# Parâmetros da lognormal serão calibrados com os dados reais (issue de coleta).
# Valores provisórios: mediana ~10 h, sigma = 1.0 (assimetria forte).
MEDIAN_H = 10.0
SIGMA = 1.0
