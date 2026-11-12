"""Configuração central do Lab04 (par P1: t de Welch x Mann-Whitney)."""
SEED = 42
N_SIM = 10_000
ALPHA = 0.05
N_PER_GROUP = 30
DATA_CSV = "data/prs.csv"
RESULTS_CSV = "data/results.csv"
EFFECTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # deslocamento padronizado (d)
CONTAM_FRAC, CONTAM_MULT = 0.10, 10.0       # outliers: 10% das obs x10
SMALL_N = 8
SAMPLE_SIZES = [8, 15, 30, 60]              # Q3: sensibilidade ao n por grupo
SENSITIVITY_CSV = "data/sensitivity.csv"
COLORS = {"Welch": "#1f77b4", "Mann-Whitney": "#d95f02"}
