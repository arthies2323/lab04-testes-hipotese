"""Geração de dados simulados e testes (P1: Welch x Mann-Whitney)."""
import numpy as np
from scipy import stats

from . import config


def make_rng(seed: int = config.SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def lognormal_group(rng, n, median, sigma):
    """Lognormal com a mediana indicada (mu = ln(mediana))."""
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=n)


def run_tests(a, b):
    """Retorna (p_welch, p_mannwhitney), ambos bicaudais."""
    p_w = stats.ttest_ind(a, b, equal_var=False).pvalue
    p_m = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    return p_w, p_m
