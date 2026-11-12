"""Geração de dados simulados e testes (P1: Welch x Mann-Whitney)."""
import numpy as np
import pandas as pd
from scipy import stats

from . import config


def make_rng(seed: int = config.SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def load_real():
    return pd.read_csv(config.DATA_CSV)


def calibrate(df=None):
    """Mediana (h) e sigma (desvio dos logs) por grupo, a partir dos dados reais."""
    df = load_real() if df is None else df
    out = {}
    for g, s in df.groupby("grupo").horas_ate_merge:
        out[g] = dict(median=float(s.median()), sigma=float(np.log(s).std(ddof=1)), n=len(s))
    return out


def pooled_params(df=None):
    c = calibrate(df)
    med = float(np.sqrt(c["com_bot"]["median"] * c["sem_bot"]["median"]))
    sig = float(np.sqrt((c["com_bot"]["sigma"] ** 2 + c["sem_bot"]["sigma"] ** 2) / 2))
    return med, sig


# Q3: cada situação altera um único aspecto da referência calibrada (ver scenario_table).
REFERENCE = "Lognormal assimétrica (n=30)"
SITUATIONS = {
    REFERENCE: dict(
        kind="lognormal", n=config.N_PER_GROUP, violacao="normalidade: assimetria à direita",
        alterado="nenhum (referência calibrada nos dados reais)"),
    "Normal, bem comportada (n=30)": dict(
        kind="normal", n=config.N_PER_GROUP, violacao="nenhum (controle)",
        alterado="forma: normal com a mesma média e DP da lognormal"),
    "Lognormal com outliers (n=30)": dict(
        kind="outliers", n=config.N_PER_GROUP, violacao="normalidade: assimetria e outliers",
        alterado=f"{config.CONTAM_FRAC:.0%} das observações de cada grupo × {config.CONTAM_MULT:g}"),
    f"Lognormal, amostra pequena (n={config.SMALL_N})": dict(
        kind="lognormal", n=config.SMALL_N, violacao="normalidade com n pequeno (o TCL não compensa)",
        alterado=f"n: {config.N_PER_GROUP} → {config.SMALL_N} por grupo"),
}


def normal_moments(median, sigma):
    """Média e DP (h) da lognormal; a normal de controle usa os mesmos valores."""
    mu = median * np.exp(sigma**2 / 2)
    return mu, mu * np.sqrt(np.exp(sigma**2) - 1)


def generate(rng, kind, n, d, size, median, sigma):
    """Retorna (a, b) com shape (size, n). d = deslocamento padronizado do grupo b."""
    if kind == "normal":
        # normal com média/dp iguais à lognormal calibrada (mesmo "tamanho de efeito" d)
        mu, sd = normal_moments(median, sigma)
        a = rng.normal(mu, sd, (size, n))
        b = rng.normal(mu + d * sd, sd, (size, n))
        return a, b
    a = rng.lognormal(np.log(median), sigma, (size, n))
    b = rng.lognormal(np.log(median) + d * sigma, sigma, (size, n))
    if kind == "outliers":
        for x in (a, b):
            m = rng.random(x.shape) < config.CONTAM_FRAC
            x[m] *= config.CONTAM_MULT
    return a, b


def effect_scale_table(median, sigma, effects=None):
    """Traduz d para as escalas originais (documenta Q2).

    Normal: o grupo b é deslocado d*dp horas, com dp o desvio-padrão da normal equivalente.
    Lognormal: o deslocamento é d*sigma na escala log, ou seja, a mediana de b é
    exp(d*sigma) vezes a de a. Poder = taxa de rejeição quando d > 0 (H1 verdadeira)."""
    effects = config.EFFECTS if effects is None else effects
    _, sd = normal_moments(median, sigma)
    return pd.DataFrame(dict(
        d=effects,
        normal_deslocamento_h=[d * sd for d in effects],
        lognormal_deslocamento_log=[d * sigma for d in effects],
        lognormal_razao_medianas=[float(np.exp(d * sigma)) for d in effects],
    ))


def scenario_table(median, sigma):
    """Q3: por situação, o pressuposto do Welch violado, o parâmetro alterado e os que ficam fixos."""
    mu, sd = normal_moments(median, sigma)
    rows = []
    for sit, spec in SITUATIONS.items():
        if spec["kind"] == "normal":
            fixed = [f"n = {spec['n']}", f"média {mu:.1f} h e DP {sd:.1f} h, iguais às da lognormal"]
        else:
            fixed = [f"mediana {median:.1f} h e σ do log {sigma:.2f}"]
            fixed += [f"n = {spec['n']}"] if spec["n"] == config.N_PER_GROUP else []
            fixed += ["sem contaminação"] if spec["kind"] != "outliers" else []
        rows.append(dict(situacao=sit, violacao=spec["violacao"], alterado=spec["alterado"],
                         fixos="; ".join(fixed)))
    return pd.DataFrame(rows)


def scenario_checks(median, sigma, size=4_000, seed=config.SEED):
    """Confere sob H0 o que cada situação gera: n, assimetria típica de uma amostra,
    simetria entre os grupos (razão das médias ≈ 1) e independência (correlação ≈ 0)."""
    rng = make_rng(seed)
    rows = []
    for sit, spec in SITUATIONS.items():
        a, b = generate(rng, spec["kind"], spec["n"], 0.0, size, median, sigma)
        rows.append(dict(situacao=sit, n=a.shape[1],
                         assimetria_tipica=float(np.median(stats.skew(np.r_[a, b], axis=1, bias=False))),
                         razao_medias_h0=float(b.mean() / a.mean()),
                         correlacao_grupos=float(np.corrcoef(a.mean(axis=1), b.mean(axis=1))[0, 1])))
    return pd.DataFrame(rows)


def run_tests(a, b):
    """p-valores bicaudais (Welch, Mann-Whitney); aceita 1D ou 2D (axis=-1)."""
    p_w = stats.ttest_ind(a, b, equal_var=False, axis=-1).pvalue
    p_m = stats.mannwhitneyu(a, b, alternative="two-sided", axis=-1).pvalue
    return np.asarray(p_w), np.asarray(p_m)


def rejection_counts(rng, kind, n, d, size, median, sigma, alpha=config.ALPHA):
    """Conta rejeições dos dois testes no mesmo conjunto de réplicas."""
    a, b = generate(rng, kind, n, d, size, median, sigma)
    pw, pm = run_tests(a, b)
    return int(np.count_nonzero(pw < alpha)), int(np.count_nonzero(pm < alpha))


def rejection_rates(rng, kind, n, d, size, median, sigma, alpha=config.ALPHA):
    counts = rejection_counts(rng, kind, n, d, size, median, sigma, alpha)
    return tuple(count / size for count in counts)


def run_all(size=config.N_SIM, seed=config.SEED):
    if size < 1:
        raise ValueError("O número de repetições deve ser positivo.")
    rng = make_rng(seed)
    med, sig = pooled_params()
    rows = []
    for sit, spec in SITUATIONS.items():
        for d in config.EFFECTS:
            counts = rejection_counts(rng, spec["kind"], spec["n"], d, size, med, sig)
            for test, count in zip(config.COLORS, counts):
                rows.append(dict(situacao=sit, d=d, teste=test, taxa_rejeicao=count / size,
                                 rejeicoes=count, repeticoes=size, alpha=config.ALPHA, semente=seed))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_all()
    df.to_csv(config.RESULTS_CSV, index=False)
    print(df.pivot_table(index=["situacao", "d"], columns="teste", values="taxa_rejeicao").round(3))
