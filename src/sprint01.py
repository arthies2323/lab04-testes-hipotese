"""LAB04S01: um exemplo reproduzível, sem executar Q1–Q3."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("com_bot", "sem_bot")


def load_real(path=None):
    df = pd.read_csv(path or ROOT / config.DATA_CSV)
    required = {"repo", "pr", "grupo", "horas_ate_merge"}
    if not required <= set(df.columns):
        raise ValueError(f"Colunas obrigatórias: {sorted(required)}")
    if set(df.grupo) != set(GROUPS):
        raise ValueError("São necessários os grupos com_bot e sem_bot.")
    values = pd.to_numeric(df.horas_ate_merge, errors="raise")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Os tempos devem ser finitos e positivos.")
    if df.duplicated(["repo", "pr"]).any() or (df.groupby("grupo").size() < 2).any():
        raise ValueError("PRs devem ser únicos e cada grupo deve ter pelo menos 2 observações.")
    return df.assign(horas_ate_merge=values)


def calibrate(real):
    rows = []
    for group in GROUPS:
        x = real.loc[real.grupo == group, "horas_ate_merge"].to_numpy()
        rows.append(dict(grupo=group, n=len(x), median=float(np.median(x)),
                         sigma=float(np.log(x).std(ddof=1))))
    return pd.DataFrame(rows).set_index("grupo")


def describe(real):
    """Forma da amostra real por grupo; é a evidência para simular com lognormal."""
    rows = []
    for group in GROUPS:
        x = real.loc[real.grupo == group, "horas_ate_merge"].to_numpy()
        q1, q3 = np.percentile(x, [25, 75])
        fence = 1.5 * (q3 - q1)
        shapiro = lambda v: float(stats.shapiro(v).pvalue) if len(v) >= 3 else np.nan
        rows.append(dict(grupo=group, n=len(x), media=float(x.mean()), mediana=float(np.median(x)),
                         dp=float(x.std(ddof=1)), assimetria=float(stats.skew(x, bias=False)),
                         minimo=float(x.min()), maximo=float(x.max()),
                         outliers=int(((x < q1 - fence) | (x > q3 + fence)).sum()),
                         shapiro_p_horas=shapiro(x), shapiro_p_log=shapiro(np.log(x))))
    return pd.DataFrame(rows).set_index("grupo")


def shape_text(desc):
    """Texto do G1: como são os dados reais e por que a simulação usa a lognormal."""
    a, b = (desc.loc[g] for g in GROUPS)
    fits = (desc.shapiro_p_horas < config.ALPHA).all() and (desc.shapiro_p_log >= config.ALPHA).all()
    text = (f"Média × mediana: {a.media:.1f} h × {a.mediana:.1f} h com bot e {b.media:.1f} h × {b.mediana:.1f} h "
            f"sem bot; assimetria de {a.assimetria:.2f} e {b.assimetria:.2f}, máximo de {desc.maximo.max():.1f} h e "
            f"{int(desc.outliers.sum())} outliers pela regra de 1,5·IQR. "
            f"O Shapiro-Wilk dá p = {a.shapiro_p_horas:.2g} e {b.shapiro_p_horas:.2g} para as horas e "
            f"p = {a.shapiro_p_log:.2g} e {b.shapiro_p_log:.2g} para o log das horas. ")
    if fits:
        return text + (f"Como a normalidade é rejeitada nas horas e não no log (α = {config.ALPHA}), cada grupo é "
                       "simulado como lognormal, com a mediana e o desvio dos logaritmos da calibração; com "
                       f"{int(desc.n.min())} PRs por grupo isso indica compatibilidade, não prova que o modelo é exato.")
    return text + "Esse padrão não sustenta claramente a lognormal, e a calibração deve ser revista."


def example(real, seed=config.SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for group, p in calibrate(real).iterrows():
        x = rng.lognormal(np.log(p['median']), p.sigma, int(p.n))
        rows.extend(dict(grupo=group, horas_ate_merge=float(v)) for v in x)
    return pd.DataFrame(rows)


def analyze(sample):
    a, b = [sample.loc[sample.grupo == g, "horas_ate_merge"].to_numpy() for g in GROUPS]
    t = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    # Fixar o método para que o p-valor corresponda à aproximação desenhada no G2.
    u = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True)
    n, m = len(a), len(b)
    _, counts = np.unique(np.r_[a, b], return_counts=True)
    tie = np.sum(counts**3 - counts) / ((n + m) * (n + m - 1))
    sd_u = np.sqrt(n * m / 12 * ((n + m + 1) - tie))
    z = max(0., (abs(float(u.statistic) - n * m / 2) - .5) / sd_u)
    pooled = np.sqrt(((n - 1) * a.var(ddof=1) + (m - 1) * b.var(ddof=1)) / (n + m - 2))
    rows = [dict(teste="Welch", estatistica=float(t.statistic), p_valor=float(t.pvalue),
                 efeito=float((a.mean() - b.mean()) / pooled), medida="d de Cohen", gl=float(t.df)),
            dict(teste="Mann-Whitney", estatistica=float(u.statistic), p_valor=float(u.pvalue),
                 efeito=float(2 * u.statistic / (n * m) - 1), medida="correlação bisserial por postos", gl=np.nan)]
    for row in rows:
        row['decisao'] = "rejeita H0" if row['p_valor'] < config.ALPHA else "não rejeita H0"
        target = "igualdade das médias" if row['teste'] == "Welch" else "igualdade das distribuições (postos)"
        row['interpretacao'] = (f"{row['teste']}: {row['decisao']} a α={config.ALPHA}; "
                                f"p={row['p_valor']:.6f}, estatística={row['estatistica']:.4f}. "
                                f"Hipótese avaliada: {target}. {row['medida']}={row['efeito']:.4f} "
                                "(sinal positivo: com_bot maior). Não rejeitar não comprova igualdade; "
                                "um conjunto simulado não mede poder nem demonstra efeito causal do bot.")
    return pd.DataFrame(rows), t, u, z


def export(output, real=None):
    real = load_real() if real is None else real
    sample = example(real)
    table, *_ = analyze(sample)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / "exemplo_simulado.csv", index=False)
    table.to_csv(output / "testes_exemplo.csv", index=False)
    calibrate(real).to_csv(output / "calibracao.csv")
    describe(real).to_csv(output / "descritivas.csv")
    from .sprint01_plots import g1, g2
    g1(real, sample).write_html(output / "G1.html", include_plotlyjs=True)
    g2(real, sample).write_html(output / "G2.html", include_plotlyjs=True)
    print('\n\n'.join(table.interpretacao))
    return sample, table


if __name__ == "__main__":
    # No Windows a saída redirecionada usa cp1252, que não codifica "α" das interpretações.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "sprint01")
    export(parser.parse_args().output)
