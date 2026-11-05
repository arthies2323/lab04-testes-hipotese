"""G1/G2 da S01; não calcula taxas de erro ou curvas de poder."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import config, sprint01

COLORS = {"com_bot": "#2ca02c", "sem_bot": "#7f7f7f"}
DESC_LABELS = {"media": "Média (h)", "mediana": "Mediana (h)", "dp": "Desvio-padrão (h)",
               "assimetria": "Assimetria", "minimo": "Mínimo (h)", "maximo": "Máximo (h)",
               "outliers": "Outliers (1,5·IQR)", "shapiro_p_horas": "Shapiro-Wilk p (horas)",
               "shapiro_p_log": "Shapiro-Wilk p (log das horas)"}


def g1(real, sample):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Dados reais", "Exemplo simulado (semente 42)"))
    xmax = max(real.horas_ate_merge.max(), sample.horas_ate_merge.max())
    for col, data in enumerate((real, sample), 1):
        for group in sprint01.GROUPS:
            x = data.loc[data.grupo == group, "horas_ate_merge"]
            fig.add_trace(go.Histogram(x=x, xbins=dict(start=0, end=xmax * 1.01, size=xmax / 20),
                          name=group, marker_color=COLORS[group], opacity=.65,
                          legendgroup=group, showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(title_text="Tempo até o merge (horas)", range=[0, xmax * 1.01])
    fig.update_yaxes(title_text="Número de PRs (contagem)")
    fig.update_layout(title="G1 — Dados reais e o mesmo conjunto simulado usado nos testes", barmode="overlay")
    return fig


def g2(real, sample, repetitions=4000):
    _, t, u, z = sprint01.analyze(sample)
    cal = sprint01.calibrate(real)
    # H0 comum: mesma distribuição nos dois grupos. Apenas ilustração da estatística nula.
    median = np.sqrt(cal.loc['com_bot', 'median'] * cal.loc['sem_bot', 'median'])
    sigma = np.sqrt(np.mean(cal.sigma**2))
    rng = np.random.default_rng(config.SEED + 1)
    n, m = [int(cal.loc[g, 'n']) for g in sprint01.GROUPS]
    a = rng.lognormal(np.log(median), sigma, (repetitions, n))
    b = rng.lognormal(np.log(median), sigma, (repetitions, m))
    null_t = stats.ttest_ind(a, b, equal_var=False, axis=-1).statistic
    null_u = stats.mannwhitneyu(a, b, axis=-1, method='asymptotic').statistic
    null_z = (null_u - n*m/2) / np.sqrt(n*m*(n+m+1)/12)
    fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Welch — gl de Satterthwaite = {t.df:.2f}",
                          f"Mann-Whitney — U = {u.statistic:.0f}, aproximação normal"), vertical_spacing=.2)
    for row, (name, dist, obs, null, p) in enumerate([
        ('Welch', stats.t(df=t.df), abs(float(t.statistic)), null_t, float(t.pvalue)),
        ('Mann-Whitney', stats.norm(), z, null_z, float(u.pvalue))], 1):
        crit = dist.ppf(1 - config.ALPHA / 2)
        bound = max(5., obs + 1., abs(null).max() + .2)
        x = np.sort(np.r_[np.linspace(-bound, bound, 2001), -obs, obs, -crit, crit])
        y = dist.pdf(x)
        fig.add_trace(go.Histogram(x=null, histnorm='probability density', opacity=.3,
                      name=f'{name}: estatística simulada sob H0', marker_color=config.COLORS[name]), row=row, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, name=f'{name}: referência aproximada',
                      line=dict(color=config.COLORS[name])), row=row, col=1)
        for cutoff, label, color in [(crit, 'Rejeição: α=5%', 'rgba(214,39,40,.35)'),
                                     (obs, f'Área bicaudal p={p:.6f}', 'rgba(80,80,80,.45)')]:
            for side in (-1, 1):
                mask = x <= -cutoff if side < 0 else x >= cutoff
                fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode='lines', line=dict(width=0),
                              fill='tozeroy', fillcolor=color, name=f'{name}: {label}', showlegend=side==1), row=row,col=1)
        fig.add_vline(x=obs, line_dash='dash', row=row, col=1)
        fig.add_vline(x=-obs, line_dash='dash', row=row, col=1)
        fig.update_xaxes(title_text='t (adimensional)' if row==1 else 'z de U (adimensional; correção de continuidade no exemplo)', row=row,col=1)
        fig.update_yaxes(title_text='Densidade',row=row,col=1)
    fig.update_layout(title='G2 — Estatísticas sob H0, rejeição e p-valor do exemplo', height=950,
                      barmode='overlay', legend=dict(orientation='h', y=-.17), margin=dict(b=210))
    return fig
