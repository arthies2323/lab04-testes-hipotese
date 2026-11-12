import pytest
import numpy as np

from src import simulate


def test_lognormal_median_close():
    rng = simulate.make_rng(1)
    a, _ = simulate.generate(rng, "lognormal", 100_000, 0.0, 1, 10.0, 1.0)
    assert abs(float(np.median(a)) - 10.0) < 0.3


def test_run_tests_returns_valid_pvalues():
    rng = simulate.make_rng(1)
    a, b = simulate.generate(rng, "lognormal", 30, 0.0, 5, 10.0, 1.0)
    pw, pm = simulate.run_tests(a, b)
    assert pw.shape == (5,) and ((0 <= pw) & (pw <= 1)).all() and ((0 <= pm) & (pm <= 1)).all()


def test_type1_error_mannwhitney_near_5pct():
    rng = simulate.make_rng(3)
    _, mw = simulate.rejection_rates(rng, "lognormal", 30, 0.0, 2000, 10.0, 1.0)
    assert 0.03 < mw < 0.07


def test_run_all_records_reproducible_rejection_counts():
    from pandas.testing import assert_frame_equal
    from src import config

    results = simulate.run_all(size=100, seed=42)
    assert_frame_equal(results, simulate.run_all(size=100, seed=42))
    null = results[results.d == 0]
    assert len(null) == 2 * len(simulate.SITUATIONS)
    assert not null.duplicated(["situacao", "teste"]).any()
    assert (results.repeticoes == 100).all()
    assert (results.semente == 42).all()
    assert (results.alpha == config.ALPHA).all()
    assert results.rejeicoes.between(0, 100).all()
    np.testing.assert_allclose(results.taxa_rejeicao, results.rejeicoes / results.repeticoes)


def test_effect_grid_starts_at_zero_and_increases():
    t = simulate.effect_scale_table(27.0, 1.24)
    assert len(t) >= 5 and t.d.iloc[0] == 0 and t.d.is_monotonic_increasing
    assert t.lognormal_razao_medianas.iloc[0] == 1.0
    assert t.normal_deslocamento_h.is_monotonic_increasing


def test_g4_reading_covers_all_situations():
    import pandas as pd
    from src import plots
    res = pd.read_csv("data/results.csv")
    assert len(plots.g4_reading(res)) == res.situacao.nunique()


def test_q3_situations_alter_one_aspect_of_reference():
    from src import config
    ref = simulate.SITUATIONS[simulate.REFERENCE]
    assert ref == simulate.SITUATIONS[next(iter(simulate.SITUATIONS))]
    for sit, spec in simulate.SITUATIONS.items():
        changed = {k for k in ("kind", "n") if spec[k] != ref[k]}
        assert len(changed) <= 1, sit
        a, b = simulate.generate(simulate.make_rng(0), spec["kind"], spec["n"], 0.0, 3, 27.0, 1.24)
        assert a.shape == b.shape == (3, spec["n"])
    small = simulate.SITUATIONS[f"Lognormal, amostra pequena (n={config.SMALL_N})"]
    assert small["kind"] == ref["kind"] and small["n"] == config.SMALL_N < config.N_PER_GROUP
    table = simulate.scenario_table(27.0, 1.24)
    assert list(table.situacao) == list(simulate.SITUATIONS)
    assert table[["violacao", "alterado", "fixos"]].ne("").all().all()


def test_contamination_is_symmetric_and_independent_under_h0():
    from src import config
    # Mesma semente: os valores-base coincidem e a razão revela quais observações foram multiplicadas.
    base = simulate.generate(simulate.make_rng(5), "lognormal", 30, 0.0, 4000, 27.0, 1.24)
    contaminated = simulate.generate(simulate.make_rng(5), "outliers", 30, 0.0, 4000, 27.0, 1.24)
    masks = []
    for x, c in zip(base, contaminated):
        mask = np.isclose(c / x, config.CONTAM_MULT)
        np.testing.assert_allclose(c[~mask], x[~mask])
        assert abs(mask.mean() - config.CONTAM_FRAC) < 0.005
        masks.append(mask.ravel())
    assert abs(np.corrcoef(*masks)[0, 1]) < 0.015


def test_scenario_checks_skewness_symmetry_and_independence():
    checks = simulate.scenario_checks(27.0, 1.24, size=4000, seed=1).set_index("situacao")
    assert list(checks.n) == [spec["n"] for spec in simulate.SITUATIONS.values()]
    np.testing.assert_allclose(checks.razao_medias_h0, 1, atol=0.08)
    assert (checks.correlacao_grupos.abs() < 0.05).all()
    skew = checks.assimetria_tipica
    assert abs(skew["Normal, bem comportada (n=30)"]) < 0.1
    assert 1 < skew[simulate.REFERENCE] < skew["Lognormal com outliers (n=30)"]


def test_generated_parameters_match_calibration():
    rng = simulate.make_rng(2)
    a, _ = simulate.generate(rng, "lognormal", 30, 0.0, 4000, 27.0, 1.24)
    assert abs(np.log(a).std() - 1.24) < 0.02 and abs(np.median(a) / 27.0 - 1) < 0.03
    mu, sd = simulate.normal_moments(27.0, 1.24)
    x, _ = simulate.generate(rng, "normal", 30, 0.0, 4000, 27.0, 1.24)
    assert abs(x.mean() / mu - 1) < 0.02 and abs(x.std() / sd - 1) < 0.02


def test_run_sample_sizes_grid_and_reproducible():
    from pandas.testing import assert_frame_equal
    from src import config
    sens = simulate.run_sample_sizes(size=50, seed=42, sizes=[8, 30])
    assert_frame_equal(sens, simulate.run_sample_sizes(size=50, seed=42, sizes=[8, 30]))
    assert len(sens) == len(simulate.SHAPES) * 2 * len(config.EFFECTS) * len(config.COLORS)
    assert not sens.duplicated(["forma", "n", "d", "teste"]).any()
    np.testing.assert_allclose(sens.taxa_rejeicao, sens.rejeicoes / sens.repeticoes)


def test_stored_csvs_are_regenerated_by_the_code():
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from src import config
    assert_frame_equal(pd.read_csv(config.RESULTS_CSV), simulate.run_all())
    assert_frame_equal(pd.read_csv(config.SENSITIVITY_CSV), simulate.run_sample_sizes())


def test_q3_results_regressions():
    """Conclusões de Q3 exibidas no dashboard: se mudarem, o texto precisa ser revisto."""
    import pandas as pd
    from src import config, plots
    res = pd.read_csv(config.RESULTS_CSV)
    assert res.groupby(["situacao", "teste"]).d.apply(lambda d: sorted(d) == config.EFFECTS).all()
    null = plots.type1_summary(res).set_index(["situacao", "teste"])
    mw = null.xs("Mann-Whitney", level="teste")
    assert ((mw.ic_inf <= config.ALPHA) & (mw.ic_sup >= config.ALPHA)).all()
    lognormal = [s for s, spec in simulate.SITUATIONS.items() if spec["kind"] != "normal"]
    assert (null.xs("Welch", level="teste").loc[lognormal].ic_sup < config.ALPHA).all()
    piv = res.pivot_table(index=["situacao", "d"], columns="teste", values="taxa_rejeicao")
    for sit in simulate.SITUATIONS:
        curve = piv.loc[sit]
        assert (curve.diff().dropna() > -0.01).all().all(), sit
        strong = curve.loc[curve.index >= 0.4]
        if sit in lognormal:
            assert (strong["Mann-Whitney"] > strong["Welch"]).all(), sit
        else:
            assert (strong["Welch"] > strong["Mann-Whitney"]).all(), sit


def test_sensitivity_regressions_and_consistency_with_main_grid():
    import pandas as pd
    from src import config
    res = pd.read_csv(config.RESULTS_CSV)
    sens = pd.read_csv(config.SENSITIVITY_CSV)
    welch_null = sens[sens.d.eq(0) & sens.teste.eq("Welch") & sens.forma.ne("Normal")]
    for _, g in welch_null.groupby("forma"):
        rates = g.sort_values("n").taxa_rejeicao
        assert rates.is_monotonic_increasing and (rates < config.ALPHA).all()
    mw_null = sens[sens.d.eq(0) & sens.teste.eq("Mann-Whitney")].taxa_rejeicao
    assert mw_null.between(0.035, 0.06).all()
    power = sens[sens.d.gt(0)].sort_values("n").groupby(["forma", "d", "teste"]).taxa_rejeicao
    assert power.apply(lambda r: (r.diff().dropna() > -0.01).all()).all()
    # Pontos em comum com a grade principal: fluxos aleatórios distintos, iguais dentro do erro de Monte Carlo.
    for sit, spec in simulate.SITUATIONS.items():
        shared = sens[sens.forma.eq(simulate.SHAPES[spec["kind"]]) & sens.n.eq(spec["n"])]
        merged = res[res.situacao.eq(sit)].merge(shared, on=["d", "teste"], suffixes=("", "_n"))
        assert len(merged) == len(config.EFFECTS) * len(config.COLORS)
        p = (merged.taxa_rejeicao + merged.taxa_rejeicao_n) / 2
        se = np.sqrt(2 * p * (1 - p) / config.N_SIM)
        assert ((merged.taxa_rejeicao - merged.taxa_rejeicao_n).abs() <= 4 * se + 1e-9).all(), sit


def test_q3_comparison_and_readings():
    import pandas as pd
    from src import config, plots
    res = pd.read_csv(config.RESULTS_CSV)
    delta = plots.q3_comparison(res)
    assert simulate.REFERENCE not in delta.index.get_level_values("situacao")
    assert list(delta.columns) == config.EFFECTS
    piv = 100 * res.pivot_table(index=["situacao", "teste"], columns="d", values="taxa_rejeicao")
    for (sit, test), row in delta.iterrows():
        np.testing.assert_allclose(row, piv.loc[(sit, test)] - piv.loc[(simulate.REFERENCE, test)])
    assert len(plots.q3_reading(res)) == len(simulate.SITUATIONS) - 1
    sens = pd.read_csv(config.SENSITIVITY_CSV)
    fig = plots.sample_size_figure(sens, 0.6)
    assert len(fig.data) == len(simulate.SHAPES) * len(config.COLORS) * 2
    assert {t.line.color for t in fig.data} == set(config.COLORS.values())
    assert len(plots.sample_size_reading(sens, 0.6)) == len(simulate.SHAPES)


def test_real_effect_is_read_on_g4_scale():
    import pandas as pd
    from scipy import stats
    from src import config, plots
    real = simulate.load_real()
    d = plots.real_effect_g4_scale(real)
    a, b = (np.log(real[real.grupo.eq(g)].horas_ate_merge) for g in ("com_bot", "sem_bot"))
    assert d == pytest.approx((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
    _, sig = simulate.pooled_params(real)
    assert np.sqrt((a.var() + b.var()) / 2) == pytest.approx(sig)
    curve = pd.DataFrame(dict(d=[0.0, 0.2, 0.4], taxa_rejeicao=[0.05, 0.15, 0.45]))
    assert plots.power_at(curve, -0.1) == pytest.approx(0.10)
    n = plots.n_for_power(0.5)
    z = stats.norm.ppf(1 - config.ALPHA / 2) + stats.norm.ppf(0.8)
    assert n == int(np.ceil(2 * (z / 0.5) ** 2)) == 63
