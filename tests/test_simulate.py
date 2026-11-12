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
