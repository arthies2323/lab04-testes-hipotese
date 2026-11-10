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
