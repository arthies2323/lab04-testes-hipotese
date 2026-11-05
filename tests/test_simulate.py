from src import simulate


def test_lognormal_median_close():
    rng = simulate.make_rng(1)
    x = simulate.lognormal_group(rng, 100_000, 10.0, 1.0)
    assert abs(float(__import__("numpy").median(x)) - 10.0) < 0.3


def test_run_tests_returns_valid_pvalues():
    rng = simulate.make_rng(1)
    a = simulate.lognormal_group(rng, 30, 10, 1)
    b = simulate.lognormal_group(rng, 30, 10, 1)
    assert all(0 <= p <= 1 for p in simulate.run_tests(a, b))


def test_type1_error_mannwhitney_near_5pct():
    rng = simulate.make_rng(3)
    _, mw = simulate.rejection_rates(rng, "lognormal", 30, 0.0, 2000, 10.0, 1.0)
    assert 0.03 < mw < 0.07
