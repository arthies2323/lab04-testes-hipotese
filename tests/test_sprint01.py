import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from src import sprint01, sprint01_plots


def test_calibration_and_example_match_source():
    real = sprint01.load_real()
    cal = sprint01.calibrate(real)
    sample = sprint01.example(real)
    pd.testing.assert_frame_equal(sample, sprint01.example(real))
    assert sample.groupby('grupo').size().to_dict() == {'com_bot': 30, 'sem_bot': 30}
    for g in sprint01.GROUPS:
        x = real.loc[real.grupo == g, 'horas_ate_merge']
        assert cal.loc[g, 'median'] == np.median(x)
        assert cal.loc[g, 'sigma'] == pytest.approx(np.std(np.log(x), ddof=1))
    assert (sample.horas_ate_merge > 0).all()
    assert not sample.equals(sprint01.example(real, seed=43))


def test_describe_justifies_lognormal():
    real = sprint01.load_real()
    desc = sprint01.describe(real)
    for g in sprint01.GROUPS:
        x = real.loc[real.grupo == g, 'horas_ate_merge'].to_numpy()
        assert desc.loc[g, 'media'] == pytest.approx(x.mean())
        assert desc.loc[g, 'assimetria'] == pytest.approx(stats.skew(x, bias=False))
        assert desc.loc[g, 'shapiro_p_log'] == pytest.approx(stats.shapiro(np.log(x)).pvalue)
    # A amostra real rejeita normalidade nas horas e não no log: é o que sustenta a lognormal.
    assert (desc.shapiro_p_horas < .05).all() and (desc.shapiro_p_log >= .05).all()
    assert 'simulado como lognormal' in sprint01.shape_text(desc)
    assert 'revista' in sprint01.shape_text(desc.assign(shapiro_p_log=.01))


def test_statistics_effects_and_reference_areas():
    sample = sprint01.example(sprint01.load_real())
    table, t, u, z = sprint01.analyze(sample)
    a, b = [sample.loc[sample.grupo == g, 'horas_ate_merge'].to_numpy() for g in sprint01.GROUPS]
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    df = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
    assert t.df == pytest.approx(df)
    assert t.pvalue == pytest.approx(2*stats.t.sf(abs(t.statistic), df))
    assert u.pvalue == pytest.approx(2*stats.norm.sf(z))
    superiority = ((a[:,None] > b).mean() + .5*(a[:,None] == b).mean())
    assert table.iloc[1].efeito == pytest.approx(2*superiority-1)
    for row in table.itertuples():
        assert row.decisao == ('rejeita H0' if row.p_valor < .05 else 'não rejeita H0')
        assert 'Não rejeitar não comprova igualdade' in row.interpretacao


def test_swap_reverses_effect_not_pvalue():
    sample = sprint01.example(sprint01.load_real())
    swapped = sample.assign(grupo=sample.grupo.map({'com_bot':'sem_bot','sem_bot':'com_bot'}))
    a, *_ = sprint01.analyze(sample)
    b, *_ = sprint01.analyze(swapped)
    np.testing.assert_allclose(a.p_valor, b.p_valor)
    np.testing.assert_allclose(a.efeito, -b.efeito)


def test_g1_preserves_tail_and_same_example():
    real = sprint01.load_real()
    sample = sprint01.example(real)
    fig = sprint01_plots.g1(real, sample)
    for trace, (data, g) in zip(fig.data, [(real,'com_bot'),(real,'sem_bot'),(sample,'com_bot'),(sample,'sem_bot')]):
        np.testing.assert_array_equal(trace.x, data.loc[data.grupo == g, 'horas_ate_merge'])


def test_g2_has_both_tests_and_both_tails():
    real = sprint01.load_real()
    sample = sprint01.example(real)
    fig = sprint01_plots.g2(real, sample, repetitions=100)
    assert len(fig.data) == 12
    for base in (0,6):
        assert np.max(fig.data[base+2].x) < 0
        assert np.min(fig.data[base+3].x) > 0
        assert 'Área bicaudal' in fig.data[base+4].name
        assert 'Área bicaudal' in fig.data[base+5].name


def test_export_is_reproducible_and_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sprint01.export(tmp_path / 'a')
    sprint01.export(tmp_path / 'b')
    for name in ('exemplo_simulado.csv','testes_exemplo.csv','calibracao.csv','descritivas.csv'):
        assert (tmp_path/'a'/name).read_bytes() == (tmp_path/'b'/name).read_bytes()
    assert (tmp_path/'a'/'G1.html').exists()
    assert (tmp_path/'a'/'G2.html').exists()


def test_cli_runs_with_non_utf8_stdout(tmp_path):
    env = {**os.environ, 'PYTHONIOENCODING': 'cp1252'}
    run = subprocess.run([sys.executable, '-m', 'src.sprint01', '--output', str(tmp_path)],
                         cwd=sprint01.ROOT, env=env, capture_output=True)
    assert run.returncode == 0, run.stderr.decode(errors='replace')
    assert 'α=0.05' in run.stdout.decode('utf-8')


@pytest.mark.parametrize('value', [0,-1,np.nan,np.inf])
def test_invalid_real_times_rejected(tmp_path, value):
    real = sprint01.load_real()
    real.loc[0,'horas_ate_merge'] = value
    path = tmp_path/'invalid.csv'
    real.to_csv(path, index=False)
    with pytest.raises(ValueError, match='finitos e positivos'):
        sprint01.load_real(path)


def test_dashboard_without_s02(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src import simulate
    def forbidden(*args, **kwargs):
        raise AssertionError('A S01 não deve executar a S02')
    monkeypatch.setattr(simulate, 'run_all', forbidden)
    original_read = pd.read_csv
    def read(path, *args, **kwargs):
        assert Path(path).name != 'results.csv'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', read)
    app = AppTest.from_file(str(sprint01.ROOT/'dashboard'/'sprint01.py')).run(timeout=30)
    assert not app.exception
    assert [x.value for x in app.header] == ['1. Cenário','2. Os dados — G1','3. Como os testes funcionam — G2']


def test_full_dashboard_uses_s01_g1_g2(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(sprint01.ROOT)  # app.py faz os.chdir; restaurado ao fim do teste
    calls = []
    for name in ('g1', 'g2'):
        original = getattr(sprint01_plots, name)
        def spy(real, sample, *args, _name=name, _original=original, **kwargs):
            calls.append((_name, sample))
            return _original(real, sample, *args, **kwargs)
        monkeypatch.setattr(sprint01_plots, name, spy)
    app = AppTest.from_file(str(sprint01.ROOT/'dashboard'/'app.py')).run(timeout=60)
    assert not app.exception
    assert [name for name, _ in calls] == ['g1', 'g2']
    expected = sprint01.example(sprint01.load_real())
    for _, sample in calls:
        pd.testing.assert_frame_equal(sample, expected)
    assert len(app.header) == 7
