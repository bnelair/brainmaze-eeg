"""Gap helpers (brainmaze_eeg.spikes._gaps, a stand-in for brainmaze_utils.gaps)."""

import numpy as np
import pytest

from brainmaze_eeg.spikes._gaps import (drop_in_gaps, fill_gaps, find_gaps, gap_intervals,
                                        mask_in_gaps)
from brainmaze_eeg.tests.spike_synth import pink_background


def test_find_gaps_runs_including_edges_and_inf():
    x = np.arange(20, dtype=float)
    x[[0, 1, 5, 9, 10, 11, 19]] = np.nan
    x[10] = np.inf
    np.testing.assert_array_equal(find_gaps(x), [[0, 2], [5, 6], [9, 12], [19, 20]])
    np.testing.assert_allclose(gap_intervals(x, 10.0), [[0, .2], [.5, .6], [.9, 1.2], [1.9, 2.0]])
    assert find_gaps(np.ones(5)).shape == (0, 2)
    with pytest.raises(ValueError):
        find_gaps(np.ones((2, 5)))


def test_short_gap_is_linear_and_valid_samples_untouched():
    x = np.arange(100, dtype=float)
    y = x.copy()
    y[40:45] = np.nan
    z = fill_gaps(y, 100.0)                          # 5 samples = 0.05 s <= 0.1 s
    np.testing.assert_allclose(z, x)
    assert np.isnan(y[40:45]).all()                  # input not modified


@pytest.mark.parametrize('method', ['mirror', 'pink'])
@pytest.mark.parametrize('where', ['middle', 'start', 'end'])
@pytest.mark.parametrize('gap_s', [0.5, 3.0, 20.0])
def test_long_gap_fill_is_continuous_and_background_like(method, where, gap_s):
    fs = 500.0
    x = pink_background(int(60 * fs), fs, 30.0, np.random.default_rng(1))
    n = int(gap_s * fs)
    s = {'middle': int(20 * fs), 'start': 0, 'end': x.size - n}[where]
    e = s + n
    y = x.copy()
    y[s:e] = np.nan
    z = fill_gaps(y, fs, method=method)
    assert np.isfinite(z).all()
    np.testing.assert_array_equal(z[:s], x[:s])
    np.testing.assert_array_equal(z[e:], x[e:])
    d = np.abs(np.diff(x)).std()
    if s > 0:
        assert abs(z[s] - z[s - 1]) < 4 * d
    if e < x.size:
        assert abs(z[e] - z[e - 1]) < 4 * d
    assert 0.4 < z[s:e].std() / x.std() < 2.5


def test_fill_reproducible_and_independent_across_channels():
    fs = 200.0
    rng = np.random.default_rng(2)
    a = pink_background(int(30 * fs), fs, 30.0, rng)
    b = pink_background(int(30 * fs), fs, 30.0, rng)
    a[2000:3000] = b[2000:3000] = np.nan
    fa = fill_gaps(a, fs, method='pink')
    np.testing.assert_array_equal(fa, fill_gaps(a, fs, method='pink'))
    fb = fill_gaps(b, fs, method='pink')
    core = slice(2200, 2800)
    assert abs(np.corrcoef(fa[core], fb[core])[0, 1]) < 0.5
    assert not np.array_equal(fa, fill_gaps(a, fs, method='pink', seed=1))


def test_fill_validation_and_all_nan():
    with pytest.raises(ValueError):
        fill_gaps(np.ones(10), 10.0, method='spectral')       # not in this stand-in
    with pytest.raises(ValueError):
        fill_gaps(np.ones(10), 0.0)
    with pytest.raises(TypeError):
        fill_gaps(np.ones(10, complex), 10.0)
    z = np.full(10, np.nan)
    with pytest.warns(RuntimeWarning):
        assert np.isnan(fill_gaps(z, 10.0)).all()
    with pytest.warns(RuntimeWarning):
        assert (fill_gaps(z, 10.0, all_nan='zero') == 0).all()
    with pytest.raises(ValueError):
        fill_gaps(z, 10.0, all_nan='raise')


def test_mask_in_gaps_explicit_units():
    gaps = np.array([[1000, 2000]])                  # samples; = [2.0, 4.0) s at 500 Hz
    fs = 500.0
    t = np.array([1.85, 1.95, 2.5, 4.05, 4.15])
    want = [False, True, True, True, False]
    np.testing.assert_array_equal(mask_in_gaps(t, gaps / fs, fs, units='seconds'), want)
    np.testing.assert_array_equal(
        mask_in_gaps(np.round(t * fs).astype(int), gaps, fs, units='samples'), want)
    np.testing.assert_allclose(drop_in_gaps(t, gaps / fs, fs, units='seconds'), [1.85, 4.15])
    # interval detections overlapping the widened gap
    np.testing.assert_array_equal(
        mask_in_gaps([1.0, 1.0], gaps / fs, fs, units='seconds', end=[1.85, 1.95]),
        [False, True])
    # unit mix-ups raise instead of silently masking nothing
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps, fs, units='seconds')            # integer gaps as seconds
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps, fs, units='samples')            # fractional samples
    with pytest.raises(TypeError):
        mask_in_gaps(t, gaps, fs)                             # units is required
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps / fs, fs, units='seconds', margin_s=-1)


def test_generator_seed_gives_independent_per_gap_streams():
    # a gap's fill must not depend on other gaps, also when seed is a Generator
    fs = 200.0
    x = pink_background(int(60 * fs), fs, 30.0, np.random.default_rng(5))
    one = x.copy()
    one[8000:9000] = np.nan
    two = one.copy()
    two[2000:2600] = np.nan
    a = fill_gaps(one, fs, method='pink', seed=np.random.default_rng(1))
    b = fill_gaps(two, fs, method='pink', seed=np.random.default_rng(1))
    np.testing.assert_array_equal(a[8000:9000], b[8000:9000])
    with pytest.raises(TypeError):
        fill_gaps(one, fs, method='pink', seed=np.random.RandomState(0))


def test_fill_nd_axis_copy_and_dtype():
    fs = 200.0
    rng = np.random.default_rng(7)
    X = np.vstack([pink_background(4000, fs, 30.0, rng) for _ in range(3)])
    X[0, 1000:1500] = np.nan
    X[2, 3000:3100] = np.inf
    rows = np.vstack([fill_gaps(r, fs) for r in X])
    np.testing.assert_array_equal(fill_gaps(X, fs), rows)          # row == 1-D call
    np.testing.assert_array_equal(fill_gaps(X.T, fs, axis=0), rows.T)
    f32 = fill_gaps(X.astype(np.float32), fs)
    assert f32.dtype == np.float32 and np.isfinite(f32).all()
    Y = X.copy()
    out = fill_gaps(Y, fs, copy=False)
    assert out is Y and np.isfinite(Y).all()
    Z = X.copy()
    Z[1] = np.nan
    with pytest.warns(RuntimeWarning, match='no finite sample'):
        assert np.isnan(fill_gaps(Z, fs)[1]).all()
    with pytest.raises(ValueError):
        fill_gaps(Z, fs, all_nan='raise')
    with pytest.raises(ValueError):
        fill_gaps(X, fs, axis=2)
