"""Gap helpers used by the spike detectors (brainmaze_eeg.spikes._gaps)."""

import numpy as np
import pytest

from brainmaze_eeg.spikes._gaps import (drop_in_gaps, fill_gaps, find_gaps, mask_in_gaps,
                                        prepare_signal)
from brainmaze_eeg.tests.spike_synth import pink_background


def test_find_gaps_runs_including_edges():
    x = np.arange(20, dtype=float)
    x[[0, 1, 5, 9, 10, 11, 19]] = np.nan
    np.testing.assert_array_equal(find_gaps(x), [[0, 2], [5, 6], [9, 12], [19, 20]])
    assert find_gaps(np.ones(5)).shape == (0, 2)
    with pytest.raises(ValueError):
        find_gaps(np.ones((2, 5)))


def test_short_gap_is_linear_and_valid_samples_untouched():
    x = np.arange(100, dtype=float)
    y = x.copy()
    y[40:45] = np.nan
    z = fill_gaps(y, fs=100.0)                       # 5 samples = 0.05 s <= 0.1 s
    np.testing.assert_allclose(z, x)
    assert np.isnan(y[40:45]).all()                  # input not modified


@pytest.mark.parametrize('gap_s', [0.5, 3.0, 20.0])
def test_long_gap_fill_is_continuous_and_background_like(gap_s):
    fs = 500.0
    x = pink_background(int(60 * fs), fs, 30.0, np.random.default_rng(1))
    y = x.copy()
    s, e = int(20 * fs), int((20 + gap_s) * fs)
    y[s:e] = np.nan
    z = fill_gaps(y, fs)
    assert np.isfinite(z).all()
    np.testing.assert_array_equal(z[:s], x[:s])
    np.testing.assert_array_equal(z[e:], x[e:])
    d = np.abs(np.diff(x)).std()
    # no step at the edges larger than the signal's own sample-to-sample variation
    assert abs(z[s] - z[s - 1]) < 4 * d and abs(z[e] - z[e - 1]) < 4 * d
    assert 0.5 < z[s:e].std() / x.std() < 2.0


def test_fill_is_reproducible_and_streams_independent():
    fs = 200.0
    x = pink_background(int(30 * fs), fs, 30.0, np.random.default_rng(2))
    x[2000:3000] = np.nan
    a = fill_gaps(x, fs)
    np.testing.assert_array_equal(a, fill_gaps(x, fs))
    b = fill_gaps(x, fs, stream=1)
    core = slice(2200, 2800)
    assert abs(np.corrcoef(a[core], b[core])[0, 1]) < 0.5


def test_prepare_signal_gives_channels_independent_fills():
    fs = 200.0
    x = pink_background(int(30 * fs), fs, 30.0, np.random.default_rng(3))
    X = np.vstack([x, x])
    X[:, 2000:3000] = np.nan
    Y, gaps, dead = prepare_signal(X, fs)
    assert not dead.any() and len(gaps) == 2
    assert abs(np.corrcoef(Y[0, 2200:2800], Y[1, 2200:2800])[0, 1]) < 0.5


@pytest.mark.parametrize('method', ['mirror', 'zeros'])
def test_unknown_fill_method_raises(method):
    with pytest.raises(ValueError):
        fill_gaps(np.ones(10), 10.0, method=method)


def test_mask_in_gaps_explicit_units():
    gaps = np.array([[1000, 2000]])                  # samples
    fs = 500.0                                       # gap = [2.0, 4.0) s
    t = np.array([1.85, 1.95, 2.5, 4.05, 4.15])
    np.testing.assert_array_equal(mask_in_gaps(t, gaps, fs, margin_s=0.1),
                                  [False, True, True, True, False])
    np.testing.assert_allclose(drop_in_gaps(t, gaps, fs, 0.1), [1.85, 4.15])
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps, None)                  # units are never inferred
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps, fs, margin_s=-1)


def test_prepare_signal_policies():
    x = np.ones((2, 100))
    x[1, 10] = np.nan
    with pytest.raises(ValueError, match='channels \\[1\\]'):
        prepare_signal(x, 100.0, 'raise')
    with pytest.raises(ValueError):
        prepare_signal(x, 100.0, 'omit')
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match='inf'):
        prepare_signal(x, 100.0, 'fill')
