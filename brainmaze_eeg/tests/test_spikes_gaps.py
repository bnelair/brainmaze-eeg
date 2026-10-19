"""Gap helpers (brainmaze_eeg.spikes._gaps: its own fill_gaps, the rest re-exported from
brainmaze_utils.gaps)."""

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
    sec = dict(units='seconds', gap_units='seconds')
    np.testing.assert_array_equal(mask_in_gaps(t, gaps / fs, fs, **sec), want)
    np.testing.assert_array_equal(
        mask_in_gaps(np.round(t * fs).astype(int), gaps, fs, units='samples',
                     gap_units='samples'), want)
    # mixed units: sample-index detections against gaps in seconds, and the reverse
    np.testing.assert_array_equal(
        mask_in_gaps(np.round(t * fs).astype(int), gaps / fs, fs, units='samples',
                     gap_units='seconds'), want)
    np.testing.assert_array_equal(mask_in_gaps(t, gaps, fs, units='seconds',
                                               gap_units='samples'), want)
    # floats within 1e-6 of an integer (t * fs) are accepted as samples
    np.testing.assert_array_equal(
        mask_in_gaps(t * fs + 1e-8, gaps, fs, units='samples', gap_units='samples'), want)
    np.testing.assert_allclose(drop_in_gaps(t, gaps / fs, fs, **sec), [1.85, 4.15])
    # interval detections overlapping the widened gap
    np.testing.assert_array_equal(
        mask_in_gaps([1.0, 1.0], gaps / fs, fs, end=[1.85, 1.95], **sec), [False, True])
    # unit mix-ups raise instead of silently masking nothing
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps, fs, units='seconds', gap_units='seconds')   # integer gaps, seconds
    with pytest.raises(ValueError):
        mask_in_gaps(np.array([100, 200]), gaps / fs, fs, units='seconds',
                     gap_units='seconds')                                 # integer det, seconds
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps, fs, units='samples', gap_units='samples')   # fractional samples
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps, fs, units='seconds', gap_units='minutes')
    with pytest.raises(TypeError):
        mask_in_gaps(t, gaps, fs, units='seconds')                        # gap_units required
    with pytest.raises(TypeError):
        mask_in_gaps(t, gaps, fs)                                         # units required
    with pytest.raises(TypeError):
        drop_in_gaps(t, gaps, fs, units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps(t, gaps / fs, fs, margin_s=-1, **sec)


def test_generator_seed_gives_independent_per_gap_streams():
    # a gap's fill must not depend on other gaps, also when seed is a Generator
    fs = 200.0
    x = pink_background(int(60 * fs), fs, 30.0, np.random.default_rng(5))
    one = x.copy()
    one[8000:9000] = np.nan
    two = one.copy()
    two[2000:2600] = np.nan
    # context_s=10: the other gap (10-13 s) lies outside the 10 s context of the one at 40 s
    a = fill_gaps(one, fs, method='pink', context_s=10.0, seed=np.random.default_rng(1))
    b = fill_gaps(two, fs, method='pink', context_s=10.0, seed=np.random.default_rng(1))
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


# ---------------------------------------------------------------- relation to brainmaze_utils.gaps
def test_shared_helpers_are_brainmaze_utils_gaps():
    """find_gaps, gap_intervals, pink_noise, mask_in_gaps and drop_in_gaps are re-exported from
    brainmaze_utils.gaps (they were bit-identical copies before)."""
    import brainmaze_utils.gaps as ug
    import brainmaze_eeg.spikes._gaps as eg
    for name in ('find_gaps', 'gap_intervals', 'pink_noise', 'mask_in_gaps', 'drop_in_gaps'):
        assert getattr(eg, name) is getattr(ug, name), name


def test_fill_gaps_is_this_packages_own_and_unchanged():
    """fill_gaps must stay eeg's own: brainmaze_utils.gaps.fill_gaps' 'mirror'/'pink' give
    different values, which would change GapAwareSpikeDetector's detections near gaps.
    Golden values from brainmaze-eeg 3.1.0 (identical before and after the switch of the other
    helpers to brainmaze_utils.gaps)."""
    import brainmaze_utils.gaps as ug
    import brainmaze_eeg.spikes._gaps as eg
    assert eg.fill_gaps is not ug.fill_gaps
    rng = np.random.default_rng(0)
    fs = 100.0
    # Exactly representable input (integers / 4): the same bytes on every platform. 'pink'
    # seeds each gap from a hash of the neighbouring samples, so an input built with np.sin
    # (last-bit differences between platform libms) gives another realisation on
    # macOS/Windows.
    x = (rng.integers(-40, 41, 400) + 30 * np.abs((np.arange(400) % 34) - 17) - 255) / 4.0
    x[150:170] = np.nan
    golden = {
        'mirror': [-34.23695420546449, -34.52328431539626, -13.301108122270424,
                   -18.961851456321675, -15.38106808197195, -7.523344591949827,
                   9.193253347262237, 9.888497921711402, 16.57322214889947, 22.443613606149448,
                   20.581140990771786, 23.035561074577096, 24.768436649559888,
                   28.351676179211086, 48.707544419910946, 50.12355805327924, 48.94597829881288,
                   56.22345351465909, 61.99926128059044, 59.61386085241393],
        'pink': [-30.65144959465621, -30.788860762145948, -22.282884007592926, -50.74255988318222,
                 32.50774830165446, 33.69132829671119, -39.35597423681201, -38.24564254870474,
                 -1.6012025867468749, 7.1655135043617015, -2.559035802450504, 103.17730636184805,
                 76.05131537563149, 50.957437617793396, 72.04601066729403, -1.4231920105472997,
                 20.536234900114753, -4.442333621063749, 37.658160737011926, 55.383063842827745],
    }
    for method, want in golden.items():
        y = eg.fill_gaps(x, fs, method=method, max_interp_s=0.02, context_s=1.0, taper_s=0.05,
                         seed=0)
        np.testing.assert_allclose(y[150:170], want, rtol=1e-9, atol=1e-9, err_msg=method)
        np.testing.assert_array_equal(y[:150], x[:150])
        np.testing.assert_array_equal(y[170:], x[170:])

    # one-sided gaps at the record start and end (the min(taper, n) branch); golden values
    # from brainmaze-eeg 3.1.0 (PyPI)
    xe = np.random.default_rng(0).integers(-40, 41, 400)
    xe = (xe + 30 * np.abs((np.arange(400) % 34) - 17) - 255) / 4.0
    xe[:15] = np.nan
    xe[385:] = np.nan
    golden_edge = {
        'mirror': ([31.0, 24.0, 1.75, 4.75, 3.5, -13.5, -28.75, -22.75, -27.25, -45.75, -40.0,
                    -55.0, -62.75, -53.5, -44.0],
                   [-6.25, -3.25, 9.5, 5.0, 23.75, 27.75, 38.0, 42.5, 50.25, 64.5, 71.75, 54.25,
                    52.5, 34.5, 40.25]),
        'pink': ([-18.14086584874253, 29.90145782159542, -4.005443863119641, 12.383299554666777,
                  30.594195904807677, 20.782786327024237, -1.165430844861354, -13.0418063000804,
                  -39.04634413323308, -28.855897042682486, -87.70150222323788, -70.5795903276702,
                  -64.05728442996818, -18.298457530746745, -39.42466367509921],
                 [-7.605277946389547, -1.4520637334016013, -21.817426557306685, 36.36907063705611,
                  -31.514982024650738, -58.484979899774956, 21.878003376016053, 13.64974612526993,
                  -44.91629470896975, -20.95299968025448, 29.789783175476778, 88.62449031440683,
                  93.41548187766081, 61.340901151002676, 26.238242631427426]),
    }
    for method, (lead, trail) in golden_edge.items():
        y = eg.fill_gaps(xe, fs, method=method, max_interp_s=0.02, context_s=1.0, taper_s=0.05,
                         seed=0)
        np.testing.assert_allclose(y[:15], lead, rtol=1e-9, atol=1e-9, err_msg=method)
        np.testing.assert_allclose(y[385:], trail, rtol=1e-9, atol=1e-9, err_msg=method)
        np.testing.assert_array_equal(y[15:385], xe[15:385])


def test_mask_in_gaps_margin_boundary():
    """Exact boundary of the widened gap [start - margin, stop + margin): closed on the left,
    open on the right. Values as given by brainmaze-eeg 3.1.0's own implementation, so a
    change of the re-exported brainmaze_utils.gaps.mask_in_gaps would be caught."""
    from brainmaze_eeg.spikes._gaps import mask_in_gaps
    fs = 100.0
    gaps = np.array([[100, 120]])                    # 1.00-1.20 s; margin 0.1 s -> [0.9, 1.3)
    want = [False, True, True, True, False, False]   # outside / at / inside ... at / outside
    det_s = np.array([0.89, 0.9, 0.91, 1.29, 1.3, 1.31])
    det_n = np.array([89, 90, 91, 129, 130, 131])
    kw = dict(margin_s=0.1)
    assert mask_in_gaps(det_s, gaps, fs, units='seconds', gap_units='samples', **kw).tolist() == want
    assert mask_in_gaps(det_n, gaps, fs, units='samples', gap_units='samples', **kw).tolist() == want
    assert mask_in_gaps(det_n, gaps / fs, fs, units='samples', gap_units='seconds', **kw).tolist() == want
    # intervals: masked when [det, end] reaches start - margin
    assert mask_in_gaps(np.array([80, 80, 131]), gaps, fs, units='samples', gap_units='samples',
                        end=np.array([89, 90, 140]), **kw).tolist() == [False, True, False]
    # no margin: the gap itself, stop exclusive
    assert mask_in_gaps(np.array([99, 100, 119, 120]), gaps, fs, units='samples',
                        gap_units='samples', margin_s=0.0).tolist() == [False, True, True, False]
