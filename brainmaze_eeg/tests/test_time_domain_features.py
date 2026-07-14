import numpy as np
import pytest

from brainmaze_eeg.features.time_domain_features import (
    TimeDomainFeatureExtractor,
    tkeo,
    line_length,
)

FS = 200


# ----------------------------------------------------------------------------------
# reference implementation: the obvious per-window computation. The extractor must
# agree with this exactly -- it is an optimisation, not a redefinition.
# ----------------------------------------------------------------------------------
def reference(x, fs, segm_size, overlap=0.0):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n = int(round(fs * segm_size))
    shift = int(round(fs * (segm_size - overlap)))

    ll, tk, dr = [], [], []
    for ch in x:
        r_ll, r_tk, r_dr = [], [], []
        for s in range(0, len(ch) - n + 1, shift):
            w = ch[s:s + n]
            inc = np.abs(np.diff(w))
            inc = inc[~np.isnan(inc)]
            r_ll.append(inc.sum() if inc.size else np.nan)
            psi = w[1:-1] ** 2 - w[2:] * w[:-2]
            psi = psi[~np.isnan(psi)]
            r_tk.append(psi.mean() if psi.size else np.nan)
            r_dr.append(np.sum(~np.isnan(w)) / n)
        ll.append(r_ll)
        tk.append(r_tk)
        dr.append(r_dr)
    return np.array(ll), np.array(tk), np.array(dr)


def features_as_dict(values, names):
    return dict(zip(names, values))


# ----------------------------------------------------------------------------------
# tkeo
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('amplitude, freq', [(1.0, 10.0), (50e-6, 2.0), (3.0, 40.0)])
def test_tkeo_matches_analytic_solution_for_a_sinusoid(amplitude, freq):
    # for x = A*sin(2*pi*f*t) the TKEO converges to A^2 * sin^2(2*pi*f/fs)
    t = np.arange(0, 10, 1 / FS)
    x = amplitude * np.sin(2 * np.pi * freq * t)

    expected = amplitude ** 2 * np.sin(2 * np.pi * freq / FS) ** 2
    assert np.nanmean(tkeo(x)) == pytest.approx(expected, rel=1e-9)


def test_tkeo_is_sample_aligned_with_nan_edges():
    x = np.arange(5, dtype=float)
    psi = tkeo(x)

    assert psi.shape == x.shape
    assert np.isnan(psi[0]) and np.isnan(psi[-1])
    # x[n]^2 - x[n-1]*x[n+1] for a ramp is exactly 1
    np.testing.assert_allclose(psi[1:-1], [1.0, 1.0, 1.0])


def test_tkeo_is_zero_for_dc_and_positive_for_oscillation():
    assert np.nanmean(tkeo(np.full(100, 7.0))) == pytest.approx(0.0, abs=1e-12)
    t = np.arange(0, 5, 1 / FS)
    assert np.nanmean(tkeo(np.sin(2 * np.pi * 10 * t))) > 0


def test_tkeo_too_short_signal_is_all_nan():
    assert np.all(np.isnan(tkeo(np.array([1.0, 2.0]))))


def test_tkeo_applies_along_requested_axis():
    x = np.random.default_rng(0).normal(size=(4, 50))
    np.testing.assert_allclose(tkeo(x.T, axis=0), tkeo(x, axis=-1).T, equal_nan=True)


# ----------------------------------------------------------------------------------
# line_length
# ----------------------------------------------------------------------------------
def test_line_length_matches_closed_form():
    # repeating ramp 0,1,2,3 -> increments 1,1,1,3 = 6 per period
    x = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), 500)
    # the very first sample has no predecessor, so the final |3->0| step is not counted
    assert np.nansum(line_length(x)) == pytest.approx(6 * 500 - 3)


def test_line_length_is_sample_aligned_with_nan_first():
    x = np.array([0.0, 3.0, 1.0])
    ll = line_length(x)
    assert ll.shape == x.shape
    assert np.isnan(ll[0])
    np.testing.assert_allclose(ll[1:], [3.0, 2.0])


def test_line_length_is_zero_for_dc():
    assert np.nansum(line_length(np.full(100, 7.0))) == pytest.approx(0.0)


def test_line_length_too_short_signal_is_all_nan():
    assert np.all(np.isnan(line_length(np.array([1.0]))))


# ----------------------------------------------------------------------------------
# extractor: agreement with the reference
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('segm_size, overlap', [
    (30, 0.0), (5, 0.0), (1, 0.0), (0.5, 0.0),   # non-overlapping
    (4, 2.0), (1, 0.5), (0.5, 0.25),             # overlapping
])
def test_extractor_matches_per_window_reference(segm_size, overlap):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 60 * FS))

    values, names = TimeDomainFeatureExtractor(
        fs=FS, segm_size=segm_size, overlap=overlap, datarate=True)(x)
    got = features_as_dict(values, names)
    ll, tk, dr = reference(x, FS, segm_size, overlap)

    np.testing.assert_allclose(got['LINE_LENGTH'], ll)
    np.testing.assert_allclose(got['TKEO_MEAN'], tk)
    np.testing.assert_allclose(got['DATA_RATE'], dr)


@pytest.mark.parametrize('segm_size, overlap', [(5, 0.0), (4, 2.0)])
def test_extractor_matches_reference_with_nans(segm_size, overlap):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 60 * FS))
    x[0, 1000:1500] = np.nan     # partially missing
    x[2, :] = np.nan             # entirely missing

    values, names = TimeDomainFeatureExtractor(
        fs=FS, segm_size=segm_size, overlap=overlap, datarate=True)(x)
    got = features_as_dict(values, names)
    ll, tk, dr = reference(x, FS, segm_size, overlap)

    np.testing.assert_allclose(got['LINE_LENGTH'], ll, equal_nan=True)
    np.testing.assert_allclose(got['TKEO_MEAN'], tk, equal_nan=True)
    np.testing.assert_allclose(got['DATA_RATE'], dr)


def test_nan_free_fast_path_agrees_with_nan_aware_path():
    # a single NaN forces the nan-aware branch; away from it results must be unchanged
    rng = np.random.default_rng(2)
    x = rng.normal(size=10 * FS)
    clean, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x)

    x_nan = x.copy()
    x_nan[-1] = np.nan                      # only touches the final window
    dirty, _ = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x_nan)

    for c, d in zip(clean, dirty):
        np.testing.assert_allclose(c[:-1], d[:-1])


# ----------------------------------------------------------------------------------
# extractor: NaN semantics
# ----------------------------------------------------------------------------------
def test_all_nan_window_yields_nan_not_zero():
    x = np.concatenate([np.ones(FS), np.full(FS, np.nan), 2 * np.ones(FS)])
    values, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1, datarate=True)(x)
    got = features_as_dict(values, names)

    # an all-NaN window is undefined for both shape features, not a flat (zero) window
    assert np.isnan(got['TKEO_MEAN'][1])
    assert np.isnan(got['LINE_LENGTH'][1])
    np.testing.assert_allclose(got['DATA_RATE'], [1.0, 0.0, 1.0])


def test_nans_are_not_counted_as_flat_signal():
    # a genuinely flat window reads 0; an all-NaN window is undefined (NaN), never 0
    x = np.concatenate([np.zeros(FS), np.full(FS, np.nan)])
    values, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x)
    got = features_as_dict(values, names)
    assert got['LINE_LENGTH'][0] == 0.0
    assert np.isnan(got['LINE_LENGTH'][1])


def test_extractor_emits_no_warnings_on_all_nan_input():
    x = np.full((2, 10 * FS), np.nan)
    with np.errstate(all='raise'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            TimeDomainFeatureExtractor(fs=FS, segm_size=1, datarate=True)(x)


# ----------------------------------------------------------------------------------
# extractor: shapes, windowing, selection
# ----------------------------------------------------------------------------------
def test_1d_input_returns_1d_features_2d_returns_2d():
    x1 = np.random.default_rng(3).normal(size=10 * FS)
    values, _ = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x1)
    assert all(v.shape == (10,) for v in values)

    x2 = np.random.default_rng(3).normal(size=(4, 10 * FS))
    values, _ = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x2)
    assert all(v.shape == (4, 10) for v in values)


def test_trailing_partial_window_is_dropped():
    x = np.zeros(int(2.7 * FS))   # 2.7 s of a 1 s window -> 2 whole windows
    values, _ = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x)
    assert values[0].shape == (2,)


def test_signal_shorter_than_one_window_yields_no_windows():
    values, _ = TimeDomainFeatureExtractor(fs=FS, segm_size=5)(np.zeros(FS))
    assert all(v.shape == (0,) for v in values)


def test_overlapping_window_count():
    x = np.zeros(10 * FS)
    values, _ = TimeDomainFeatureExtractor(fs=FS, segm_size=1, overlap=0.5)(x)
    assert values[0].shape == (19,)   # (10 - 1)/0.5 + 1


def test_feature_selection_controls_output_and_order():
    x = np.random.default_rng(4).normal(size=5 * FS)

    _, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1, features=('TKEO_MEAN',))(x)
    assert names == ['TKEO_MEAN']

    _, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1, features=('LINE_LENGTH',))(x)
    assert names == ['LINE_LENGTH']

    _, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1, datarate=True)(x)
    assert names == ['DATA_RATE', 'LINE_LENGTH', 'TKEO_MEAN']


def test_datarate_is_off_by_default():
    x = np.random.default_rng(5).normal(size=5 * FS)
    _, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x)
    assert 'DATA_RATE' not in names


def test_integer_input_is_accepted():
    x = np.arange(5 * FS, dtype=np.int64)
    values, names = TimeDomainFeatureExtractor(fs=FS, segm_size=1)(x)
    got = features_as_dict(values, names)
    # a unit ramp: every increment is 1, so line length is n-1 per window
    np.testing.assert_allclose(got['LINE_LENGTH'], np.full(5, FS - 1.0))


def test_input_is_not_mutated():
    x = np.random.default_rng(6).normal(size=5 * FS)
    x[10] = np.nan
    before = x.copy()
    TimeDomainFeatureExtractor(fs=FS, segm_size=1, datarate=True)(x)
    np.testing.assert_array_equal(x, before)


# ----------------------------------------------------------------------------------
# extractor: input validation
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('kwargs', [
    {'fs': 0, 'segm_size': 1},
    {'fs': -1, 'segm_size': 1},
    {'fs': FS, 'segm_size': 0},
    {'fs': FS, 'segm_size': -1},
    {'fs': FS, 'segm_size': np.inf},
    {'fs': FS, 'segm_size': 1, 'overlap': -0.1},
    {'fs': FS, 'segm_size': 1, 'overlap': 1.0},    # overlap == segm_size
    {'fs': FS, 'segm_size': 1, 'overlap': 2.0},
    {'fs': FS, 'segm_size': 1, 'features': ()},
    {'fs': FS, 'segm_size': 1, 'features': ('NOPE',)},
    {'fs': FS, 'segm_size': 0.01},                 # 2 samples -> TKEO needs 3
    {'fs': FS, 'segm_size': 1, 'overlap': 0.999},  # step rounds to 0 samples
])
def test_invalid_construction_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        TimeDomainFeatureExtractor(**kwargs)


@pytest.mark.parametrize('bad', [
    np.zeros((2, 2, 100)),
    np.array(1.0),
])
def test_invalid_input_dimensionality_raises(bad):
    with pytest.raises(ValueError):
        TimeDomainFeatureExtractor(fs=FS, segm_size=1)(bad)


# ----------------------------------------------------------------------------------
# physical sanity
# ----------------------------------------------------------------------------------
def test_line_length_and_tkeo_increase_with_amplitude_and_frequency():
    t = np.arange(0, 10, 1 / FS)
    e = TimeDomainFeatureExtractor(fs=FS, segm_size=10)

    base = features_as_dict(*e(np.sin(2 * np.pi * 5 * t)))
    louder = features_as_dict(*e(3 * np.sin(2 * np.pi * 5 * t)))
    faster = features_as_dict(*e(np.sin(2 * np.pi * 20 * t)))

    assert louder['LINE_LENGTH'][0] > base['LINE_LENGTH'][0]
    assert faster['LINE_LENGTH'][0] > base['LINE_LENGTH'][0]
    assert louder['TKEO_MEAN'][0] > base['TKEO_MEAN'][0]
    assert faster['TKEO_MEAN'][0] > base['TKEO_MEAN'][0]


def test_channels_are_independent():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=10 * FS), 5 * rng.normal(size=10 * FS)
    e = TimeDomainFeatureExtractor(fs=FS, segm_size=1)

    stacked, names = e(np.vstack([a, b]))
    got = features_as_dict(stacked, names)
    alone_a = features_as_dict(*e(a))
    alone_b = features_as_dict(*e(b))

    for key in ('LINE_LENGTH', 'TKEO_MEAN'):
        np.testing.assert_allclose(got[key][0], alone_a[key])
        np.testing.assert_allclose(got[key][1], alone_b[key])
