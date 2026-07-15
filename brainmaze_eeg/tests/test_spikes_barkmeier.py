import numpy as np
import pytest

from brainmaze_eeg.spikes import detect_spikes_barkmeier, DEFAULT_THRESHOLDS

FS = 512


def _biphasic(x, center, amp=600.0):
    """Add a sharp biphasic spike: central lobe with flanking troughs, into x in place."""
    w = int(0.02 * FS)
    k = np.arange(-w, w + 1)
    lobe = amp * np.exp(-(k / (0.004 * FS)) ** 2)
    troughs = (0.33 * amp * np.exp(-((k - 0.02 * FS) / (0.004 * FS)) ** 2)
               + 0.33 * amp * np.exp(-((k + 0.02 * FS) / (0.004 * FS)) ** 2))
    seg = lobe - troughs
    s = x[center - w:center + w + 1]
    s += seg[:len(s)]


def _noise(n, seed=0, sd=20.0):
    return np.random.default_rng(seed).normal(0, sd, n)


def test_detects_clean_positive_spikes_without_duplicates():
    x = _noise(10 * FS)
    times = [1, 3, 5, 7, 9]
    for t in times:
        _biphasic(x, int(t * FS))
    out = detect_spikes_barkmeier(x, FS)
    assert len(out) == len(times)
    got = sorted(d['peak_time'] for d in out)
    np.testing.assert_allclose(got, times, atol=0.03)


def test_detects_negative_going_spike():
    # regression: candidate detection must use |narrow band|, not narrow band > 0
    x = _noise(4 * FS, seed=1)
    _biphasic(x, int(2 * FS), amp=-600.0)
    assert len(detect_spikes_barkmeier(x, FS)) == 1


def test_pure_noise_false_positive_rate_is_bounded():
    # Block-scaling normalises the channel's median amplitude to `scale`, so on an
    # isolated single channel it inflates the noise floor and a fixed threshold fires at
    # a low but non-zero rate. This is a property of the Barkmeier method (its block
    # scaling is designed to be MULTICHANNEL); documented in the module. Here we only pin
    # that the rate stays low, not zero.
    out = detect_spikes_barkmeier(_noise(20 * FS, seed=2), FS)
    assert len(out) / 20.0 < 1.0   # fewer than 1 false positive per second


def test_high_threshold_rejects_everything_conjunctive_and():
    # the acceptance test is a single conjunctive AND: raising any one threshold out of
    # reach must reject the spike. A spurious "all-below" acceptance branch (the bug this
    # replaces) would instead resurrect it.
    x = _noise(4 * FS, seed=4)
    _biphasic(x, int(2 * FS), amp=600.0)
    huge = {'total_amp': 1e9, 'slope': DEFAULT_THRESHOLDS['slope'], 'half_dur': DEFAULT_THRESHOLDS['half_dur']}
    assert detect_spikes_barkmeier(x, FS, thresholds=huge) == []
    huge = {'total_amp': DEFAULT_THRESHOLDS['total_amp'], 'slope': 1e12, 'half_dur': DEFAULT_THRESHOLDS['half_dur']}
    assert detect_spikes_barkmeier(x, FS, thresholds=huge) == []


def test_refractory_is_in_seconds():
    # two discharges 120 ms apart: kept with no refractory, merged with a 300 ms one
    x = _noise(4 * FS, seed=3)
    _biphasic(x, int(1.0 * FS))
    _biphasic(x, int(1.0 * FS) + int(0.12 * FS))
    assert len(detect_spikes_barkmeier(x, FS, refractory=0.0)) == 2
    assert len(detect_spikes_barkmeier(x, FS, refractory=0.30)) == 1


def test_multichannel_block_scaling_preserves_amplitude_ratio():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 20, 6 * FS), rng.normal(0, 20, 6 * FS)])
    _biphasic(X[0], int(3 * FS), amp=800.0)
    _biphasic(X[1], int(3 * FS), amp=300.0)
    out = detect_spikes_barkmeier(X, FS)
    amp = {d['channel']: d['total_amp'] for d in out}
    assert 0 in amp and 1 in amp
    assert amp[0] / amp[1] == pytest.approx(800 / 300, rel=0.25)


def test_1d_and_2d_shapes():
    x = _noise(4 * FS, seed=6)
    _biphasic(x, int(2 * FS))
    out1d = detect_spikes_barkmeier(x, FS)
    assert all(d['channel'] == 0 for d in out1d)

    X = np.vstack([x, _noise(4 * FS, seed=7)])
    out2d = detect_spikes_barkmeier(X, FS)
    assert set(d['channel'] for d in out2d) <= {0, 1}


def test_spike_near_end_does_not_crash():
    x = _noise(4 * FS, seed=8)
    _biphasic(x, x.size - int(0.01 * FS))
    detect_spikes_barkmeier(x, FS)   # must not raise


def test_detection_fields_are_consistent():
    x = _noise(6 * FS, seed=9)
    _biphasic(x, int(3 * FS))
    d = detect_spikes_barkmeier(x, FS)[0]
    assert d['total_amp'] == pytest.approx(d['left_amp'] + d['right_amp'])
    assert d['left_slope'] == pytest.approx(d['left_amp'] / d['left_dur'])
    assert d['peak_time'] == pytest.approx(d['peak_index'] / FS)


@pytest.mark.parametrize('bad_band', [(0, 50), (20, 300), (50, 20)])
def test_invalid_band_raises(bad_band):
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(_noise(FS), FS, narrow_band=bad_band)


def test_non_1d_2d_input_raises():
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(np.zeros((2, 3, FS)), FS)


def test_default_thresholds_not_mutated_by_call():
    before = dict(DEFAULT_THRESHOLDS)
    detect_spikes_barkmeier(_noise(2 * FS), FS, thresholds={'total_amp': 1, 'slope': 1, 'half_dur': 0})
    assert DEFAULT_THRESHOLDS == before


def test_partial_thresholds_dict_merges_with_defaults():
    # review #61: a partial dict must fill from defaults, not raise a later KeyError
    x = _noise(6 * FS, seed=11)
    _biphasic(x, int(3 * FS))
    out = detect_spikes_barkmeier(x, FS, thresholds={'total_amp': 600.0})  # slope/half_dur from defaults
    assert isinstance(out, list)


def test_unknown_threshold_key_raises():
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(_noise(2 * FS), FS, thresholds={'TAMP': 600})  # wrong key name
