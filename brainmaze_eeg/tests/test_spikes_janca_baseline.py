"""
Janca reference baseline (JancaBaseline, baseline=/combine=) and gap-aware local statistics
(gap_aware_stats=), brainmaze_eeg 3.1.0. The original algorithm must stay the default and
unchanged; that is covered here and by the eeg_forge parity tests in test_spikes_janca.py.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from brainmaze_eeg.spikes import (GapAwareSpikeDetector, JancaBaseline, JancaDetector,
                                  detect_spikes_janca, janca_threshold)
from brainmaze_eeg.spikes.janca import _masked_log_stats
from brainmaze_eeg.spikes.janca_baseline import BASELINE_SCHEMA_VERSION
from brainmaze_eeg.tests.spike_synth import dense_ieeg, pink_background, synth_ieeg

FS = 500
_REF = np.load(Path(__file__).parent / 'data' / 'janca_eeg_forge_reference.npz')


def _hits(det, truth, tol):
    det = np.asarray(det)
    return sum(bool(np.any(np.abs(det - t) <= tol)) for t in truth)


def _false(det, truth, tol):
    return sum(not np.any(np.abs(truth - d) <= tol) for d in det)


@pytest.fixture(scope='module')
def background():
    """300 s of spike-free background (the reference) at 500 Hz."""
    return dense_ieeg(FS, dur=300.0, rate=0, seed=100)[0]


@pytest.fixture(scope='module')
def baseline(background):
    return JancaBaseline.from_signal(background, FS, units='uV')


# ============================================================ the default is unchanged
@pytest.mark.parametrize('key', sorted(k for k in _REF.files if k.startswith('fs')))
def test_explicit_defaults_equal_eeg_forge_fixture(key):
    fs, seed = (int(v) for v in key[2:].split('_seed'))
    x, _ = synth_ieeg(fs, dur=float(_REF['dur']), seed=seed)
    out = detect_spikes_janca(x, fs, baseline=None, combine='reference',
                              broadcast_baseline=False, gap_aware_stats=False)
    np.testing.assert_array_equal(out, _REF[key])
    np.testing.assert_array_equal(JancaDetector().detect(x[None], fs)[0], _REF[key])


def test_default_details_keys_unchanged():
    x, _ = synth_ieeg(FS, dur=30.0, seed=0)
    _, d = detect_spikes_janca(x, FS, return_details=True)
    assert set(d) == {'fs_analysis', 'up', 'down', 'envelope', 'threshold', 'filters',
                      'preset', 'params'}


@pytest.mark.parametrize('key', sorted(k for k in _REF.files if k.startswith('fs')))
def test_gap_aware_stats_without_gaps_equals_original(key):
    """No invalid samples: same detections, threshold equal to rounding (cumsum vs filter)."""
    fs, seed = (int(v) for v in key[2:].split('_seed'))
    x, _ = synth_ieeg(fs, dur=float(_REF['dur']), seed=seed)
    d0, t0 = detect_spikes_janca(x, fs, return_details=True)
    d1, t1 = detect_spikes_janca(x, fs, return_details=True, gap_aware_stats=True)
    np.testing.assert_array_equal(d1, _REF[key])
    np.testing.assert_allclose(t1['threshold'], t0['threshold'], rtol=1e-10)
    assert t1['stats_valid'].all()


# ============================================================ dense spiking: the use case
@pytest.mark.parametrize('rate, local_max, ref_min', [(1, 0.99, 0.95), (3, 0.6, 0.95),
                                                      (5, 0.1, 0.95)])
def test_reference_baseline_detects_permanent_spiking(baseline, rate, local_max, ref_min):
    """The prototype's evidence (local 93/42/0.5 %, reference 97/97/96 %, 0 false) with
    fixed seeds: the local model loses the spikes as the rate grows, the reference does not."""
    x, pos = dense_ieeg(FS, dur=300.0, rate=rate, amp=150.0, seed=rate)
    tol = 0.05 * FS
    local = detect_spikes_janca(x, FS)
    ref = detect_spikes_janca(x, FS, baseline=baseline)
    assert _hits(local, pos, tol) / pos.size <= local_max
    assert _hits(ref, pos, tol) / pos.size >= ref_min
    assert _false(ref, pos, tol) == 0
    assert _false(local, pos, tol) == 0


def test_reference_baseline_gives_no_detections_on_background(baseline):
    x, _ = dense_ieeg(FS, dur=300.0, rate=0, seed=7)          # new background, no spikes
    assert detect_spikes_janca(x, FS, baseline=baseline).size <= 1


# ============================================================ combine semantics
def test_combine_threshold_curves(baseline):
    x, _ = dense_ieeg(FS, dur=60.0, rate=2, seed=3)
    _, loc = detect_spikes_janca(x, FS, return_details=True)
    t_ref = float(janca_threshold(baseline.mu[0], baseline.sd[0], 3.65))
    for mode, f in (('reference', None), ('min', np.minimum), ('max', np.maximum)):
        _, d = detect_spikes_janca(x, FS, baseline=baseline, combine=mode, return_details=True)
        assert d['threshold_reference'] == pytest.approx(t_ref, rel=1e-15)
        assert d['combine'] == mode
        if f is None:
            assert d['threshold_local'] is None
            np.testing.assert_array_equal(d['threshold'], np.full(x.size // 2, t_ref))
        else:
            np.testing.assert_array_equal(d['threshold_local'], loc['threshold'])
            np.testing.assert_array_equal(d['threshold'], f(loc['threshold'], t_ref))


def test_combine_min_max_order(baseline):
    """'min' detects at least what either model detects alone, 'max' at most."""
    x, pos = dense_ieeg(FS, dur=300.0, rate=2, seed=11)
    n = {m: detect_spikes_janca(x, FS, baseline=baseline, combine=m).size
         for m in ('reference', 'min', 'max')}
    n['local'] = detect_spikes_janca(x, FS).size
    assert n['min'] >= max(n['reference'], n['local'])
    assert n['max'] <= min(n['reference'], n['local'])


def test_threshold_multiplier_is_free(baseline):
    x, _ = dense_ieeg(FS, dur=60.0, rate=2, seed=3)
    _, d = detect_spikes_janca(x, FS, baseline=baseline, threshold=5.0, min_distance_s=0.2,
                               window_s=20.0, return_details=True)
    assert d['threshold_reference'] == pytest.approx(float(baseline.threshold(5.0)[0]))


# ============================================================ from_signal: gaps excluded
def _with_drops(x, fs, value=np.nan):
    """x with three dropouts (start, 20 s inside, end) set to ``value``; and the kept runs."""
    y = x.copy()
    spans = [(0, int(3.0 * fs)), (int(100 * fs), int(120 * fs)), (int(250 * fs), int(251 * fs)),
             (x.size - int(2 * fs), x.size)]
    for a, b in spans:
        y[a:b] = value
    keep, prev = [], 0
    for a, b in spans:
        if a > prev:
            keep.append((prev, a))
        prev = b
    return y, keep


@pytest.mark.parametrize('fill', [np.nan, np.inf, 0.0, 1e6])
def test_baseline_excludes_gaps_and_ignores_their_values(background, fill):
    """A baseline from data with drops (NaN, inf, or a constant of any value) equals the
    baseline from the same data cut into the valid runs, and is close to the baseline of the
    data without drops."""
    y, keep = _with_drops(background, FS, fill)
    b_drop = JancaBaseline.from_signal(y, FS)
    b_runs = JancaBaseline.from_signal([background[a:b] for a, b in keep], FS)
    b_full = JancaBaseline.from_signal(background, FS)
    np.testing.assert_array_equal(b_drop.mu, b_runs.mu)
    np.testing.assert_array_equal(b_drop.sd, b_runs.sd)
    np.testing.assert_array_equal(b_drop.valid_s, b_runs.valid_s)
    assert abs(b_drop.mu[0] - b_full.mu[0]) < 0.02
    assert abs(b_drop.sd[0] - b_full.sd[0]) < 0.02
    assert b_drop.threshold()[0] == pytest.approx(b_full.threshold()[0], rel=0.02)


def test_baseline_valid_time_accounts_for_margins(background):
    y, keep = _with_drops(background, FS)
    for margin in (0.0, 0.5, 2.0):
        b = JancaBaseline.from_signal(y, FS, stats_margin_s=margin)
        fa, m = b.fs_analysis, int(np.ceil(margin * b.fs_analysis - 1e-9))
        expect = sum(-(-(bb - a) // 2) - 2 * m for a, bb in keep) / fa   # q = 2 at 500 Hz
        assert b.valid_s[0] == pytest.approx(expect, abs=1e-12)
        assert b.info['n_runs'] == [len(keep)]


def test_baseline_margin_removes_transients_at_dc_steps(background):
    """Runs with large DC offsets (amplifier re-settling after a dropout): with the margin
    the statistics match those of the offset-free data."""
    y, keep = _with_drops(background, FS)
    for k, (a, b) in enumerate(keep):
        y[a:b] += 5000.0 * (-1) ** k + 20.0 * np.linspace(0, 1, b - a)
    b_off = JancaBaseline.from_signal(y, FS)
    b_ref = JancaBaseline.from_signal(_with_drops(background, FS)[0], FS)
    assert b_off.threshold()[0] == pytest.approx(b_ref.threshold()[0], rel=0.01)


def test_baseline_segments_select_intervals(background):
    segs = [(10.0, 70.0), (150.0, 200.5)]
    b = JancaBaseline.from_signal(background, FS, segments=segs)
    cut = JancaBaseline.from_signal([background[int(a * FS):int(b_ * FS)] for a, b_ in segs], FS)
    np.testing.assert_array_equal(b.mu, cut.mu)
    np.testing.assert_array_equal(b.sd, cut.sd)
    with pytest.raises(ValueError, match='outside'):
        JancaBaseline.from_signal(background, FS, segments=[(250.0, 400.0)])
    with pytest.raises(ValueError, match='list of segments'):
        JancaBaseline.from_signal([background], FS, segments=segs)


def test_baseline_segments_of_different_lengths_and_channels():
    a = np.vstack([pink_background(FS * 40, FS, 30.0, np.random.default_rng(s)) for s in (1, 2)])
    b = np.vstack([pink_background(FS * 75, FS, 30.0, np.random.default_rng(s)) for s in (3, 4)])
    bl = JancaBaseline.from_signal([a, b], FS, min_valid_s=0, channel_names=['A1', 'A2'])
    assert bl.n_channels == 2 and bl.channel_names == ('A1', 'A2')
    np.testing.assert_allclose(bl.valid_s, [(40 - 1) + (75 - 1)] * 2)
    one = JancaBaseline.from_signal([a[1], b[1]], FS, min_valid_s=0)
    assert one.mu[0] == bl.mu[1] and one.sd[0] == bl.sd[1]
    with pytest.raises(ValueError, match='channel'):
        JancaBaseline.from_signal([a, b[:1]], FS)


def test_baseline_warns_on_short_reference_and_raises_without_any(background):
    with pytest.warns(UserWarning, match='min_valid_s'):
        b = JancaBaseline.from_signal(background[:FS * 20], FS)
    assert b.valid_s[0] == pytest.approx(19.0)
    x = np.vstack([background[:FS * 100], np.full(FS * 100, np.nan)])
    with pytest.raises(ValueError, match=r'channel\(s\) \[1\]'):
        JancaBaseline.from_signal(x, FS)
    with pytest.raises(ValueError, match='no valid reference'):
        JancaBaseline.from_signal(np.zeros(FS * 100), FS)     # constant: all missing
    with pytest.raises(ValueError, match='no valid reference'):   # runs too short
        JancaBaseline.from_signal(np.where(np.arange(FS * 100) % FS < 10, np.nan,
                                           background[:FS * 100]), FS)


def test_robust_statistic(background):
    """Robust (median, 1.4826 MAD) is a different statistic: the log of a band-passed noise
    envelope is skewed, so on clean background its threshold is ~14 % higher than mean/SD
    (measured: scratch/janca-baseline/robust_probe.out); it is pulled up less by spikes in
    the reference (3 spikes/s: x1.31 vs x1.44)."""
    b_mean = JancaBaseline.from_signal(background, FS)
    b_rob = JancaBaseline.from_signal(background, FS, robust=True)
    assert b_rob.info['statistic'] == 'robust' and b_mean.info['statistic'] == 'mean'
    assert 1.05 < b_rob.threshold()[0] / b_mean.threshold()[0] < 1.25
    spiky, _ = dense_ieeg(FS, dur=300.0, rate=3, amp=150.0, seed=100)
    t_mean = JancaBaseline.from_signal(spiky, FS).threshold()[0] / b_mean.threshold()[0]
    t_rob = (JancaBaseline.from_signal(spiky, FS, robust=True).threshold()[0]
             / b_rob.threshold()[0])
    assert 1 < t_rob < t_mean


def test_prototype_statistic_definition(background):
    """mu/sd are the mean and population SD of log(e + eps_rel * median(e)) over the valid
    envelope (stats_margin_s=0, no gaps: exactly the prototype's reference_stats)."""
    _, d = detect_spikes_janca(background, FS, return_details=True)
    e = d['envelope']
    L = np.log(e + 1e-6 * np.median(e))
    b = JancaBaseline.from_signal(background, FS, stats_margin_s=0)
    assert b.mu[0] == pytest.approx(L.mean(), rel=1e-12)
    assert b.sd[0] == pytest.approx(L.std(), rel=1e-12)


# ============================================================ matching rules
@pytest.mark.parametrize('kw, name', [
    (dict(band=(10, 50)), 'band'), (dict(filter_order=4), 'filter_order'),
    (dict(powerline=60), 'powerline'), (dict(notch_width=4.0), 'notch_width'),
    (dict(notch_order=2), 'notch_order'), (dict(notch_harmonics=2), 'notch_harmonics'),
    (dict(target_fs=250), 'target_fs'), (dict(decimation='exact'), 'decimation'),
    (dict(eps_rel=1e-5), 'eps_rel')])
def test_signal_path_mismatch_raises(baseline, kw, name):
    x, _ = dense_ieeg(FS, dur=30.0, rate=1, seed=0)
    with pytest.raises(ValueError, match=name):
        detect_spikes_janca(x, FS, baseline=baseline, **kw)
    with pytest.raises(ValueError, match=name):
        JancaDetector(baseline=baseline, **kw)


def test_mismatch_message_names_every_parameter(baseline):
    x, _ = dense_ieeg(FS, dur=30.0, rate=1, seed=0)
    with pytest.raises(ValueError) as err:
        detect_spikes_janca(x, FS, baseline=baseline, band=(12, 60), powerline=60, eps_rel=0)
    for name in ('band', 'powerline', 'eps_rel'):
        assert name in str(err.value)


def test_analysis_rate_mismatch_raises(baseline):
    x, _ = dense_ieeg(1000, dur=30.0, rate=1, seed=0)      # 1000 Hz -> 200 Hz analysis
    with pytest.raises(ValueError, match='fs_analysis'):
        detect_spikes_janca(x, 1000, baseline=baseline)
    det = JancaDetector(baseline=baseline)                 # construction OK (no fs yet)
    with pytest.raises(ValueError, match='fs_analysis'):
        det.detect(x[None], 1000)


def test_exact_decimation_shares_analysis_rate_across_input_rates():
    """decimation='exact': a baseline from a 512 Hz recording serves a 1000 Hz one (both
    resampled from the same 2 kHz 'analog' process)."""
    from scipy.signal import resample_poly
    ref2k, _ = dense_ieeg(2000, dur=120.0, rate=0, seed=1)
    x2k, pos2k = dense_ieeg(2000, dur=60.0, rate=3, seed=2)
    b = JancaBaseline.from_signal(resample_poly(ref2k, 32, 125), 512, decimation='exact')
    assert b.fs_analysis == pytest.approx(200.0)
    d = detect_spikes_janca(resample_poly(x2k, 1, 2), 1000, baseline=b, decimation='exact')
    pos = pos2k // 2
    assert _hits(d, pos, 50) / pos.size > 0.95
    assert _false(d, pos, 50) == 0
    with pytest.raises(ValueError, match='fs_analysis'):          # 'integer': 256 vs 200 Hz
        detect_spikes_janca(resample_poly(x2k, 1, 2), 1000,
                            baseline=JancaBaseline.from_signal(resample_poly(ref2k, 32, 125),
                                                               512))


def test_notch_settings_ignored_without_powerline(background):
    b = JancaBaseline.from_signal(background[:FS * 100], FS, powerline=None)
    x, _ = dense_ieeg(FS, dur=30.0, rate=1, seed=0)
    detect_spikes_janca(x, FS, baseline=b, powerline=None, notch_width=3.0)


def test_powerline_none_vs_set_mismatch(background):
    b = JancaBaseline.from_signal(background[:FS * 100], FS, powerline=None)
    with pytest.raises(ValueError, match='powerline'):
        JancaDetector(baseline=b)


def test_channel_count_rules(baseline):
    x2 = np.vstack([dense_ieeg(FS, dur=30.0, rate=1, seed=s)[0] for s in (0, 1)])
    with pytest.raises(ValueError, match='broadcast_baseline'):
        detect_spikes_janca(x2, FS, baseline=baseline)
    out = detect_spikes_janca(x2, FS, baseline=baseline, broadcast_baseline=True)
    for c in range(2):
        np.testing.assert_array_equal(out[c], detect_spikes_janca(x2[c], FS, baseline=baseline))
    b3 = JancaBaseline(mu=[2.5, 2.5, 2.5], sd=[0.6] * 3, fs=FS)
    with pytest.raises(ValueError, match='3 channel'):
        detect_spikes_janca(x2, FS, baseline=b3, broadcast_baseline=True)   # never broadcast
    np.testing.assert_array_equal(
        detect_spikes_janca(x2, FS, baseline=b3.select([0, 2]))[1],
        detect_spikes_janca(x2[1], FS, baseline=b3.select(2)))


def test_options_that_would_be_ignored_raise(baseline):
    x, _ = dense_ieeg(FS, dur=30.0, rate=1, seed=0)
    for kw in (dict(combine='max'), dict(broadcast_baseline=True),
               dict(valid=np.ones(x.size, bool)), dict(min_valid_fraction=0.3),
               dict(stats_margin_s=1.0)):
        with pytest.raises(ValueError):
            detect_spikes_janca(x, FS, **kw)
    with pytest.raises(ValueError):
        detect_spikes_janca(x, FS, baseline=baseline, combine='mean')
    with pytest.raises(TypeError):
        detect_spikes_janca(x, FS, baseline={'mu': 1, 'sd': 1})
    with pytest.raises(ValueError):
        JancaDetector(combine='min')
    with pytest.raises(ValueError, match='gap_aware_stats'):
        JancaDetector().detect(x[None], FS, valid=np.ones((1, x.size), bool))


def test_units_mismatch_warns(baseline):
    x, _ = dense_ieeg(FS, dur=60.0, rate=1, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _, d = detect_spikes_janca(x, FS, baseline=baseline, return_details=True)
    assert 0.5 < d['level_ratio'] < 2
    with pytest.warns(UserWarning, match='unit'):
        detect_spikes_janca(x * 1e-6, FS, baseline=baseline)       # volts vs uV


# ============================================================ manual baseline, persistence
def test_manual_baseline_equals_measured(baseline):
    m = JancaBaseline(mu=baseline.mu, sd=baseline.sd, fs=FS)
    assert m.fs_analysis == baseline.fs_analysis == 250.0
    x, _ = dense_ieeg(FS, dur=60.0, rate=3, seed=4)
    np.testing.assert_array_equal(detect_spikes_janca(x, FS, baseline=m),
                                  detect_spikes_janca(x, FS, baseline=baseline))
    m2 = JancaBaseline(mu=float(baseline.mu[0]), sd=float(baseline.sd[0]), fs_analysis=250.0)
    np.testing.assert_array_equal(m2.threshold(), baseline.threshold())


def test_manual_baseline_validation():
    with pytest.raises(TypeError, match='fs'):
        JancaBaseline(mu=2.0, sd=0.5)
    with pytest.raises(ValueError, match='fs_analysis'):
        JancaBaseline(mu=2.0, sd=0.5, fs=500, fs_analysis=200)
    with pytest.raises(ValueError, match='finite'):
        JancaBaseline(mu=np.nan, sd=0.5, fs=500)
    with pytest.raises(ValueError, match='>= 0'):
        JancaBaseline(mu=2.0, sd=-0.5, fs=500)
    with pytest.raises(ValueError, match='Nyquist'):
        JancaBaseline(mu=2.0, sd=0.5, fs_analysis=100)
    with pytest.raises(ValueError, match='channel_names'):
        JancaBaseline(mu=[1, 2], sd=[1, 1], fs=500, channel_names=['a'])
    b = JancaBaseline(mu=[1.0, 2.0], sd=0.5, fs=500)
    assert b.n_channels == 2 and b.sd.tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        b.mu[0] = 3.0                                       # read-only


def test_envelope_levels(baseline):
    lv = baseline.envelope_levels(threshold=3.65)
    assert lv['median'][0] == pytest.approx(np.exp(baseline.mu[0]))
    assert lv['mode'][0] == pytest.approx(np.exp(baseline.mu[0] - baseline.sd[0] ** 2))
    assert lv['threshold'][0] == pytest.approx(3.65 * (lv['mode'][0] + lv['median'][0]))
    # in the input unit: about 1.06 x the SD of the band-passed background
    from scipy.signal import sosfiltfilt
    from brainmaze_eeg.spikes import design_janca_filters
    f = design_janca_filters(FS)
    y = sosfiltfilt(f['bandpass'], baseline_bg := dense_ieeg(FS, 300.0, rate=0, seed=100)[0])
    y = sosfiltfilt(f['notches'][0][1], y)
    assert lv['median'][0] / y.std() == pytest.approx(1.06, abs=0.06)
    assert baseline_bg.size == 300 * FS


def test_save_load_round_trip(tmp_path, baseline):
    b = JancaBaseline.from_signal(np.vstack([baseline_signal(s) for s in (1, 2)]), FS,
                                  channel_names=['LA1', 'LA2'], units='uV', powerline=60)
    path = tmp_path / 'b.json'
    b.save(path)
    c = JancaBaseline.load(path)
    assert c == b
    np.testing.assert_array_equal(c.mu, b.mu)          # floats exact
    assert c.params == b.params and c.channel_names == b.channel_names and c.units == 'uV'
    assert c.info['statistic'] == 'mean' and c.fs == FS
    x = np.vstack([dense_ieeg(FS, dur=60.0, rate=3, seed=s)[0] for s in (5, 6)])
    for k, v in zip(detect_spikes_janca(x, FS, baseline=c, powerline=60),
                    detect_spikes_janca(x, FS, baseline=b, powerline=60)):
        np.testing.assert_array_equal(k, v)
    d = json.loads(path.read_text())
    assert d['schema_version'] == BASELINE_SCHEMA_VERSION == 1
    assert d['format'] == 'brainmaze_eeg.spikes.JancaBaseline'
    assert d['params']['band'] == [10.0, 60.0] and d['params']['powerline'] == 60.0
    d['schema_version'] = 2
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='newer'):
        JancaBaseline.load(path)
    d['schema_version'] = 1
    d['format'] = 'something else'
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='not a JancaBaseline'):
        JancaBaseline.load(path)


def baseline_signal(seed, dur=120.0):
    return pink_background(int(dur * FS), FS, 30.0, np.random.default_rng(seed))


def test_repr_and_select(baseline):
    b = JancaBaseline(mu=[1.0, 2.0, 3.0], sd=[0.1, 0.2, 0.3], fs=FS, powerline=60,
                      channel_names=['a', 'b', 'c'], units='uV')
    r = repr(b)
    for s in ('n_channels=3', 'powerline=60.0', "units='uV'", 'fs_analysis=250'):
        assert s in r
    s = b.select(['c', 'a'])
    assert s.mu.tolist() == [3.0, 1.0] and s.channel_names == ('c', 'a')
    assert b.select(1).mu.tolist() == [2.0]
    with pytest.raises(KeyError):
        b.select(['z'])
    assert 'JancaDetector(' in repr(JancaDetector(baseline=baseline, combine='max'))


# ============================================================ gap-aware local statistics
def _brute_masked(L, ok, W, frac):
    """Direct per-sample evaluation of the masked statistics (reflect at the ends)."""
    n, h = L.size, W // 2

    def refl(i):
        i %= 2 * n
        return i if i < n else 2 * n - 1 - i

    idx = [np.array([refl(j) for j in range(k - h, k + h + 1)]) for k in range(n)]
    mu = np.array([L[i[ok[i]]].mean() if ok[i].any() else np.nan for i in idx])
    sd, cnt = np.empty(n), np.empty(n)
    for k, i in enumerate(idx):
        sel = i[ok[i]]
        cnt[k] = sel.size
        sd[k] = np.sqrt(np.mean((L[sel] - mu[sel]) ** 2)) if sel.size else np.nan
    low = cnt < frac * W
    mu[low] = np.nan
    sd[low] = np.nan
    return mu, sd


@pytest.mark.parametrize('W', [3, 51, 301, 1001])
@pytest.mark.parametrize('frac', [0.2, 0.5, 1.0])
def test_masked_stats_match_brute_force(W, frac):
    rng = np.random.default_rng(W)
    n = 400
    L = rng.normal(2.5, 0.7, n)
    ok = np.ones(n, bool)
    ok[50:120] = False
    ok[200:205] = False
    ok[rng.integers(0, n, 30)] = False
    mu, sd = _masked_log_stats(L, ok, W, frac)
    bm, bs = _brute_masked(L, ok, W, frac)
    np.testing.assert_array_equal(np.isnan(mu), np.isnan(bm))
    np.testing.assert_allclose(mu, bm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, bs, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('W', [3, 51, 1001])
def test_masked_stats_without_gaps_equal_uniform_filter(W):
    L = np.random.default_rng(1).normal(2.5, 0.7, 400)
    mu, sd = _masked_log_stats(L, np.ones(L.size, bool), W, 0.5)
    m0 = uniform_filter1d(L, W, mode='reflect')
    s0 = np.sqrt(uniform_filter1d((L - m0) ** 2, W, mode='reflect'))
    np.testing.assert_allclose(mu, m0, rtol=1e-12)
    np.testing.assert_allclose(sd, s0, rtol=1e-10)


def test_gap_aware_stats_long_window_with_dropout():
    """60 s window, 70 s dropout (stored as a constant): the threshold is NaN where less than
    half a window is valid, there are no detections there, the dropout never enters the
    statistics, and the result matches the brute-force masked statistics of the envelope."""
    x, truth = synth_ieeg(FS, dur=300.0, seed=2, n_spikes=60)
    x[100 * FS:170 * FS] = 0.0
    det, d = detect_spikes_janca(x, FS, window_s=60.0, gap_aware_stats=True,
                                 return_details=True)
    fa, ok, thr = d['fs_analysis'], d['stats_valid'], d['threshold']
    m = int(np.ceil(0.5 * fa))
    assert not ok[int(100 * fa) - m:int(170 * fa) + m].any()
    assert ok[:int(100 * fa) - m].all() and ok[int(170 * fa) + m:].all()
    undefined = np.isnan(thr)
    assert undefined[int(135 * fa)] and not undefined[int(50 * fa)]
    assert not np.any(undefined[np.round(det / 2).astype(int)])
    # NaN exactly where fewer than half of the window's samples are valid (brute force on a
    # grid of centres; the values themselves: test_gap_aware_threshold_equals_masked_stats...)
    W = int(60 * fa) + 1
    for k in range(0, thr.size, 997):
        win = np.arange(k - W // 2, k + W // 2 + 1)
        win = np.where(win < 0, -win - 1, win)
        win = np.where(win >= thr.size, 2 * thr.size - win - 1, win)
        assert np.isnan(thr[k]) == (ok[win].sum() < 0.5 * W)
    # original statistics: the constant dropout drags the background down -> many detections
    assert detect_spikes_janca(x, FS, window_s=60.0).size > 3 * det.size
    assert _false(det, truth, 25) == 0


def test_gap_aware_threshold_equals_masked_stats_of_envelope():
    x, _ = synth_ieeg(FS, dur=200.0, seed=5)
    x[60 * FS:75 * FS] = -3.0
    _, d = detect_spikes_janca(x, FS, window_s=30.0, gap_aware_stats=True, return_details=True)
    e, ok = d['envelope'], d['stats_valid']
    L = np.log(e + 1e-6 * np.median(e[ok]))
    W = int(30 * d['fs_analysis']) + 1
    mu, sd = _masked_log_stats(L, ok, W, 0.5)
    np.testing.assert_array_equal(d['threshold'], 3.65 * (np.exp(mu - sd ** 2) + np.exp(mu)))
    np.testing.assert_array_equal(d['threshold_local'], d['threshold'])


def test_gap_aware_valid_mask_and_parameters():
    x, _ = synth_ieeg(FS, dur=120.0, seed=6)
    valid = np.ones(x.size, bool)
    valid[30 * FS:31 * FS] = False
    _, d = detect_spikes_janca(x, FS, gap_aware_stats=True, valid=valid, stats_margin_s=1.0,
                               return_details=True)
    ok = d['stats_valid']
    assert (~ok).sum() == (1 + 2 * 1.0) * 250
    with pytest.raises(ValueError, match='shape'):
        detect_spikes_janca(x, FS, gap_aware_stats=True, valid=valid[:-1])
    with pytest.raises(TypeError, match='boolean'):
        detect_spikes_janca(x, FS, gap_aware_stats=True, valid=valid.astype(int))
    with pytest.raises(ValueError, match='min_valid_fraction'):
        detect_spikes_janca(x, FS, gap_aware_stats=True, min_valid_fraction=0)
    # fewer valid samples than required anywhere -> no detections at all
    _, d = detect_spikes_janca(x, FS, gap_aware_stats=True, valid=valid, window_s=2.0,
                               min_valid_fraction=1.0, return_details=True)
    assert np.isnan(d['threshold'][int(30.5 * 250)])


def test_gap_aware_combine_where_local_undefined(baseline):
    """Local threshold undefined (window coverage too low, but the sample itself usable):
    'min' falls back to the reference (np.fmin: never less sensitive than 'reference'),
    'max' stays undefined; samples excluded from the statistics are undefined in every mode."""
    x, pos = dense_ieeg(FS, dur=200.0, rate=2, seed=8)
    x[50 * FS:120 * FS] = 0.0
    valid = np.ones(x.size, bool)
    valid[50 * FS:120 * FS] = False
    x[50 * FS:120 * FS] = dense_ieeg(FS, dur=70.0, rate=2, seed=18)[0]   # finite, excluded
    out = {}
    for mode in ('reference', 'min', 'max'):
        out[mode] = detect_spikes_janca(x, FS, baseline=baseline, combine=mode, window_s=60.0,
                                        gap_aware_stats=True, valid=valid, stats_margin_s=0.0,
                                        min_valid_fraction=0.8, return_details=True)
    fa = out['min'][1]['fs_analysis']
    t_ref = out['min'][1]['threshold_reference']
    k = int(85 * fa)                          # inside the excluded run: undefined everywhere
    for mode in out:
        assert np.isnan(out[mode][1]['threshold'][k])
    loc = out['min'][1]['threshold_local']
    j = np.flatnonzero(np.isnan(loc) & out['min'][1]['stats_valid'])
    assert j.size > 0                         # usable samples where the local model is undefined
    assert np.all(out['min'][1]['threshold'][j] == t_ref)
    assert np.all(np.isnan(out['max'][1]['threshold'][j]))
    thr_min, thr_ref = out['min'][1]['threshold'], out['reference'][1]['threshold']
    both = ~np.isnan(thr_ref)
    np.testing.assert_array_equal(np.isnan(thr_min), np.isnan(thr_ref))
    assert np.all(thr_min[both] <= thr_ref[both])
    assert set(out['reference'][0].tolist()) <= set(out['min'][0].tolist())
    for mode in out:                          # never inside the excluded run
        d = out[mode][0]
        assert not np.any((d >= 50 * FS) & (d < 120 * FS))


@pytest.mark.filterwarnings('ignore:window_s=600 s')
@pytest.mark.parametrize('dc', [500.0, 5000.0])
@pytest.mark.parametrize('window_s', [30.0, 60.0, 600.0])
def test_gap_aware_dc_offset_zero_dropouts_no_false_detections(dc, window_s):
    """Review R1: zero dropouts in a signal with a DC offset (DC-coupled amplifier, raw ADC
    counts) step by the offset at both edges; the filter transient there must not be
    detected (those samples are excluded from the statistics and from detection)."""
    x = dense_ieeg(FS, dur=600.0, rate=0, seed=0)[0] + dc
    rng = np.random.default_rng(3)
    for a in rng.integers(10 * FS, x.size - 30 * FS, 20):
        x[a:a + rng.integers(FS, 10 * FS)] = 0.0
    det, d = detect_spikes_janca(x, FS, window_s=window_s, gap_aware_stats=True,
                                 return_details=True)
    assert det.size == 0
    # without the fix the transients were detected (the original path: hundreds)
    assert detect_spikes_janca(x, FS, window_s=window_s).size > 100


# ============================================================ through GapAwareSpikeDetector
class _Spy(JancaDetector):
    def detect(self, x, fs, **kw):
        self.calls = getattr(self, 'calls', []) + [kw]
        return super().detect(x, fs, **kw)


def test_wrapper_passes_gap_mask_only_with_gap_aware_stats():
    x, _ = synth_ieeg(FS, dur=60.0, seed=1)
    x[10 * FS:12 * FS] = np.nan
    plain = _Spy()
    GapAwareSpikeDetector(plain).detect(x, FS)
    assert plain.calls == [{}]                       # default: protocol call unchanged
    spy = _Spy(gap_aware_stats=True)
    GapAwareSpikeDetector(spy).detect(x, FS)
    (kw,) = spy.calls
    v = kw['valid']
    assert v.shape == (1, x.size) and not v[0, 10 * FS:12 * FS].any() and v.sum() == x.size - 2 * FS


def test_wrapper_filled_samples_never_enter_statistics():
    """Two different fills of the same gaps: the samples in the statistics are the same and
    exclude every filled sample (+ margin); the thresholds agree away from the gaps."""
    x, _ = synth_ieeg(FS, dur=120.0, seed=3)
    x[40 * FS:46 * FS] = np.nan
    x[80 * FS:80 * FS + 50] = np.nan
    det = JancaDetector(window_s=30.0, gap_aware_stats=True)
    seen = {}
    for fill in ('mirror', 'pink'):
        class Grab(JancaDetector):
            def detect(self, y, fs, **kw):
                seen[fill] = detect_spikes_janca(y[0], fs, window_s=30.0, gap_aware_stats=True,
                                                 valid=kw['valid'][0], return_details=True)[1]
                return super().detect(y, fs, **kw)
        GapAwareSpikeDetector(Grab(window_s=30.0, gap_aware_stats=True), fill=fill).detect(x, FS)
    a, b = seen['mirror'], seen['pink']
    np.testing.assert_array_equal(a['stats_valid'], b['stats_valid'])
    ok = a['stats_valid']
    assert not ok[40 * 250 - 125:46 * 250 + 125].any()
    assert not ok[80 * 250 - 125:80 * 250 + 25 + 125].any()
    np.testing.assert_allclose(a['threshold'][ok], b['threshold'][ok], rtol=2e-3)
    assert repr(det).endswith('gap_aware_stats=True)')


@pytest.mark.filterwarnings(r'ignore:channel\(s\) \[0\] contain no usable sample')
def test_wrapper_with_baseline_keeps_channel_alignment():
    """Per-channel baselines through the wrapper: channels fed singly, an all-missing channel
    left out; every channel must still use its own baseline row."""
    scales = np.array([1.0, 10.0, 0.1])
    ref = np.vstack([dense_ieeg(FS, dur=120.0, rate=0, seed=20 + k)[0] * sc
                     for k, sc in enumerate(scales)])
    b = JancaBaseline.from_signal(ref, FS)
    x = np.vstack([dense_ieeg(FS, dur=60.0, rate=3, seed=s)[0] * sc
                   for s, sc in zip((1, 2, 3), scales)])
    x[0] = np.nan                                     # channel 0 entirely missing
    x[2, 20 * FS:21 * FS] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)   # no level warning: rows aligned
        warnings.simplefilter('ignore', RuntimeWarning)
        out = GapAwareSpikeDetector(JancaDetector(baseline=b)).detect(x, FS)
    assert out[0].size == 0
    direct = detect_spikes_janca(x[1], FS, baseline=b.select(1))
    np.testing.assert_array_equal(out[1], direct)
    assert out[2].size > 100
    with pytest.raises(ValueError, match='2 channel'):
        GapAwareSpikeDetector(JancaDetector(baseline=b.select([0, 1]))).detect(x, FS)
    one = b.select(1)
    with pytest.raises(ValueError, match='broadcast_baseline'):
        GapAwareSpikeDetector(JancaDetector(baseline=one)).detect(x[1:], FS)
    out_b = GapAwareSpikeDetector(JancaDetector(baseline=one, broadcast_baseline=True)).detect(
        x[1:2], FS)
    np.testing.assert_array_equal(out_b[0], direct)


def test_wrapper_with_baseline_and_gap_aware_stats():
    x, pos = dense_ieeg(FS, dur=200.0, rate=4, seed=9)
    x[50 * FS:60 * FS] = np.nan
    ref, _ = dense_ieeg(FS, dur=200.0, rate=0, seed=10)
    ref[30 * FS:35 * FS] = np.nan
    b = JancaBaseline.from_signal(ref, FS)
    det = GapAwareSpikeDetector(JancaDetector(baseline=b, combine='min', gap_aware_stats=True,
                                              window_s=60.0))
    out = det.detect(x, FS)
    keep = (pos < 50 * FS - 0.3 * FS) | (pos > 60 * FS + 0.3 * FS)
    assert _hits(out, pos[keep], 25) / keep.sum() > 0.95
    assert _false(out, pos, 25) == 0


# ============================================================ performance
def test_masked_stats_are_linear_time_for_long_windows():
    """24 h at 500 Hz -> 21.6 M analysis samples at 250 Hz; window_s = 3600 s. The sliding
    statistics must not depend on the window length (cumulative sums)."""
    n = 24 * 3600 * 250 // 8                  # 3 h of analysis samples keeps CI memory low
    rng = np.random.default_rng(0)
    L = rng.normal(2.5, 0.6, n)
    ok = np.ones(n, bool)
    for a in rng.integers(0, n - 250 * 600, 30):
        ok[a:a + rng.integers(250, 250 * 600)] = False
    t = time.perf_counter()
    _masked_log_stats(L, ok, 3600 * 250 + 1, 0.5)
    t_long = time.perf_counter() - t
    t = time.perf_counter()
    _masked_log_stats(L, ok, 5 * 250 + 1, 0.5)
    t_short = time.perf_counter() - t
    assert t_long < 3 * t_short + 1.0
    assert t_long < 10.0


# ============================================================ round-2 review fixes
def _montage(scales, dur=120.0, seed=0, rate=0, amp_factor=5.0):
    """Channels of background rms 30 * scale; with ``rate``, spikes of ``amp_factor`` x 30 uV
    (before the channel's scale)."""
    return np.vstack([dense_ieeg(FS, dur=dur, rate=rate, amp=amp_factor * 30.0,
                                 seed=seed + k)[0] * sc for k, sc in enumerate(scales)])


SCALES = (0.3, 1.0, 2.0, 3.0)


@pytest.fixture(scope='module')
def montage_baseline():
    return JancaBaseline.from_signal(_montage(SCALES, dur=300.0, seed=50), FS,
                                     channel_names=['A', 'B', 'C', 'D'])


def _warnings(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = fn()
    return out, [str(m.message) for m in w]


def test_montage_level_warning_gain_mismatch(montage_baseline):
    """Review R3: a gain mismatch of x3.5 (below the per-channel x10) warns via the montage
    median of the background level; matching gain and x2 do not."""
    b = montage_baseline
    x = _montage(SCALES, seed=60)
    (_, d), msgs = _warnings(lambda: detect_spikes_janca(x, FS, baseline=b,
                                                         return_details=True))
    assert msgs == []
    assert all(0.8 < c['background_ratio'] < 1.25 for c in d)
    _, msgs = _warnings(lambda: detect_spikes_janca(x * 2.0, FS, baseline=b))
    assert msgs == []
    for g in (3.5, 1 / 3.5):
        _, msgs = _warnings(lambda: detect_spikes_janca(x * g, FS, baseline=b))
        assert len(msgs) == 1 and 'background envelope level' in msgs[0]


@pytest.mark.parametrize('rate,amp_factor', [(5, 5.0), (10, 5.0), (3, 20.0)])
def test_montage_level_warning_not_fooled_by_dense_spiking(montage_baseline, rate, amp_factor):
    """Dense spiking on every channel raises the mean-log level ratio up to ~x4 (beyond the
    x3 limit) but the low-quantile background level much less: no warning (README,
    scratch/janca-baseline/r2/r3_probe.out)."""
    x = _montage(SCALES, seed=60, rate=rate, amp_factor=amp_factor)
    (_, d), msgs = _warnings(lambda: detect_spikes_janca(x, FS, baseline=montage_baseline,
                                                         return_details=True))
    assert msgs == []
    assert max(c['level_ratio'] for c in d) > 1.4
    assert np.median([c['background_ratio'] for c in d]) < 3.0
    if (rate, amp_factor) == (3, 20.0):
        assert np.median([c['level_ratio'] for c in d]) > 3.0    # mean-log would have warned


def test_channel_names_order_checked(montage_baseline):
    b = montage_baseline
    x = _montage(SCALES, seed=60)
    ok = detect_spikes_janca(x, FS, baseline=b, channel_names=['A', 'B', 'C', 'D'])
    np.testing.assert_array_equal(ok[2], detect_spikes_janca(x, FS, baseline=b)[2])
    with pytest.raises(ValueError, match=r"select\(\['D', 'C', 'B', 'A'\]\)"):
        detect_spikes_janca(x[::-1], FS, baseline=b, channel_names=['D', 'C', 'B', 'A'])
    with pytest.raises(ValueError, match='do not match'):
        detect_spikes_janca(x, FS, baseline=b, channel_names=['A', 'B', 'C', 'X'])
    with pytest.raises(ValueError, match='4 channel'):
        detect_spikes_janca(x, FS, baseline=b, channel_names=['A', 'B'])
    with pytest.raises(ValueError, match='only apply with a baseline'):
        detect_spikes_janca(x, FS, channel_names=['A', 'B', 'C', 'D'])
    rev = b.select(['D', 'C', 'B', 'A'])
    out = detect_spikes_janca(x[::-1], FS, baseline=rev, channel_names=['D', 'C', 'B', 'A'])
    np.testing.assert_array_equal(out[3], ok[0])
    # a broadcast 1-channel baseline is not name-checked
    detect_spikes_janca(x[1:2], FS, baseline=b.select('B'), broadcast_baseline=True,
                        channel_names=['other'])
    # baseline without names: nothing to check against
    nameless = JancaBaseline(mu=b.mu, sd=b.sd, fs=FS)
    detect_spikes_janca(x, FS, baseline=nameless, channel_names=['D', 'C', 'B', 'A'])


@pytest.mark.filterwarnings(r'ignore:channel\(s\) \[1\] contain no usable sample')
def test_detector_channel_names_and_channel_types(montage_baseline):
    b = montage_baseline
    x = _montage(SCALES, seed=60)
    with pytest.raises(ValueError, match='do not match'):
        JancaDetector(baseline=b, channel_names=['D', 'C', 'B', 'A'])
    with pytest.raises(ValueError, match='only apply with a baseline'):
        JancaDetector(channel_names=['A'])
    det = JancaDetector(baseline=b, channel_names=['A', 'B', 'C', 'D'])
    ref = detect_spikes_janca(x, FS, baseline=b)
    xg = x.copy()
    xg[1] = np.nan                                    # wrapper feeds channels singly, skips 1
    out = GapAwareSpikeDetector(det).detect(xg, FS)
    np.testing.assert_array_equal(out[2], ref[2])
    # review C1: channel ids must be integers (or names), never truncated floats / bools
    with pytest.raises(TypeError, match='integer'):
        det.detect(x[:1], FS, channels=[1.9], n_channels=4)
    with pytest.raises(TypeError, match='integer'):
        det.detect(x[:1], FS, channels=[True], n_channels=4)
    np.testing.assert_array_equal(det.detect(x[2:3], FS, channels=np.array([2]),
                                             n_channels=4)[0], ref[2])
    np.testing.assert_array_equal(det.detect(x[2:4], FS, channels=['C', 'D'])[1], ref[3])
    with pytest.raises(KeyError):
        JancaDetector(baseline=b).detect(x[:1], FS, channels=['Z'])
    with pytest.raises(ValueError, match='channel_names'):
        JancaDetector(baseline=JancaBaseline(mu=b.mu, sd=b.sd, fs=FS)).detect(
            x[:1], FS, channels=['A'])


def test_from_dict_missing_or_invalid_keys_raise_value_error(baseline):
    d = baseline.to_dict()
    for key in ('params', 'mu', 'sd', 'fs_analysis', 'preset', 'n_channels'):
        bad = dict(d)
        del bad[key]
        with pytest.raises(ValueError, match=f"missing key.*'{key}'"):
            JancaBaseline.from_dict(bad)
    for key, val in (('params', [1, 2]), ('mu', 3.0), ('mu', []), ('preset', 'nope'),
                     ('fs_analysis', 'x')):
        bad = dict(d)
        bad[key] = val
        with pytest.raises(ValueError):
            JancaBaseline.from_dict(bad)


@pytest.mark.parametrize('W', [4001, 40001, 4_000_001])
def test_sliding_sum_long_window_bounded_memory(W):
    """Review C2: windows much longer than the record give the reflected (repeated) result
    of scipy's uniform_filter1d, with O(record) memory."""
    import tracemalloc
    L = np.random.default_rng(2).normal(2.0, 0.6, 1000)
    ok = np.ones(L.size, bool)
    tracemalloc.start()
    mu, sd = _masked_log_stats(L, ok, W, 0.5)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 2e6
    m0 = uniform_filter1d(L, W, mode='reflect')
    s0 = np.sqrt(uniform_filter1d((L - m0) ** 2, W, mode='reflect'))
    np.testing.assert_allclose(mu, m0, rtol=1e-10)
    np.testing.assert_allclose(sd, s0, rtol=1e-9)


def test_sliding_sum_unchanged_for_windows_up_to_twice_the_record():
    from brainmaze_eeg.spikes.janca import _sliding_sum
    a = np.random.default_rng(4).normal(size=300)
    for W in (3, 301, 601):
        h = W // 2
        cs = np.cumsum(np.pad(a, h, mode='symmetric'))
        ref = cs[W - 1:].copy()
        ref[1:] -= cs[:-W]
        np.testing.assert_array_equal(_sliding_sum(a, W), ref)
    for W in (603, 1201, 5001):                   # periodic path vs explicit padding
        h = W // 2
        ap = np.pad(a, h, mode='symmetric')
        ref = np.array([ap[i:i + W].sum() for i in range(a.size)])
        np.testing.assert_allclose(_sliding_sum(a, W), ref, rtol=1e-10, atol=1e-9)
        b = (a > 0)
        bp = np.pad(b, h, mode='symmetric')
        np.testing.assert_array_equal(_sliding_sum(b, W),
                                      [bp[i:i + W].sum() for i in range(b.size)])


@pytest.mark.parametrize('fs_b', [511.99, 24414.0625, 512.0])
def test_exact_decimation_baseline_across_input_rates(fs_b):
    """Review C3: with decimation='exact' baselines made at 511.99 / 24414.0625 Hz (analysis
    200.00008 / 199.99982 Hz) apply to a 1000 Hz recording (200 Hz)."""
    b = JancaBaseline(mu=2.0, sd=0.6, fs=fs_b, decimation='exact')
    x, _ = dense_ieeg(1000, dur=30.0, rate=1, seed=1)
    detect_spikes_janca(x, 1000, decimation='exact', baseline=b)
    JancaBaseline(mu=2.0, sd=0.6, fs=fs_b, fs_analysis=200.0, decimation='exact')


def test_analysis_rate_mismatch_message_has_enough_digits():
    b = JancaBaseline(mu=2.0, sd=0.6, fs_analysis=200.001, decimation='exact')
    x, _ = dense_ieeg(1000, dur=30.0, rate=1, seed=1)
    with pytest.raises(ValueError, match=r'200\.001 Hz, detector 200 Hz'):
        detect_spikes_janca(x, 1000, decimation='exact', baseline=b)
    bi = JancaBaseline(mu=2.0, sd=0.6, fs_analysis=200.000001)        # integer: strict
    with pytest.raises(ValueError, match=r'200\.000001 Hz'):
        detect_spikes_janca(x, 1000, baseline=bi)
