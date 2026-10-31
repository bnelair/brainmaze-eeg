"""Multitaper spectra against known answers.

Bounds are tight on purpose: each is narrow enough that a factor-of-two scaling error,
the wrong degrees of freedom or quantile, a halved p-value, or a half-sample time shift
fails it. Sizes and coverages are estimated over many seeded realizations, so the tests
are exact to reproduce and not flaky.
"""
import warnings

import numpy as np
import pytest
from scipy import stats

from brainmaze_eeg.spectral import (band_power, multitaper_psd, multitaper_spectrogram,
                                    normalize_spectrogram, two_spectrum_test)

FS = 256.0


def _noise(seed, n, sigma=1.0):
    return np.random.default_rng(seed).normal(0, sigma, n)


def _interior(s):
    return (s.freqs > 2 * s.bandwidth) & (s.freqs < FS / 2 - 2 * s.bandwidth)


def test_a_sine_peaks_at_its_frequency_and_carries_its_power():
    t = np.arange(int(20 * FS)) / FS
    amp, f0 = 3.0, 12.0
    s = multitaper_psd(amp * np.sin(2 * np.pi * f0 * t), FS, nw=4)
    assert abs(s.freqs[np.argmax(s.psd)] - f0) <= s.freqs[1]
    total = band_power(s.freqs, s.psd, {"line": (f0 - 2, f0 + 2)})["line"]
    assert total == pytest.approx(amp ** 2 / 2, rel=0.03)       # a sine's variance


def test_white_noise_is_flat_at_two_sigma_squared_over_fs_and_integrates_to_its_variance():
    sigma = 2.0
    s = multitaper_psd(_noise(0, int(60 * FS), sigma), FS, nw=4)
    # The mean, not the median: a chi2(2K) estimate's median sits ~5% below its mean.
    assert np.mean(s.psd[_interior(s)]) == pytest.approx(2 * sigma ** 2 / FS, rel=0.02)
    assert band_power(s.freqs, s.psd, {"all": (0, FS / 2)})["all"] == pytest.approx(sigma ** 2, rel=0.03)


@pytest.mark.parametrize("method", ["jackknife", "chi2"])
def test_the_confidence_intervals_cover_the_truth_at_their_level(method):
    truth = 2 / FS
    hits = []
    for seed in range(12):
        s = multitaper_psd(_noise(seed, int(16 * FS)), FS, nw=4, ci_method=method)
        m = _interior(s)
        hits.append((s.ci_low[m] <= truth) & (truth <= s.ci_high[m]))
    assert 0.925 <= np.mean(np.concatenate(hits)) <= 0.97, method


def test_the_degrees_of_freedom_halve_within_the_bandwidth_of_dc_and_nyquist():
    s = multitaper_psd(_noise(1, 4096), FS, nw=4)
    k = 7
    assert s.n_tapers == k and s.bandwidth == pytest.approx(4 * FS / 4096)
    assert s.dof[0] == k and s.dof[-1] == k
    assert s.dof[_interior(s)].min() == 2 * k


def test_missing_data_is_refused_by_default():
    x = _noise(2, 4096)
    x[100] = np.nan
    with pytest.raises(ValueError, match="nan_policy"):
        multitaper_psd(x, FS)


def test_omitted_gaps_leave_equal_segments_and_an_unbiased_level():
    x = _noise(3, int(40 * FS))
    x[int(10 * FS):int(11 * FS)] = np.nan                     # runs of 10 s and 29 s
    s = multitaper_psd(x, FS, nw=3, nan_policy="omit", segment_samples=int(5 * FS))
    assert s.n_segments == 2 + 5 and s.n_tapers == 7 * 5
    assert np.mean(s.psd[_interior(s)]) == pytest.approx(2 / FS, rel=0.03)


def test_short_runs_never_bias_a_red_spectrum():
    """The review's case: one long run and many short ones of 1/f^2 noise. Pooling
    short runs at equal weight read 2.5x the truth at 2-5 Hz; now they are not pooled
    (the default segment is the longest run) and a warning says how much was left out."""
    rng = np.random.default_rng(4)
    long_run = np.cumsum(rng.normal(size=2048))
    shorts = [np.cumsum(rng.normal(size=256)) for _ in range(10)]
    gappy = np.concatenate([long_run] + [np.concatenate(([np.nan] * 8, r)) for r in shorts])
    with pytest.warns(RuntimeWarning, match="finite samples"):
        s = multitaper_psd(gappy, FS, nw=4, nan_policy="omit")
    alone = multitaper_psd(long_run, FS, nw=4)
    assert s.n_segments == 1
    np.testing.assert_allclose(s.psd, alone.psd)


@pytest.mark.parametrize("method", ["jackknife", "f"])
def test_the_two_spectrum_test_holds_its_size_in_the_interior(method):
    rejected = []
    for seed in range(12):
        a = multitaper_psd(_noise(100 + seed, int(16 * FS)), FS, nw=4)
        b = multitaper_psd(_noise(200 + seed, int(16 * FS)), FS, nw=4)
        _, p = two_spectrum_test(a, b, method=method)
        rejected.append(p[_interior(a)] < 0.05)
    assert 0.035 <= np.mean(np.concatenate(rejected)) <= 0.065, method


def test_the_f_test_is_not_liberal_next_to_dc_and_nyquist():
    edge = []
    for seed in range(40):
        a = multitaper_psd(_noise(300 + seed, 1024), FS, nw=4)
        b = multitaper_psd(_noise(400 + seed, 1024), FS, nw=4)
        _, p = two_spectrum_test(a, b, method="f")
        near = (a.freqs < a.bandwidth) | (a.freqs > FS / 2 - a.bandwidth)
        edge.append(p[near] < 0.05)
    assert np.mean(np.concatenate(edge)) <= 0.08


@pytest.mark.parametrize("method", ["jackknife", "f"])
def test_the_two_spectrum_test_finds_a_real_difference(method):
    a = multitaper_psd(_noise(5, int(30 * FS)), FS, nw=4)
    c = multitaper_psd(_noise(6, int(30 * FS), sigma=3.0), FS, nw=4)
    _, p = two_spectrum_test(a, c, method=method)
    assert np.mean(p[_interior(a)] < 0.05) > 0.85


def test_each_spectrogram_column_is_the_psd_of_its_window_centred_on_it():
    x = _noise(7, int(20 * FS))
    times, freqs, p = multitaper_spectrogram(x, FS, window_s=2.0, step_s=1.0, nw=3, chunk=3)
    n_win = int(2 * FS)
    assert times[0] == pytest.approx((n_win - 1) / 2 / FS)
    assert times[1] - times[0] == pytest.approx(1.0)
    for j in (0, 5, p.shape[-1] - 1):
        s = int(round((times[j] * FS) - (n_win - 1) / 2))
        np.testing.assert_allclose(p[:, j], multitaper_psd(x[s:s + n_win], FS, nw=3).psd, rtol=1e-12)


def test_a_spectrogram_follows_a_frequency_change_and_marks_gaps_missing():
    t = np.arange(int(20 * FS)) / FS
    x = np.where(t < 10, np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 30 * t)) + 0.05 * _noise(8, t.size)
    x[int(4 * FS)] = np.nan
    times, freqs, p = multitaper_spectrogram(x, FS, window_s=2.0, step_s=1.0)
    peak = freqs[np.nanargmax(np.nan_to_num(p, nan=-1), axis=0)]
    assert np.all(np.abs(peak[(times > 6) & (times < 9)] - 8) < 1.5)
    assert np.all(np.abs(peak[times > 11] - 30) < 1.5)
    gap = np.abs(times - 4) < 1.0
    assert np.isnan(p[:, gap]).all() and np.isfinite(p[:, ~gap]).all()


def test_spectrogram_normalisation():
    rng = np.random.default_rng(9)
    power = rng.uniform(1, 10, (5, 40))
    base = np.zeros(40, dtype=bool)
    base[:10] = True
    np.testing.assert_allclose(normalize_spectrogram(power, "db"), 10 * np.log10(power))
    z = normalize_spectrogram(power, "zscore", baseline=base)
    np.testing.assert_allclose(z[:, base].mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z[:, base].std(axis=1), 1, atol=1e-12)
    db = normalize_spectrogram(power, "baseline", baseline=base)
    np.testing.assert_allclose((10 ** (db[:, base] / 10)).mean(axis=1), 1, atol=1e-12)
    assert db[0, 20] == pytest.approx(10 * np.log10(power[0, 20] / power[0, :10].mean()))
    with pytest.raises(ValueError):
        normalize_spectrogram(power, "baseline")


def test_band_power_integrates_exactly_over_the_band_edges():
    freqs = np.linspace(0, 128, 257)                            # 0.5 Hz bins
    flat = np.full((2, freqs.size), 3.0)
    assert band_power(freqs, flat, {"b": (4.2, 7.9)})["b"] == pytest.approx(3.0 * 3.7)
    with pytest.raises(ValueError):
        band_power(freqs, flat, {"b": (100, 200)})


def test_impossible_parameters_are_refused():
    x = _noise(10, 4096)
    with pytest.raises(ValueError, match="poorly concentrated"):
        multitaper_psd(x, FS, nw=4, k=10)
    with pytest.raises(ValueError, match="nfft"):
        multitaper_psd(x, FS, nfft=1024)
    with pytest.raises(ValueError, match="nfft"):
        multitaper_spectrogram(x, FS, window_s=2.0, nfft=128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multitaper_psd(x, FS, nw=4, k=8)                       # 2*NW is allowed
    assert stats is not None


def test_the_nyquist_bin_of_an_even_fft_is_not_doubled():
    """One-sided PSD: every bin is doubled except 0 Hz and, for an even FFT, Nyquist --
    so for white noise the Nyquist bin reads half the interior level."""
    ratios = []
    for seed in range(40):
        s = multitaper_psd(_noise(500 + seed, 2048), FS, nw=4)
        ratios.append(s.psd[-1] / np.mean(s.psd[_interior(s)]))
    assert np.mean(ratios) == pytest.approx(0.5, abs=0.1)
