"""Multitaper spectra against known answers: a sine's power, white noise's level and the
coverage of its confidence intervals, the two-spectrum test's false-positive rate, and
gaps that never enter an estimate."""

import numpy as np
import pytest

from brainmaze_eeg.spectral import (band_power, multitaper_psd, multitaper_spectrogram,
                                    normalize_spectrogram, two_spectrum_test)

FS = 256.0


def test_a_sine_peaks_at_its_frequency_and_carries_its_power():
    t = np.arange(int(20 * FS)) / FS
    amp, f0 = 3.0, 12.0
    s = multitaper_psd(amp * np.sin(2 * np.pi * f0 * t), FS, nw=4)
    assert abs(s.freqs[np.argmax(s.psd)] - f0) <= s.freqs[1]
    total = band_power(s.freqs, s.psd, {"line": (f0 - 2, f0 + 2)})["line"]
    assert total == pytest.approx(amp ** 2 / 2, rel=0.05)       # a sine's variance


def test_white_noise_is_flat_at_two_sigma_squared_over_fs_and_its_ci_covers_it():
    rng = np.random.default_rng(0)
    sigma = 2.0
    x = rng.normal(0, sigma, int(60 * FS))
    truth = 2 * sigma ** 2 / FS
    for method in ("jackknife", "chi2"):
        s = multitaper_psd(x, FS, nw=4, ci_method=method)
        inner = slice(5, -5)
        assert np.median(s.psd[inner]) == pytest.approx(truth, rel=0.1)
        covered = np.mean((s.ci_low[inner] <= truth) & (truth <= s.ci_high[inner]))
        assert 0.88 <= covered <= 0.995, (method, covered)
    # Integrating the PSD returns the variance.
    assert band_power(s.freqs, s.psd, {"all": (0, FS / 2)})["all"] == pytest.approx(sigma ** 2, rel=0.05)


def test_gaps_are_refused_by_default_and_never_estimated_across_when_omitted():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, int(30 * FS))
    with pytest.raises(ValueError, match="nan_policy"):
        multitaper_psd(np.where(np.arange(x.size) == 100, np.nan, x), FS)
    gappy = x.copy()
    gappy[int(10 * FS):int(12 * FS)] = np.nan
    gappy[int(12 * FS):int(13 * FS)] = 1e6                      # a step that is NOT a gap...
    gappy[int(12 * FS):int(13 * FS)] = np.nan                   # ...is never seen once it is one
    s = multitaper_psd(gappy, FS, nw=3, nan_policy="omit")
    assert s.n_tapers == 2 * 5                                  # two finite runs x 5 tapers
    assert np.median(s.psd[5:-5]) == pytest.approx(2 / FS, rel=0.15)


def test_the_two_spectrum_test_holds_its_size_and_finds_a_real_difference():
    rng = np.random.default_rng(2)
    a = multitaper_psd(rng.normal(0, 1, int(30 * FS)), FS, nw=4)
    b = multitaper_psd(rng.normal(0, 1, int(30 * FS)), FS, nw=4)
    _, p = two_spectrum_test(a, b)
    assert 0.01 <= np.mean(p[5:-5] < 0.05) <= 0.12              # ~alpha under the null
    c = multitaper_psd(rng.normal(0, 3, int(30 * FS)), FS, nw=4)
    _, p = two_spectrum_test(a, c)
    assert np.mean(p[5:-5] < 0.05) > 0.95


def test_a_spectrogram_follows_a_frequency_change_and_marks_gaps_missing():
    t = np.arange(int(20 * FS)) / FS
    x = np.where(t < 10, np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 30 * t))
    x = x + 0.05 * np.random.default_rng(3).normal(size=t.size)
    x[int(4 * FS)] = np.nan
    times, freqs, p = multitaper_spectrogram(x, FS, window_s=2.0, step_s=1.0)
    peak = freqs[np.nanargmax(np.nan_to_num(p, nan=-1), axis=0)]
    assert np.all(np.abs(peak[(times > 6) & (times < 9)] - 8) < 1.5)
    assert np.all(np.abs(peak[times > 11] - 30) < 1.5)
    assert np.isnan(p[:, np.abs(times - 4) < 1.0]).all()       # windows holding the gap
    db = normalize_spectrogram(p, "db")
    assert np.allclose(db[:, 0], 10 * np.log10(p[:, 0]))
    z = normalize_spectrogram(p[:, times > 11], "zscore")
    assert np.allclose(np.nanmean(z, axis=-1), 0, atol=1e-6)
    assert np.allclose(np.nanstd(z, axis=-1), 1, atol=1e-6)
