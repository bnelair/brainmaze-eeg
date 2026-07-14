import numpy as np
import pytest

from brainmaze_utils.types import ObjDict
from brainmaze_eeg.features.spectral_features import median_frequency, mean_frequency


def _args(psd, band, freq):
    return ObjDict({'psd': np.atleast_2d(psd), 'fbands': np.array(band), 'freq': freq})


FREQ = np.arange(0, 100, 0.5)


def test_median_frequency_flat_spectrum_is_band_midpoint():
    # for a flat spectrum, half the band power sits at the band midpoint
    band = [[1.0, 20.0]]
    out = median_frequency(_args(np.ones(FREQ.size), band, FREQ))[0]
    assert out[0] == pytest.approx(10.5, abs=0.5)


def test_median_frequency_uses_band_power_not_full_spectrum():
    # regression: the reference power must be the band total, not the whole spectrum.
    # with the old full-spectrum reference this returned the band's 2nd bin (~1.5 Hz).
    band = [[1.0, 20.0]]
    out = median_frequency(_args(np.ones(FREQ.size), band, FREQ))[0]
    assert out[0] > 5.0   # not stuck near the bottom of the band


def test_median_frequency_single_spectral_line():
    psd = np.zeros(FREQ.size)
    psd[np.argmin(np.abs(FREQ - 8.0))] = 1.0
    out = median_frequency(_args(psd, [[1.0, 20.0]], FREQ))[0]
    assert out[0] == pytest.approx(8.0)


def test_median_frequency_zero_power_band_is_nan():
    out = median_frequency(_args(np.zeros(FREQ.size), [[1.0, 20.0]], FREQ))[0]
    assert np.isnan(out[0])


def test_median_frequency_full_band_flat_spectrum():
    # sanity for the non-narrowed case: median of a flat full band is its midpoint
    band = [[0.5, 49.5]]
    out = median_frequency(_args(np.ones((3, FREQ.size)), band, FREQ))[0]
    np.testing.assert_allclose(out, 25.0, atol=0.5)


def test_median_frequency_is_monotonic_in_spectral_tilt():
    # shifting power toward high frequencies must raise the median frequency
    band = [[1.0, 40.0]]
    lowtilt = np.where(FREQ < 10, 1.0, 0.1)
    hightilt = np.where(FREQ > 30, 1.0, 0.1)
    m_low = median_frequency(_args(lowtilt, band, FREQ))[0][0]
    m_high = median_frequency(_args(hightilt, band, FREQ))[0][0]
    assert m_high > m_low


def test_mean_frequency_flat_spectrum_is_band_midpoint():
    # cross-check: mean frequency of a flat band is also the midpoint
    out = mean_frequency(_args(np.ones(FREQ.size), np.array([[1.0, 20.0]]), FREQ))[0]
    assert np.asarray(out).ravel()[0] == pytest.approx(10.5, abs=0.5)
