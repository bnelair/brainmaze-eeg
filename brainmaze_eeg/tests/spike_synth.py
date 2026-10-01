"""
Seeded synthetic iEEG used by the spike-detector tests and by the eeg_forge parity fixture
generator (``brainmaze_eeg/tests/data/make_janca_reference_fixtures.py``).

Only numpy's ``default_rng`` (PCG64, stream-stable across numpy 1.x/2.x) and ``numpy.fft``
are used, so the same seed gives the same signal in every supported environment.
"""

import numpy as np


def pink_background(n, fs, rms=30.0, rng=None, f_lo=0.5):
    """1/f (power) background with the given rms, energy below ``f_lo`` Hz removed."""
    rng = np.random.default_rng(0) if rng is None else rng
    f = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.zeros_like(f)
    amp[f > f_lo] = 1.0 / np.sqrt(f[f > f_lo])
    x = np.fft.irfft(amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)), n)
    return x / x.std() * rms


def ied_waveform(fs, amp=200.0):
    """IED-like transient: sharp (~25 ms) negative spike followed by a slower positive wave."""
    t = np.arange(-0.05, 0.15, 1.0 / fs)
    return -amp * np.exp(-(t / 0.012) ** 2) + 0.3 * amp * np.exp(-((t - 0.06) / 0.03) ** 2)


def synth_ieeg(fs, dur=60.0, seed=0, n_spikes=25, rms=30.0, mains_hz=50.0, mains_amp=5.0,
               amp_range=(80.0, 300.0)):
    """
    1/f background + mains hum + ``n_spikes`` IED-like transients of random amplitude and
    polarity at random times in ``[2, dur-2]`` s.

    Returns
    -------
    x : np.ndarray (n_samples,)
    spike_samples : np.ndarray (n_spikes,) int, sample index of each transient's peak
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    x = pink_background(n, fs, rms, rng)
    if mains_hz:
        x += mains_amp * np.sin(2 * np.pi * mains_hz * np.arange(n) / fs)
    times = np.sort(rng.uniform(2.0, dur - 2.0, n_spikes))
    amps = rng.uniform(*amp_range, n_spikes) * rng.choice([-1.0, 1.0], n_spikes)
    i0 = int(round(0.05 * fs))
    peaks = []
    for t, a in zip(times, amps):
        w = ied_waveform(fs, a)
        i = int(round(t * fs)) - i0
        m = min(w.size, n - i)
        x[i:i + m] += w[:m]
        peaks.append(i + i0)
    return x, np.asarray(peaks, dtype=np.int64)
