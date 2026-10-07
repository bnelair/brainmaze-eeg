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


def dense_ieeg(fs, dur=300.0, rate=3.0, amp=150.0, seed=0, rms=30.0, jitter=0.3):
    """
    **Permanently spiking** iEEG (the case of the Janca reference baseline): pink background of
    the given rms plus one sharp-and-slow-wave transient (~70 ms, ``amp`` uV) every
    ``1 / rate`` s, jittered by +/- ``jitter`` of the period, from 1 s to ``dur - 1`` s. The
    generator of the 3.1.0 evidence (brainmaze-work scratch/janca-dense/dense.py), seeded per
    call. ``rate=0`` or ``amp=0`` gives background only.

    Returns
    -------
    x : np.ndarray (n_samples,)
    spike_samples : np.ndarray int, sample index of each transient's centre
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    # the prototype's background: white noise shaped by 1/sqrt(f) (DC removed, all other
    # frequencies kept), scaled to ``rms``. Unlike pink_background (no power below 0.5 Hz)
    # most of the power is slow, so the 10-60 Hz band holds ~1/3 of the rms (SD ~12 uV at
    # rms 30), as in the evidence of the 3.1.0 README.
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    x = x / x.std() * rms
    if rate <= 0 or amp == 0:
        return x, np.zeros(0, dtype=np.int64)
    t = np.arange(-0.15, 0.15, 1.0 / fs)
    s = amp * (-np.exp(-(t / 0.012) ** 2) + 0.35 * np.exp(-((t - 0.05) / 0.04) ** 2))
    h = s.size // 2
    period = 1.0 / rate
    times = np.arange(1.0, dur - 1.0, period)
    times = times + rng.uniform(-jitter, jitter, times.size) * period
    pos = (times * fs).astype(np.int64)
    pos = pos[(pos > h) & (pos < n - s.size)]
    for p in pos:
        x[p - h:p - h + s.size] += s
    return x, pos
