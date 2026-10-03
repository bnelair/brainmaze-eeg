# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Seeded synthetic iEEG for the spike-detection demos.

Not a physiological model: 1/f background plus mains hum plus stereotyped transients, so that
every detection can be scored against a known ground truth. Only numpy is used, and a seed
gives the same signal with numpy 1.x and 2.x.
"""

import numpy as np


def pink_noise(n, fs, rms, rng, f_lo=0.5):
    """1/f (power) noise with the given rms; no energy below ``f_lo`` Hz."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.zeros_like(f)
    amp[f > f_lo] = 1.0 / np.sqrt(f[f > f_lo])
    x = np.fft.irfft(amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)), n)
    return x / x.std() * rms


def ied(fs, amp):
    """IED-like transient: sharp (~25 ms) negative spike and a slower positive after-wave.
    Peak (the spike) at sample ``round(0.05 * fs)`` of the returned waveform."""
    t = np.arange(-0.05, 0.15, 1.0 / fs)
    return -amp * np.exp(-(t / 0.012) ** 2) + 0.3 * amp * np.exp(-((t - 0.06) / 0.03) ** 2)


def ripple(fs, f0, amp, dur=0.06):
    """Ripple-like burst: ``f0`` Hz sine under a Gaussian envelope of about ``dur`` s.
    Centre at the middle sample of the returned waveform."""
    t = np.arange(-2 * dur, 2 * dur, 1.0 / fs)
    return amp * np.exp(-(t / (dur / 2.5)) ** 2) * np.sin(2 * np.pi * f0 * t)


def random_times(rng, n, dur, min_sep, margin=2.0):
    """``n`` sorted random times in ``[margin, dur - margin]`` s, then thinned so that
    consecutive times are at least ``min_sep`` s apart."""
    t = np.sort(rng.uniform(margin, dur - margin, n))
    keep = [t[0]]
    for ti in t[1:]:
        if ti - keep[-1] >= min_sep:
            keep.append(ti)
    return np.asarray(keep)


def synth_ieeg(fs=1000, dur=120.0, n_channels=8, spiking=(0, 1, 2, 3), n_spikes=40,
               amp_range=(150.0, 350.0), rms=30.0, mains_hz=60.0, mains_amp=5.0, seed=0):
    """
    ``n_channels`` of 1/f background (``rms`` uV) + mains hum; the channels in ``spiking``
    also carry about ``n_spikes`` IED-like transients (>= 0.5 s apart).

    Returns
    -------
    x : np.ndarray, (n_channels, n_samples), uV
    truth : list of np.ndarray, sample index of every planted spike, per channel
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    x = np.stack([pink_noise(n, fs, rms, rng) for _ in range(n_channels)])
    x += mains_amp * np.sin(2 * np.pi * mains_hz * t)
    i0 = int(round(0.05 * fs))
    truth = []
    for c in range(n_channels):
        peaks = []
        if c in spiking:
            for ts in random_times(rng, n_spikes, dur, 0.5):
                w = ied(fs, rng.uniform(*amp_range))
                i = int(round(ts * fs)) - i0
                x[c, i:i + w.size] += w
                peaks.append(i + i0)
        truth.append(np.asarray(peaks, dtype=np.int64))
    return x, truth


def score(detections, truth, fs, tol_s=0.05, exclude=None):
    """
    Match detections to planted events one-to-one within ``tol_s``.

    Greedy nearest matching: candidate pairs closer than ``tol_s`` are taken in order of
    increasing distance, and each detection and each planted event is used at most once.
    A second detection of the same event is therefore a false detection.

    ``exclude``: optional ``(n, 2)`` array of ``[start, stop)`` samples; planted events in
    there are not counted (they are not in the data any more, e.g. inside a dropout).

    Returns ``(hits, n_planted, false_detections)``.
    """
    det = np.asarray(detections, dtype=np.int64)
    tru = np.asarray(truth, dtype=np.int64)
    if exclude is not None and len(exclude):
        inside = np.zeros(tru.size, dtype=bool)
        for a, b in exclude:
            inside |= (tru >= a) & (tru < b)
        tru = tru[~inside]
    if det.size == 0 or tru.size == 0:
        return 0, int(tru.size), int(det.size)
    dist = np.abs(det[:, None] - tru[None, :])
    di, ti = np.nonzero(dist <= tol_s * fs)
    used_d, used_t = set(), set()
    for k in np.argsort(dist[di, ti], kind='stable'):
        if di[k] not in used_d and ti[k] not in used_t:
            used_d.add(di[k])
            used_t.add(ti[k])
    return len(used_t), int(tru.size), int(det.size - len(used_d))


def peak_indices(per_channel):
    """Sample indices of one channel's detections: an index array (Janca) or a list of
    Barkmeier detection dicts (``'peak_index'``)."""
    if isinstance(per_channel, np.ndarray):
        return per_channel
    return np.asarray([d['peak_index'] for d in per_channel], dtype=np.int64)
