# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Epochs around events
^^^^^^^^^^^^^^^^^^^^

Event-locked analysis -- spikes, stimuli, spindles -- for a review viewer's
"epoch average" and for scripts:

* :func:`extract_epochs` -- cut fixed windows around event times;
* :func:`epoch_average` -- mean and standard error;
* :func:`cluster_permutation_test` -- where in time does the average differ from zero,
  or two conditions from each other (Maris & Oostenveld, 2007), with the family-wise
  error controlled over the whole window;
* :func:`ersp_itc` -- event-related spectral perturbation and inter-trial coherence from
  complex Morlet wavelets.

An epoch that would run past either end of the record, or that holds any NaN/inf, is
dropped rather than padded: the returned ``kept`` mask says which events were used.
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = ["extract_epochs", "epoch_average", "Cluster", "cluster_permutation_test", "ersp_itc"]


def extract_epochs(x, fs: float, events_s: Sequence[float], pre_s: float, post_s: float):
    """Windows ``[t - pre_s, t + post_s)`` of ``x`` (``(..., n_samples)``) around each event.

    ``events_s`` are seconds from the first sample. Returns ``(epochs, times, kept)``:
    ``epochs`` is ``(n_kept, ..., n_window)``, ``times`` the window's time axis in seconds
    relative to the event, and ``kept`` a boolean mask over ``events_s``.
    """
    x = np.asarray(x, dtype=float)
    n_pre, n_post = int(round(pre_s * fs)), int(round(post_s * fs))
    if n_pre < 0 or n_post <= 0:
        raise ValueError("pre_s must be >= 0 and post_s > 0")
    centres = np.round(np.asarray(events_s, dtype=float) * fs).astype(int)
    kept = (centres - n_pre >= 0) & (centres + n_post <= x.shape[-1])
    epochs = [x[..., c - n_pre:c + n_post] for c in centres[kept]]
    finite = np.array([np.isfinite(e).all() for e in epochs], dtype=bool)
    kept[np.flatnonzero(kept)[~finite]] = False
    epochs = [e for e, ok in zip(epochs, finite) if ok]
    times = (np.arange(-n_pre, n_post)) / fs
    shape = (0,) + x.shape[:-1] + (n_pre + n_post,)
    return (np.stack(epochs) if epochs else np.empty(shape)), times, kept


def epoch_average(epochs) -> Tuple[np.ndarray, np.ndarray]:
    """Mean over epochs (axis 0) and its standard error."""
    epochs = np.asarray(epochs, dtype=float)
    if epochs.shape[0] == 0:
        raise ValueError("no epochs")
    n = epochs.shape[0]
    sem = epochs.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(epochs.shape[1:], np.nan)
    return epochs.mean(axis=0), sem


class Cluster(NamedTuple):
    """A run of time bins ``[start, stop)`` past the threshold, its summed t, and its
    permutation p-value (family-wise over the window)."""
    start: int
    stop: int
    stat: float
    p: float


def _t(a: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
    if b is None:
        return a.mean(0) / (a.std(0, ddof=1) / np.sqrt(a.shape[0]))
    return stats.ttest_ind(a, b, axis=0, equal_var=False).statistic


def _clusters(t: np.ndarray, threshold: float, tail: int) -> List[Tuple[int, int, float]]:
    out = []
    for sign in ((1, -1) if tail == 0 else (tail,)):
        above = sign * t > threshold
        edges = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            out.append((int(a), int(b), float(t[a:b].sum())))
    return out


def cluster_permutation_test(a, b=None, n_permutations: int = 1000, threshold: Optional[float] = None,
                             tail: int = 0, alpha: float = 0.05, seed=None):
    """Cluster-based permutation test over the last axis (time), Maris & Oostenveld (2007).

    ``a`` is ``(n_epochs, n_times)``. With ``b`` (``(m_epochs, n_times)``) the two
    conditions are compared (Welch t; epochs shuffled between conditions); without it,
    ``a`` is tested against zero (one-sample t; epochs sign-flipped).

    ``threshold`` is the t value a bin must pass to join a cluster, default the two-sided
    ``alpha`` critical value. ``tail``: 0 both signs, 1 positive, -1 negative. A
    cluster's p is the share of permutations whose LARGEST cluster is at least as
    large, so the family-wise error over the window is controlled.

    Returns ``(t_obs, clusters)``.
    """
    a = np.asarray(a, dtype=float)
    b = None if b is None else np.asarray(b, dtype=float)
    if a.ndim != 2 or (b is not None and (b.ndim != 2 or b.shape[1] != a.shape[1])):
        raise ValueError("a (and b) must be (n_epochs, n_times) with the same n_times")
    if a.shape[0] < 2 or (b is not None and b.shape[0] < 2):
        raise ValueError("each condition needs at least two epochs")
    if tail not in (-1, 0, 1):
        raise ValueError("tail is -1, 0 or 1")
    dof = a.shape[0] - 1 if b is None else a.shape[0] + b.shape[0] - 2
    if threshold is None:
        threshold = stats.t.ppf(1 - (alpha / 2 if tail == 0 else alpha), dof)
    rng = np.random.default_rng(seed)
    t_obs = _t(a, b)
    observed = _clusters(t_obs, threshold, tail)

    def largest(t):
        found = _clusters(t, threshold, tail)
        return max((abs(s) for _, _, s in found), default=0.0)

    null = np.empty(n_permutations)
    pooled = a if b is None else np.concatenate([a, b])
    for i in range(n_permutations):
        if b is None:
            flips = rng.choice([-1.0, 1.0], size=(a.shape[0], 1))
            null[i] = largest(_t(a * flips, None))
        else:
            order = rng.permutation(pooled.shape[0])
            null[i] = largest(_t(pooled[order[:a.shape[0]]], pooled[order[a.shape[0]:]]))
    clusters = [Cluster(s, e, stat, float((1 + np.sum(null >= abs(stat))) / (1 + n_permutations)))
                for s, e, stat in observed]
    return t_obs, sorted(clusters, key=lambda c: c.start)


def _morlet(fs: float, freq: float, n_cycles: float) -> np.ndarray:
    sigma_t = n_cycles / (2 * np.pi * freq)
    t = np.arange(-4 * sigma_t, 4 * sigma_t + 1 / fs, 1 / fs)
    w = np.exp(2j * np.pi * freq * t) * np.exp(-t ** 2 / (2 * sigma_t ** 2))
    return w / np.sqrt(np.sum(np.abs(w) ** 2))


def ersp_itc(epochs, fs: float, freqs: Sequence[float], n_cycles=7.0, baseline=None):
    """Event-related spectral perturbation and inter-trial coherence.

    ``epochs`` is ``(n_epochs, n_times)``. Each epoch is convolved with a unit-energy
    complex Morlet wavelet per frequency (``n_cycles`` per frequency, or one value).
    Returns ``(ersp, itc)``, both ``(n_freqs, n_times)``:

    * ``ersp`` -- mean power in dB relative to the mean over ``baseline`` (a boolean mask
      over time bins), or in dB of absolute power without one;
    * ``itc`` -- ``|mean(exp(i * phase))|`` over epochs, 0 (random phase) to 1 (locked).

    The wavelet is longer than the epoch at low frequencies; those edge bins are NaN
    rather than estimated from zero padding.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 2 or epochs.shape[0] < 1:
        raise ValueError("epochs must be (n_epochs, n_times)")
    freqs = np.asarray(freqs, dtype=float)
    cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), freqs.shape)
    n_times = epochs.shape[1]
    power = np.full((freqs.size, n_times), np.nan)
    itc = np.full((freqs.size, n_times), np.nan)
    for i, (f, c) in enumerate(zip(freqs, cycles)):
        w = _morlet(fs, f, c)
        half = w.size // 2
        conv = np.array([np.convolve(e, w, mode="same") for e in epochs])
        valid = slice(half, n_times - half) if n_times > 2 * half else slice(0, 0)
        power[i, valid] = (np.abs(conv[:, valid]) ** 2).mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            itc[i, valid] = np.abs((conv[:, valid] / np.abs(conv[:, valid])).mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        if baseline is None:
            return 10 * np.log10(power), itc
        ref = np.nanmean(power[:, np.asarray(baseline, dtype=bool)], axis=1, keepdims=True)
        return 10 * np.log10(power / ref), itc
