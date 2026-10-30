# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Multitaper spectral estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Thomson's multitaper method with Slepian (DPSS) tapers, for the quick analyses of a
review viewer and for scripts alike:

* :func:`multitaper_psd` -- a power spectral density with confidence intervals
  (jackknife over the tapers, or chi-square);
* :func:`two_spectrum_test` -- are two spectra the same, frequency by frequency?
* :func:`multitaper_spectrogram` and :func:`normalize_spectrogram` -- time-frequency
  power, in dB, as a z-score, or as change from a baseline;
* :func:`band_power` -- power integrated over named bands.

Scaling: one-sided PSD in ``units**2 / Hz``. The tapers have unit energy, so white
noise of variance ``s2`` has a PSD of ``2 * s2 / fs`` and integrating the PSD over
``[0, fs/2]`` returns the signal's variance.

Missing data
------------
NaN/inf marks missing data. ``nan_policy='raise'`` (the default) refuses it.
``nan_policy='omit'`` estimates from every finite run of at least ``min_samples``
samples: each run gets its own tapers (on a common FFT grid), and the eigenspectra of
all runs are pooled, so a gap never enters the estimate as a step. Shorter runs have a
wider bandwidth (``W = NW / T_run``) -- the pooled estimate is as smooth as its
shortest run.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.signal import windows

__all__ = ["MultitaperSpectrum", "multitaper_psd", "two_spectrum_test", "multitaper_spectrogram",
           "normalize_spectrogram", "band_power"]


@dataclass(frozen=True)
class MultitaperSpectrum:
    """A multitaper PSD estimate.

    ``psd``, ``ci_low``, ``ci_high`` have the input's leading shape plus a last axis of
    frequencies. ``n_tapers`` is the number of eigenspectra averaged per series (pooled
    over runs for ``nan_policy='omit'``) -- the degrees of freedom are ``2 * n_tapers``.
    """
    freqs: np.ndarray
    psd: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_tapers: np.ndarray
    nw: float
    confidence: float


def _finite_runs(x: np.ndarray, min_samples: int):
    """(start, stop) of every run of finite samples at least ``min_samples`` long."""
    ok = np.isfinite(x)
    if ok.all():
        return [(0, x.size)] if x.size >= min_samples else []
    edges = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= min_samples]


def _eigenspectra(x: np.ndarray, fs: float, nw: float, k: int, nfft: int) -> np.ndarray:
    """(k, nfft//2 + 1) one-sided eigenspectra of one finite series."""
    n = x.size
    tapers = windows.dpss(n, nw, Kmax=k)                        # (k, n), unit energy
    tapers = np.atleast_2d(tapers)
    xf = np.fft.rfft(tapers * (x - x.mean())[None, :], n=nfft, axis=-1)
    s = (np.abs(xf) ** 2) / fs
    s[:, 1:] *= 2.0
    if nfft % 2 == 0:
        s[:, -1] /= 2.0                                          # the Nyquist bin is not doubled
    return s


def _confidence(eig: np.ndarray, confidence: float, method: str):
    """(psd, low, high) from (K, F) eigenspectra."""
    k = eig.shape[0]
    psd = eig.mean(axis=0)
    alpha = 1.0 - confidence
    if method == "chi2" or k < 2:
        dof = 2 * k
        low = dof * psd / stats.chi2.ppf(1 - alpha / 2, dof)
        high = dof * psd / stats.chi2.ppf(alpha / 2, dof)
        return psd, low, high
    if method != "jackknife":
        raise ValueError("ci_method is 'jackknife' or 'chi2'")
    # Thomson & Chave (1991): jackknife the log spectrum over the tapers.
    with np.errstate(divide="ignore"):
        loo = np.log((eig.sum(axis=0)[None, :] - eig) / (k - 1))
    var = (k - 1) / k * ((loo - loo.mean(axis=0)) ** 2).sum(axis=0)
    t = stats.t.ppf(1 - alpha / 2, k - 1)
    with np.errstate(divide="ignore"):
        log_psd = np.log(psd)
    return psd, np.exp(log_psd - t * np.sqrt(var)), np.exp(log_psd + t * np.sqrt(var))


def multitaper_psd(x, fs: float, nw: float = 4.0, k: Optional[int] = None, nfft: Optional[int] = None,
                   confidence: float = 0.95, ci_method: str = "jackknife", nan_policy: str = "raise",
                   min_samples: Optional[int] = None) -> MultitaperSpectrum:
    """Multitaper PSD of ``x`` along its last axis.

    Parameters
    ----------
    x : array_like
        Signal, ``(..., n_samples)``.
    fs : float
        Sampling rate, Hz.
    nw : float
        Time-half-bandwidth product; the frequency resolution is ``2 * nw / T``.
    k : int, optional
        Number of tapers, default ``2 * nw - 1``.
    nfft : int, optional
        FFT length, default the next power of two of the longest run.
    confidence : float
        Confidence level of ``ci_low`` / ``ci_high``.
    ci_method : {'jackknife', 'chi2'}
    nan_policy : {'raise', 'omit'}
        See the module notes on missing data.
    min_samples : int, optional
        ``'omit'`` only: the shortest finite run used, default ``max(2 * nw * k, 16)``
        samples (enough for ``k`` well-concentrated tapers).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0 or x.shape[-1] < 2:
        raise ValueError("x needs at least two samples along its last axis")
    if fs <= 0:
        raise ValueError("fs must be positive")
    k = int(2 * nw - 1) if k is None else int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if nan_policy not in ("raise", "omit"):
        raise ValueError("nan_policy is 'raise' or 'omit'")
    if nan_policy == "raise" and not np.isfinite(x).all():
        raise ValueError("x contains NaN/inf; pass nan_policy='omit' to use the finite runs")
    min_samples = int(max(2 * nw * k, 16)) if min_samples is None else int(min_samples)

    flat = x.reshape(-1, x.shape[-1])
    runs = [_finite_runs(row, min_samples) for row in flat]
    longest = max([b - a for r in runs for a, b in r], default=0)
    if longest == 0:
        raise ValueError("no finite run of at least min_samples samples")
    nfft = int(2 ** np.ceil(np.log2(longest))) if nfft is None else int(nfft)
    if nfft < longest:
        raise ValueError("nfft is shorter than the data")
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    psd = np.full((flat.shape[0], freqs.size), np.nan)
    low, high = psd.copy(), psd.copy()
    n_tapers = np.zeros(flat.shape[0], dtype=int)
    for i, (row, row_runs) in enumerate(zip(flat, runs)):
        if not row_runs:
            continue
        eig = np.concatenate([_eigenspectra(row[a:b], fs, nw, k, nfft) for a, b in row_runs], axis=0)
        psd[i], low[i], high[i] = _confidence(eig, confidence, ci_method)
        n_tapers[i] = eig.shape[0]
    shape = x.shape[:-1] + (freqs.size,)
    return MultitaperSpectrum(freqs, psd.reshape(shape), low.reshape(shape), high.reshape(shape),
                              n_tapers.reshape(x.shape[:-1]), float(nw), float(confidence))


def two_spectrum_test(a: MultitaperSpectrum, b: MultitaperSpectrum) -> Tuple[np.ndarray, np.ndarray]:
    """Are ``a`` and ``b`` estimates of the same spectrum? Frequency by frequency.

    Under the null each multitaper estimate is ``S * chi2(2K) / 2K``, so the ratio
    ``a.psd / b.psd`` is ``F(2 K_a, 2 K_b)``-distributed. Returns ``(ratio, p)`` with a
    two-sided p-value per frequency. The frequency grids must match (same ``fs`` and
    ``nfft``). Neighbouring frequencies within the bandwidth are not independent --
    correct for multiple comparisons accordingly.
    """
    if a.freqs.shape != b.freqs.shape or not np.allclose(a.freqs, b.freqs):
        raise ValueError("the two spectra are on different frequency grids")
    ka = np.asarray(a.n_tapers)[..., None]
    kb = np.asarray(b.n_tapers)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a.psd / b.psd
    cdf = stats.f.cdf(ratio, 2 * ka, 2 * kb)
    p = np.minimum(1.0, 2 * np.minimum(cdf, 1 - cdf))
    return ratio, p


def multitaper_spectrogram(x, fs: float, window_s: float, step_s: Optional[float] = None, nw: float = 3.0,
                           k: Optional[int] = None, nfft: Optional[int] = None):
    """Multitaper spectrogram of ``x`` along its last axis.

    Returns ``(times, freqs, power)`` -- ``times`` the window centres in seconds from the
    first sample, ``power`` of shape ``(..., n_freqs, n_times)``. A window holding any
    NaN/inf is NaN (missing), never estimated across the gap.
    """
    x = np.asarray(x, dtype=float)
    n_win = int(round(window_s * fs))
    step = int(round((step_s if step_s is not None else window_s / 2) * fs))
    if n_win < 2 or step < 1:
        raise ValueError("window_s and step_s are too short for fs")
    if x.shape[-1] < n_win:
        raise ValueError("the signal is shorter than one window")
    starts = np.arange(0, x.shape[-1] - n_win + 1, step)
    k = int(2 * nw - 1) if k is None else int(k)
    nfft = int(2 ** np.ceil(np.log2(n_win))) if nfft is None else int(nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    tapers = np.atleast_2d(windows.dpss(n_win, nw, Kmax=k))
    seg = np.stack([x[..., s:s + n_win] for s in starts], axis=-2)           # (..., T, n_win)
    seg = seg - seg.mean(axis=-1, keepdims=True)
    xf = np.fft.rfft(seg[..., None, :] * tapers, n=nfft, axis=-1)           # (..., T, K, F)
    power = (np.abs(xf) ** 2).mean(axis=-2) / fs
    power[..., 1:] *= 2.0
    if nfft % 2 == 0:
        power[..., -1] /= 2.0
    power[~np.isfinite(seg).all(axis=-1)] = np.nan
    times = (starts + n_win / 2.0) / fs
    return times, freqs, np.swapaxes(power, -1, -2)


def normalize_spectrogram(power, mode: str = "db", baseline: Optional[np.ndarray] = None):
    """Express a spectrogram ``(..., n_freqs, n_times)`` for display.

    ``mode``: ``'db'`` -- ``10 log10(power)``; ``'zscore'`` -- per frequency, against the
    mean and SD over ``baseline`` time bins (default all); ``'baseline'`` -- dB change from
    the mean over ``baseline`` time bins. ``baseline`` is a boolean mask over time bins.
    """
    power = np.asarray(power, dtype=float)
    ref = power if baseline is None else power[..., np.asarray(baseline, dtype=bool)]
    if mode == "db":
        with np.errstate(divide="ignore"):
            return 10 * np.log10(power)
    if mode == "zscore":
        mu = np.nanmean(ref, axis=-1, keepdims=True)
        sd = np.nanstd(ref, axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (power - mu) / sd
    if mode == "baseline":
        if baseline is None:
            raise ValueError("mode='baseline' needs a baseline mask")
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10 * np.log10(power / np.nanmean(ref, axis=-1, keepdims=True))
    raise ValueError("mode is 'db', 'zscore' or 'baseline'")


def band_power(freqs, psd, bands: Dict[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    """Power in each named band ``(lo, hi)`` Hz, integrated over the PSD's last axis."""
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    out = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs <= hi)
        if sel.sum() < 2:
            raise ValueError(f"band {name!r} ({lo}-{hi} Hz) holds fewer than two frequency bins")
        out[name] = np.trapezoid(psd[..., sel], freqs[sel], axis=-1) if hasattr(np, "trapezoid") \
            else np.trapz(psd[..., sel], freqs[sel], axis=-1)
    return out
