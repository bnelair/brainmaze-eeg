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
``[0, fs/2]`` returns the signal's variance. Each segment is demeaned first, which
removes power within the half-bandwidth ``W = NW / T`` of 0 Hz as well (by about
``1 / (2 NW)`` at the first bins).

Assumptions behind the intervals and the test
--------------------------------------------
The chi-square interval and the F test treat the K eigenspectra as independent
estimates. That holds for spectra that are locally flat; on steep spectra (a dynamic
range of more than ~40 dB, e.g. raw 1/f^3 iEEG) taper leakage correlates them and both
become too liberal -- prefer the jackknife interval and test (the defaults), or
prewhiten. Within ``W`` of 0 Hz and of Nyquist the eigenspectra are real-valued or
folded, so their degrees of freedom are about K, not 2K; the chi-square interval and
the F test use K there.

Missing data
------------
NaN/inf marks missing data. ``nan_policy='raise'`` (the default) refuses it.
``nan_policy='omit'`` cuts the finite runs into non-overlapping segments of ONE length
(``segment_samples``, default the longest run) and averages the segments' eigenspectra:
every estimate then has the same resolution and the same weight, so the level is not
biased by short runs and the degrees of freedom are honest. Finite data in runs
shorter than a segment, and the remainder of each run, are not used; a
``RuntimeWarning`` says so when that is more than half of it.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.integrate import trapezoid
from scipy.signal import windows

__all__ = ["MultitaperSpectrum", "multitaper_psd", "two_spectrum_test", "multitaper_spectrogram",
           "normalize_spectrogram", "band_power"]


@dataclass(frozen=True)
class MultitaperSpectrum:
    """A multitaper PSD estimate.

    ``psd``, ``ci_low``, ``ci_high``, ``log_var`` and ``dof`` have the input's leading shape
    plus a last axis of frequencies. ``n_tapers`` is the number of eigenspectra averaged
    per series (``k`` times ``n_segments``); ``dof`` the chi-square degrees of freedom per
    frequency (``2 * n_tapers``, ``n_tapers`` within ``bandwidth`` of 0 Hz and Nyquist);
    ``log_var`` the jackknife variance of ``ln(psd)``; ``bandwidth`` the half-bandwidth
    ``W`` in Hz.
    """
    freqs: np.ndarray
    psd: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    log_var: np.ndarray
    dof: np.ndarray
    n_tapers: np.ndarray
    n_segments: np.ndarray
    nw: float
    bandwidth: float
    confidence: float


def _check_tapers(nw: float, k: Optional[int]) -> int:
    k = int(2 * nw - 1) if k is None else int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > int(2 * nw):
        raise ValueError(f"k={k} tapers for NW={nw}: beyond 2*NW the tapers are poorly concentrated")
    return k


def _finite_runs(x: np.ndarray):
    """(start, stop) of every run of finite samples."""
    ok = np.isfinite(x)
    edges = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _eigenspectra(seg: np.ndarray, tapers: np.ndarray, fs: float, nfft: int) -> np.ndarray:
    """One-sided eigenspectra ``(..., k, nfft//2 + 1)`` of segments ``(..., n)``."""
    seg = seg - seg.mean(axis=-1, keepdims=True)
    xf = np.fft.rfft(seg[..., None, :] * tapers, n=nfft, axis=-1)
    s = (np.abs(xf) ** 2) / fs
    s[..., 1:] *= 2.0
    if nfft % 2 == 0:
        s[..., -1] /= 2.0                                        # the Nyquist bin is not doubled
    return s


def _edge_dof(freqs: np.ndarray, fs: float, w: float, k_total: int) -> np.ndarray:
    dof = np.full(freqs.shape, 2.0 * k_total)
    dof[(freqs < w) | (freqs > fs / 2 - w)] = float(k_total)
    return dof


def _estimate(eig: np.ndarray, dof: np.ndarray, confidence: float, method: str):
    """(psd, low, high, log_var) from ``(K, F)`` eigenspectra."""
    k = eig.shape[0]
    psd = eig.mean(axis=0)
    alpha = 1.0 - confidence
    log_var = np.full(psd.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        if k >= 2:
            # Thomson & Chave (1991): jackknife the log spectrum over the eigenspectra.
            loo = np.log((eig.sum(axis=0)[None, :] - eig) / (k - 1))
            log_var = (k - 1) / k * ((loo - loo.mean(axis=0)) ** 2).sum(axis=0)
        if method == "chi2" or k < 2:
            low = dof * psd / stats.chi2.ppf(1 - alpha / 2, dof)
            high = dof * psd / stats.chi2.ppf(alpha / 2, dof)
        elif method == "jackknife":
            t = stats.t.ppf(1 - alpha / 2, k - 1)
            low, high = np.exp(np.log(psd) - t * np.sqrt(log_var)), np.exp(np.log(psd) + t * np.sqrt(log_var))
        else:
            raise ValueError("ci_method is 'jackknife' or 'chi2'")
    return psd, low, high, log_var


def multitaper_psd(x, fs: float, nw: float = 4.0, k: Optional[int] = None, nfft: Optional[int] = None,
                   confidence: float = 0.95, ci_method: str = "jackknife", nan_policy: str = "raise",
                   segment_samples: Optional[int] = None) -> MultitaperSpectrum:
    """Multitaper PSD of ``x`` along its last axis.

    Parameters
    ----------
    x : array_like
        Signal, ``(..., n_samples)``.
    fs : float
        Sampling rate, Hz.
    nw : float
        Time-half-bandwidth product; the half-bandwidth is ``W = nw / T`` Hz.
    k : int, optional
        Number of tapers, default ``2 * nw - 1``; at most ``2 * nw``.
    nfft : int, optional
        FFT length, default the next power of two of the segment length.
    confidence : float
        Confidence level of ``ci_low`` / ``ci_high``.
    ci_method : {'jackknife', 'chi2'}
    nan_policy : {'raise', 'omit'}
        See the module notes on missing data.
    segment_samples : int, optional
        ``'omit'`` only: the length every segment is cut to, default the longest finite
        run (of any series).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0 or x.shape[-1] < 2:
        raise ValueError("x needs at least two samples along its last axis")
    if fs <= 0:
        raise ValueError("fs must be positive")
    k = _check_tapers(nw, k)
    if nan_policy not in ("raise", "omit"):
        raise ValueError("nan_policy is 'raise' or 'omit'")
    if ci_method not in ("jackknife", "chi2"):
        raise ValueError("ci_method is 'jackknife' or 'chi2'")
    flat = x.reshape(-1, x.shape[-1])
    if nan_policy == "raise":
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/inf; pass nan_policy='omit' to use the finite runs")
        runs = [[(0, x.shape[-1])] for _ in flat]
        length = x.shape[-1]
    else:
        runs = [_finite_runs(row) for row in flat]
        longest = max([b - a for r in runs for a, b in r], default=0)
        length = longest if segment_samples is None else int(segment_samples)
    if length < max(2 * k + 2, 16):
        raise ValueError(f"segments of {length} samples are too short for {k} tapers")
    nfft = int(2 ** np.ceil(np.log2(length))) if nfft is None else int(nfft)
    if nfft < length:
        raise ValueError("nfft is shorter than a segment")
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    w = nw * fs / length
    tapers = np.atleast_2d(windows.dpss(length, nw, Kmax=k))

    shape = x.shape[:-1] + (freqs.size,)
    out = {name: np.full((flat.shape[0], freqs.size), np.nan) for name in ("psd", "low", "high", "log_var", "dof")}
    n_tapers = np.zeros(flat.shape[0], dtype=int)
    n_segments = np.zeros(flat.shape[0], dtype=int)
    for i, (row, row_runs) in enumerate(zip(flat, runs)):
        starts = [a + j * length for a, b in row_runs for j in range((b - a) // length)]
        finite = sum(b - a for a, b in row_runs)
        if nan_policy == "omit" and finite and len(starts) * length < finite / 2:
            warnings.warn(f"only {len(starts) * length} of {finite} finite samples fit whole segments of "
                          f"{length}; pass a shorter segment_samples to use more", RuntimeWarning, stacklevel=2)
        if not starts:
            continue
        segs = np.stack([row[s:s + length] for s in starts])
        eig = _eigenspectra(segs, tapers, fs, nfft).reshape(-1, freqs.size)
        dof = _edge_dof(freqs, fs, w, eig.shape[0])
        out["psd"][i], out["low"][i], out["high"][i], out["log_var"][i] = _estimate(eig, dof, confidence, ci_method)
        out["dof"][i] = dof
        n_tapers[i], n_segments[i] = eig.shape[0], len(starts)
    return MultitaperSpectrum(freqs, out["psd"].reshape(shape), out["low"].reshape(shape),
                              out["high"].reshape(shape), out["log_var"].reshape(shape), out["dof"].reshape(shape),
                              n_tapers.reshape(x.shape[:-1]), n_segments.reshape(x.shape[:-1]),
                              float(nw), float(w), float(confidence))


def two_spectrum_test(a: MultitaperSpectrum, b: MultitaperSpectrum, method: str = "jackknife"
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Are ``a`` and ``b`` estimates of the same spectrum? Frequency by frequency.

    Returns ``(ratio, p)``: ``a.psd / b.psd`` and a two-sided p-value per frequency.

    * ``method='jackknife'`` (default): ``ln(a) - ln(b)`` over the jackknife standard
      error of both, against a t distribution with ``K_a + K_b - 2`` degrees of freedom
      (in the spirit of Bokil et al., 2007). Robust to correlated eigenspectra.
    * ``method='f'``: the ratio against ``F(dof_a, dof_b)`` -- exact for independent
      eigenspectra of a locally flat spectrum, too liberal on steep ones (see the module
      notes).

    The frequency grids must match (same ``fs`` and ``nfft``) and the two estimates should
    share ``nw`` and segment length, or one compares different resolutions. Neighbouring
    frequencies within the bandwidth are not independent -- correct for multiple
    comparisons accordingly.
    """
    if a.freqs.shape != b.freqs.shape or not np.allclose(a.freqs, b.freqs):
        raise ValueError("the two spectra are on different frequency grids")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a.psd / b.psd
        if method == "jackknife":
            df = (np.asarray(a.n_tapers) + np.asarray(b.n_tapers) - 2)[..., None]
            z = np.log(ratio) / np.sqrt(a.log_var + b.log_var)
            return ratio, 2 * stats.t.sf(np.abs(z), df)
        if method == "f":
            cdf = stats.f.cdf(ratio, a.dof, b.dof)
            return ratio, np.minimum(1.0, 2 * np.minimum(cdf, 1 - cdf))
    raise ValueError("method is 'jackknife' or 'f'")


def multitaper_spectrogram(x, fs: float, window_s: float, step_s: Optional[float] = None, nw: float = 3.0,
                           k: Optional[int] = None, nfft: Optional[int] = None, chunk: int = 64):
    """Multitaper spectrogram of ``x`` along its last axis.

    Returns ``(times, freqs, power)`` -- ``times`` the window centres in seconds from the
    first sample, ``power`` of shape ``(..., n_freqs, n_times)``, each column exactly
    :func:`multitaper_psd` of that window. A window holding any NaN/inf is NaN (missing),
    never estimated across the gap. Windows are processed ``chunk`` at a time to bound
    memory.
    """
    x = np.asarray(x, dtype=float)
    n_win = int(round(window_s * fs))
    step = int(round((step_s if step_s is not None else window_s / 2) * fs))
    k = _check_tapers(nw, k)
    if n_win < max(2 * k + 2, 16) or step < 1:
        raise ValueError("window_s and step_s are too short for fs and the tapers")
    if x.shape[-1] < n_win:
        raise ValueError("the signal is shorter than one window")
    nfft = int(2 ** np.ceil(np.log2(n_win))) if nfft is None else int(nfft)
    if nfft < n_win:
        raise ValueError("nfft is shorter than a window")
    starts = np.arange(0, x.shape[-1] - n_win + 1, step)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    tapers = np.atleast_2d(windows.dpss(n_win, nw, Kmax=k))
    power = np.empty(x.shape[:-1] + (starts.size, freqs.size))
    for c in range(0, starts.size, max(1, int(chunk))):
        part = starts[c:c + chunk]
        seg = np.stack([x[..., s:s + n_win] for s in part], axis=-2)            # (..., T, n_win)
        p = _eigenspectra(seg, tapers, fs, nfft).mean(axis=-2)
        p[~np.isfinite(seg).all(axis=-1)] = np.nan
        power[..., c:c + part.size, :] = p
    times = (starts + (n_win - 1) / 2.0) / fs
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


def _at(freqs: np.ndarray, psd: np.ndarray, f: float) -> np.ndarray:
    """``psd`` linearly interpolated at frequency ``f`` along the last axis."""
    i = int(np.clip(np.searchsorted(freqs, f) - 1, 0, freqs.size - 2))
    w = (f - freqs[i]) / (freqs[i + 1] - freqs[i])
    return psd[..., i] * (1 - w) + psd[..., i + 1] * w


def band_power(freqs, psd, bands: Dict[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    """Power in each named band ``[lo, hi]`` Hz: the PSD integrated over exactly that
    interval (interpolated at the edges), along the last axis."""
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    out = {}
    for name, (lo, hi) in bands.items():
        if not freqs[0] <= lo < hi <= freqs[-1]:
            raise ValueError(f"band {name!r} ({lo}-{hi} Hz) is not inside {freqs[0]}-{freqs[-1]} Hz")
        inner = (freqs > lo) & (freqs < hi)
        grid = np.concatenate(([lo], freqs[inner], [hi]))
        values = np.concatenate((_at(freqs, psd, lo)[..., None], psd[..., inner], _at(freqs, psd, hi)[..., None]),
                                axis=-1)
        out[name] = trapezoid(values, grid, axis=-1)
    return out
