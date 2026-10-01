# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NaN / gap handling shared by the spike detectors (pre- and post-processing).

The detectors themselves never see a NaN. Missing data are handled in two dedicated steps
around the unchanged detection algorithm:

1. :func:`prepare_signal` (pre) -- find the gaps (runs of NaN) per channel on the
   **original** signal and, with ``nan_policy='fill'``, fill them with
   :func:`brainmaze_utils.gaps.fill_gaps` (short gaps linearly, long gaps with pink noise
   cross-faded into the mirrored neighbouring signal, so the fill looks like background and
   does not distort the detectors' running background estimates or ring through filters).
   With ``nan_policy='raise'`` any NaN raises :class:`ValueError`.
2. :func:`in_gap_mask` (post) -- flag detections that fall inside a gap or within
   ``margin_s`` of one (via :func:`brainmaze_utils.gaps.mask_in_gaps`) so the detector can
   drop them. Detections on filled samples are never real.

Without this, a single NaN propagates through ``filtfilt``/FFT/Hilbert to the whole channel
(Janca: zero detections, silently) or through ``np.median`` to the block scaling factor of
every channel (Barkmeier).

``+/-inf`` is never treated as a gap: it raises :class:`ValueError` under every policy.
"""

import warnings

import numpy as np

NAN_POLICIES = ('fill', 'raise')


def _gaps_module():
    try:
        from brainmaze_utils import gaps
    except ImportError as exc:      # pragma: no cover - depends on installed version
        raise ImportError(
            "nan_policy='fill' needs brainmaze_utils.gaps (brainmaze_utils>=2.1.0). Upgrade "
            "brainmaze_utils, or remove the NaNs yourself and/or use nan_policy='raise'."
        ) from exc
    return gaps


def prepare_signal(x, fs, nan_policy='fill', fill_kwargs=None):
    """
    Pre-processing step: validate finiteness and fill NaN gaps.

    Parameters
    ----------
    x : np.ndarray, shape (n_channels, n_samples)
        Signal, float64, time along the last axis.
    fs : float
        Sampling frequency (Hz).
    nan_policy : {'fill', 'raise'}
        See module docstring.
    fill_kwargs : dict, optional
        Extra keyword arguments for :func:`brainmaze_utils.gaps.fill_gaps`
        (e.g. ``max_interp_s``, ``method``, ``seed``).

    Returns
    -------
    y : np.ndarray, shape (n_channels, n_samples)
        Finite signal (a copy when anything was filled). Channels that are entirely NaN are
        returned as zeros and listed in ``dead``.
    gaps : list of np.ndarray
        Per channel, ``(n_gaps, 2)`` ``[start, stop)`` sample indices of the NaN runs of the
        **original** signal (empty arrays when there are none).
    dead : np.ndarray of bool, shape (n_channels,)
        True for channels that contain no finite sample at all (no detections are possible).
    """
    if nan_policy not in NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {NAN_POLICIES}, got {nan_policy!r}")
    if np.isinf(x).any():
        raise ValueError('signal contains +/-inf; only NaN is accepted as a gap marker')
    nan = np.isnan(x)
    n_ch = x.shape[0]
    empty = [np.zeros((0, 2), dtype=np.int64) for _ in range(n_ch)]
    if not nan.any():
        return x, empty, np.zeros(n_ch, dtype=bool)
    if nan_policy == 'raise':
        bad = np.flatnonzero(nan.any(axis=1)).tolist()
        raise ValueError(f"signal contains NaN (channels {bad}); use nan_policy='fill' or "
                         "remove the gaps first")

    g = _gaps_module()
    dead = nan.all(axis=1)
    gaps = [g.find_gaps(x[c]) if nan[c].any() else empty[c] for c in range(n_ch)]
    y = np.array(x, dtype=np.float64, copy=True)
    kw = dict(fill_kwargs or {})
    for c in np.flatnonzero(nan.any(axis=1) & ~dead):
        y[c] = g.fill_gaps(x[c], fs, **kw)
    if dead.any():
        warnings.warn(f'channel(s) {np.flatnonzero(dead).tolist()} are entirely NaN; '
                      'no detections are reported for them', RuntimeWarning, stacklevel=3)
        y[dead] = 0.0
    return y, gaps, dead


def in_gap_mask(times_s, gaps, fs, margin_s):
    """
    Post-processing step: boolean mask of detections (times in seconds) that fall inside a
    gap of the original signal or within ``margin_s`` seconds of one.
    """
    times_s = np.atleast_1d(np.asarray(times_s, dtype=float))
    if gaps is None or len(gaps) == 0 or times_s.size == 0:
        return np.zeros(times_s.shape, dtype=bool)
    return _gaps_module().mask_in_gaps(times_s, gaps, fs=fs, margin_s=margin_s)


def gap_sample_mask(gaps, n_samples):
    """Boolean ``(n_samples,)`` mask, True on samples that were NaN in the original signal."""
    m = np.zeros(n_samples, dtype=bool)
    for s, e in np.asarray(gaps, dtype=np.int64).reshape(-1, 2):
        m[s:e] = True
    return m
