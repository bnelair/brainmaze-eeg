# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gap-aware wrapper for the spike detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The detectors in this package are **raw**: they run their algorithm on a finite signal and
raise ``ValueError`` on NaN/inf. :class:`GapAwareSpikeDetector` is the layer for signals with
missing data (e.g. artifacts replaced by NaN). Around any detector that follows the protocol
below, it

1. finds the gaps (runs of NaN/inf) of every channel of the **original** signal;
2. fills them -- gaps up to ``short_gap_s`` by linear interpolation, longer ones with
   ``fill`` (``'mirror'``, ``'pink'`` or ``'linear'``, see
   :mod:`brainmaze_eeg.spikes._gaps`) -- so that filters and background estimates run on a
   signal without steps or silent stretches;
3. runs the detector on the filled signal (channels that are entirely NaN/inf are left out,
   so they cannot bias cross-channel statistics such as Barkmeier's median scaling);
4. removes every detection inside a gap or within ``edge_margin_s`` of it;
5. reports the gaps and the valid time per channel, so rates can be normalised by the time
   in which a detection could have been reported.

On a gap-free signal the result is identical to calling the detector directly.

Detector protocol
-----------------
An object with a method ``detect(x, fs)`` where ``x`` is a finite ``float64`` array of shape
``(n_channels, n_samples)``, returning a list of length ``n_channels``; element ``c`` is
either

- a 1-D integer array of detection **sample indices** into ``x[c]``, or
- a list of dicts, each with an integer ``'peak_index'`` (sample index); a ``'channel'`` key,
  if present, is rewritten to the channel index of the caller's array.

If the object has ``accepts_valid = True``, ``detect`` is called as
``detect(x, fs, valid=mask)`` with a boolean ``(n_channels, n_samples)`` mask that is False
on filled samples, so the detector can exclude them from its statistics.

Implementations: :class:`~brainmaze_eeg.spikes.janca.JancaDetector`,
:class:`~brainmaze_eeg.spikes.barkmeier.BarkmeierDetector` and
:class:`~brainmaze_eeg.spikes.janca.SpikeDetectorHilbert` (its ``detect`` method).

Example
-------
>>> from brainmaze_eeg.spikes import GapAwareSpikeDetector, JancaDetector
>>> det = GapAwareSpikeDetector(JancaDetector(powerline=60))
>>> spikes, info = det.detect(x, fs, return_info=True)    # x: (n_channels, n_samples)
>>> rate_per_min = [len(s) / (v / 60) for s, v in zip(spikes, info['valid_s'])]
"""

import warnings

import numpy as np

from brainmaze_eeg.spikes._gaps import FILL_METHODS, fill_gaps, find_gaps, gap_mask, mask_in_gaps

__all__ = ['GapAwareSpikeDetector']


def _indices_of(item):
    """Sample indices of one channel's detections (array or list of dicts)."""
    if isinstance(item, np.ndarray):
        return item.astype(np.int64, copy=False), False
    item = list(item)
    if item and isinstance(item[0], dict):
        return np.array([d['peak_index'] for d in item], dtype=np.int64), True
    return np.asarray(item, dtype=np.int64).reshape(-1), False


class GapAwareSpikeDetector:
    """
    Run a spike detector on a signal with gaps (NaN/inf); see the module docstring.

    Parameters
    ----------
    detector : object or class
        A detector instance following the protocol (e.g. ``JancaDetector()``,
        ``BarkmeierDetector()``), or a detector class, instantiated with
        ``detector_kwargs``.
    detector_kwargs : dict, optional
        Constructor arguments when ``detector`` is a class.
    short_gap_s : float
        Gaps up to this length (s) are linearly interpolated (default 0.1).
    fill : {'mirror', 'pink', 'linear'}
        Fill of longer gaps (default ``'mirror'``; see the README for the measurements
        behind this choice).
    edge_margin_s : float
        Detections within this distance (s) of a gap, or inside it, are removed
        (default 0.1).
    seed : int or None
        Seed of the ``'pink'`` fill noise (default 0: reproducible).
    fill_kwargs : dict, optional
        Further options of :func:`brainmaze_eeg.spikes._gaps.fill_gaps` (``context_s``,
        ``taper_s``, ``beta``).
    """

    def __init__(self, detector, detector_kwargs=None, *, short_gap_s=0.1, fill='mirror',
                 edge_margin_s=0.1, seed=0, fill_kwargs=None):
        if isinstance(detector, type):
            detector = detector(**(detector_kwargs or {}))
        elif detector_kwargs:
            raise TypeError('detector_kwargs is only used when detector is a class')
        if not callable(getattr(detector, 'detect', None)):
            raise TypeError(f'{detector!r} does not implement detect(x, fs) (see the detector '
                            'protocol in brainmaze_eeg.spikes.gap_aware)')
        if fill not in FILL_METHODS:
            raise ValueError(f'fill must be one of {FILL_METHODS}, got {fill!r}')
        if not short_gap_s >= 0:
            raise ValueError(f'short_gap_s must be >= 0, got {short_gap_s}')
        if not edge_margin_s >= 0:
            raise ValueError(f'edge_margin_s must be >= 0, got {edge_margin_s}')
        fill_kwargs = dict(fill_kwargs or {})
        bad = set(fill_kwargs) - {'context_s', 'taper_s', 'beta'}
        if bad:
            raise TypeError(f'unsupported fill_kwargs {sorted(bad)}; use the constructor '
                            'arguments short_gap_s, fill and seed')
        self.detector = detector
        self.short_gap_s = short_gap_s
        self.fill = fill
        self.edge_margin_s = edge_margin_s
        self.seed = seed
        self.fill_kwargs = fill_kwargs

    def __repr__(self):
        return (f'GapAwareSpikeDetector({self.detector!r}, short_gap_s={self.short_gap_s}, '
                f'fill={self.fill!r}, edge_margin_s={self.edge_margin_s}, seed={self.seed})')

    def _excluded_samples(self, gaps, n, fs):
        """Number of samples in which a detection would be removed (gaps + margins)."""
        if len(gaps) == 0:
            return 0
        m = self.edge_margin_s * fs
        lo = np.clip(np.ceil(gaps[:, 0] - m), 0, n).astype(np.int64)
        hi = np.clip(np.ceil(gaps[:, 1] + m), 0, n).astype(np.int64)
        total, cur_lo, cur_hi = 0, lo[0], hi[0]
        for a, b in zip(lo[1:], hi[1:]):
            if a > cur_hi:
                total += cur_hi - cur_lo
                cur_lo, cur_hi = a, b
            else:
                cur_hi = max(cur_hi, b)
        return int(total + cur_hi - cur_lo)

    def detect(self, x, fs, return_info=False, return_mask=False):
        """
        Detect spikes in ``x`` with gaps handled.

        Parameters
        ----------
        x : np.ndarray
            ``(n_samples,)`` or ``(n_channels, n_samples)``; NaN/inf mark missing data.
        fs : float
            Sampling frequency (Hz).
        return_info : bool
            Also return a dict (see Returns).
        return_mask : bool
            Include ``info['gap_mask']`` (bool, shape of ``x``; True on missing samples).

        Returns
        -------
        detections
            The detector's output with in/near-gap detections removed: for 2-D input a list
            with one entry per channel (int64 sample-index array, or list of dicts), for 1-D
            input that single entry. Entirely missing channels give an empty entry.
        info : dict
            Only with ``return_info=True``. Per channel: ``gaps`` (``(n_gaps, 2)``
            ``[start, stop)`` samples), ``gap_intervals_s`` (same in seconds), ``all_nan``
            (bool), ``n_removed`` (detections dropped in/near gaps), ``valid_s`` (seconds in
            which a detection can be reported: record length minus gaps widened by
            ``edge_margin_s``; normalise rates by this), ``valid_fraction``; and
            ``gap_mask`` with ``return_mask=True``.
        """
        fs = float(fs)
        if not fs > 0:
            raise ValueError(f'fs must be > 0, got {fs}')
        x = np.asarray(x, dtype=np.float64)
        one_d = x.ndim == 1
        if one_d:
            x = x[np.newaxis, :]
        elif x.ndim != 2:
            raise ValueError(f'x must be 1-D or 2-D (n_channels, n_samples), got {x.ndim}-D')
        elif x.shape[0] > x.shape[1]:
            raise ValueError(f'x has shape {x.shape}: more channels than samples. The layout is '
                             '(n_channels, n_samples); transpose your array (x.T).')
        n_ch, n = x.shape

        finite = np.isfinite(x)
        all_nan = ~finite.any(axis=1)
        gaps = [find_gaps(x[c]) if not finite[c].all() else np.zeros((0, 2), np.int64)
                for c in range(n_ch)]
        if all_nan.any():
            warnings.warn(f'channel(s) {np.flatnonzero(all_nan).tolist()} contain no finite '
                          'sample; no detections are reported for them (info["all_nan"])',
                          RuntimeWarning, stacklevel=2)
        live = np.flatnonzero(~all_nan)

        out = [None] * n_ch
        n_removed = np.zeros(n_ch, dtype=np.int64)
        if live.size:
            y = x[live].copy()
            for k, c in enumerate(live):
                if len(gaps[c]):
                    y[k] = fill_gaps(x[c], fs, max_interp_s=self.short_gap_s,
                                     method=self.fill, seed=self.seed, **self.fill_kwargs)
            if getattr(self.detector, 'accepts_valid', False):
                res = self.detector.detect(y, fs, valid=finite[live])
            else:
                res = self.detector.detect(y, fs)
            if not isinstance(res, (list, tuple)) or len(res) != live.size:
                raise TypeError(f'{self.detector!r}.detect returned {type(res).__name__} of '
                                f'length {getattr(res, "__len__", lambda: "?")()}; expected a '
                                f'list with one entry per channel ({live.size})')
            for k, c in enumerate(live):
                idx, records = _indices_of(res[k])
                keep = ~mask_in_gaps(idx, gaps[c], fs, units='samples',
                                     margin_s=self.edge_margin_s)
                n_removed[c] = int((~keep).sum())
                if records:
                    item = []
                    for d, kp in zip(res[k], keep):
                        if kp:
                            d = dict(d)
                            if 'channel' in d:
                                d['channel'] = int(c)
                            item.append(d)
                    out[c] = item
                elif isinstance(res[k], np.ndarray):
                    out[c] = res[k][keep]
                else:
                    out[c] = idx[keep]
        empty_records = live.size and isinstance(out[live[0]], list)
        for c in np.flatnonzero(all_nan):
            out[c] = [] if empty_records else np.zeros(0, dtype=np.int64)

        result = out[0] if one_d else out
        if not return_info:
            return result
        valid_s = np.array([0.0 if all_nan[c] else
                            (n - self._excluded_samples(gaps[c], n, fs)) / fs
                            for c in range(n_ch)])
        info = {'gaps': gaps, 'gap_intervals_s': [g / fs for g in gaps], 'all_nan': all_nan,
                'n_removed': n_removed, 'valid_s': valid_s, 'valid_fraction': valid_s / (n / fs)}
        if return_mask:
            info['gap_mask'] = np.vstack([gap_mask(g, n) for g in gaps])
        if one_d:
            info = {k: v[0] for k, v in info.items()}
        return result, info
