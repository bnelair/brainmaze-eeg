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
missing data. Around any detector that follows the protocol below, it

1. finds the gaps of every channel of the **original** signal: runs of NaN/inf and, by
   default, runs of an exactly constant value lasting at least ``flat_as_gap_s`` (0.1 s;
   see "Missing data stored as a constant" below);
2. fills them -- gaps up to ``short_gap_s`` (0.02 s) by linear interpolation, longer ones
   with ``fill`` (``'mirror'``, ``'pink'`` or ``'linear'``, see
   :mod:`brainmaze_eeg.spikes._gaps`) -- so that filters and background estimates run on a
   signal without steps or silent stretches;
3. runs the detector on the filled signal (channels that are entirely missing are left out,
   so they cannot bias cross-channel statistics such as Barkmeier's median scaling;
   detectors that declare ``channel_independent = True`` are fed one channel at a time to
   bound memory);
4. removes every detection inside a gap or within ``edge_margin_s`` (0.2 s) of it -- for
   short, interpolated gaps as well as long ones;
5. reports the gaps and the valid time per channel, so rates can be normalised by the time
   in which a detection could have been reported (a ``RuntimeWarning`` names channels with
   less than half of the record valid).

On a gap-free signal the result is identical to calling the detector directly.

Missing data stored as a constant
---------------------------------
Many recordings do **not** store missing data as NaN: MEF/EDF exports and acquisition
systems often write a constant (0, the last value, or a fixed code) during dropouts. A
raw detector sees steps at both ends of such a flat run and fires trains of false
detections there (on the 6.8 h eeg_forge parity recording, constant runs inside its
``data_present == 0`` intervals produced ~70-85 of the 1494 detections). Convert missing
data to NaN when you know where it is (e.g. ``x[~data_present] = np.nan``). As a safety net
the wrapper treats any run of exactly equal consecutive samples lasting at least
``flat_as_gap_s`` seconds as a gap (default 0.1 s; ``None`` disables it). Evidence for the
default: in that recording every constant run of >= 0.1 s is missing data, and constant
runs within real signal last at most 4 ms (2 samples). Saturated (clipped) stretches are
caught too, which is intended. A channel that is constant throughout becomes an
all-missing channel (flagged, no detections).

Short gaps and the margin (trade-off)
-------------------------------------
Linear interpolation is right for very short dropouts but carries no band power, so with
many longer interpolated gaps a background-modelling detector (Janca) sees a lower
background and a lower threshold *everywhere*: on 30 min of real Fz-Cz with 100 ms
dropouts every 1 s, the old default ``short_gap_s=0.1`` gave 173 detections in valid time
vs 108 without gaps (+67). With ``short_gap_s=0.02`` (gaps > 20 ms mirrored), the same
probe gives 110 vs 108, and every tested dropout pattern (4-200 ms, every 0.1-2 s) stays
within a few detections of the gap-free run (``brainmaze-work/scratch/eeg-spikes/r2/``).
Even 1-2 sample gaps move nearby detections by > 20 ms, so the margin applies to every gap.
The price is valid time: with ``edge_margin_s=0.2`` a gap every 0.4 s or less leaves no
valid time at all. That is reported (``info['valid_s']``, warning) rather than hidden;
lower ``edge_margin_s`` only if you accept detections influenced by the fill.

Detector protocol
-----------------
An object with a method ``detect(x, fs)`` where ``x`` is a finite ``float64`` array of shape
``(n_channels, n_samples)``, returning a list of length ``n_channels``; element ``c`` is
either

- a 1-D integer array of detection **sample indices** into ``x[c]``, or
- a list of dicts, each with an integer ``'peak_index'`` (sample index); a ``'channel'`` key,
  if present, is rewritten to the channel index of the caller's array. Such detectors set
  the class attribute ``output = 'records'`` so that empty channels keep the list type
  (``output = 'indices'`` otherwise).

If the object has ``accepts_valid = True``, ``detect`` is called as
``detect(x, fs, valid=mask)`` with a boolean ``(n_channels, n_samples)`` mask that is False
on filled samples, so the detector can exclude them from its statistics. If it has
``channel_independent = True`` (its result for one channel never depends on the others),
the wrapper calls it once per channel with a ``(1, n_samples)`` array.

Implementations: :class:`~brainmaze_eeg.spikes.janca.JancaDetector` (also with
``preset='ripple'``), :class:`~brainmaze_eeg.spikes.barkmeier.BarkmeierDetector` and
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

from brainmaze_eeg.spikes import _checks as chk
from brainmaze_eeg.spikes._gaps import FILL_METHODS, fill_gaps, find_gaps, gap_mask, mask_in_gaps

__all__ = ['GapAwareSpikeDetector']


def _indices_of(item, records_hint):
    """Sample indices of one channel's detections (array or list of dicts)."""
    if isinstance(item, np.ndarray):
        return item.astype(np.int64, copy=False), False
    item = list(item)
    if (item and isinstance(item[0], dict)) or (not item and records_hint):
        return np.array([d['peak_index'] for d in item], dtype=np.int64), True
    return np.asarray(item, dtype=np.int64).reshape(-1), False


class GapAwareSpikeDetector:
    """
    Run a spike detector on a signal with gaps (NaN/inf, and constant runs); see the module
    docstring.

    Parameters
    ----------
    detector : object or class
        A detector instance following the protocol (e.g. ``JancaDetector()``,
        ``JancaDetector('ripple')``, ``BarkmeierDetector()``), or a detector class,
        instantiated with ``detector_kwargs``.
    detector_kwargs : dict, optional
        Constructor arguments when ``detector`` is a class.
    short_gap_s : float
        Gaps up to this length (s) are linearly interpolated (default 0.02; see the module
        docstring for why not longer). Finite, >= 0.
    fill : {'mirror', 'pink', 'linear'}
        Fill of longer gaps (default ``'mirror'``; see the README for the measurements
        behind this choice). Always passed explicitly to the fill function.
    edge_margin_s : float
        Detections within this distance (s) of a gap, or inside it, are removed
        (default 0.2: covers the mirror fill's influence on peak selection, measured at
        0.10-0.16 s from the edge, and ``min_distance_s`` + half a spike). Finite, >= 0.
    flat_as_gap_s : float or None
        Runs of exactly equal consecutive samples lasting at least this long (s) are treated
        as gaps (default 0.1; ``None`` disables). See "Missing data stored as a constant".
    seed : None, int, sequence of int, np.random.SeedSequence or np.random.Generator
        Seed of the ``'pink'`` fill noise (default 0: reproducible). Validated at
        construction.
    fill_kwargs : dict, optional
        Further options of :func:`brainmaze_eeg.spikes._gaps.fill_gaps` (``context_s`` > 0,
        ``taper_s`` >= 0, ``beta``; all finite). Validated at construction.
    """

    def __init__(self, detector, detector_kwargs=None, *, short_gap_s=0.02, fill='mirror',
                 edge_margin_s=0.2, flat_as_gap_s=0.1, seed=0, fill_kwargs=None):
        if isinstance(detector, type):
            detector = detector(**(detector_kwargs or {}))
        elif detector_kwargs:
            raise TypeError('detector_kwargs is only used when detector is a class')
        if not callable(getattr(detector, 'detect', None)):
            raise TypeError(f'{detector!r} does not implement detect(x, fs) (see the detector '
                            'protocol in brainmaze_eeg.spikes.gap_aware)')
        if not isinstance(fill, str) or fill not in FILL_METHODS:
            raise ValueError(f'fill must be one of {FILL_METHODS}, got {fill!r}')
        short_gap_s = chk.number('short_gap_s', short_gap_s, ge=0)
        edge_margin_s = chk.number('edge_margin_s', edge_margin_s, ge=0)
        flat_as_gap_s = chk.number('flat_as_gap_s', flat_as_gap_s, gt=0, allow_none=True)
        fill_kwargs = dict(fill_kwargs or {})
        bad = set(fill_kwargs) - {'context_s', 'taper_s', 'beta'}
        if bad:
            raise TypeError(f'unsupported fill_kwargs {sorted(bad)}; use the constructor '
                            'arguments short_gap_s, fill and seed')
        if 'context_s' in fill_kwargs:
            fill_kwargs['context_s'] = chk.number('context_s', fill_kwargs['context_s'], gt=0)
        if 'taper_s' in fill_kwargs:
            fill_kwargs['taper_s'] = chk.number('taper_s', fill_kwargs['taper_s'], ge=0)
        if 'beta' in fill_kwargs:
            fill_kwargs['beta'] = chk.number('beta', fill_kwargs['beta'])
        _check_seed(seed)
        self.detector = detector
        self.short_gap_s = short_gap_s
        self.fill = fill
        self.edge_margin_s = edge_margin_s
        self.flat_as_gap_s = flat_as_gap_s
        self.seed = seed
        self.fill_kwargs = fill_kwargs

    def __repr__(self):
        return (f'GapAwareSpikeDetector({self.detector!r}, short_gap_s={self.short_gap_s}, '
                f'fill={self.fill!r}, edge_margin_s={self.edge_margin_s}, '
                f'flat_as_gap_s={self.flat_as_gap_s}, seed={self.seed})')

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

    def _missing_row(self, row, fs):
        """
        ``(working_row, flat_runs)``: the row as float64 with constant runs of at least
        ``flat_as_gap_s`` set to NaN, and those runs as ``[start, stop)`` samples.
        """
        y = np.array(row, dtype=np.float64, copy=True)
        if self.flat_as_gap_s is None or y.size < 2:
            return y, np.zeros((0, 2), np.int64)
        min_len = max(int(np.ceil(self.flat_as_gap_s * fs - 1e-9)), 2)
        eq = y[1:] == y[:-1]                       # NaN never equals anything
        d = np.diff(np.concatenate(([0], eq.astype(np.int8), [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1) + 1   # sample runs
        keep = (en - st) >= min_len
        runs = np.stack([st[keep], en[keep]], axis=1).astype(np.int64)
        for a, b in runs:
            y[a:b] = np.nan
        return y, runs

    def _fill(self, y, fs):
        # method and max_interp_s are ALWAYS explicit: a different default in the fill
        # module (e.g. brainmaze_utils.gaps' 'spectral') must not change this wrapper.
        return fill_gaps(y, fs, max_interp_s=self.short_gap_s, method=self.fill,
                         seed=self.seed, **self.fill_kwargs)

    def detect(self, x, fs, return_info=False, return_mask=False):
        """
        Detect spikes in ``x`` with gaps handled.

        Parameters
        ----------
        x : np.ndarray
            ``(n_samples,)`` or ``(n_channels, n_samples)``; NaN/inf (and constant runs of
            at least ``flat_as_gap_s``) mark missing data. Not modified; any real dtype
            (float32 is converted one channel at a time).
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
            ``[start, stop)`` samples, NaN/inf and flat runs merged), ``flat_runs`` (the
            constant runs treated as gaps), ``gap_intervals_s`` (gaps in seconds),
            ``all_nan`` (bool: no usable sample), ``n_removed`` (detections dropped in/near
            gaps), ``valid_s`` (seconds in which a detection can be reported: record length
            minus gaps widened by ``edge_margin_s``; normalise rates by this),
            ``valid_fraction``; and ``gap_mask`` with ``return_mask=True``.
        """
        fs = chk.number('fs', fs, gt=0)
        x = np.asarray(x)
        if np.iscomplexobj(x) or not (np.issubdtype(x.dtype, np.number)
                                      or np.issubdtype(x.dtype, np.bool_)):
            raise TypeError(f'x must be a real numeric array, got dtype {x.dtype}')
        one_d = x.ndim == 1
        if one_d:
            x = x[np.newaxis, :]
        elif x.ndim != 2:
            raise ValueError(f'x must be 1-D or 2-D (n_channels, n_samples), got {x.ndim}-D')
        elif x.shape[0] > x.shape[1]:
            raise ValueError(f'x has shape {x.shape}: more channels than samples. The layout is '
                             '(n_channels, n_samples); transpose your array (x.T).')
        n_ch, n = x.shape

        # -- 1. gaps per channel (NaN/inf + flat runs), on the original signal -------------
        gaps, flats, all_nan = [], [], np.zeros(n_ch, dtype=bool)
        for c in range(n_ch):
            row, runs = self._missing_row(x[c], fs)
            fin = np.isfinite(row)
            all_nan[c] = not fin.any()
            gaps.append(find_gaps(row) if not fin.all() else np.zeros((0, 2), np.int64))
            flats.append(runs)
        if all_nan.any():
            warnings.warn(f'channel(s) {np.flatnonzero(all_nan).tolist()} contain no usable '
                          'sample (no finite sample, or constant throughout); no detections '
                          'are reported for them (info["all_nan"])', RuntimeWarning,
                          stacklevel=2)
        live = np.flatnonzero(~all_nan)

        def filled(c):
            row, _ = self._missing_row(x[c], fs)
            return self._fill(row, fs) if len(gaps[c]) else row

        # -- 2./3. fill and detect ----------------------------------------------------------
        out = [None] * n_ch
        records_hint = getattr(self.detector, 'output', None) == 'records'
        accepts_valid = getattr(self.detector, 'accepts_valid', False)
        n_removed = np.zeros(n_ch, dtype=np.int64)
        res = {}
        if live.size and getattr(self.detector, 'channel_independent', False):
            for c in live:                       # one channel at a time: bounded memory
                yc = filled(c)[np.newaxis, :]
                if accepts_valid:
                    r = self.detector.detect(yc, fs, valid=~gap_mask(gaps[c], n)[None])
                else:
                    r = self.detector.detect(yc, fs)
                self._check_result(r, 1)
                res[c] = r[0]
                del yc
        elif live.size:
            y = np.empty((live.size, n), dtype=np.float64)
            for k, c in enumerate(live):
                y[k] = filled(c)
            if accepts_valid:
                valid = np.vstack([~gap_mask(gaps[c], n) for c in live])
                r = self.detector.detect(y, fs, valid=valid)
            else:
                r = self.detector.detect(y, fs)
            del y
            self._check_result(r, live.size)
            res = {c: r[k] for k, c in enumerate(live)}

        # -- 4. drop detections in / near gaps ---------------------------------------------
        for c in live:
            item = res[c]
            idx, records = _indices_of(item, records_hint)
            keep = ~mask_in_gaps(idx, gaps[c], fs, units='samples', margin_s=self.edge_margin_s)
            n_removed[c] = int((~keep).sum())
            if records:
                kept = []
                for d, kp in zip(item, keep):
                    if kp:
                        d = dict(d)
                        if 'channel' in d:
                            d['channel'] = int(c)
                        kept.append(d)
                out[c] = kept
            elif isinstance(item, np.ndarray):
                out[c] = item[keep]
            else:
                out[c] = idx[keep]
        for c in np.flatnonzero(all_nan):
            out[c] = [] if records_hint else np.zeros(0, dtype=np.int64)

        # -- 5. report -----------------------------------------------------------------------
        valid_s = np.array([0.0 if all_nan[c] else
                            (n - self._excluded_samples(gaps[c], n, fs)) / fs
                            for c in range(n_ch)])
        valid_fraction = valid_s / (n / fs)
        low = np.flatnonzero(~all_nan & (valid_fraction < 0.5))
        if low.size:
            warnings.warn(f'channel(s) {low.tolist()}: less than half of the record is valid '
                          f'(gaps widened by edge_margin_s={self.edge_margin_s} s; valid '
                          f'fractions {np.round(valid_fraction[low], 3).tolist()}); normalise '
                          'rates by info["valid_s"]', RuntimeWarning, stacklevel=2)
        result = out[0] if one_d else out
        if not return_info:
            return result
        info = {'gaps': gaps, 'flat_runs': flats, 'gap_intervals_s': [g / fs for g in gaps],
                'all_nan': all_nan, 'n_removed': n_removed, 'valid_s': valid_s,
                'valid_fraction': valid_fraction}
        if return_mask:
            info['gap_mask'] = np.vstack([gap_mask(g, n) for g in gaps])
        if one_d:
            info = {k: v[0] for k, v in info.items()}
        return result, info

    def _check_result(self, res, n_expected):
        if not isinstance(res, (list, tuple)) or len(res) != n_expected:
            raise TypeError(f'{self.detector!r}.detect returned {type(res).__name__} of '
                            f'length {getattr(res, "__len__", lambda: "?")()}; expected a '
                            f'list with one entry per channel ({n_expected})')


def _check_seed(seed):
    """Reject seeds :func:`numpy.random.SeedSequence` would refuse, at construction."""
    if seed is None or isinstance(seed, (np.random.Generator, np.random.SeedSequence)):
        return
    if isinstance(seed, (bool, np.bool_)) or isinstance(seed, (str, bytes, float)):
        raise TypeError(f'seed must be None, an int, a sequence of ints, a SeedSequence or a '
                        f'Generator, got {seed!r}')
    try:
        np.random.SeedSequence(seed)
    except (TypeError, ValueError) as e:
        raise type(e)(f'invalid seed {seed!r}: {e}') from None
