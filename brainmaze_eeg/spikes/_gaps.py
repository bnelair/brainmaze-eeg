# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gap (NaN/inf) helpers used by :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`.

Three steps around an unchanged detector:

1. :func:`find_gaps` -- runs of non-finite samples (NaN or +/-inf) of the **original**
   signal, as ``[start, stop)`` sample indices.
2. :func:`fill_gaps` -- make the signal finite so filters/FFT/Hilbert can run:

   - gaps up to ``max_interp_s`` (default 0.1 s here; the wrapper passes its own
     ``short_gap_s``, 0.02 s) are linearly interpolated;
   - longer gaps, ``method='pink'``: 1/f noise scaled to the robust amplitude (MAD) of the
     neighbouring ``context_s`` seconds, offset to the local level (bridged linearly between
     the two sides) and cross-faded with a raised-cosine taper of ``taper_s`` seconds into
     the mirror image of the neighbouring signal at each edge (continuous at the edges);
   - longer gaps, ``method='mirror'``: the neighbouring signal mirrored into the gap from both
     sides, the two images cross-faded over the whole gap (keeps the local spectrum). Real
     events next to the gap are copied into it. Detections inside the gap are removed by
     the wrapper, but the mirrored copies still compete with real maxima in peak selection
     (``find_peaks(distance=...)``) just outside the gap: on spike-dense real data the
     extra/missing detections sit 0.10-0.16 s from the edge, which is why the wrapper's
     ``edge_margin_s`` default is 0.2 s (independent review of PR #67, R7);
   - ``method='linear'``: straight line for every gap (not recommended for gaps > ~0.1 s
     before a background-modelling detector such as Janca).

   Each gap draws its noise from its own generator, seeded from ``seed`` and the gap's
   position and neighbouring data, so fills are reproducible, independent across channels
   and independent of the other gaps.
3. :func:`mask_in_gaps` / :func:`drop_in_gaps` -- remove detections inside a gap or within
   ``margin_s`` of it. Detections on filled samples are never real, whatever the fill.

Relation to ``brainmaze_utils.gaps``
------------------------------------
:func:`find_gaps`, :func:`gap_intervals`, :func:`pink_noise`, :func:`mask_in_gaps` and
:func:`drop_in_gaps` **are** ``brainmaze_utils.gaps``' functions (imported from
brainmaze-utils >= 3.0.0 and re-exported here; this module used to carry copies of them, and
they gave bit-identical results; the one difference is input validation: ``fs`` given as a
numeric string such as ``'256'`` is now accepted instead of raising ``ValueError``).

:func:`fill_gaps` stays here on purpose. ``brainmaze_utils.gaps.fill_gaps`` has the same
signature, but its ``'mirror'`` and ``'pink'`` fills are a later, different design (odd
reflection at the edges and a correlation-aware cross-fade normalisation for ``'mirror'``;
kriging-conditioned edges, a different per-gap seeding and a ``'spectral'`` default for the
noise fills). Switching would change the fill values, and with them the detections of
:class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` next to gaps (on synthetic and
real data, 66 of 68 tested ``'mirror'``/``'pink'`` configurations differ in the kept
detections, the Barkmeier records or ``info['n_removed']``; all ``'linear'`` ones are
identical). The detections of this package must not change in a minor release, so the
wrapper keeps the fill it was validated with. Only ``'linear'`` and short-gap interpolation are identical in both.
Differences from ``brainmaze_utils.gaps.fill_gaps``: no ``'spectral'`` method, default
``method='mirror'`` (there ``'spectral'``), and different fill values for ``'mirror'`` and
``'pink'``. :func:`gap_mask` and :func:`flat_runs` are specific to this package.
"""

import hashlib
import warnings

import numpy as np
# Identical in brainmaze_utils.gaps (>= 3.0.0): use the family's implementation.
from brainmaze_utils.gaps import drop_in_gaps, find_gaps, gap_intervals, mask_in_gaps, pink_noise

__all__ = ['FILL_METHODS', 'find_gaps', 'gap_intervals', 'fill_gaps', 'pink_noise',
           'mask_in_gaps', 'drop_in_gaps', 'gap_mask', 'flat_runs', 'FLAT_AS_GAP_S']

FILL_METHODS = ('pink', 'mirror', 'linear')


def _check_fs(fs):
    try:
        ok = np.isfinite(fs) and fs > 0
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f'fs must be a finite number > 0, got {fs!r}')


def gap_mask(gaps, n_samples):
    """Boolean ``(n_samples,)`` mask, True on the samples covered by ``gaps`` (samples)."""
    m = np.zeros(n_samples, dtype=bool)
    for s, e in np.asarray(gaps, dtype=np.int64).reshape(-1, 2):
        m[s:e] = True
    return m


#: Default minimum length (s) of a run of exactly equal samples that counts as missing data
#: (``GapAwareSpikeDetector(flat_as_gap_s=...)``; see its module docstring for the evidence).
FLAT_AS_GAP_S = 0.1


def flat_runs(x, fs, flat_as_gap_s=FLAT_AS_GAP_S):
    """
    Runs of exactly equal consecutive samples lasting at least ``flat_as_gap_s`` seconds
    (at least 2 samples) in a 1-D signal: the rule by which
    :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` treats constant stretches
    (dropouts stored as a constant, clipping) as missing data. NaN never equals anything, so
    NaN runs are not reported here (use :func:`find_gaps`). ``flat_as_gap_s=None`` disables
    the rule (no runs). Not part of ``brainmaze_utils.gaps``.

    Returns
    -------
    np.ndarray, shape (n_runs, 2), int64
        ``[start, stop)`` sample indices.
    """
    y = np.asarray(x)
    if flat_as_gap_s is None or y.size < 2:
        return np.zeros((0, 2), np.int64)
    min_len = max(int(np.ceil(flat_as_gap_s * fs - 1e-9)), 2)
    eq = y[1:] == y[:-1]                       # NaN never equals anything
    d = np.diff(np.concatenate(([0], eq.astype(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1) + 1   # sample runs
    keep = (en - st) >= min_len
    return np.stack([st[keep], en[keep]], axis=1).astype(np.int64)


def _robust_sd(v):
    if v.size < 2:
        return 0.0
    sd = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float(sd if sd > 0 else v.std())


def _raised_cosine(n):
    """Weights going 1 -> 0 over ``n`` samples (end points excluded)."""
    return 0.5 * (1 + np.cos(np.pi * np.arange(1, n + 1) / (n + 1)))


def _reflect(k, length):
    """Triangle-wave index 0..L-1, L-1..0, 0.. so a mirror image can exceed its context."""
    if length <= 1:
        return np.zeros_like(k)
    k = k % (2 * length)
    return np.where(k < length, k, 2 * length - 1 - k)


def _root_entropy(seed):
    """Entropy of the root SeedSequence of one fill operation (drawn once per call)."""
    if isinstance(seed, np.random.Generator):
        return int(seed.integers(0, 2 ** 63))       # one draw: per-gap streams stay independent
    if isinstance(seed, np.random.SeedSequence):
        return seed.entropy
    if seed is None:
        return np.random.SeedSequence().entropy
    if isinstance(seed, np.random.RandomState):
        raise TypeError('seed must be None, an int, a sequence of ints, a SeedSequence or a '
                        'Generator (not RandomState)')
    return seed


def _gap_rng(entropy, s, e, context):
    """Generator for one gap: root entropy + (start, stop, hash of the neighbouring data)."""
    h = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(context).tobytes(),
                                       digest_size=8).digest(), 'little')
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(int(s), int(e), h)))


def _fill_1d(x, fs, max_interp_s, method, context_s, taper_s, beta, entropy):
    y = np.array(x, dtype=np.float64, copy=True)
    gaps = find_gaps(y)
    n_total = y.size
    if gaps.size == 0 or (len(gaps) == 1 and gaps[0, 0] == 0 and gaps[0, 1] == n_total):
        return y
    max_interp = int(np.floor(max_interp_s * fs + 1e-9))
    ctx = max(int(round(context_s * fs)), 2)
    taper = int(round(taper_s * fs))
    near = max(min(int(round(0.5 * fs)), ctx), 1)
    nxt_start = np.r_[gaps[1:, 0], n_total]
    for (s, e), nxt in zip(gaps, nxt_start):
        n = e - s
        has_l, has_r = s > 0, e < n_total
        r = np.arange(1, n + 1) / (n + 1)
        if n <= max_interp or method == 'linear':
            a = y[s - 1] if has_l else y[e]
            b = y[e] if has_r else y[s - 1]
            y[s:e] = a + (b - a) * r
            continue
        # context: finite samples next to the gap (left side already filled -> finite)
        left = y[max(s - ctx, 0):s]
        right = y[e:min(e + ctx, nxt)]
        lvl_a = np.median(left[-near:]) if left.size else np.median(right[:near])
        lvl_b = np.median(right[:near]) if right.size else np.median(left[-near:])
        base = lvl_a + (lvl_b - lvl_a) * r
        if method == 'mirror':
            k = np.arange(n)
            imgs = []
            if left.size:
                imgs.append(y[s - 1 - _reflect(k, left.size)] - lvl_a)
            if right.size:
                imgs.append(y[e + _reflect(n - 1 - k, right.size)] - lvl_b)
            if len(imgs) == 2:
                w = _raised_cosine(n)
                dev = (w * imgs[0] + (1 - w) * imgs[1]) / np.sqrt(w ** 2 + (1 - w) ** 2)
            else:
                dev = imgs[0]
            y[s:e] = base + dev
            continue
        # pink: independent, reproducible stream per gap
        context = np.concatenate([left, right])
        rng = _gap_rng(entropy, s, e, context)
        fill = base + _robust_sd(context) * pink_noise(n, beta, rng=rng)
        t = min(taper, n // 2) if (has_l and has_r) else min(taper, n)
        tl = min(t, left.size)
        if has_l and tl > 0:
            w = _raised_cosine(tl)
            fill[:tl] = w * y[s - 1 - np.arange(tl)] + (1 - w) * fill[:tl]
        tr = min(t, right.size)
        if has_r and tr > 0:
            w = _raised_cosine(tr)[::-1]
            fill[n - tr:] = w * y[e + np.arange(tr)][::-1] + (1 - w) * fill[n - tr:]
        y[s:e] = fill
    return y


def fill_gaps(x, fs, *, max_interp_s=0.1, method='mirror', context_s=30.0, taper_s=0.5,
              beta=1.0, seed=0, axis=-1, all_nan='keep', copy=True):
    """
    Fill the non-finite gaps of a signal (see module docstring).

    Parameters
    ----------
    x : array_like
        Signal, 1-D ``(n_samples,)`` or N-D with time along ``axis``; every other index is
        filled independently (a row of an N-D array is filled exactly as the same 1-D
        signal).
    fs : float
        Sampling frequency (Hz).
    max_interp_s : float
        Gaps up to this length (s) are linearly interpolated (default 0.1 s).
    method : {'mirror', 'pink', 'linear'}
        Fill of longer gaps (default ``'mirror'``; ``brainmaze_utils.gaps`` additionally has
        ``'spectral'``, its default).
    context_s : float
        Seconds of valid data on each side used for amplitude, level and mirroring
        (default 30, as in ``brainmaze_utils.gaps``; the wrapper passes 10 explicitly, its
        tuned value).
    taper_s : float
        ``'pink'``: raised-cosine cross-fade length (s) into the mirrored signal at each
        edge, capped at half the gap (default 0.5).
    beta : float
        ``'pink'``: spectral exponent, ``P(f) ~ 1/f**beta`` (default 1).
    seed : None, int, sequence of int, np.random.SeedSequence or np.random.Generator
        Noise seed (default 0: reproducible). Each gap gets its own stream derived from the
        seed, the gap's position and a hash of its neighbouring data, so channels with the
        same seed still get independent noise.
    axis : int
        Time axis (default -1).
    all_nan : {'keep', 'zero', 'raise'}
        A channel without any finite sample: left unchanged (``'keep'``), set to zeros
        (``'zero'``) -- both with one warning listing the channels -- or ``ValueError``
        (``'raise'``).
    copy : bool
        ``True`` (default): ``x`` is not modified. ``False``: a writable floating-point
        ndarray is filled in place and returned.

    Returns
    -------
    np.ndarray
        Same shape as ``x``; floating input keeps its dtype, anything else becomes float64.
    """
    _check_fs(fs)
    if method not in FILL_METHODS:
        raise ValueError(f'fill method must be one of {FILL_METHODS}, got {method!r}')
    if all_nan not in ('keep', 'zero', 'raise'):
        raise ValueError(f"all_nan must be 'keep', 'zero' or 'raise', got {all_nan!r}")
    if not (np.isfinite(max_interp_s) and max_interp_s >= 0 and np.isfinite(taper_s)
            and taper_s >= 0 and context_s > 0 and np.isfinite(beta)):
        raise ValueError('max_interp_s and taper_s must be finite >= 0, context_s > 0 and '
                         'beta finite')
    x_in = x
    x = np.asarray(x)
    if np.iscomplexobj(x) or not (np.issubdtype(x.dtype, np.number)
                                  or np.issubdtype(x.dtype, np.bool_)):
        raise TypeError(f'x must be a real numeric array, got dtype {x.dtype}')
    if x.ndim == 0:
        raise ValueError('x must have at least one dimension')
    if not -x.ndim <= axis < x.ndim:
        raise ValueError(f'axis {axis} out of range for {x.ndim}-D input')
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    in_place = (not copy and isinstance(x_in, np.ndarray) and x_in.dtype == dtype
                and x_in.flags.writeable)
    out = x_in if in_place else np.array(x, dtype=dtype, copy=True)
    entropy = _root_entropy(seed)
    view = np.moveaxis(out, axis, -1)
    dead = []
    for idx in np.ndindex(view.shape[:-1]):
        row = view[idx]
        if row.size == 0 or np.isfinite(row).all():
            continue
        if not np.isfinite(row).any():
            dead.append(idx)
            if all_nan == 'zero':
                row[...] = 0
            continue
        row[...] = _fill_1d(row, fs, max_interp_s, method, context_s, taper_s, beta, entropy)
    if dead:
        if all_nan == 'raise':
            raise ValueError(f'channel(s) {dead} contain no finite sample')
        warnings.warn(f'fill_gaps: channel(s) {dead} contain no finite sample; '
                      + ('left unchanged' if all_nan == 'keep' else 'set to zeros'),
                      RuntimeWarning, stacklevel=2)
    return out
