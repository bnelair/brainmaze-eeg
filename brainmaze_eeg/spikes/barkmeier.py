# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Barkmeier interictal spike detector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multichannel amplitude/slope/duration half-wave spike detector after:

    Barkmeier, D.T., Shah, A.K., Flanagan, D., Atkinson, M.D., Agarwal, R.,
    Fuerst, D.R., Jafari-Khouzani, K., Loeb, J.A. (2012). *High inter-reviewer
    variability of spike detection on intracranial EEG addressed by an automated
    multi-channel algorithm.* Clinical Neurophysiology 123(6), 1088-1095.
    https://doi.org/10.1016/j.clinph.2011.09.023  (PMC3277646)

Written from the paper's Methods ("Spike detection algorithm"); no third-party code.
Each step below is marked **[paper]** when it is specified by the paper and **[ours]** when
the paper leaves it open and this implementation makes a documented choice.

Algorithm
---------
0. **[paper] One-minute blocks.** The record is processed in successive blocks of
   ``block_s`` seconds (60). The scaling factor, the artifact-channel rule and the candidate
   threshold are computed **per block**. **[ours]** Filtering is done once on the whole
   record (no block-edge transients); a trailing remainder shorter than half a block is
   merged into the previous block; ``block_s=None`` treats the whole record as one block.
1. **[paper] Artifact channels -- OFF by default here (opt-in).** The paper excludes, in
   each block, a channel whose average slope is more than 10 standard deviations from the
   mean slope of the channels. **[ours]** Average slope = mean ``|dx/dt|`` of the input
   signal over the (valid samples of the) block. No formulation of this rule we tested
   keeps every real detection *and* removes realistic artifacts, so the default is
   ``artifact_sd=None, artifact_ratio=None`` (no channel is ever excluded) and two
   formulations are offered as opt-in. Measured on 1/f background with 60 s blocks at 256
   and 1000 Hz (``brainmaze-work/scratch/eeg-spikes/r3/v1_*.py``; table in the README):

   - *Literal paper rule* (mean/SD over all channels): the tested channel is part of the
     SD, so its z-score is bounded by ``sqrt(n_channels - 1)``; it cannot fire below 102
     channels (flagged nothing in any test).
   - *Leave-one-out mean/SD* (round 1 of this module): the reference SD of similar
     channels is tiny, so a genuinely spiking channel is excluded (8 channels, 300 uV
     IEDs at 1/s on one: 120 of 598 kept at 256 Hz; 1000 uV at 3/s: 0 of 1794 kept), and
     3-4 channel noise montages have 3-12 % false flags.
   - **Spatial robust rule** (``artifact_sd``, e.g. 10; round 2 of this module): centre =
     median slope of the usable channels, spread = ``max(1.4826 * MAD, artifact_rel_floor
     * median)``, flagged when ``|slope - median| > artifact_sd * spread``. With the floor
     0.2 a channel is flagged whenever its slope exceeds **3x the median channel's,
     whatever the cause**: in a montage of 12 contacts at 1x and 4 at 3.5-6x amplitude
     (grey vs white matter, no artifact) the 4 large channels are flagged in every block
     and a spiking large channel loses **all** its detections (0 of 221, 181, 154 and 145
     kept at 3.5x, 4x, 5x, 6x). It also removes strong IED bursts (2000 uV at 3/s:
     541 -> 4 at 256 Hz). Use it only for montages of similar contacts.
   - **Self-referenced rule** (``artifact_ratio``, e.g. 3; ours): each channel's slope is
     divided by its own median slope over all blocks, and a channel-block is flagged when
     that ratio exceeds ``artifact_ratio`` times the median ratio of the montage in that
     block. Heterogeneous montages and steady spiking are never flagged (all cases above
     kept), but it needs >= 3 blocks, misses an artifact present in more than half of the
     blocks, and also removes strong IED bursts confined to a few blocks (2000 uV at 3/s
     in 3 of 10 blocks at 256 Hz: 541 -> 4).

   What either rule catches: broadband noise (white noise at 3x the background rms:
   slope ratio 5.5-6.2) and mains pick-up (300 uV, 60 Hz). What neither catches (mean slope
   barely changes, ratio 0.03-1.9): slow drifts, EMG bursts, electrode pops, flat
   stretches with jumps -- although EMG, pops and jumps cause false detections (17-94 in a
   3-minute stretch). On real 15-channel iEEG (2 x 1 h, 256 Hz, contact slopes 0.14-4.1x
   the median) no rule flagged anything. Only usable channels count (>= 3 needed, positive
   median); excluded channel-blocks get no detections and do not enter the scaling, and a
   ``UserWarning`` names them. Both rules may be combined (a channel-block is excluded if
   either flags it).
2. **[paper] Candidates.** Band-pass 20-50 Hz (``narrow_band``); candidates are local maxima
   of the rectified narrow-band signal above a threshold of ``std_coeff`` (4) standard
   deviations. **[ours: interpretation]** The paper's wording ("four standard deviations of
   the channel mean amplitude") is ambiguous; the threshold here is ``mean + std_coeff *
   std`` of the *rectified* narrow-band signal in the block (on Gaussian noise about
   3.2 SD of the narrow-band signal itself). **[ours]** Filter type/order are not given in
   the paper: Butterworth, prototype order ``narrow_order`` = 2 (the paper's broad-band
   design), zero-phase.
3. **[paper] Morphology band and block scaling.** Band-pass 1-35 Hz, **2nd-order
   Butterworth** (``broad_band``, ``broad_order``). All channels of a block are multiplied by
   one factor that brings the median (across channels) of the channel mean rectified
   amplitudes to ``scale`` (70 uV). **[ours]** zero-phase application (``sosfiltfilt``):
   the paper only says "second order digital Butterworth". The band edges are design edges
   (-3 dB single pass); applied forward-backward the attenuation doubles, so the realised
   response is -6 dB at 1 and 35 Hz.
4. **[paper]** For each candidate, the broad-band peak (largest ``|x|`` within +/-2 ms
   **[ours]**) and the flanking opposite extrema within ``trough_search`` (50 ms **[ours]**)
   define two half-waves. A spike is accepted iff total amplitude of both half-waves
   ``> 600``, each half-wave slope ``> 7 uV/ms`` and each half-wave duration ``> 10 ms``,
   in the block-scaled domain (:data:`DEFAULT_THRESHOLDS`).
5. **[ours]** Accepted detections on a channel closer than ``trough_search`` are merged
   (largest total amplitude kept; one discharge produces several narrow-band maxima); then
   the optional ``refractory`` (s) is applied to the merged list (default 0, the paper
   defines none).

Filters (verified by the test-suite at 200-32000 Hz)
----------------------------------------------------
=========  ======================  =====  ==========================================
signal     type                    order  zero-phase response
=========  ======================  =====  ==========================================
narrow     Butterworth band-pass   2      -6 dB at 20 / 50 Hz, ~0 dB at 30-35 Hz,
                                          -15 dB at 60 Hz
broad      Butterworth band-pass   2      -6 dB at 1 / 35 Hz, ~0 dB at 5-10 Hz
=========  ======================  =====  ==========================================

Differences from the earlier version of this module
---------------------------------------------------
- Broad band was 1-80 Hz (2nd-order high-pass + 4th-order low-pass) and its docstring
  claimed "per the paper"; the paper specifies 1-35 Hz 2nd-order Butterworth. The 80 Hz
  band admits sharper noise transients: on 1/f noise the false-positive rate drops ~5x with
  the paper's band (see the README for numbers).
- No blocks: one scaling factor and one threshold per channel for the whole record. Now per
  ``block_s`` (paper: one minute).
- No artifact-channel rule. Now available as two opt-in formulations (see step 1); off by
  default because none keeps every real detection.
- The refractory period was applied before the 50 ms merge, so a merge could pick a
  detection the refractory had already used to suppress its neighbour. Now merge first.
- A single NaN anywhere silently made the median scaling factor NaN for **every** channel.
  Non-finite input now raises ``ValueError``; data with gaps go through
  :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` (with
  :class:`BarkmeierDetector`), which fills the gaps, passes the gap mask as ``valid`` so
  that block statistics use only real samples, and drops detections in/near gaps.
- A transposed ``(n_samples, n_channels)`` array was silently accepted. A 2-D input with
  more rows than columns, or shorter than 1 s, now raises ``ValueError``.

Note on false positives
-----------------------
Block scaling normalises the median channel to ``scale`` whatever the channel count, so the
fixed thresholds are relative to the *typical* channel of the montage. On pure 1/f
background the detector fires at a low but non-zero rate (0.015-0.027/s per channel with
the paper's 1-35 Hz band; 8 channels of 30 uV 1/f noise) **independently of the number of
channels**; the multichannel design
makes the result comparable *between* channels, it does not by itself remove noise
detections. The earlier statement that this was a single-channel artefact was wrong.
"""

import warnings

import numpy as np
from scipy.signal import find_peaks, sosfiltfilt

from brainmaze_eeg.spikes import _checks as chk
from brainmaze_eeg.spikes import _filters as flt

__all__ = ['detect_spikes_barkmeier', 'BarkmeierDetector', 'design_barkmeier_filters',
           'DEFAULT_THRESHOLDS']

# Thresholds are evaluated in the block-scaled domain (median channel amplitude -> `scale`).
# total_amp and slope are the paper's 600 uV and 7 uV/ms (= 7000 uV/s); half_dur is the
# physical 10 ms half-wave duration.
DEFAULT_THRESHOLDS = {
    'total_amp': 600.0,   # total amplitude of both half-waves (scaled units)
    'slope': 7000.0,      # each half-wave slope, scaled-units per second (7 uV/ms)
    'half_dur': 0.010,    # each half-wave duration, seconds (10 ms)
}


def design_barkmeier_filters(fs, narrow_band=(20.0, 50.0), broad_band=(1.0, 35.0),
                             narrow_order=2, broad_order=2):
    """
    Design the band-pass filters used by :func:`detect_spikes_barkmeier`.

    Returns
    -------
    dict
        ``{'narrow': sos, 'broad': sos}`` (Butterworth band-passes; applied zero-phase).

    Raises
    ------
    ValueError
        Unless ``0 < low < high < fs/2`` for both bands and the orders are positive integers.
    """
    fs = float(fs)
    flt.check_band(narrow_band, fs, name='narrow_band')
    flt.check_band(broad_band, fs, name='broad_band')
    return {'narrow': flt.butter_bandpass(narrow_band, fs, narrow_order),
            'broad': flt.butter_bandpass(broad_band, fs, broad_order)}


def _blocks(n, fs, block_s):
    if block_s is None:
        return np.array([[0, n]])
    if not block_s > 0:
        raise ValueError(f'block_s must be > 0 seconds or None, got {block_s}')
    bn = max(int(round(block_s * fs)), 1)
    starts = list(range(0, n, bn))
    if len(starts) > 1 and n - starts[-1] < bn / 2:
        starts.pop()                       # merge a short tail into the previous block
    stops = starts[1:] + [n]
    return np.array(list(zip(starts, stops)), dtype=np.int64)


def _artifact_channels(slopes, usable, n_sd, rel_floor=0.2):
    """
    Robust artifact-channel rule (see the module docstring, step 1).

    A usable channel is flagged when ``|slope - median| > n_sd * spread`` with ``median``
    and ``spread = max(1.4826 * MAD, rel_floor * median)`` taken over all usable channels.
    Needs >= 3 usable channels and a positive median slope; otherwise nothing is flagged.
    """
    flag = np.zeros(slopes.shape, dtype=bool)
    idx = np.flatnonzero(usable)
    if n_sd is None or idx.size < 3:
        return flag
    s = slopes[idx]
    centre = np.median(s)
    if not centre > 0:
        return flag
    spread = max(1.4826 * np.median(np.abs(s - centre)), rel_floor * centre)
    flag[idx] = np.abs(s - centre) > n_sd * spread
    return flag


def _artifact_self_referenced(slopes, usable, ratio):
    """
    Self-referenced artifact rule (see the module docstring, step 1).

    ``slopes`` and ``usable`` are ``(n_blocks, n_channels)``. Each usable channel-block slope
    is divided by the channel's median slope over its usable blocks (>= 3 needed); a
    channel-block is flagged when that ratio exceeds ``ratio`` times the median ratio of the
    eligible channels in the block (>= 3 needed). Returns a bool ``(n_blocks, n_channels)``.
    """
    flag = np.zeros(slopes.shape, dtype=bool)
    if ratio is None or slopes.shape[0] < 3:
        return flag
    s = np.where(usable, slopes, np.nan)
    enough = np.isfinite(s).sum(axis=0) >= 3
    base = np.full(s.shape[1], np.nan)
    if enough.any():
        base[enough] = np.nanmedian(s[:, enough], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = s / base
    rel[:, ~(base > 0)] = np.nan
    for b in range(s.shape[0]):
        idx = np.flatnonzero(np.isfinite(rel[b]))
        if idx.size < 3:
            continue
        m = np.median(rel[b, idx])
        if m > 0:
            flag[b, idx] = rel[b, idx] > ratio * m
    return flag


def _check_params(scale, std_coeff, trough_search, thresholds, narrow_band, broad_band,
                  refractory, narrow_order, broad_order, block_s, artifact_sd,
                  artifact_rel_floor, artifact_ratio=None):
    """Validate every Barkmeier parameter that does not need ``fs``; return the thresholds."""
    chk.number('scale', scale, gt=0)
    chk.number('std_coeff', std_coeff, ge=0)
    chk.number('trough_search', trough_search, gt=0)
    chk.number('refractory', refractory, ge=0)
    chk.pair('narrow_band', narrow_band, gt=0)
    chk.pair('broad_band', broad_band, gt=0)
    chk.integer('narrow_order', narrow_order, ge=1, le=20)
    chk.integer('broad_order', broad_order, ge=1, le=20)
    if block_s is not None and isinstance(block_s, (int, float)) and np.isinf(block_s):
        raise ValueError('block_s must be finite; use block_s=None for one block')
    chk.number('block_s', block_s, ge=1.0, allow_none=True)
    chk.number('artifact_sd', artifact_sd, gt=0, allow_none=True)
    chk.number('artifact_rel_floor', artifact_rel_floor, ge=0)
    chk.number('artifact_ratio', artifact_ratio, gt=1, allow_none=True)
    if thresholds is None:
        return dict(DEFAULT_THRESHOLDS)
    if not isinstance(thresholds, dict):
        raise TypeError(f'thresholds must be a dict, got {type(thresholds).__name__}')
    unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(
            f"unknown threshold key(s) {sorted(unknown)}; "
            f"expected {sorted(DEFAULT_THRESHOLDS)}.")
    thr = {**DEFAULT_THRESHOLDS, **thresholds}   # partial dict -> fill from defaults
    for k, v in thr.items():
        thr[k] = chk.number(f"thresholds['{k}']", v, ge=0)
    return thr


def detect_spikes_barkmeier(sig, fs, scale=70.0, std_coeff=4.0, trough_search=0.05,
                            thresholds=None, narrow_band=(20.0, 50.0), broad_band=(1.0, 35.0),
                            refractory=0.0, *, narrow_order=2, broad_order=2, block_s=60.0,
                            artifact_sd=None, artifact_rel_floor=0.2, artifact_ratio=None,
                            valid=None,
                            return_info=False):
    """
    Detect interictal spikes with the Barkmeier (2012) multichannel half-wave criteria.

    See the module docstring for the algorithm and which choices are the paper's.

    Parameters
    ----------
    sig : np.ndarray
        iEEG in **uV**, ``(n_samples,)`` or ``(n_channels, n_samples)`` -- pass the whole
        montage: scaling and the artifact rule are across channels. Must be finite:
        NaN/inf raise ``ValueError`` (use
        :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` for data with gaps).
    fs : float
        Sampling frequency in Hz.
    scale : float
        Block-scaling target for the median channel mean rectified amplitude (paper: 70 uV).
        Finite, > 0.
    std_coeff : float
        Candidate threshold in SDs of the rectified narrow-band signal (paper: 4). Finite,
        >= 0.
    trough_search : float
        Half-window (s) each side of the peak in which to find the flanking troughs, and
        merge distance of detections (ours: 0.05). Finite, > 0 and at least 2 samples.
    thresholds : dict, optional
        ``{'total_amp', 'slope', 'half_dur'}`` in the block-scaled domain; partial dicts are
        completed from :data:`DEFAULT_THRESHOLDS` (paper: 600 uV, 7 uV/ms = 7000 uV/s, 10 ms).
        Each finite, >= 0.
    narrow_band : (float, float)
        Candidate band design edges (paper: 20-50 Hz); -6 dB realised (zero-phase).
    broad_band : (float, float)
        Morphology/scaling band design edges (paper: 1-35 Hz); -6 dB realised.
    refractory : float
        Minimum time (s) between accepted spikes on a channel, after merging (default 0).
        Finite, >= 0.
    narrow_order, broad_order : int
        Butterworth prototype orders (broad: paper 2; narrow: ours 2). 1-20.
    block_s : float or None
        Block length in seconds (paper: 60), finite and >= 1. ``None``: one block for the
        whole record.
    artifact_sd : float or None
        **Opt-in** spatial robust artifact-channel rule (paper: 10 SD; default ``None`` =
        off), finite and > 0. Flags any channel whose block slope exceeds ``1 + artifact_sd
        * artifact_rel_floor`` (3x) times the median channel's **whatever the cause**, so
        large normal channels of a heterogeneous montage lose all detections; use only for
        montages of similar contacts. See the module docstring, step 1.
    artifact_rel_floor : float
        Floor of the spatial rule's spread, relative to the median channel slope (ours:
        0.2). Finite, >= 0.
    artifact_ratio : float or None
        **Opt-in** self-referenced artifact rule (ours; e.g. 3; default ``None`` = off),
        finite and > 1: flags a channel-block whose slope, relative to the channel's own
        median over blocks, exceeds ``artifact_ratio`` times the montage's median relative
        slope in that block. Safe for heterogeneous montages; needs >= 3 blocks; misses
        artifacts present in most blocks and removes strong IED bursts confined to a few
        blocks. See the module docstring, step 1.
    valid : np.ndarray of bool, optional
        Same shape as ``sig``; samples that are real data (default: all). Only valid samples
        enter the per-block statistics (scaling factor, candidate threshold, artifact slope);
        a channel with no valid sample in a block is excluded from that block. Used by
        :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` so that filled gaps do
        not bias the statistics; detections are not filtered by it.
    return_info : bool
        Also return a dict with per-block diagnostics (see Returns).

    Returns
    -------
    detections : list of dict
        One dict per spike, sorted by (channel, time): ``channel, peak_index, peak_time,
        block, peak_amp, left_amp, left_dur, left_slope, right_amp, right_dur, right_slope,
        total_amp``. Index/time refer to ``sig`` (samples / seconds). Amplitudes and slopes
        are in the **block-scaled domain** (multiply by ``1/info['scale_factor'][block]`` for
        input units); durations are seconds.
    info : dict
        Only with ``return_info=True``: ``blocks`` ``(n_blocks, 2)`` ``[start, stop)``
        samples; ``scale_factor`` ``(n_blocks,)``; ``artifact`` ``(n_blocks, n_channels)``
        bool; ``channel_slope`` and ``candidate_threshold`` ``(n_blocks, n_channels)`` in
        input units; ``filters``.

    Raises
    ------
    ValueError, TypeError
        Invalid parameter (type, NaN/inf, range), band at/above Nyquist, input not 1-D/2-D, a 2-D input with more rows than
        columns (probably transposed), a record shorter than 1 s, a non-finite value, or a
        ``valid`` mask of the wrong shape.
    """
    thr = _check_params(scale, std_coeff, trough_search, thresholds, narrow_band, broad_band,
                        refractory, narrow_order, broad_order, block_s, artifact_sd,
                        artifact_rel_floor, artifact_ratio)
    fs = chk.number('fs', fs, gt=0)
    if int(round(fs * trough_search)) < 2:
        raise ValueError(f'trough_search={trough_search} s is shorter than 2 samples at '
                         f'fs={fs:g} Hz')
    sig = np.asarray(sig)
    if np.iscomplexobj(sig) or not (np.issubdtype(sig.dtype, np.number)
                                    or np.issubdtype(sig.dtype, np.bool_)):
        raise TypeError(f"'sig' must be a real numeric array, got dtype {sig.dtype}")
    if sig.ndim == 1:
        x = sig[np.newaxis, :]
    elif sig.ndim == 2:
        x = sig
        if x.shape[0] > x.shape[1]:
            raise ValueError(f"'sig' has shape {x.shape}: more channels than samples. The "
                             'layout is (n_channels, n_samples); transpose your array (sig.T).')
    else:
        raise ValueError(f"'sig' must be 1-D or 2-D, got {sig.ndim}-D.")
    n_ch, n_samples = x.shape
    if n_samples < fs:
        raise ValueError(f'record has {n_samples} samples (< 1 s at fs={fs} Hz); check fs and '
                         'the array layout (n_channels, n_samples).')

    filters = design_barkmeier_filters(fs, narrow_band, broad_band, narrow_order, broad_order)
    if not np.isfinite(x).all():
        ch = np.flatnonzero(~np.isfinite(x).all(axis=1)).tolist()
        raise ValueError(f"'sig' contains NaN/inf (channels {ch}). The raw detector needs "
                         'finite input; use brainmaze_eeg.spikes.GapAwareSpikeDetector('
                         'BarkmeierDetector()) to fill gaps and drop detections near them.')
    if valid is None:
        valid = np.ones(x.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool).reshape(-1, n_samples) if sig.ndim == 1 else \
            np.asarray(valid, dtype=bool)
        if valid.shape != x.shape:
            raise ValueError(f'valid has shape {valid.shape}, expected {sig.shape}')

    blocks = _blocks(n_samples, fs, block_s)
    nb = len(blocks)
    factor = np.zeros(nb)
    artifact = np.zeros((nb, n_ch), dtype=bool)
    slopes = np.full((nb, n_ch), np.nan)
    amps = np.full((nb, n_ch), np.nan)
    cand_thr = np.full((nb, n_ch), np.inf)
    block_of = np.zeros(n_samples, dtype=np.int64)
    for b, (s, e) in enumerate(blocks):
        block_of[s:e] = b

    # -- filtering and per-block statistics, one channel at a time (bounded memory) --------
    fx_broad = np.empty((n_ch, n_samples))
    rect = np.empty((n_ch, n_samples))
    for c in range(n_ch):
        xc = np.asarray(x[c], dtype=np.float64)
        fx_broad[c] = sosfiltfilt(filters['broad'], xc)
        rect[c] = np.abs(sosfiltfilt(filters['narrow'], xc))
        for b, (s, e) in enumerate(blocks):
            v = valid[c, s:e]
            if not v.any():
                continue
            pair = v[1:] & v[:-1]
            if pair.any():
                slopes[b, c] = np.abs(np.diff(xc[s:e]))[pair].mean() * fs
            amps[b, c] = np.abs(fx_broad[c, s:e][v]).mean()
            r = rect[c, s:e][v]
            cand_thr[b, c] = r.mean() + std_coeff * r.std()

    # -- cross-channel decisions per block ----------------------------------------------
    usable_all = np.isfinite(slopes) & np.isfinite(amps)
    artifact |= _artifact_self_referenced(slopes, usable_all, artifact_ratio)
    for b in range(nb):
        amp = amps[b]
        usable = usable_all[b].copy()
        artifact[b] |= _artifact_channels(np.nan_to_num(slopes[b]), usable, artifact_sd,
                                          artifact_rel_floor)
        usable &= ~artifact[b]
        cand_thr[b, ~usable] = np.inf
        med = np.median(amp[usable]) if usable.any() else 0.0
        factor[b] = scale / med if med > 0 else 0.0
        if factor[b] == 0:
            cand_thr[b] = np.inf

    if artifact.any():
        bad = sorted(set(np.nonzero(artifact)[1].tolist()))
        warnings.warn(f'Barkmeier artifact-channel rule excluded {int(artifact.sum())} '
                      f'channel-block(s) (channels {bad}); use return_info=True for details',
                      UserWarning, stacklevel=2)

    half_peak = int(round(fs * 0.002))            # +/- 2 ms search for the broad-band peak
    n_trough = int(round(fs * trough_search))
    refractory_n = int(round(fs * refractory))

    detections = []
    for ch in range(n_ch):
        broad = fx_broad[ch]
        peak_idx, _ = find_peaks(rect[ch], height=cand_thr[block_of, ch])

        ch_detections = []
        for pi in peak_idx:
            l = max(pi - half_peak, 0)
            r = min(pi + half_peak + 1, n_samples)
            # spike polarity from the broad band: whichever extreme is larger in magnitude
            seg = broad[l:r]
            spike_i = l + int(np.argmax(np.abs(seg)))
            sign = np.sign(broad[spike_i]) or 1.0
            b = block_of[pi]           # the candidate's block: same statistics throughout
            k = factor[b]
            if k == 0 or artifact[b, ch]:
                continue

            # flanking troughs (opposite extreme) within trough_search on each side
            ll = max(spike_i - n_trough, 0)
            rr = min(spike_i + n_trough + 1, n_samples)
            if spike_i - ll < 1 or rr - spike_i < 2:
                continue
            left_i = ll + int(np.argmin(sign * broad[ll:spike_i]))
            right_i = spike_i + int(np.argmin(sign * broad[spike_i:rr]))

            spike_V = k * broad[spike_i]
            left_amp = k * abs(broad[spike_i] - broad[left_i])
            right_amp = k * abs(broad[spike_i] - broad[right_i])
            left_dur = (spike_i - left_i) / fs
            right_dur = (right_i - spike_i) / fs
            if left_dur <= 0 or right_dur <= 0:
                continue
            left_slope = left_amp / left_dur
            right_slope = right_amp / right_dur
            total_amp = left_amp + right_amp

            accept = (total_amp > thr['total_amp'] and
                      left_slope > thr['slope'] and right_slope > thr['slope'] and
                      left_dur > thr['half_dur'] and right_dur > thr['half_dur'])
            if not accept:
                continue
            ch_detections.append({
                'channel': ch,
                'peak_index': int(spike_i),
                'peak_time': spike_i / fs,
                'block': int(b),
                'peak_amp': float(spike_V),
                'left_amp': float(left_amp), 'left_dur': float(left_dur),
                'left_slope': float(left_slope),
                'right_amp': float(right_amp), 'right_dur': float(right_dur),
                'right_slope': float(right_slope),
                'total_amp': float(total_amp),
            })

        # One discharge produces a cluster of supra-threshold narrow-band maxima: merge
        # detections closer than `trough_search` (keep the largest), THEN apply refractory.
        merged = _apply_refractory(_merge_close(ch_detections, n_trough), refractory_n)
        detections.extend(merged)

    detections.sort(key=lambda d: (d['channel'], d['peak_index']))
    if return_info:
        info = {'blocks': blocks, 'scale_factor': factor, 'artifact': artifact,
                'channel_slope': slopes, 'candidate_threshold': cand_thr, 'filters': filters}
        return detections, info
    return detections


class BarkmeierDetector:
    """
    :func:`detect_spikes_barkmeier` as a detector object (the protocol of
    :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`).

    ``BarkmeierDetector(**params).detect(x, fs)`` runs :func:`detect_spikes_barkmeier` on
    the whole montage ``x`` ``(n_channels, n_samples)`` and returns a list with one list of
    detection dicts per channel (same dicts as :func:`detect_spikes_barkmeier`, sorted by
    ``peak_index``). Parameters are those of :func:`detect_spikes_barkmeier` except
    ``valid`` and ``return_info``; they are validated at construction (names, types,
    finiteness, ranges); the checks that need ``fs`` (band vs Nyquist, ``trough_search`` of
    at least 2 samples) run in :meth:`detect`.
    """

    output = 'records'           # per-channel lists of dicts with a 'peak_index' key
    accepts_valid = True         # detect(x, fs, valid=mask) excludes gaps from statistics

    def __init__(self, **params):
        import inspect
        allowed = set(inspect.signature(detect_spikes_barkmeier).parameters) - {
            'sig', 'fs', 'valid', 'return_info'}
        unknown = set(params) - allowed
        if unknown:
            raise TypeError(f'unknown BarkmeierDetector parameter(s) {sorted(unknown)}')
        sig = inspect.signature(detect_spikes_barkmeier).parameters
        full = {k: params.get(k, sig[k].default) for k in allowed}
        _check_params(**{k: full[k] for k in inspect.signature(_check_params).parameters})
        self.params = dict(params)

    def __repr__(self):
        args = ', '.join(f'{k}={v!r}' for k, v in self.params.items())
        return f'BarkmeierDetector({args})'

    def detect(self, x, fs, valid=None):
        """Per-channel lists of detection dicts for ``x`` ``(n_channels, n_samples)``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f'BarkmeierDetector.detect expects (n_channels, n_samples), '
                             f'got {x.shape}')
        dets = detect_spikes_barkmeier(x, fs, valid=valid, **self.params)
        out = [[] for _ in range(x.shape[0])]
        for d in dets:
            out[d['channel']].append(d)
        return out


def _merge_close(dets, min_gap):
    """Keep the max-total_amp detection within each run of peaks closer than `min_gap`."""
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d['peak_index'])
    merged = []
    cluster = [dets[0]]
    for d in dets[1:]:
        if d['peak_index'] - cluster[-1]['peak_index'] < min_gap:
            cluster.append(d)
        else:
            merged.append(max(cluster, key=lambda c: c['total_amp']))
            cluster = [d]
    merged.append(max(cluster, key=lambda c: c['total_amp']))
    return merged


def _apply_refractory(dets, refractory_n):
    """Drop detections closer than `refractory_n` samples to the previous kept one."""
    if refractory_n <= 0 or not dets:
        return dets
    out = [dets[0]]
    for d in dets[1:]:
        if d['peak_index'] - out[-1]['peak_index'] >= refractory_n:
            out.append(d)
    return out
