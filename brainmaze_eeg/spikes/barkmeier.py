# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Barkmeier interictal spike detector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Amplitude/slope/duration half-wave spike detector after:

    Barkmeier, D.T., Shah, A.K., Flanagan, D., Atkinson, M.D., Agarwal, R.,
    Fuerst, D.R., Jafari-Khouzani, K., Loeb, J.A. (2012). *High inter-reviewer
    variability of spike detection on intracranial EEG addressed by an automated
    multi-channel algorithm.* Clinical Neurophysiology 123(6), 1088-1095.
    https://doi.org/10.1016/j.clinph.2011.09.023  (PMC3277646)

This is a from-scratch implementation written against the paper's Methods. It is not
a copy of any third-party source. It supersedes an earlier internal port that carried
several defects relative to the paper; see the module notes below and the accompanying
PR for the point-by-point differences.

Algorithm (per the paper)
-------------------------
1. Band-pass the signal 20-50 Hz (``narrow_band``). Candidate spike times are samples
   where the **absolute** narrow-band amplitude exceeds ``mean(|x|) + std_coeff*std(|x|)``
   (the paper: "absolute amplitudes of peaks greater than four standard deviations").
2. Band-pass the signal 1-80 Hz (``broad_band``) for morphology, then **block-scale**:
   divide by the median across channels of each channel's mean absolute amplitude and
   multiply by ``scale`` (default 70), bringing the median channel amplitude to a fixed
   value while preserving relative differences between channels. Thresholds are therefore
   evaluated in this scaled domain.
3. For each candidate, locate the broad-band peak, then the flanking troughs within
   ``trough_search`` seconds. Compute each half-wave's amplitude, duration and slope.
4. Accept the candidate as a spike iff **all** of: total amplitude of both half-waves
   ``> total_amp``, each half-wave slope ``> slope``, and each half-wave duration
   ``> half_dur``. (Conjunctive; the paper excludes candidates falling below threshold.)

Differences from the earlier internal port (all verified against the paper)
---------------------------------------------------------------------------
- Candidate detection used ``x > thresh`` (positive only); the paper uses ``|x| > thresh``,
  so negative-dominant spikes were missed. Fixed.
- The acceptance test had a second branch accepting candidates with **all metrics below**
  the thresholds, contradicting the paper (which excludes sub-threshold candidates). Removed.
- The refractory test ``spike_i - last_idx > 0.005`` compared a **sample count** to
  ``0.005`` seconds, so at any realistic ``fs`` it never rejected anything. Reimplemented
  as a correctly-united, optional refractory (``refractory`` seconds; the paper defines
  none, so the default is 0).
- Block-scaling was applied per channel using that channel's own mean, which destroys the
  cross-channel amplitude relationships the paper's block-scaling exists to preserve. This
  implementation scales all channels by a single median-based factor.
- Peak picking ran ``find_peaks`` on the gappy supra-threshold subsequence (indices are not
  contiguous in time), which can invent maxima. It now runs on the continuous rectified band.

Caveat: block-scaling is inherently multi-channel
--------------------------------------------------
The paper's block-scaling brings the **median across channels** of the channel amplitudes
to ``scale``, which is what makes the amplitude thresholds comparable across a montage. On a
single isolated channel the median degenerates to that channel's own amplitude, so the noise
floor is normalised to ``scale`` and the fixed thresholds fire on noise at a low but non-zero
rate (order 0.3/s on pure noise at the defaults). Pass the full ``(n_channels, n_samples)``
montage so the population median does the scaling; interpret single-channel output with this
in mind. This is a property of the method, not of this implementation.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

__all__ = ['detect_spikes_barkmeier', 'DEFAULT_THRESHOLDS']

# Thresholds are evaluated in the block-scaled domain (median channel amplitude -> `scale`).
# total_amp and slope track the paper's 600 uV and 7 uV/ms (= 7000 uV/s); half_dur is the
# physical 10 ms half-wave duration.
DEFAULT_THRESHOLDS = {
    'total_amp': 600.0,   # total amplitude of both half-waves (scaled units)
    'slope': 7000.0,      # each half-wave slope, scaled-units per second (7 uV/ms)
    'half_dur': 0.010,    # each half-wave duration, seconds (10 ms)
}


def _bandpass(x, fs, band, axis=-1):
    """Zero-phase band-pass: 2nd-order high-pass then 4th-order low-pass, per the paper."""
    bh, ah = butter(2, band[0] / (fs / 2), 'highpass')
    bl, al = butter(4, band[1] / (fs / 2), 'lowpass')
    x = filtfilt(bh, ah, x, axis=axis)
    x = filtfilt(bl, al, x, axis=axis)
    return x


def detect_spikes_barkmeier(sig, fs, scale=70.0, std_coeff=4.0, trough_search=0.05,
                            thresholds=None, narrow_band=(20, 50), broad_band=(1, 80),
                            refractory=0.0):
    """
    Detect interictal spikes with the Barkmeier (2012) half-wave criteria.

    Parameters
    ----------
    sig : np.ndarray
        EEG signal, ``(n_samples,)`` or ``(n_channels, n_samples)``. Amplitudes in uV.
    fs : float
        Sampling frequency in Hz.
    scale : float
        Block-scaling target: the median channel amplitude is scaled to this value
        (default 70, per the paper).
    std_coeff : float
        Candidate threshold in standard deviations of the rectified narrow-band signal
        (default 4).
    trough_search : float
        Half-window (seconds) each side of the peak in which to find the flanking troughs.
    thresholds : dict, optional
        Acceptance thresholds ``{'total_amp', 'slope', 'half_dur'}`` in the block-scaled
        domain. Defaults to :data:`DEFAULT_THRESHOLDS`.
    narrow_band, broad_band : tuple(float, float)
        Band-pass edges (Hz) for candidate detection and morphology, respectively.
    refractory : float
        Minimum time (seconds) between accepted spikes on a channel. The paper defines no
        refractory; default 0 disables it.

    Returns
    -------
    list of dict
        One dict per detected spike, sorted by time, with keys:
        ``channel, peak_index, peak_time, peak_amp, left_amp, left_dur, left_slope,
        right_amp, right_dur, right_slope, total_amp``. Indices/times are into ``sig``.

    Raises
    ------
    ValueError
        If ``sig`` is not 1-D or 2-D, or a band edge is at/above Nyquist.
    """
    thr = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)

    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        x = sig[np.newaxis, :]
    elif sig.ndim == 2:
        x = sig
    else:
        raise ValueError(f"'sig' must be 1-D or 2-D, got {sig.ndim}-D.")

    nyq = fs / 2
    for band in (narrow_band, broad_band):
        if not (0 < band[0] < band[1] < nyq):
            raise ValueError(f"band {band} must satisfy 0 < low < high < fs/2 ({nyq}).")

    fx_narrow = _bandpass(x, fs, narrow_band, axis=-1)
    fx_broad = _bandpass(x, fs, broad_band, axis=-1)

    # Block-scaling: one factor for the whole array, from the median (across channels) of
    # each channel's mean absolute broad-band amplitude. Preserves inter-channel ratios.
    chan_mean_amp = np.mean(np.abs(fx_broad), axis=-1)
    denom = np.median(chan_mean_amp)
    if denom > 0:
        fx_broad = fx_broad * (scale / denom)

    n_samples = x.shape[-1]
    half_peak = int(round(fs * 0.002))            # +/- 2 ms search for the broad-band peak
    n_trough = int(round(fs * trough_search))
    refractory_n = int(round(fs * refractory))

    detections = []
    for ch in range(x.shape[0]):
        narrow = fx_narrow[ch]
        broad = fx_broad[ch]
        rect = np.abs(narrow)

        thresh = rect.mean() + std_coeff * rect.std()
        # candidate peaks: local maxima of the rectified narrow band that clear threshold
        peak_idx, _ = find_peaks(rect, height=thresh)

        last_idx = -np.inf
        ch_detections = []
        for pi in peak_idx:
            l = max(pi - half_peak, 0)
            r = min(pi + half_peak + 1, n_samples)
            # spike polarity from the broad band: whichever extreme is larger in magnitude
            seg = broad[l:r]
            spike_i = l + int(np.argmax(np.abs(seg)))
            spike_V = broad[spike_i]
            sign = np.sign(spike_V) or 1.0

            # flanking troughs (opposite extreme) within trough_search on each side
            ll = max(spike_i - n_trough, 0)
            rr = min(spike_i + n_trough + 1, n_samples)
            if spike_i - ll < 1 or rr - spike_i < 2:
                continue
            left_i = ll + int(np.argmin(sign * broad[ll:spike_i]))
            right_i = spike_i + int(np.argmin(sign * broad[spike_i:rr]))

            left_amp = abs(spike_V - broad[left_i])
            right_amp = abs(spike_V - broad[right_i])
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
            if spike_i - last_idx < refractory_n:
                continue
            last_idx = spike_i

            ch_detections.append({
                'channel': ch,
                'peak_index': int(spike_i),
                'peak_time': spike_i / fs,
                'peak_amp': float(spike_V),
                'left_amp': float(left_amp), 'left_dur': float(left_dur),
                'left_slope': float(left_slope),
                'right_amp': float(right_amp), 'right_dur': float(right_dur),
                'right_slope': float(right_slope),
                'total_amp': float(total_amp),
            })

        # One epileptiform discharge produces a cluster of supra-threshold narrow-band
        # maxima; collapse detections whose peaks fall within `trough_search` of each
        # other, keeping the largest-amplitude one, so a discharge yields a single spike.
        detections.extend(_merge_close(ch_detections, n_trough))

    detections.sort(key=lambda d: (d['channel'], d['peak_index']))
    return detections


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
