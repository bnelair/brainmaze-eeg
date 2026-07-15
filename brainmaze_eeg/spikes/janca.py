# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Janca (Hilbert-envelope) interictal spike detector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interictal epileptiform discharge (IED) detector based on modelling the distribution of
the band-passed signal's Hilbert envelope as log-normal, and flagging envelope maxima that
exceed an adaptive threshold derived from that distribution.

    Janca, R., Jezdik, P., Cmejla, R., Tomasek, M., Worrell, G.A., Stead, M., Wagenaar, J.,
    Jefferys, J.G.R., Krsek, P., Komarek, V., Jiruska, P., Marusic, P. (2015). *Detection of
    Interictal Epileptiform Discharges Using Signal Envelope Distribution Modelling:
    Application to Epileptic and Non-Epileptic Intracranial Recordings.* Brain Topography
    28(1), 172-183. https://doi.org/10.1007/s10548-014-0379-1

This is an independent implementation written against the paper and the algorithm's public
MATLAB reference (EpiReC-ISARG/IED_detector ``spike_detector_hilbert_v24.m``, cross-checked
against Lab-Frauscher/Spike-Gamma ``v25.m``). **No third-party source is vendored** -- see
``brainmaze_eeg/spikes/README.md`` for the licensing of those repositories.

Relationship to the earlier internal Python port
------------------------------------------------
An earlier draft translation had two load-bearing defects that this implementation fixes:

- **Decimation.** The draft resampled with ``scipy.signal.decimate(x, int(q/p), ...)``, where
  ``int(q/p)`` truncates the non-integer ``fs/decimation`` ratio to an integer, so the target
  rate was wrong for most input rates (and silently a no-op when ``int(q/p) == 1``). Here the
  signal is resampled with :func:`scipy.signal.resample_poly` at the exact rational ratio
  ``decimation/fs`` reduced by its gcd, which handles arbitrary input rates.
- **Segment overlap.** The draft's boundary-trim conditions were off by one relative to the
  MATLAB (0- vs 1-based): ``(i > 1)`` should be ``(i > 0)`` ("not the first segment") and the
  right-margin test ``(i < len(index_stop))`` is always true and should be ``(i < N - 1)``
  ("not the last segment"). The net effect was duplicated / dropped detections at segment
  joins. Here buffering uses a core-partition scheme (below) that is overlap-invariant by
  construction, verified by a whole-vs-buffered test.

Buffering scheme
----------------
The record is partitioned into contiguous *core* windows that tile ``[0, N)`` exactly. Each
core is analysed inside a block extended by ``margin = 3 * winsize`` samples on each side (for
filter/threshold context), and only detections whose position lands inside the core are kept.
Because the cores partition the signal, every detection is produced exactly once, with full
two-sided context except at the true signal ends.

Not implemented
---------------
Beta/mu-activity rejection (``beta`` parameter) and the ``ti_switch == 2`` timing mode are not
implemented; both were untested in the source. Requesting beta detection raises.
"""

import warnings
from math import gcd

import numpy as np
from scipy.signal import (butter, cheby2, cheb2ord, filtfilt, firwin, hilbert,
                          resample_poly)
from scipy.interpolate import interp1d
from scipy.special import erf

__all__ = ['SpikeDetectorHilbert', 'spike_detector_hilbert_v24']


class SpikeDetectorHilbert:
    """
    Janca envelope-distribution IED detector.

    Parameters (defaults follow the reference implementation)
    ---------------------------------------------------------
    bandwidth : (float, float)
        Band-pass edges [low, high] in Hz. Default [10, 60].
    k1 : float
        Threshold multiplier for obvious spikes. Default 3.65.
    k2 : float
        Threshold multiplier for ambiguous spikes (accepted only near an obvious detection).
        Default equals ``k1`` (ambiguous detection disabled).
    k3 : float
        Threshold tilt term. Default 0.
    main_hum_freq : float
        Mains frequency to notch (Hz). Default 50.
    decimation : float
        Target analysis sampling rate (Hz). Default 200. Set 0 to keep the input rate.
    buffering : float
        Core window length for batch processing (seconds). Default 300.
    winsize, noverlap : float
        Envelope-model window and overlap in **seconds**. Defaults 5 and 4.
    polyspike_union_time : float
        Poly-spike union interval (seconds). Default 0.12.
    discharge_tol : float
        Grouping tolerance for multichannel events (seconds). Default 0.005.
    f_type : int
        Band-pass family: 1 = Chebyshev-II (default), 2 = Butterworth, 3 = FIR.
    beta : float
        Low edge (Hz) of beta rejection. ``inf`` (default) disables it; any finite value
        raises ``NotImplementedError``.
    """

    def __init__(self, **kwargs):
        self.bandwidth = [10.0, 60.0]
        self.k1 = 3.65
        self.k2 = self.k1
        self.k3 = 0.0
        self.main_hum_freq = 50.0
        self.decimation = 200.0
        self.buffering = 300.0
        self.winsize = 5.0          # seconds
        self.noverlap = 4.0         # seconds
        self.polyspike_union_time = 0.12
        self.discharge_tol = 0.005
        self.f_type = 1
        self.beta = np.inf
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f'unknown parameter {key!r}')
            setattr(self, key, value)
        if self.k2 < self.k1:
            raise ValueError('k2 must be >= k1')

    # ------------------------------------------------------------------ public
    def run(self, d, fs):
        """
        Detect IEDs in ``d`` sampled at ``fs``.

        Parameters
        ----------
        d : np.ndarray
            Signal, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
        fs : float
            Input sampling frequency (Hz).

        Returns
        -------
        out : dict
            Per-detection arrays: ``pos`` (s), ``dur`` (s), ``chan`` (int),
            ``con`` (1 obvious / 0.5 ambiguous), ``weight`` (envelope CDF), ``pdf``.
        discharges : dict
            Per multichannel event: ``MV`` [n_events, n_chan] spike type, ``MA`` max envelope
            above background, ``MP`` start position (s), ``MD`` duration (s), ``MW`` CDF weight,
            ``MPDF`` pdf.
        d_decim : np.ndarray
            Decimated, hum-notched signal ``(n_samples_dec, n_chan)``.
        envelope : np.ndarray
            Hilbert envelope of the band-passed signal ``(n_samples_dec, n_chan)``.
        background : np.ndarray
            Threshold curves ``(n_samples_dec, n_chan, 2)`` for k1 and k2.
        envelope_pdf : np.ndarray
            Log-normal PDF of the envelope ``(n_samples_dec, n_chan)``.
        """
        fs = float(fs)
        if np.isfinite(self.beta):
            raise NotImplementedError('beta/mu rejection is not implemented')

        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 1:
            d = d[:, None]
        elif d.ndim != 2:
            raise ValueError("'d' must be 1-D or 2-D (n_samples, n_channels)")

        target = fs if self.decimation in (0, None) else float(self.decimation)
        if self.bandwidth[1] >= target / 2:
            raise ValueError(f'bandwidth high edge {self.bandwidth[1]} >= target Nyquist {target/2}')

        # -- decimate (per channel), then notch mains + 1 Hz high-pass -----------
        d_dec = self._resample(d, fs, target)
        fsd = target
        d_dec = self._filt_hum(d_dec, fsd)
        bb, aa = butter(2, 2 * 1.0 / fsd, 'highpass')
        d_decim = filtfilt(bb, aa, d_dec, axis=0)

        n = d_decim.shape[0]
        winsize = int(round(self.winsize * fsd))
        margin = 3 * winsize
        core = max(int(round(self.buffering * fsd)), winsize)

        # -- core-partition buffering (overlap-invariant) ------------------------
        out = _empty_out()
        markers_high = np.zeros((n, d_decim.shape[1]), dtype=bool)
        markers_low = np.zeros((n, d_decim.shape[1]), dtype=bool)
        envelope = np.zeros_like(d_decim)
        background = np.zeros((n, d_decim.shape[1], 2))
        envelope_cdf = np.zeros_like(d_decim)
        envelope_pdf = np.zeros_like(d_decim)

        starts = list(range(0, n, core)) if n > core else [0]
        for a in starts:
            b = min(a + core, n)
            bs = max(a - margin, 0)
            be = min(b + margin, n)
            blk = d_decim[bs:be]
            env, mh, ml, bg, cdf, pdf = self._detect_block(blk, fsd, winsize)

            # write back the core slice [a, b) from block-local coords
            lo, hi = a - bs, b - bs
            envelope[a:b] = env[lo:hi]
            background[a:b] = bg[lo:hi]
            envelope_cdf[a:b] = cdf[lo:hi]
            envelope_pdf[a:b] = pdf[lo:hi]
            markers_high[a:b] = mh[lo:hi]
            markers_low[a:b] = ml[lo:hi]

        # blank first/last second (filter transient), per the reference
        edge = int(round(fsd))
        if n > 2 * edge:
            markers_high[:edge] = markers_high[-edge:] = False
            markers_low[:edge] = markers_low[-edge:] = False

        out = self._markers_to_out(markers_high, markers_low, envelope_cdf, envelope_pdf, fsd)
        discharges = self._group_discharges(out, envelope, background, envelope_cdf,
                                            envelope_pdf, d_decim, fsd)
        return out, discharges, d_decim, envelope, background, envelope_pdf

    # ------------------------------------------------------------- core science
    def _detect_block(self, d, fs, winsize):
        """Band-pass, envelope, threshold and mark one block of shape (n_samples, n_chan)."""
        d_bp = self._bandpass(d, fs)
        n, nch = d_bp.shape
        envelope = np.zeros((n, nch))
        markers_high = np.zeros((n, nch), dtype=bool)
        markers_low = np.zeros((n, nch), dtype=bool)
        background = np.zeros((n, nch, 2))
        cdf = np.zeros((n, nch))
        pdf = np.zeros((n, nch))

        step = max(winsize - int(round(self.noverlap * fs)), 1)
        index = np.arange(0, max(n - winsize + 1, 1), step)

        for ch in range(nch):
            if not d_bp[:, ch].any():
                continue
            (envelope[:, ch], markers_high[:, ch], markers_low[:, ch],
             background[:, ch, :], cdf[:, ch], pdf[:, ch]) = self._one_channel(
                d_bp[:, ch], fs, index, winsize)
        return envelope, markers_high, markers_low, background, cdf, pdf

    def _one_channel(self, d, fs, index, winsize):
        envelope = np.abs(hilbert(d))
        k1, k2, k3 = self.k1, self.k2, self.k3

        # per-window MLE of log-normal params on the envelope
        phat = np.zeros((len(index), 2))
        for k, i0 in enumerate(index):
            seg = envelope[i0:i0 + winsize]
            seg = seg[seg > 0]
            if seg.size == 0:
                phat[k] = [0.0, 1.0]
            else:
                logs = np.log(seg)
                phat[k] = [logs.mean(), logs.std()]

        r = envelope.shape[0] / max(len(index), 1)
        n_avg = int(round((winsize / fs) * (fs / r))) if r > 0 else 1
        # smooth the per-window params; filtfilt needs len(phat) > padlen (= 3*(n_avg-1))
        if n_avg > 1 and phat.shape[0] > 3 * (n_avg - 1) + 1:
            phat = filtfilt(np.ones(n_avg) / n_avg, 1, phat, axis=0)

        # interpolate window params to a full-length threshold "background" curve
        if phat.shape[0] > 1:
            centers = index + round(winsize / 2)
            xs = np.arange(centers[0], centers[-1] + 1)
            # cubic interpolation needs >= 4 points; short recordings (2-3 window centers)
            # fall back to linear rather than raising a ValueError inside interp1d.
            kind = 'cubic' if centers.size >= 4 else 'linear'
            pi = np.empty((xs.size, 2))
            for c in range(2):
                pi[:, c] = interp1d(centers, phat[:, c], kind=kind,
                                    fill_value='extrapolate')(xs)
            top = int(np.floor(winsize / 2))
            head = np.repeat(pi[:1], top, axis=0)
            tail = np.repeat(pi[-1:], max(envelope.shape[0] - pi.shape[0] - top, 0), axis=0)
            phat_int = np.vstack([head, pi, tail])[:envelope.shape[0]]
            if phat_int.shape[0] < envelope.shape[0]:
                phat_int = np.vstack([phat_int,
                                      np.repeat(phat_int[-1:], envelope.shape[0] - phat_int.shape[0], axis=0)])
        else:
            phat_int = np.repeat(phat[:1], envelope.shape[0], axis=0)

        mu, sigma = phat_int[:, 0], phat_int[:, 1]
        mode = np.exp(mu - sigma ** 2)
        median = np.exp(mu)
        mean = np.exp(mu + sigma ** 2 / 2)

        prah = np.zeros((envelope.shape[0], 2))
        prah[:, 0] = k1 * (mode + median) - k3 * (mean - mode)
        prah[:, 1] = (k2 * (mode + median) - k3 * (mean - mode)) if k2 != k1 else prah[:, 0]

        with np.errstate(divide='ignore', invalid='ignore'):
            log_env = np.log(np.where(envelope > 0, envelope, np.nan))
            cdf = 0.5 + 0.5 * erf((log_env - mu) / np.sqrt(2 * sigma ** 2))
            pdf = np.exp(-0.5 * ((log_env - mu) / sigma) ** 2) / (envelope * sigma * np.sqrt(2 * np.pi))
        cdf = np.nan_to_num(cdf)
        pdf = np.nan_to_num(pdf)

        mh = self._local_maxima(envelope, prah[:, 0], fs)
        mh = self._detection_union(mh, envelope, self.polyspike_union_time * fs)
        if k2 != k1:
            ml = self._local_maxima(envelope, prah[:, 1], fs)
            ml = self._detection_union(ml, envelope, self.polyspike_union_time * fs)
        else:
            ml = mh
        return envelope, mh, ml, prah, cdf, pdf

    # -------------------------------------------------- maxima / union helpers
    @staticmethod
    def _runs(mask):
        """Start (inclusive) / stop (exclusive) indices of True runs in a boolean mask."""
        m = mask.astype(int)
        starts = np.flatnonzero(np.diff(np.concatenate(([0], m))) > 0)
        stops = np.flatnonzero(np.diff(np.concatenate((m, [0]))) < 0) + 1
        return starts, stops

    def _local_maxima(self, envelope, prah, fs):
        above = envelope > prah
        starts, stops = self._runs(above)

        marker = np.zeros(envelope.shape[0], dtype=bool)
        for s, e in zip(starts, stops):
            if e - s > 2:
                seg = envelope[s:e]
                sgn = np.sign(np.diff(seg))
                loc = np.flatnonzero(np.diff(np.concatenate(([0], sgn))) < 0)
                marker[s + loc] = True
            else:
                marker[s + int(np.argmax(envelope[s:e]))] = True

        # union runs of maxima closer than polyspike_union_time
        pointer = np.flatnonzero(marker)
        pu = self.polyspike_union_time * fs
        state = False
        start = 0
        for k in range(len(pointer)):
            hi = int(np.ceil(pointer[k] + pu))
            seg = marker[pointer[k] + 1: hi + 1] if hi < marker.shape[0] else marker[pointer[k] + 1:]
            if state:
                if seg.sum() > 0:
                    state = True
                else:
                    state = False
                    marker[start:pointer[k]] = True
            else:
                if seg.sum() > 0:
                    state = True
                    start = pointer[k]

        # keep only the local-max peaks within each (now unioned) run
        starts, stops = self._runs(marker)
        for s, e in zip(starts, stops):
            if e - s > 1:
                lm = pointer[(pointer >= s) & (pointer < e)]
                marker[s:e] = False
                if lm.size:
                    vals = envelope[lm]
                    keep = np.flatnonzero(np.diff((np.diff(np.concatenate(([0], vals, [0]))) < 0).astype(int)) > 0)
                    marker[lm[keep]] = True
        return marker

    def _detection_union(self, marker, envelope, union_samples):
        u = int(np.ceil(union_samples))
        if u % 2 == 0:
            u += 1
        mask = np.ones(u)
        dil = np.convolve(marker.astype(float), mask, mode='same') > 0     # dilation
        ero = np.convolve((~dil).astype(float), mask, mode='same') > 0     # erosion
        closed = ~ero

        out = np.zeros(marker.shape[0], dtype=bool)
        starts, stops = self._runs(closed)
        for s, e in zip(starts, stops):
            out[s + int(np.argmax(envelope[s:e]))] = True
        return out

    # ----------------------------------------------------------- output plumbing
    def _markers_to_out(self, markers_high, markers_low, cdf, pdf, fs):
        out = _empty_out()
        t_dur = 0.005
        obvious_any = markers_high.any(axis=1)
        for ch in range(markers_high.shape[1]):
            idx = np.flatnonzero(markers_high[:, ch])
            if idx.size:
                _append_out(out, idx / fs, t_dur, ch, 1.0, cdf[idx, ch], pdf[idx, ch])
        if self.k2 != self.k1:
            for ch in range(markers_low.shape[1]):
                idx = np.flatnonzero(markers_low[:, ch] & ~markers_high[:, ch])
                for i in idx:
                    lo = max(int(i - 0.01 * fs), 0)
                    if obvious_any[lo:i + 1].any():   # ambiguous accepted near an obvious spike
                        _append_out(out, np.array([i / fs]), t_dur, ch, 0.5,
                                    np.array([cdf[i, ch]]), np.array([pdf[i, ch]]))
        order = np.argsort(out['pos']) if len(out['pos']) else []
        for key in out:
            out[key] = np.asarray(out[key])[order] if len(out[key]) else np.array([])
        return out

    def _group_discharges(self, out, envelope, background, cdf, pdf, d_decim, fs):
        nch = envelope.shape[1]
        disc = {k: [] for k in ('MV', 'MA', 'MP', 'MD', 'MW', 'MPDF')}
        if not len(out['pos']):
            return {k: np.zeros((0, nch)) for k in disc}

        tol = int(round(self.discharge_tol * fs))
        M = np.zeros((envelope.shape[0], nch))
        for p, ch, con in zip(out['pos'], out['chan'], out['con']):
            s = int(round(p * fs))
            M[s:s + tol + 1, int(ch)] = con

        active = M.sum(axis=1) > 0
        starts, stops = self._runs(active)
        for s, e in zip(starts, stops):
            seg = M[s:e]
            mv = seg.max(axis=0)
            env_seg = envelope[s:e] - background[s:e, :, 0] / self.k1
            ma = np.abs(env_seg).max(axis=0)
            mw = cdf[s:e].max(axis=0)
            mpdf = (pdf[s:e] * (seg > 0)).max(axis=0)
            mp = np.full(nch, np.nan)
            rows, cols = np.where(seg > 0)
            for rr, cc in zip(rows, cols):
                if np.isnan(mp[cc]):
                    mp[cc] = (s + rr) / fs
            disc['MV'].append(mv)
            disc['MA'].append(ma)
            disc['MW'].append(mw)
            disc['MPDF'].append(mpdf)
            disc['MP'].append(mp)
            disc['MD'].append(np.full(nch, (e - s) / fs))
        return {k: np.array(v) for k, v in disc.items()}

    # ---------------------------------------------------------------- filtering
    def _resample(self, d, fs, target):
        if target == fs:
            return d.copy()
        g = gcd(int(round(target)), int(round(fs)))
        up, down = int(round(target)) // g, int(round(fs)) // g
        return resample_poly(d, up, down, axis=0)

    def _filt_hum(self, d, fs):
        """Comb of 2nd-order notches at the mains frequency and harmonics up to ~1.1*high."""
        R, r = 1.0, 0.985
        f = self.main_hum_freq
        while f <= 1.1 * self.bandwidth[1] and f < fs / 2:
            w = 2 * np.pi * f / fs
            b = np.array([1.0, -2 * R * np.cos(w), R * R])
            a = np.array([1.0, -2 * r * np.cos(w), r * r])
            d = filtfilt(b, a, d, axis=0)
            f += self.main_hum_freq
        return d

    def _bandpass(self, d, fs):
        ftype = self.f_type
        if ftype == 1 and self.decimation not in (0, None) and self.decimation != 200:
            warnings.warn('f_type switched to Butterworth for non-200 Hz decimation')
            ftype = 2
        lo, hi = self.bandwidth
        if ftype == 1:      # Chebyshev-II
            n, _ = cheb2ord(2 * hi / fs, 2 * hi / fs + 0.1, 6, 60)
            bl, al = cheby2(n, 60, 2 * hi / fs)
            n, _ = cheb2ord(2 * lo / fs, 2 * lo / fs - 0.05, 6, 60)
            bh, ah = cheby2(n, 60, 2 * lo / fs, 'highpass')
        elif ftype == 2:    # Butterworth
            bh, ah = butter(4, 2 * lo / fs, 'highpass')
            bl, al = butter(4, 2 * hi / fs, 'lowpass')
        else:               # FIR
            bh = firwin(int(fs / 2) | 1, 2 * lo / fs, pass_zero='highpass')
            ah = 1.0
            bl = firwin(int(fs / 2) | 1, 2 * hi / fs)
            al = 1.0
        d = filtfilt(bh, ah, d, axis=0)
        if hi < fs / 2:
            d = filtfilt(bl, al, d, axis=0)
        return d


# Backwards-compatible alias for the name used by the original port.
spike_detector_hilbert_v24 = SpikeDetectorHilbert


def _empty_out():
    return {'pos': [], 'dur': [], 'chan': [], 'con': [], 'weight': [], 'pdf': []}


def _append_out(out, pos, dur, chan, con, weight, pdf):
    pos = np.atleast_1d(pos)
    out['pos'] = np.concatenate([out['pos'], pos])
    out['dur'] = np.concatenate([out['dur'], np.full(pos.size, dur)])
    out['chan'] = np.concatenate([out['chan'], np.full(pos.size, chan)])
    out['con'] = np.concatenate([out['con'], np.full(pos.size, con)])
    out['weight'] = np.concatenate([out['weight'], np.atleast_1d(weight)])
    out['pdf'] = np.concatenate([out['pdf'], np.atleast_1d(pdf)])
