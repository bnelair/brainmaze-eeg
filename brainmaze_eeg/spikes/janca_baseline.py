# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Reference baseline for the Janca detector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Janca detector (:func:`~brainmaze_eeg.spikes.janca.detect_spikes_janca`) models the
background as the log-normal distribution of the band-passed Hilbert envelope ``e`` in a
sliding window (5 s by default) and detects envelope maxima above
``threshold * (exp(mu - sd**2) + exp(mu))`` (mode + median of the model), where ``mu`` and
``sd`` are the mean and standard deviation of ``log(e + eps)``. On a channel that spikes
**permanently** (several spikes per second, e.g. after an injury) the local window learns
the spikes as background and the threshold rises with the spike rate: on synthetic iEEG with
150 uV spikes in 30 uV pink background, the local model finds 93 / 42 / 0.5 % of the spikes
at 1 / 3 / 5 spikes/s, and a longer window does not help.

A :class:`JancaBaseline` holds ``mu`` and ``sd`` per channel measured on a **reference**
recording instead: a quiet or pre-injury recording of the same channels, with the same
amplitude unit, gain and montage. With it the detector uses a fixed threshold per channel
(``combine='reference'``), or the lower (``'min'``, more sensitive) or higher (``'max'``,
stricter) of the fixed and the local threshold. Same data: 97 / 97 / 96 % at 1 / 3 / 5
spikes/s, no false detections.

How the statistics are made (:meth:`JancaBaseline.from_signal`)
---------------------------------------------------------------
1. Missing data is **excluded**, not filled: NaN/inf samples and runs of an exactly constant
   value lasting at least ``flat_as_gap_s`` (0.1 s; dropouts stored as a constant, clipping;
   the rule of :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`).
2. Each remaining contiguous run is filtered, resampled and enveloped **on its own**, exactly
   as the detector does (same filters, same resampling, same Hilbert envelope).
3. ``stats_margin_s`` (0.5 s) of analysis samples are dropped at both ends of every run (also
   at the ends of the record), so filter and Hilbert transients at gaps and edges do not
   enter the statistics. Runs too short for the zero-phase filters or for the two margins
   are skipped. Measured on pink noise (5-20 s runs, 200-1000 Hz analysis rates): beyond
   0.1 s from a run edge the pooled statistics equal those of the uninterrupted signal
   (threshold ratio 1.0000-1.0001); 0.5 s is a safety factor against DC steps and amplifier
   recovery after a dropout, at little cost of valid time.
4. ``eps = eps_rel * median(e)`` over the pooled valid envelope samples of the channel, then
   ``mu = mean(log(e + eps))``, ``sd = std(log(e + eps))`` (population SD) over the pooled
   samples: the definition of the local model with the window being the whole valid
   reference. ``robust=True`` uses ``median`` and ``1.4826 * MAD`` instead. This is a
   **different statistic**, not a drop-in: the log of a real (band-passed noise) envelope is
   skewed, so on clean background the robust threshold is about 14 % higher than the
   mean/SD one (synthetic background, 500 Hz). It is pulled up somewhat less by spikes in
   the reference (3 spikes/s in the reference: threshold x1.31 vs x1.44). Prefer a clean
   reference over relying on it.

The valid reference time per channel is stored (``valid_s``); less than ``min_valid_s``
(60 s) warns, none raises.

What must match, what is free
-----------------------------
The **signal path** must be identical between the baseline and the detection, because the
envelope level depends on it: ``band``, ``filter_order``, ``powerline``, ``notch_width``,
``notch_order``, ``notch_harmonics``, ``target_fs``, ``decimation``, ``eps_rel``
(:data:`~brainmaze_eeg.spikes.janca.SIGNAL_PATH_PARAMS`) and the analysis rate
``fs_analysis`` (which follows from the input rate, ``target_fs`` and ``decimation``; with
``decimation='exact'`` recordings at different input rates share one analysis rate). The
detector raises ``ValueError`` naming every mismatch. The **decision** parameters are free:
``threshold``, ``min_distance_s``, ``window_s`` (local model), ``combine``.

Not checkable, and the user's responsibility: the amplitude **unit** and gain (uV vs V), the
**montage**/reference and the **channel order**. The detector warns when a channel's envelope
level differs from the baseline's by more than 10x (``BASELINE_LEVEL_WARN_RATIO``).

Limitations
-----------
- A fixed baseline does not follow slow changes of the background amplitude (electrode
  impedance, drift over days, sleep/wake, medication). Refresh the baseline from a recent
  quiet stretch, or combine it with the local model: ``'max'`` (never more sensitive than
  either model) or ``'min'`` (never less sensitive).
- A reference that itself contains spikes or artifacts inflates ``mu``/``sd`` and raises the
  threshold; use quiet data, ``segments=`` to pick spike-free stretches, or ``robust=True``.
"""

import json
import math
import warnings

import numpy as np

from brainmaze_eeg.spikes import _checks as chk
from brainmaze_eeg.spikes._gaps import FLAT_AS_GAP_S, flat_runs
from brainmaze_eeg.spikes.janca import (_P, DEFAULT_STATS_MARGIN_S, SIGNAL_PATH_PARAMS,
                                        _as_channels_first, _channel_envelope,
                                        _envelope_scale, _signal_path, janca_params,
                                        janca_resampling, janca_threshold)

__all__ = ['JancaBaseline', 'BASELINE_SCHEMA', 'BASELINE_SCHEMA_VERSION']

#: ``format`` field of the JSON file written by :meth:`JancaBaseline.save`.
BASELINE_SCHEMA = 'brainmaze_eeg.spikes.JancaBaseline'
#: Version of that JSON schema (files with a newer version are refused).
BASELINE_SCHEMA_VERSION = 1

_NOTCH_PARAMS = ('notch_width', 'notch_order', 'notch_harmonics')
_FS_REL_TOL = 1e-9
_MAD_TO_SD = 1.4826


def _vector(name, v):
    a = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f'{name} must be a scalar or a non-empty 1-D array (one value per '
                         f'channel), got shape {np.shape(v)}')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite, got {a.tolist()}')
    return a


def _readonly(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a.flags.writeable = False
    return a


def _jsonable_params(params):
    out = dict(params)
    out['band'] = list(out['band'])
    return out


class JancaBaseline:
    """
    Per-channel reference statistics of the Janca log-envelope model, with the signal-path
    parameters they were made with. See the module docstring.

    Make one from a reference recording with :meth:`from_signal`, load one with
    :meth:`load`, or give the values directly::

        JancaBaseline(mu=[3.4, 3.1], sd=[0.55, 0.6], fs=500)           # 'spike' path at 500 Hz
        JancaBaseline(mu=3.4, sd=0.55, fs_analysis=250, powerline=60)

    Parameters
    ----------
    mu, sd : float or array_like
        Mean and standard deviation of ``log(e + eps)`` per channel (``e``: Hilbert envelope
        of the band-passed signal, in the input unit). One value each per channel; finite,
        ``sd >= 0``. The fixed threshold is ``threshold * (exp(mu - sd**2) + exp(mu))``.
    fs_analysis : float, optional
        Analysis rate (Hz) of the envelope the statistics describe.
    fs : float, optional
        Input sampling rate (Hz) of the reference; ``fs_analysis`` is then derived from it
        (:func:`~brainmaze_eeg.spikes.janca.janca_resampling`) and the signal path is
        validated at it. At least one of ``fs`` / ``fs_analysis`` is required; if both are
        given they must agree.
    preset, band, filter_order, powerline, notch_width, notch_order, notch_harmonics, \
target_fs, decimation, eps_rel
        The signal path, as in :func:`~brainmaze_eeg.spikes.janca.detect_spikes_janca`
        (defaults: the preset's values).
    valid_s : float or array_like, optional
        Seconds of valid reference data per channel (metadata).
    channel_names : sequence of str, optional
        One name per channel (metadata; also usable in :meth:`select`).
    units : str, optional
        Amplitude unit of the reference signal, e.g. ``'uV'`` (metadata; not checked against
        the detection input, which carries no unit).
    info : dict, optional
        Further JSON-serialisable provenance (``from_signal`` fills it).

    Attributes
    ----------
    mu, sd : np.ndarray (read-only)
    n_channels : int
    params : dict
        The resolved signal-path parameters (:data:`~brainmaze_eeg.spikes.janca.SIGNAL_PATH_PARAMS`).
    preset : str
    fs_analysis : float
    fs : float or None
    valid_s : np.ndarray or None
    channel_names : tuple of str or None
    units : str or None
    info : dict
    """

    def __init__(self, mu, sd, *, fs_analysis=None, fs=None, preset='spike', band=_P,
                 filter_order=_P, powerline=_P, notch_width=_P, notch_order=_P,
                 notch_harmonics=_P, target_fs=_P, decimation=_P, eps_rel=_P, valid_s=None,
                 channel_names=None, units=None, info=None):
        p = janca_params(preset, band=band, filter_order=filter_order, powerline=powerline,
                         notch_width=notch_width, notch_order=notch_order,
                         notch_harmonics=notch_harmonics, target_fs=target_fs,
                         decimation=decimation, eps_rel=eps_rel)
        self._params = {k: p[k] for k in SIGNAL_PATH_PARAMS}
        self._preset = preset
        mu, sd = _vector('mu', mu), _vector('sd', sd)
        if mu.size != sd.size:
            if mu.size == 1:
                mu = np.repeat(mu, sd.size)
            elif sd.size == 1:
                sd = np.repeat(sd, mu.size)
            else:
                raise ValueError(f'mu has {mu.size} channels, sd has {sd.size}')
        if (sd < 0).any():
            raise ValueError(f'sd must be >= 0, got {sd.tolist()}')
        self._mu, self._sd = _readonly(mu), _readonly(sd)
        n_ch = mu.size

        if fs is None and fs_analysis is None:
            raise TypeError('give fs (input rate of the reference) or fs_analysis (analysis '
                            'rate of the envelope): the baseline is only valid at one '
                            'analysis rate')
        if fs is not None:
            fs = chk.number('fs', fs, gt=0)
            _signal_path(p, fs)                     # validates band/notch/resampling at fs
            fa = janca_resampling(fs, p['target_fs'], p['decimation'])[2]
            if fs_analysis is not None:
                fs_analysis = chk.number('fs_analysis', fs_analysis, gt=0)
                if not math.isclose(fa, fs_analysis, rel_tol=_FS_REL_TOL):
                    raise ValueError(f'fs={fs:g} Hz gives an analysis rate of {fa:g} Hz with '
                                     f"target_fs={p['target_fs']}, decimation="
                                     f"{p['decimation']!r}, not fs_analysis={fs_analysis:g}")
            fs_analysis = fa
        fs_analysis = chk.number('fs_analysis', fs_analysis, gt=0)
        if not p['band'][1] < fs_analysis / 2:
            raise ValueError(f"band high edge ({p['band'][1]} Hz) must be < Nyquist of "
                             f'fs_analysis ({fs_analysis / 2} Hz)')
        self._fs_analysis = float(fs_analysis)
        self._fs = fs

        if valid_s is not None:
            valid_s = _vector('valid_s', valid_s)
            if valid_s.size == 1 and n_ch > 1:
                valid_s = np.repeat(valid_s, n_ch)
            if valid_s.size != n_ch or (valid_s < 0).any():
                raise ValueError(f'valid_s must be >= 0 with one value per channel ({n_ch})')
            valid_s = _readonly(valid_s)
        self._valid_s = valid_s
        if channel_names is not None:
            if isinstance(channel_names, str):
                channel_names = [channel_names]
            channel_names = tuple(channel_names)
            if len(channel_names) != n_ch or not all(isinstance(c, str) for c in channel_names):
                raise ValueError(f'channel_names must be {n_ch} string(s), got {channel_names!r}')
            if len(set(channel_names)) != n_ch:
                raise ValueError(f'channel_names must be unique, got {channel_names!r}')
        self._channel_names = channel_names
        if units is not None and not isinstance(units, str):
            raise TypeError(f'units must be a string or None, got {units!r}')
        self._units = units
        info = {} if info is None else dict(info)
        try:
            json.dumps(info, allow_nan=False)
        except (TypeError, ValueError) as err:
            raise ValueError(f'info must be JSON-serialisable: {err}') from None
        self._info = info

    # ------------------------------------------------------------------ attributes
    mu = property(lambda self: self._mu, doc='Mean of log(e + eps) per channel (read-only).')
    sd = property(lambda self: self._sd, doc='SD of log(e + eps) per channel (read-only).')
    n_channels = property(lambda self: int(self._mu.size), doc='Number of channels.')
    params = property(lambda self: dict(self._params),
                      doc='Resolved signal-path parameters (a copy).')
    preset = property(lambda self: self._preset, doc='Preset the parameters were resolved from.')
    fs_analysis = property(lambda self: self._fs_analysis, doc='Analysis rate (Hz).')
    fs = property(lambda self: self._fs, doc='Input rate of the reference (Hz) or None.')
    valid_s = property(lambda self: self._valid_s,
                       doc='Valid reference seconds per channel (read-only) or None.')
    channel_names = property(lambda self: self._channel_names, doc='Channel names or None.')
    units = property(lambda self: self._units, doc='Amplitude unit (metadata) or None.')
    info = property(lambda self: dict(self._info), doc='Provenance (a copy).')

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_signal(cls, x_ref, fs, *, preset='spike', band=_P, filter_order=_P,
                    powerline=_P, notch_width=_P, notch_order=_P, notch_harmonics=_P,
                    target_fs=_P, decimation=_P, eps_rel=_P, segments=None, min_valid_s=60.0,
                    flat_as_gap_s=FLAT_AS_GAP_S, stats_margin_s=DEFAULT_STATS_MARGIN_S,
                    robust=False, channel_names=None, units=None):
        """
        Measure the baseline on a reference recording (see the module docstring).

        Parameters
        ----------
        x_ref : array_like or list of array_like
            The reference: ``(n_samples,)`` or ``(n_channels, n_samples)``, or a **list of
            segments**, each ``(n_samples_i,)`` or ``(n_channels, n_samples_i)`` with the same
            channels and any lengths (e.g. quiet stretches cut from several days); the
            statistics are pooled over all segments. NaN/inf and constant runs (see
            ``flat_as_gap_s``) are excluded. Note that a list is always a list of
            *segments*, never of channels: stack channels into a 2-D array.
        fs : float
            Sampling rate (Hz) of the reference.
        preset, band, filter_order, powerline, notch_width, notch_order, notch_harmonics, \
target_fs, decimation, eps_rel
            The signal path, exactly as it will be given to the detector (see
            :func:`~brainmaze_eeg.spikes.janca.detect_spikes_janca`). Decision parameters
            (``threshold``, ``min_distance_s``, ``window_s``) are not part of a baseline.
        segments : sequence of (start_s, stop_s), optional
            Use only these time intervals (seconds from the start of ``x_ref``), e.g.
            annotated spike-free background. Only with an array ``x_ref``.
        min_valid_s : float
            Warn (``UserWarning``) for channels with less valid reference time than this
            (default 60 s). A channel without any valid time raises ``ValueError``.
        flat_as_gap_s : float or None
            Runs of exactly equal consecutive samples at least this long (s) are missing
            data (default 0.1, as in ``GapAwareSpikeDetector``; ``None`` disables).
        stats_margin_s : float
            Seconds dropped at both ends of every valid run (default 0.5).
        robust : bool
            ``False`` (default): mean and SD of the log-envelope (the definition of the
            local model). ``True``: median and ``1.4826 * MAD`` (a different statistic: about
            14 % higher threshold on clean background; see the module docstring).
        channel_names : sequence of str, optional
        units : str, optional
            Metadata stored with the baseline.

        Returns
        -------
        JancaBaseline
        """
        p = janca_params(preset, band=band, filter_order=filter_order, powerline=powerline,
                         notch_width=notch_width, notch_order=notch_order,
                         notch_harmonics=notch_harmonics, target_fs=target_fs,
                         decimation=decimation, eps_rel=eps_rel)
        fs = chk.number('fs', fs, gt=0)
        min_valid_s = chk.number('min_valid_s', min_valid_s, ge=0)
        flat_as_gap_s = chk.number('flat_as_gap_s', flat_as_gap_s, gt=0, allow_none=True)
        stats_margin_s = chk.number('stats_margin_s', stats_margin_s, ge=0)
        if not isinstance(robust, (bool, np.bool_)):
            raise TypeError(f'robust must be a bool, got {robust!r}')
        filters, up, down, fs_a, padlen = _signal_path(p, fs)
        margin_a = int(np.ceil(stats_margin_s * fs_a - 1e-9))

        if isinstance(x_ref, (list, tuple)):
            if segments is not None:
                raise ValueError('segments= selects intervals of one array; with a list of '
                                 'segments cut them yourself')
            if len(x_ref) == 0:
                raise ValueError('x_ref is an empty list')
            pieces = [_as_channels_first(s, 'x_ref segment')[0] for s in x_ref]
        else:
            pieces = [_as_channels_first(x_ref, 'x_ref')[0]]
        n_ch = pieces[0].shape[0]
        for k, s in enumerate(pieces):
            if s.shape[0] != n_ch:
                raise ValueError(f'segment {k} has {s.shape[0]} channel(s), segment 0 has {n_ch}')
        selected = None
        if segments is not None:
            selected = _segment_mask(segments, pieces[0].shape[1], fs)

        mu, sd, valid_s = np.empty(n_ch), np.empty(n_ch), np.empty(n_ch)
        n_runs, n_skipped, empty = [], [], []
        for c in range(n_ch):
            env, used, skipped = [], 0, 0
            for s in pieces:
                row = np.asarray(s[c], dtype=np.float64)
                usable = np.isfinite(row)
                for a, b in flat_runs(row, fs, flat_as_gap_s):
                    usable[a:b] = False
                if selected is not None:
                    usable &= selected
                for a, b in _true_runs(usable):
                    if b - a <= padlen:
                        skipped += 1
                        continue
                    e = _channel_envelope(row[a:b], filters, up, down)
                    if e.size <= 2 * margin_a:
                        skipped += 1
                        continue
                    env.append(e[margin_a:e.size - margin_a])
                    used += 1
            n_runs.append(used)
            n_skipped.append(skipped)
            if not env:
                empty.append(c)
                continue
            e = np.concatenate(env)
            del env
            scale = _envelope_scale(e)
            if not scale > 0:
                empty.append(c)
                continue
            log_e = np.log(e + p['eps_rel'] * scale)
            if robust:
                mu[c] = np.median(log_e)
                sd[c] = _MAD_TO_SD * np.median(np.abs(log_e - mu[c]))
            else:
                mu[c] = log_e.mean()
                sd[c] = log_e.std()
            valid_s[c] = e.size / fs_a
        if empty:
            raise ValueError(
                f'no valid reference data in channel(s) {empty}: after removing NaN/inf, '
                f'constant runs >= {flat_as_gap_s} s and {stats_margin_s} s at both ends of '
                f'every valid run, nothing is left (runs shorter than the zero-phase filters '
                f'need ({padlen / fs:g} s) or than the two margins are skipped), or the '
                'envelope is identically zero')
        low = np.flatnonzero(valid_s < min_valid_s)
        if low.size:
            warnings.warn(f'channel(s) {low.tolist()} have only '
                          f'{np.round(valid_s[low], 1).tolist()} s of valid reference data '
                          f'(< min_valid_s={min_valid_s:g} s); the baseline statistics may be '
                          'unreliable', UserWarning, stacklevel=2)
        from brainmaze_eeg import __version__ as version
        info = {'method': 'from_signal', 'statistic': 'robust' if robust else 'mean',
                'stats_margin_s': stats_margin_s, 'flat_as_gap_s': flat_as_gap_s,
                'n_segments': len(pieces), 'n_runs': n_runs, 'n_runs_skipped': n_skipped,
                'brainmaze_eeg_version': version}
        return cls(mu, sd, fs=fs, preset=preset, **_p_signal(p), valid_s=valid_s,
                   channel_names=channel_names, units=units, info=info)

    # ------------------------------------------------------------------ use
    def check_compatible(self, params, fs_analysis=None):
        """
        Raise ``ValueError`` naming every signal-path parameter (and the analysis rate, if
        given) in which the resolved detector parameters ``params`` differ from the
        baseline's. The notch settings are not compared when both have ``powerline=None``.
        """
        bad = []
        no_notch = self._params['powerline'] is None and params['powerline'] is None
        for k in SIGNAL_PATH_PARAMS:
            if no_notch and k in _NOTCH_PARAMS:
                continue
            a, b = self._params[k], params[k]
            if k == 'band':
                a, b = tuple(a), tuple(b)
            if a != b:
                bad.append(f'{k}: baseline {a!r}, detector {b!r}')
        if fs_analysis is not None and not math.isclose(self._fs_analysis, fs_analysis,
                                                        rel_tol=_FS_REL_TOL):
            bad.append(f'fs_analysis: baseline {self._fs_analysis:g} Hz, detector '
                       f'{fs_analysis:g} Hz (it follows from the input rate, target_fs and '
                       "decimation; decimation='exact' gives the same analysis rate for "
                       'different input rates)')
        if bad:
            raise ValueError('the baseline was made with a different signal path: '
                             + '; '.join(bad) + '. The signal-path parameters '
                             f'{list(SIGNAL_PATH_PARAMS)} and the analysis rate must equal '
                             "the baseline's (threshold, min_distance_s, window_s and combine "
                             'are free).')

    def threshold(self, threshold=3.65):
        """The fixed Janca threshold per channel (input unit) for multiplier ``threshold``."""
        threshold = chk.number('threshold', threshold, gt=0)
        return janca_threshold(self._mu, self._sd, threshold)

    def envelope_levels(self, threshold=None):
        """
        The baseline as envelope levels in the **input unit** (for intuition and plots).

        Returns a dict of per-channel arrays: ``'median'`` = ``exp(mu)`` (median of the
        log-normal model; for a Gaussian background whose band-passed SD is ``s`` this is
        about ``1.06 * s``), ``'mode'`` = ``exp(mu - sd**2)``, and with ``threshold`` given
        ``'threshold'`` = ``threshold * (mode + median)``, the detection level. These are
        levels of the Hilbert envelope of the band-passed signal, not of the raw signal.
        """
        out = {'median': np.exp(self._mu), 'mode': np.exp(self._mu - self._sd ** 2)}
        if threshold is not None:
            out['threshold'] = self.threshold(threshold)
        return out

    def select(self, channels):
        """
        A baseline with only ``channels`` (indices, or names if ``channel_names`` is set), in
        that order, e.g. to match the channel order of a detection montage.
        """
        if isinstance(channels, (str, int, np.integer)):
            channels = [channels]
        idx = []
        for ch in channels:
            if isinstance(ch, str):
                if self._channel_names is None or ch not in self._channel_names:
                    raise KeyError(f'no channel named {ch!r} (channel_names: '
                                   f'{self._channel_names})')
                idx.append(self._channel_names.index(ch))
            else:
                k = chk.integer('channel', ch, ge=-self.n_channels, le=self.n_channels - 1)
                idx.append(k % self.n_channels)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            raise ValueError('select() needs at least one channel')
        return type(self)(
            self._mu[idx], self._sd[idx], fs_analysis=self._fs_analysis, fs=self._fs,
            preset=self._preset, **self._params,
            valid_s=None if self._valid_s is None else self._valid_s[idx],
            channel_names=(None if self._channel_names is None
                           else [self._channel_names[k] for k in idx]),
            units=self._units, info={**self._info, 'selected_from': idx.tolist()})

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        """JSON-serialisable representation (the content of :meth:`save`)."""
        return {
            'format': BASELINE_SCHEMA, 'schema_version': BASELINE_SCHEMA_VERSION,
            'n_channels': self.n_channels, 'mu': self._mu.tolist(), 'sd': self._sd.tolist(),
            'fs_analysis': self._fs_analysis, 'fs': self._fs, 'preset': self._preset,
            'params': _jsonable_params(self._params),
            'valid_s': None if self._valid_s is None else self._valid_s.tolist(),
            'channel_names': None if self._channel_names is None else list(self._channel_names),
            'units': self._units, 'info': dict(self._info),
        }

    @classmethod
    def from_dict(cls, d):
        """Inverse of :meth:`to_dict` (schema checked)."""
        if not isinstance(d, dict) or d.get('format') != BASELINE_SCHEMA:
            raise ValueError(f'not a JancaBaseline (format must be {BASELINE_SCHEMA!r})')
        ver = d.get('schema_version')
        if not isinstance(ver, int) or isinstance(ver, bool) or ver < 1:
            raise ValueError(f'invalid schema_version {ver!r}')
        if ver > BASELINE_SCHEMA_VERSION:
            raise ValueError(f'schema_version {ver} is newer than this brainmaze_eeg '
                             f'supports ({BASELINE_SCHEMA_VERSION}); upgrade brainmaze_eeg')
        params = dict(d['params'])
        unknown = set(params) - set(SIGNAL_PATH_PARAMS)
        missing = set(SIGNAL_PATH_PARAMS) - set(params)
        if unknown or missing:
            raise ValueError(f'params: unknown {sorted(unknown)}, missing {sorted(missing)}')
        params['band'] = tuple(params['band'])
        b = cls(d['mu'], d['sd'], fs_analysis=d['fs_analysis'], fs=d.get('fs'),
                preset=d['preset'], **params, valid_s=d.get('valid_s'),
                channel_names=d.get('channel_names'), units=d.get('units'),
                info=d.get('info'))
        if b.n_channels != d.get('n_channels') or len(d['mu']) != len(d['sd']):
            raise ValueError(f"n_channels {d.get('n_channels')!r} does not match mu/sd "
                             f"({len(d['mu'])}/{len(d['sd'])})")
        return b

    def save(self, path):
        """Write the baseline to ``path`` as JSON (schema :data:`BASELINE_SCHEMA`, version
        :data:`BASELINE_SCHEMA_VERSION`). Floats are written exactly (round trip is lossless)."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=1, allow_nan=False)
            f.write('\n')

    @classmethod
    def load(cls, path):
        """Read a baseline written by :meth:`save`."""
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ misc
    def __eq__(self, other):
        if not isinstance(other, JancaBaseline):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None

    def __repr__(self):
        def arr(a, fmt='{:.4g}'):
            v = [fmt.format(x) for x in a[:6]]
            return '[' + ', '.join(v) + (', ...' if a.size > 6 else '') + ']'
        changed = {k: v for k, v in self._params.items()
                   if v != janca_params(self._preset)[k]}
        parts = [f'n_channels={self.n_channels}', f'mu={arr(self._mu)}', f'sd={arr(self._sd)}',
                 f'fs_analysis={self._fs_analysis:g}', f'preset={self._preset!r}']
        parts += [f'{k}={v!r}' for k, v in changed.items()]
        if self._valid_s is not None:
            parts.append(f"valid_s={arr(self._valid_s, '{:.1f}')}")
        if self._channel_names is not None:
            names = list(self._channel_names[:6]) + (['...'] if self.n_channels > 6 else [])
            parts.append(f'channel_names={names}')
        if self._units is not None:
            parts.append(f'units={self._units!r}')
        return f'JancaBaseline({", ".join(parts)})'


def _p_signal(p):
    """The signal-path subset of resolved detector parameters ``p``."""
    return {k: p[k] for k in SIGNAL_PATH_PARAMS}


def _true_runs(mask):
    """``[start, stop)`` runs of True in a boolean row."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist())


def _segment_mask(segments, n, fs):
    sel = np.zeros(n, dtype=bool)
    try:
        segs = [tuple(s) for s in segments]
    except TypeError:
        raise ValueError('segments must be a sequence of (start_s, stop_s) pairs') from None
    if not segs:
        raise ValueError('segments is empty')
    for k, s in enumerate(segs):
        a, b = chk.pair(f'segments[{k}]', s)
        if a < 0 or b > n / fs + 1e-9:
            raise ValueError(f'segments[{k}] = {s} lies outside the record (0 .. {n / fs:g} s)')
        sel[int(round(a * fs)):int(round(b * fs))] = True
    return sel
