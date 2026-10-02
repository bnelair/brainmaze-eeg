# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tools for analyzing hypnograms such as number of cycles, hypnogram time etc.

Plotting (:func:`plot_hypnogram`, ``score_night(..., plot=True)``) needs matplotlib, an
optional dependency: ``pip install "brainmaze-eeg[plot]"``. Everything else in this
module works without it.

**Input conventions** (since 2.0.1):

A hypnogram is a :class:`pandas.DataFrame` with one row per scored epoch and the columns
``annotation`` (sleep-state label), ``start`` and ``end``, optionally ``duration``.

* ``start``/``end`` are either **timezone-aware** datetimes (``datetime.datetime``,
  :class:`pandas.Timestamp` or a ``datetime64[ns, tz]`` column) or **numeric POSIX
  timestamps in seconds**. Naive datetimes raise ``TypeError`` (elapsed time across a DST
  change would be wrong; localize them first). Numeric values whose magnitude exceeds
  ``1e11`` (the year 5138 in seconds) raise ``ValueError``, because they are almost
  certainly milliseconds or finer.
* ``duration``, if present, must equal ``end - start`` in seconds (to 1 ms), otherwise
  ``ValueError`` is raised. This also catches millisecond timestamps with durations in
  seconds. Where a function needs durations it uses ``end - start``.
* Rows may come in any order; functions sort by ``start`` (stable) and never modify the
  caller's frame.
* Arousals are events, not states: rows labelled ``'Arousal'`` (the label written by
  :mod:`brainmaze_utils`) or ``'Arrousal'`` (the legacy spelling of older data) are removed
  before scoring.
* The scoring functions (:func:`get_fell_asleep_time`, :func:`get_awakening_time`,
  :func:`get_rem_latency`, :func:`score_night`) need a valid hypnogram: after arousals are
  removed, epochs must not overlap (``ValueError`` otherwise). They first merge touching
  epochs with the same label into **bouts** (one row per continuous period), so a hypnogram
  tiled into 30-s epochs and the same hypnogram with merged rows give the same results.
"""


import numpy as np
import pandas as pd
import datetime
from copy import deepcopy
from tqdm import tqdm

# merge_annotations and filter_by_key are no longer used here; they stay importable from
# this module for backward compatibility.
from brainmaze_utils.annotations import merge_annotations, filter_by_key, create_day_indexes  # noqa: F401


AROUSAL_TAGS = ('Arousal', 'Arrousal')
"""Labels treated as arousal events: ``'Arousal'`` (brainmaze-utils) and the legacy
spelling ``'Arrousal'``."""

_NUMERIC_TIMESTAMP_LIMIT = 1e11  # seconds; larger values are taken for ms/us/ns
_DURATION_TOL = 1e-3  # s, tolerance of the duration == end - start check
_TOL = 1e-6  # s, tolerance for "touching" epochs and time comparisons
_EPOCH_UTC = pd.Timestamp(0, tz='UTC')


def _pyplot():
    """Import matplotlib.pyplot on first use (optional dependency, extra ``[plot]``)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting in brainmaze_eeg.hypnogram needs matplotlib, which is an optional "
            "dependency. Install it with: pip install \"brainmaze-eeg[plot]\" "
            "(or pip install matplotlib)."
        ) from e
    return plt


def _delta_seconds(delta):
    """Length of a time difference in seconds (timedelta, or a difference of numeric
    timestamps in seconds). Unlike ``timedelta.seconds`` this keeps the days component
    and the sign."""
    if isinstance(delta, (int, float, np.integer, np.floating)):
        return float(delta)
    return pd.Timedelta(delta).total_seconds()


def _is_real_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _to_seconds(values, name, where):
    """Convert a time column to float POSIX seconds.

    Returns ``(seconds, kind)`` with ``kind`` ``'datetime'`` (timezone-aware datetimes) or
    ``'numeric'`` (numbers, taken as seconds). Raises on naive datetimes, mixed types,
    NaN/NaT, and numbers that look like milliseconds or finer.
    """
    s = pd.Series(values).reset_index(drop=True)
    dtype = s.dtype
    if len(s) == 0:
        kind = 'numeric' if pd.api.types.is_numeric_dtype(dtype) else 'datetime'
        return np.zeros(0), kind
    if pd.api.types.is_bool_dtype(dtype):
        raise TypeError(f'{where}: "{name}" holds booleans, not times.')
    if isinstance(dtype, pd.DatetimeTZDtype):
        sec = ((s.dt.tz_convert('UTC') - _EPOCH_UTC) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
        kind = 'datetime'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        raise TypeError(f'{where}: "{name}" holds timezone-naive datetimes. Localize them first '
                        '(e.g. df[col].dt.tz_localize(...)) or pass POSIX timestamps in seconds.')
    elif pd.api.types.is_numeric_dtype(dtype):
        sec = s.to_numpy(dtype=float)
        kind = 'numeric'
    else:
        vals = list(s)
        if all(isinstance(v, datetime.datetime) for v in vals):
            if any(v.tzinfo is None or v.utcoffset() is None for v in vals):
                raise TypeError(f'{where}: "{name}" holds timezone-naive datetimes. Localize them '
                                'first or pass POSIX timestamps in seconds.')
            sec = np.array([v.timestamp() for v in vals], dtype=float)
            kind = 'datetime'
        elif all(_is_real_number(v) for v in vals):
            sec = np.array(vals, dtype=float)
            kind = 'numeric'
        else:
            raise TypeError(f'{where}: "{name}" must hold timezone-aware datetimes or numeric POSIX '
                            'timestamps in seconds, not a mixture or other types.')
    if not np.all(np.isfinite(sec)):
        raise ValueError(f'{where}: "{name}" contains missing (NaN/NaT) or infinite values.')
    if kind == 'numeric' and np.max(np.abs(sec)) > _NUMERIC_TIMESTAMP_LIMIT:
        raise ValueError(f'{where}: numeric "{name}" values exceed {_NUMERIC_TIMESTAMP_LIMIT:.0e}; '
                         'they look like milliseconds (or finer). Timestamps must be in seconds.')
    return sec, kind


def _hypnogram(df, where, drop_arousals=True, allow_overlap=False):
    """Validated, sorted copy of a hypnogram with float-second columns ``_s``/``_e`` and a
    ``duration`` column in seconds (see the module docstring)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'{where}: expected a pandas DataFrame, got {type(df).__name__}.')
    missing = [c for c in ('annotation', 'start', 'end') if c not in df.columns]
    if missing:
        raise ValueError(f'{where}: the hypnogram has no column(s) {missing}.')
    h = df.copy()
    if drop_arousals:
        h = h.loc[~h['annotation'].isin(AROUSAL_TAGS)]
    h = h.reset_index(drop=True)
    s, kind_s = _to_seconds(h['start'], 'start', where)
    e, kind_e = _to_seconds(h['end'], 'end', where)
    if len(h) and kind_s != kind_e:
        raise TypeError(f'{where}: "start" is {kind_s} but "end" is {kind_e}.')
    dur = e - s
    if np.any(dur < -_TOL):
        i = int(np.flatnonzero(dur < -_TOL)[0])
        raise ValueError(f'{where}: an epoch ends before it starts (start={h["start"].iloc[i]}).')
    if 'duration' in h.columns and len(h):
        d = pd.to_numeric(h['duration'], errors='coerce').to_numpy(dtype=float)
        bad = ~(np.abs(d - dur) <= _DURATION_TOL)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f'{where}: "duration" does not equal end - start in seconds for {int(bad.sum())} '
                f'epoch(s) (first: duration={d[i]!r}, end - start={dur[i]!r} s). Timestamps and '
                'durations must both be in seconds.')
    h['duration'] = dur
    order = np.argsort(s, kind='mergesort')
    h = h.iloc[order].reset_index(drop=True)
    h['_s'] = s[order]
    h['_e'] = e[order]
    if not allow_overlap and len(h) > 1:
        prev_end = np.maximum.accumulate(h['_e'].to_numpy())[:-1]
        ov = np.flatnonzero(h['_s'].to_numpy()[1:] < prev_end - _TOL)
        if ov.size:
            i = int(ov[0]) + 1
            raise ValueError(
                f'{where}: epochs overlap (the {h["annotation"].iloc[i]!r} epoch starting at '
                f'{h["start"].iloc[i]} begins before an earlier epoch ends). A hypnogram must have '
                f'one state at a time; arousals {AROUSAL_TAGS} are removed before this check.')
    return h


def _merge_bouts(h):
    """Merge touching epochs with the same label of a validated, non-overlapping hypnogram
    (from :func:`_hypnogram`) into bouts. Keeps the original ``start``/``end`` values."""
    if len(h) == 0:
        return h
    ann = h['annotation'].to_numpy()
    s = h['_s'].to_numpy()
    e = h['_e'].to_numpy()
    new = np.ones(len(h), dtype=bool)
    new[1:] = (ann[1:] != ann[:-1]) | (s[1:] - e[:-1] > _TOL)
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(h) - 1]
    b = h.iloc[first].reset_index(drop=True)
    b['end'] = h['end'].iloc[last].reset_index(drop=True)
    b['_e'] = e[last]
    b['duration'] = b['_e'] - b['_s']
    return b


def _bouts(df, where):
    return _merge_bouts(_hypnogram(df, where))


def _tag_list(tags):
    return [tags] if isinstance(tags, str) else list(tags)


def _fell_asleep_index(b, t_sleep_check, t_awake_threshold, awake_tag, sleep_cycle_tags, where):
    if len(b) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    ann = b['annotation'].to_numpy()
    s = b['_s'].to_numpy()
    dur = b['duration'].to_numpy()
    sleep = np.isin(ann, _tag_list(sleep_cycle_tags))
    awake = ann == awake_tag
    candidates = ([0] if sleep[0] else []) + list(np.flatnonzero(awake[:-1] & sleep[1:]) + 1)
    window = t_sleep_check * 60.0
    limit = t_awake_threshold * 60.0
    for k in candidates:
        in_window = (s >= s[k]) & (s < s[k] + window)
        if dur[in_window & awake].sum() < limit:
            return int(k)
    raise ValueError(
        f'{where}: no sleep onset found: no transition into {_tag_list(sleep_cycle_tags)} is '
        f'followed by less than {t_awake_threshold} min of {awake_tag!r} within {t_sleep_check} min.')


def _awakening(b, t_awake_threshold, t_sleep_threshold, awake_tag, sleep_cycle_tags, after_s, where):
    """Returns ``(time, seconds)`` of the awakening."""
    if len(b) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    ann = b['annotation'].to_numpy()
    s = b['_s'].to_numpy()
    dur = b['duration'].to_numpy()
    sleep_tags = _tag_list(sleep_cycle_tags)
    sleep = np.isin(ann, sleep_tags)
    asleep = np.isin(ann, list(dict.fromkeys(['N1', 'N2', 'N3', 'REM'] + sleep_tags)))
    awake = ann == awake_tag
    candidates = np.flatnonzero(sleep[:-1] & awake[1:]) + 1
    if after_s is not None:
        candidates = candidates[s[candidates] >= after_s - _TOL]
    window = t_awake_threshold * 60.0
    for k in candidates:
        in_window = (s >= s[k]) & (s < s[k] + window)
        if dur[in_window & awake].sum() >= window - _TOL and dur[in_window & asleep].sum() <= t_sleep_threshold * 60.0 + _TOL:
            return b['start'].iloc[k], s[k]
    if awake[-1]:
        if candidates.size == 0:
            raise ValueError(f'{where}: no awakening found: the hypnogram ends {awake_tag!r} but has no '
                             f'transition from {sleep_tags} to {awake_tag!r}'
                             + ('' if after_s is None else ' after the given time') + '.')
        k = candidates[-1]
        return b['start'].iloc[k], s[k]
    return b['end'].iloc[-1], b['_e'].iloc[-1]


def get_hypnogram_datarate(df):
    """
    Calculate the data rate of a hypnogram.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start`` and ``end`` columns (optionally
        ``duration``); see the module docstring for the accepted time formats.

    Returns
    -------
    float
        Scored time (sum of ``end - start`` of all non-arousal epochs) divided by the time
        span from the earliest ``start`` to the latest ``end``. 1.0 means no gaps.

    Raises
    ------
    ValueError
        If the hypnogram is empty, spans zero time, or epochs overlap.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       The span used ``timedelta.seconds``, which drops whole days: a fully scored 25 h
       hypnogram returned 25.0 instead of 1.0. Numeric timestamps (seconds) now work (they
       raised ``AttributeError``), arousal rows are no longer counted as scored time, the
       input may be unsorted, and the input is validated (module docstring).
    """
    where = 'get_hypnogram_datarate'
    h = _hypnogram(df, where)
    if len(h) == 0:
        raise ValueError(f'{where}: the hypnogram is empty.')
    span = h['_e'].max() - h['_s'].min()
    if span <= 0:
        raise ValueError(f'{where}: the hypnogram spans zero time.')
    return float(h['duration'].sum() / span)


def get_fell_asleep_time(df, t_sleep_check=60, t_awake_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N1', 'N2', 'N3']):
    """
    Determine when the subject fell asleep based on hypnogram data.

    Touching epochs with the same label are merged into bouts first (module docstring).
    Candidates are the first bout, if it is a sleep stage, and every sleep-stage bout that
    directly follows an ``awake_tag`` bout, in time order. The first candidate for which
    the ``awake_tag`` bouts starting within ``[candidate start, candidate start +
    t_sleep_check)`` last less than ``t_awake_threshold`` in total is the sleep onset. A
    bout that starts inside the window counts in full.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    t_sleep_check : int, optional
        Time window in minutes to check for sustained sleep. Default is 60.
    t_awake_threshold : int, optional
        Awake time in minutes within the window must be below this. Default is 10.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.
    sleep_cycle_tags : list, optional
        List of tags indicating sleep states. Default is ['REM', 'N1', 'N2', 'N3'].

    Returns
    -------
    datetime or float
        Start of the sleep-onset bout, in the type of the ``start`` column.

    Raises
    ------
    ValueError
        If no candidate qualifies, the hypnogram is empty, or epochs overlap.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       If no candidate qualified, v1.0.0 returned the start of the recording, which looks
       like a valid onset; it now raises ``ValueError``. Rows labelled ``'Arousal'`` are
       removed like ``'Arrousal'`` (only the latter was, so an ``'Arousal'`` epoch between
       AWAKE and sleep hid the onset). Epochs are sorted and merged into bouts, so tiled and
       merged hypnograms agree. ``awake_tag`` is compared for equality (it was a substring
       test). Numeric timestamps (seconds) work.
    """
    where = 'get_fell_asleep_time'
    b = _bouts(df, where)
    k = _fell_asleep_index(b, t_sleep_check, t_awake_threshold, awake_tag, sleep_cycle_tags, where)
    return b['start'].iloc[k]


def get_awakening_time(df, t_awake_threshold=90, t_sleep_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N2', 'N3'], after=None):
    """
    Determine when the subject woke up based on hypnogram data.

    Touching epochs with the same label are merged into bouts first (module docstring).
    Candidates are the ``awake_tag`` bouts that directly follow a ``sleep_cycle_tags``
    bout (and start at or after ``after``, if given), in time order. Of the bouts that
    start within ``[candidate start, candidate start + t_awake_threshold)`` (counted in
    full), the first candidate whose ``awake_tag`` bouts last at least
    ``t_awake_threshold`` and whose sleep bouts (N1, N2, N3, REM and ``sleep_cycle_tags``)
    last at most ``t_sleep_threshold`` is the awakening.

    If no candidate qualifies: when the hypnogram ends with ``awake_tag`` the last
    candidate is returned (``ValueError`` if there is none); otherwise the end of the
    recording is returned (the subject was still asleep when it ended).

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    t_awake_threshold : int, optional
        Minimum sustained awake time in minutes to be considered awakened. Default is 90.
    t_sleep_threshold : int, optional
        Maximum sleep time in minutes allowed in the awakening window. Default is 10.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.
    sleep_cycle_tags : list, optional
        List of tags indicating sleep states. Default is ['REM', 'N2', 'N3'].
    after : datetime or float, optional
        Only consider awakenings starting at or after this time (same type as ``start``).
        :func:`score_night` passes the sleep onset. New in 2.0.1.

    Returns
    -------
    datetime or float
        Time of the awakening, in the type of the ``start`` column.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       ``t_sleep_threshold`` (minutes) was compared with the sleep time in seconds, so
       10 meant 10 s: one minute of N2 in the window rejected an awakening. A first epoch
       in a sleep stage is no longer a candidate (it returned the start of that sleep
       epoch), ``'Arousal'`` rows are removed like ``'Arrousal'``, epochs are sorted and
       merged into bouts, and ``ValueError`` replaces an ``IndexError`` when no
       transition exists. Numeric timestamps (seconds) work.
    """
    where = 'get_awakening_time'
    h = _hypnogram(df, where)
    after_s = None
    if after is not None:
        after_s = _to_seconds([after], 'after', where)[0][0]
    return _awakening(_merge_bouts(h), t_awake_threshold, t_sleep_threshold, awake_tag,
                      sleep_cycle_tags, after_s, where)[0]


def is_sleep_complete(df, awake_tag='AWAKE'):
    """
    Check if the sleep session is complete (starts and ends with awake state).

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.

    Returns
    -------
    bool
        True if the earliest and the latest epoch (arousals excluded) are ``awake_tag``.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Uses the earliest and latest epoch by ``start`` (it used the first and last rows),
       ignores arousal rows, and raises ``ValueError`` on an empty hypnogram.
    """
    where = 'is_sleep_complete'
    h = _hypnogram(df, where)
    if len(h) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    return bool(h['annotation'].iloc[0] == awake_tag == h['annotation'].iloc[-1])


def _rem_latency(b, k_onset, rem_tag, awake_tag, where):
    ann = b['annotation'].to_numpy()
    rem = np.flatnonzero(ann == rem_tag)
    if rem.size == 0:
        raise ValueError(f'{where}: the hypnogram has no {rem_tag!r} epoch.')
    k_rem = rem[0]
    first_rem_start = b['start'].iloc[k_rem]
    before = np.flatnonzero((ann == awake_tag) & (b['_e'].to_numpy() <= b['_s'].iloc[k_rem] + _TOL))
    last_awake_end = b['end'].iloc[before[-1]] if before.size else b['start'].iloc[k_onset]
    return {'last_awake': first_rem_start - last_awake_end,
            'fall_asleep': first_rem_start - b['start'].iloc[k_onset]}


def get_rem_latency(df, rem_tag='REM', awake_tag='AWAKE'):
    """
    Calculate REM sleep latency (time from falling asleep or last awake to first REM).

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    rem_tag : str, optional
        Tag for REM sleep state. Default is 'REM'.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.

    Returns
    -------
    dict
        ``'fall_asleep'``: start of the first REM epoch minus the sleep onset
        (:func:`get_fell_asleep_time` with its defaults). ``'last_awake'``: start of the
        first REM epoch minus the end of the last ``awake_tag`` epoch before it (the sleep
        onset if there is none). Timedeltas for datetime input, seconds (float) for
        numeric input. A latency is **negative** when the first REM precedes the reference
        (e.g. REM before the sustained sleep onset).

    Raises
    ------
    ValueError
        If there is no ``rem_tag`` epoch or no sleep onset.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Sorted, validated input; arousal rows (both spellings) removed; ``ValueError``
       instead of ``IndexError`` without REM. The values themselves are computed as before.
    """
    where = 'get_rem_latency'
    b = _bouts(df, where)
    k_onset = _fell_asleep_index(b, 60, 10, awake_tag, ['REM', 'N1', 'N2', 'N3'], where)
    return _rem_latency(b, k_onset, rem_tag, awake_tag, where)


def get_number_of_sleep_stages(df, tags='REM', delay=30):
    """
    Count the bouts (periods) of a sleep stage, ignoring bouts that start too soon after
    the previous counted one.

    1. Epochs whose ``annotation`` is in ``tags`` are sorted by ``start``. Epochs that
       touch or overlap in time are merged into one **bout** (several ``tags`` are treated
       as one class: touching epochs of any of them form one bout). A 40-min REM period
       counts as one bout whether it is one row or eighty 30-s epochs.
    2. The first bout is counted. A later bout is counted if its start is at least
       ``delay`` minutes after the **end of the last counted bout** (bouts that were not
       counted do not move this reference).

    Example (delay 30 min): REM 40 min, N2 20 min, REM 40 min gives 1 (the second REM bout
    starts 20 min after the first ends); with N2 30 min instead it gives 2.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``); see the
        module docstring for the accepted time formats. Rows may be unsorted.
    tags : str or list, optional
        Sleep stage tag(s) to count. Default is 'REM'.
    delay : float, optional
        Minimum time in minutes (``>= 0``) between the end of the last counted bout and the
        start of the next counted one. Default is 30. Any value is allowed, including
        ``>= 1440`` (one day).

    Returns
    -------
    int
        Number of counted bouts (0 if no epoch has one of the tags).

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       v1.0.0 always raised under pandas >= 2 (it used ``DataFrame.append``). Its results
       under pandas < 2 differ from 2.0.1 as follows:

       - It worked **per epoch**: the delay was measured from the end of the single epoch
         that started the last counted occurrence, so on a hypnogram tiled into 30-s epochs
         one long period was counted several times (one 40-min REM period: 2; 90 min: 3;
         REM 40 / N2 20 / REM 40 min: 4). Now bouts are counted (1, 1 and 1). On input
         that is already merged into bouts (as ``score_night`` passed it) the result is
         unchanged.
       - With several distinct ``tags`` it required an epoch to equal all of them (none
         did) and then raised ``AttributeError``; the same happened when no epoch matched.
         Now several tags are counted together and no match gives 0.
       - It compared ``timedelta.seconds``, which drops whole days and wraps negatives:
         a gap of 1 day + 1 min counted as 1 min, ``delay >= 1440`` wrapped (``delay=1440``
         acted as 0, so every epoch counted), and unsorted or overlapping rows gave
         arbitrary counts. Now the rows are sorted, overlapping epochs join one bout, and
         gaps and ``delay`` are exact.
       - Numeric timestamps must be in seconds; values that look like milliseconds raise
         ``ValueError`` (they were silently read as seconds). Naive datetimes raise
         ``TypeError``.
    """
    where = 'get_number_of_sleep_stages'
    tags = _tag_list(tags)
    if not delay >= 0:
        raise ValueError(f'{where}: delay must be >= 0 minutes, got {delay!r}.')
    delay_s = float(delay) * 60.0

    h = _hypnogram(df, where, drop_arousals=False, allow_overlap=True)
    sel = h.loc[h['annotation'].isin(tags)]

    bouts = []  # [start_s, end_s]
    for s, e in zip(sel['_s'].to_numpy(), sel['_e'].to_numpy()):
        if bouts and s <= bouts[-1][1] + _TOL:
            bouts[-1][1] = max(bouts[-1][1], e)
        else:
            bouts.append([s, e])

    n_occurrences = 0
    last_end = None
    for s, e in bouts:
        if last_end is None or s - last_end >= delay_s - _TOL:
            n_occurrences += 1
            last_end = e
    return n_occurrences


def _count_awakenings(ann, awake_tag, n1_tag, sleep_tags):
    n_awakenings = 0
    sleep_happened = False
    awake_happened = False
    sleep_tags = set(_tag_list(sleep_tags))
    for a in ann:
        if a in sleep_tags:
            sleep_happened = True
        if a == awake_tag:
            awake_happened = True
        elif a != n1_tag:
            awake_happened = False
        if sleep_happened and awake_happened:
            n_awakenings += 1
            sleep_happened = False
            awake_happened = False
    return n_awakenings


def get_number_of_awakenings(df, awake_tag='AWAKE', n1_tag='N1', sleep_tags=['N2', 'N3', 'REM']):
    """
    Count the number of awakenings after sleep onset.

    The epochs are taken in time order. An awakening is counted at the first ``awake_tag``
    epoch after a ``sleep_tags`` epoch, with only ``awake_tag``/``n1_tag`` epochs in
    between; the next awakening needs another ``sleep_tags`` epoch first.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with an ``annotation`` column. With ``start``/``end`` columns the input
        is validated and sorted by ``start``; without them the row order is used.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.
    n1_tag : str, optional
        Tag for N1 sleep stage. Default is 'N1'.
    sleep_tags : list, optional
        List of tags for sleep states. Default is ['N2', 'N3', 'REM'].

    Returns
    -------
    int
        Number of awakenings.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Sorted by ``start`` when available (it used the row order and label-based
       indexing, which broke on a non-default index) and arousal rows (both spellings)
       are ignored (an arousal epoch between sleep and AWAKE hid the awakening).
    """
    where = 'get_number_of_awakenings'
    if 'start' in df.columns and 'end' in df.columns:
        ann = _hypnogram(df, where)['annotation'].to_numpy()
    else:
        ann = df['annotation'].loc[~df['annotation'].isin(AROUSAL_TAGS)].to_numpy()
    return _count_awakenings(ann, awake_tag, n1_tag, sleep_tags)


def get_time_by_key(df, key):
    """
    Calculate total time for a specific sleep stage or combination of stages.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation' and 'duration' columns.
    key : str or list
        Sleep stage tag(s) to sum durations for.

    Returns
    -------
    float
        Total duration in the unit of the ``duration`` column (seconds by convention).
    """
    if isinstance(key, (list, tuple)):
        value = 0
        for single_key in key:
            value += (df.duration[(df.annotation == single_key)]).sum()
        return value
    else:
        return (df.duration[(df.annotation == key)]).sum()


def get_stage_times(df, keys):
    """
    Get total times for multiple sleep stages.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation' and 'duration' columns.
    keys : list
        List of sleep stage tags.

    Returns
    -------
    list
        List of total durations for each stage in seconds.
    """
    return [get_time_by_key(df, key) for key in keys]


def get_stage_times_dataset(hypnograms:list, keys, verbose=True):
    """
    Get stage times for a dataset of hypnograms.

    Parameters
    ----------
    hypnograms : list
        List of hypnogram dataframes.
    keys : list
        List of sleep stage tags.
    verbose : bool, optional
        If True, show progress bar. Default is True.

    Returns
    -------
    pd.DataFrame
        Dataframe with stage times for each hypnogram.
    """
    if verbose:
        return pd.DataFrame([dict([(state, get_time_by_key(hyp, state)) for state in keys]) for hyp in tqdm(hypnograms)])
    return pd.DataFrame([dict([(state, get_time_by_key(hyp, state)) for state in keys]) for hyp in hypnograms])


def score_night(df, plot=False):
    """
    Compute comprehensive sleep metrics for a night's hypnogram.

    The hypnogram is validated, sorted, arousal rows (``'Arousal'``/``'Arrousal'``) are
    removed and touching epochs with the same label are merged into bouts (module
    docstring). Then, with the defaults of the individual functions:

    - ``fell_asleep_time``: :func:`get_fell_asleep_time`.
    - ``awakening_time``: :func:`get_awakening_time` with ``after=fell_asleep_time``.
    - The **sleep period** is made of the bouts starting at or after ``fell_asleep_time``
      and before ``awakening_time``; the counts and stage times below are taken over it.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``):
        timezone-aware datetimes or numeric POSIX timestamps in seconds.
    plot : bool, optional
        If True, plot the hypnogram with sleep onset and awakening markers. Default is False.
        Needs matplotlib (optional extra ``brainmaze-eeg[plot]``) and datetime input.

    Returns
    -------
    dict
        - sleep_complete: the earliest and latest epochs are AWAKE (:func:`is_sleep_complete`)
        - fell_asleep_time: time of sleep onset (type of ``start``)
        - rem_latency_fell_asleep: first REM start minus sleep onset, in seconds (float;
          negative if REM came first; NaN if there is no REM)
        - rem_latency_last_awake: first REM start minus the end of the last AWAKE before it,
          in seconds (float; NaN if there is no REM); see :func:`get_rem_latency`
        - awakening_time: time of final awakening (type of ``start``)
        - n_complete_sleep_cycles: REM bouts in the sleep period,
          :func:`get_number_of_sleep_stages` with ``tags='REM', delay=30``
        - n_awakenings: :func:`get_number_of_awakenings` over the sleep period
        - n1_sleep_time, n2_sleep_time, n3_sleep_time, rem_sleep_time, awake_sleep_time:
          time in each stage within the sleep period, in seconds

    Raises
    ------
    ValueError
        If the hypnogram is empty, invalid (module docstring), or no sleep onset or
        awakening can be determined.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       v1.0.0 raised with brainmaze-utils 3 (its ``merge_annotations`` accepts only
       numeric timestamps, while the rest of the function needed datetimes). Now datetime
       and numeric input both work. Numbers that v1.0.0 would have got wrong are fixed:
       REM latencies used ``timedelta.seconds`` (a latency of -23 min became 85020 s; now
       negative values stay negative), the awakening ignored ``t_sleep_threshold`` units
       (see :func:`get_awakening_time`) and could be found before the sleep onset (now
       only after it), ``'Arousal'`` rows were not removed, and a night without REM raised
       ``IndexError`` (now the latencies are NaN).
    """
    where = 'score_night'
    h = _hypnogram(df, where)
    if len(h) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    b = _merge_bouts(h)

    k_onset = _fell_asleep_index(b, 60, 10, 'AWAKE', ['REM', 'N1', 'N2', 'N3'], where)
    fell_asleep_time = b['start'].iloc[k_onset]
    fell_s = b['_s'].iloc[k_onset]
    awakening_time, awake_s = _awakening(b, 90, 10, 'AWAKE', ['REM', 'N2', 'N3'], fell_s, where)
    sleep_complete = bool(b['annotation'].iloc[0] == 'AWAKE' == b['annotation'].iloc[-1])

    sleep_df = b.loc[(b['_s'] >= fell_s - _TOL) & (b['_s'] < awake_s - _TOL)].reset_index(drop=True)

    n_complete_sleep_cycles = get_number_of_sleep_stages(sleep_df, tags='REM', delay=30)
    n_awakenings = _count_awakenings(sleep_df['annotation'].to_numpy(), 'AWAKE', 'N1', ['N2', 'N3', 'REM'])
    if (b['annotation'] == 'REM').any():
        rem_latency = _rem_latency(b, k_onset, 'REM', 'AWAKE', where)
        rem_latency = {k: _delta_seconds(v) for k, v in rem_latency.items()}
    else:
        rem_latency = {'fall_asleep': float('nan'), 'last_awake': float('nan')}

    n1_sleep_time = float(get_time_by_key(sleep_df, 'N1'))
    n2_sleep_time = float(get_time_by_key(sleep_df, 'N2'))
    n3_sleep_time = float(get_time_by_key(sleep_df, 'N3'))
    rem_sleep_time = float(get_time_by_key(sleep_df, 'REM'))
    awake_sleep_time = float(get_time_by_key(sleep_df, 'AWAKE'))

    if plot == True:
        plot_hypnogram(df)
        plt = _pyplot()
        plt.stem([fell_asleep_time, awakening_time], [7, 7], linefmt='r', markerfmt='or', basefmt='r')

    return {
        'sleep_complete': sleep_complete,
        'fell_asleep_time': fell_asleep_time,
        'rem_latency_fell_asleep': rem_latency['fall_asleep'],
        'rem_latency_last_awake': rem_latency['last_awake'],
        'awakening_time': awakening_time,
        'n_complete_sleep_cycles': n_complete_sleep_cycles,
        'n_awakenings': n_awakenings,
        'n1_sleep_time': n1_sleep_time,
        'n2_sleep_time': n2_sleep_time,
        'n3_sleep_time': n3_sleep_time,
        'rem_sleep_time': rem_sleep_time,
        'awake_sleep_time': awake_sleep_time
    }


def _format_hms(seconds):
    """``[-]HH:MM:SS`` of a number of seconds (keeps the sign, hours may exceed 24)."""
    if seconds is None or not np.isfinite(seconds):
        return 'n/a'
    sign = '-' if seconds < 0 else ''
    hours, remainder = divmod(int(round(abs(seconds))), 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{sign}{hours:02}:{minutes:02}:{secs:02}'


def _format_clock(t):
    return t.strftime('%H:%M:%S') if hasattr(t, 'strftime') else f'{t} s'


def print_sleep_score(score):
    """
    Print a formatted summary of sleep metrics.

    Parameters
    ----------
    score : dict
        Sleep score dictionary from score_night function.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Works with the floats that :func:`score_night` returns (it called ``.seconds`` on
       them). Durations keep whole days and negative latencies keep their sign. The
       relative times are fractions of the sleep period (sleep onset to awakening), which
       is now labelled as such (it was printed as "Total Sleep Time").
    """
    sleep_period = _delta_seconds(score['awakening_time'] - score['fell_asleep_time'])

    def rel(value):
        return value / sleep_period if sleep_period > 0 else float('nan')

    non_REM = score['n3_sleep_time'] + score['n2_sleep_time'] + score['n1_sleep_time']

    print('Sleep Complete: ', score['sleep_complete'])
    print('Falling asleep: ', _format_clock(score['fell_asleep_time']))
    print('Awakening: ', _format_clock(score['awakening_time']))
    print('Sleep period (onset to awakening): ' + _format_hms(sleep_period))
    print('Rem latency - last_awake: ' + _format_hms(score['rem_latency_last_awake']))
    print('Rem latency - fall_asleep: ' + _format_hms(score['rem_latency_fell_asleep']))

    print('Number of hypnogram cycles: ', score['n_complete_sleep_cycles'])
    print('Number of awakenings', score['n_awakenings'])
    for name, value in [('non-REM', non_REM), ('REM', score['rem_sleep_time']),
                        ('awake', score['awake_sleep_time']), ('N1', score['n1_sleep_time']),
                        ('N2', score['n2_sleep_time']), ('N3', score['n3_sleep_time'])]:
        print()
        print('Sleep-time ' + name)
        print('Absolute: {0}  Relative: {1:0.3f}'.format(int(value), rel(value)))


def get_transition_counts(hyp, states=['AWAKE', 'N1', 'N2', 'N3', 'REM']):
    """
    Count transitions between sleep stages.

    Consecutive rows (in ``start`` order) form a transition; a row whose label is not in
    ``states`` breaks the chain (no transition is counted into or out of it). Time gaps
    between rows are not considered. On a hypnogram tiled into epochs the diagonal holds
    the epoch-to-epoch persistence; on merged bouts it is (nearly) zero.

    Parameters
    ----------
    hyp : pd.DataFrame
        Hypnogram with an ``annotation`` column. With ``start``/``end`` columns it is
        validated and sorted by ``start``; without them the row order is used.
    states : list, optional
        List of sleep states to include. Default is ['AWAKE', 'N1', 'N2', 'N3', 'REM'].

    Returns
    -------
    np.ndarray
        Matrix of transition counts where [i,j] is count of transitions from state i to state j.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Rows are sorted by ``start`` (it used label-based indexing on the row order, which
       broke on a non-default index), and arousal rows (``'Arousal'``/``'Arrousal'``) are
       ignored unless listed in ``states`` (an arousal between two epochs hid their
       transition).
    """
    where = 'get_transition_counts'
    states = np.array(states)
    drop_arousals = not np.isin(AROUSAL_TAGS, states).any()
    if 'start' in hyp.columns and 'end' in hyp.columns:
        ann = _hypnogram(hyp, where, drop_arousals=drop_arousals)['annotation'].to_numpy()
    else:
        ann = hyp['annotation']
        if drop_arousals:
            ann = ann.loc[~ann.isin(AROUSAL_TAGS)]
        ann = ann.to_numpy()
    index = {s: i for i, s in enumerate(states)}
    matrix = np.zeros((states.__len__(), states.__len__()))
    for s1, s2 in zip(ann[:-1], ann[1:]):
        if s1 in index and s2 in index:
            matrix[index[s1], index[s2]] += 1
    return matrix


def get_transition_matrix(hyp, states=['AWAKE', 'N1', 'N2', 'N3', 'REM']):
    """
    Calculate transition probability matrix between sleep stages.

    Parameters
    ----------
    hyp : pd.DataFrame
        Hypnogram dataframe with 'annotation' column.
    states : list, optional
        List of sleep states to include. Default is ['AWAKE', 'N1', 'N2', 'N3', 'REM'].

    Returns
    -------
    np.ndarray
        Normalized transition probability matrix.
    """
    m = get_transition_counts(hyp, states)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = m / m.sum(axis=1).reshape(-1, 1)
    m[np.isnan(m)] = 0
    return m


def get_transition_matrix_dataset(hypnograms, states=['AWAKE', 'N1', 'N2', 'N3', 'REM']):
    """
    Calculate mean and std of transition matrices across a dataset of hypnograms.

    Parameters
    ----------
    hypnograms : list
        List of hypnogram dataframes.
    states : list, optional
        List of sleep states to include. Default is ['AWAKE', 'N1', 'N2', 'N3', 'REM'].

    Returns
    -------
    tuple
        Mean transition matrix and standard deviation matrix (rows of states that never
        occur in a hypnogram are left out of the mean of that row).
    """
    ms = []
    for hyp in hypnograms:
        m = get_transition_counts(hyp, states)
        with np.errstate(invalid='ignore', divide='ignore'):
            m = m / m.sum(axis=1).reshape(-1, 1)
        ms += [m]
    ms = np.array(ms)
    return np.nanmean(ms, axis=0), np.nanstd(ms, axis=0)


def transition_matrix_to_change_matrix(m):
    """
    Convert transition matrix to change matrix (excluding diagonal/same-state transitions).

    Parameters
    ----------
    m : np.ndarray
        Transition probability matrix.

    Returns
    -------
    np.ndarray
        Change matrix with diagonal set to zero and rows renormalized.
    """
    m = deepcopy(m)
    np.fill_diagonal(m, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = m / m.sum(axis=1).reshape(-1, 1)
    m[np.isnan(m)] = 0
    return m


def valid_dataset_index_by_duration(hypnograms:list, filt_dict:dict):
    """
    Filter hypnograms by minimum duration requirements for specific stages.

    Parameters
    ----------
    hypnograms : list
        List of hypnogram dataframes.
    filt_dict : dict
        Dictionary mapping stage names to minimum total duration (unit of the ``duration``
        column, seconds by convention). A stage that does not occur has duration 0.

    Returns
    -------
    list
        Indices of hypnograms in which every listed stage lasts at least its threshold.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       v1.0.0 compared the total duration of the **whole** hypnogram (all stages) with
       each stage's threshold, so e.g. ``{'REM': 3600}`` accepted any night longer than
       1 h that contained any REM.
    """
    return [idx for idx, hyp in enumerate(hypnograms)
            if all(get_time_by_key(hyp, k) >= v for k, v in filt_dict.items())]


def _correction_input(df, where):
    return _hypnogram(df, where, drop_arousals=False, allow_overlap=True)


def _correction_output(rows, h):
    if not rows:
        return h.drop(columns=['_s', '_e'])
    return pd.DataFrame(rows).drop(columns=['_s', '_e']).reset_index(drop=True)




def do_median_filtration(df):
    """
    Apply median filtering to remove isolated 30-second stage annotations.

    An epoch of 30 s whose neighbours (in ``start`` order, after earlier replacements)
    have the same label takes that label.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation' and 'start'/'end' (or 'duration') columns.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the hypnogram. The input is not modified.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       It assigned through chained indexing (``df.iloc[k]['annotation'] = ...``), which
       does not write into the frame, so it returned the input unchanged. It now returns a
       filtered copy, in ``start`` order.
    """
    where = 'do_median_filtration'
    if 'start' in df.columns and 'end' in df.columns:
        h = _correction_input(df, where).drop(columns=['_s', '_e'])
    else:
        h = df.copy().reset_index(drop=True)
    ann = h['annotation'].to_numpy(dtype=object).copy()
    dur = h['duration'].to_numpy(dtype=float)
    for k in range(1, len(h) - 1):
        if ann[k - 1] == ann[k + 1] and abs(dur[k] - 30) <= _DURATION_TOL:
            ann[k] = ann[k - 1]
    h['annotation'] = ann
    return h


def fill_same_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    """
    Fill temporal gaps between consecutive same-state annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    time_threshold : int, optional
        Maximum gap duration in seconds to fill. Default is 300 (5 minutes).
    initial_state : str, optional
        Initial assumed state. Default is 'AWAKE'.

    Returns
    -------
    pd.DataFrame
        Hypnogram with filled gaps.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
    """
    h = _correction_input(df, 'fill_same_voids')
    rows = []
    current_state = initial_state
    last_end, last_end_s = (h['start'].iloc[0], h['_s'].iloc[0]) if len(h) else (None, None)
    for idx in range(len(h)):
        crow = h.iloc[idx].copy()
        void = crow['_s'] - last_end_s
        if 0 < void <= time_threshold and current_state == crow['annotation']:
            crow['start'] = last_end
            crow['_s'] = last_end_s
            crow['duration'] = crow['_e'] - crow['_s']
        current_state = crow['annotation']
        last_end, last_end_s = h['end'].iloc[idx], h['_e'].iloc[idx]
        rows.append(crow)
    return _correction_output(rows, h)


def _fill_voids_with(df, where, time_threshold, initial_state, condition, label):
    h = _correction_input(df, where)
    rows = []
    current_state = initial_state
    last_end, last_end_s = (h['start'].iloc[0], h['_s'].iloc[0]) if len(h) else (None, None)
    for idx in range(len(h)):
        crow = h.iloc[idx].copy()
        void = crow['_s'] - last_end_s
        if 0 < void <= time_threshold and condition(current_state, crow['annotation']):
            vrow = crow.copy()
            vrow['annotation'] = label
            vrow['start'], vrow['_s'] = last_end, last_end_s
            vrow['end'], vrow['_e'] = crow['start'], crow['_s']
            vrow['duration'] = void
            rows.append(vrow)
        current_state = crow['annotation']
        last_end, last_end_s = h['end'].iloc[idx], h['_e'].iloc[idx]
        rows.append(crow)
    return _correction_output(rows, h)




def fill_wakerem_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    """
    Fill gaps between AWAKE and REM states with AWAKE annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    time_threshold : int, optional
        Maximum gap duration in seconds to fill. Default is 300 (5 minutes).
    initial_state : str, optional
        Initial assumed state. Default is 'AWAKE'.

    Returns
    -------
    pd.DataFrame
        Hypnogram with filled wake-REM gaps.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
       The inserted epoch's ``duration`` was the length of the following epoch instead of
       the gap.
    """
    return _fill_voids_with(df, 'fill_wakerem_voids', time_threshold, initial_state,
                            lambda cur, new: cur == 'AWAKE' and new == 'REM', 'AWAKE')


def fill_nonrem_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    """
    Fill gaps between non-REM sleep stages (N2, N3) with 'N' annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    time_threshold : int, optional
        Maximum gap duration in seconds to fill. Default is 300 (5 minutes).
    initial_state : str, optional
        Initial assumed state. Default is 'AWAKE'.

    Returns
    -------
    pd.DataFrame
        Hypnogram with filled non-REM gaps.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
       The inserted epoch's ``duration`` was the length of the following epoch instead of
       the gap.
    """
    nrem = ('N2', 'N3', 'N')
    return _fill_voids_with(df, 'fill_nonrem_voids', time_threshold, initial_state,
                            lambda cur, new: cur in nrem and new in nrem, 'N')


def fill_sleep_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    """
    Fill gaps between any sleep states (non-AWAKE) with 'SLP' annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    time_threshold : int, optional
        Maximum gap duration in seconds to fill. Default is 300 (5 minutes).
    initial_state : str, optional
        Initial assumed state. Default is 'AWAKE'.

    Returns
    -------
    pd.DataFrame
        Hypnogram with filled sleep gaps.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
       The inserted epoch's ``duration`` was the length of the following epoch instead of
       the gap.
    """
    return _fill_voids_with(df, 'fill_sleep_voids', time_threshold, initial_state,
                            lambda cur, new: cur != 'AWAKE' and new != 'AWAKE', 'SLP')


def correct_rem(df, time_threshold=5*60, initial_state='AWAKE'):
    """
    Correct REM annotations that occur shortly after prolonged awake periods.

    A REM epoch that starts at most 60 s after an AWAKE epoch, when the AWAKE epochs right
    before it last at least ``time_threshold`` in total, is relabelled AWAKE.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    time_threshold : int, optional
        Minimum sustained awake duration in seconds. Default is 300 (5 minutes).
    initial_state : str, optional
        Initial assumed state. Default is 'AWAKE'.

    Returns
    -------
    pd.DataFrame
        Hypnogram with corrected REM annotations.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
    """
    h = _correction_input(df, 'correct_rem')
    rows = []
    current_state = initial_state
    awake_duration = time_threshold
    last_end_s = h['_s'].iloc[0] if len(h) else None
    for idx in range(len(h)):
        crow = h.iloc[idx].copy()
        void = crow['_s'] - last_end_s
        if void <= 60 and current_state == 'AWAKE' and crow['annotation'] == 'REM' and awake_duration >= time_threshold:
            crow['annotation'] = 'AWAKE'
        if crow['annotation'] == 'AWAKE':
            awake_duration += crow['duration']
        else:
            awake_duration = 0
        current_state = crow['annotation']
        last_end_s = crow['_e']
        rows.append(crow)
    return _correction_output(rows, h)


def correct_hypnogram(df, time_threshold=60):
    """
    Apply a series of corrections to hypnogram data.

    This function applies multiple correction steps: fills gaps between same states,
    fills wake-REM gaps, fills non-REM gaps, fills sleep gaps, and corrects REM annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    time_threshold : int, optional
        Time threshold in seconds for gap filling. Default is 60.

    Returns
    -------
    pd.DataFrame
        Corrected hypnogram dataframe.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
    """
    new_df = deepcopy(df)
    new_df = fill_same_voids(new_df, time_threshold=time_threshold)
    new_df = fill_wakerem_voids(new_df, time_threshold=time_threshold)
    new_df = fill_nonrem_voids(new_df, time_threshold=time_threshold)
    new_df = fill_sleep_voids(new_df, time_threshold=time_threshold)
    new_df = correct_rem(new_df, time_threshold=time_threshold)
    # new_df = do_median_filtration(new_df)
    return new_df


def _night_spans(t_lo, t_hi, tz, night_start):
    """Night intervals ``[D night_start, D night_start + 12 h]`` (wall clock of ``tz``) for
    every calendar date D from the day before ``t_lo`` to the day of ``t_hi``, clipped to
    ``[t_lo, t_hi]``; empty ones are skipped."""
    spans = []
    d = (t_lo - pd.Timedelta(days=1)).date()
    last = t_hi.date()
    while d <= last:
        base = datetime.datetime(d.year, d.month, d.day)
        a = pd.Timestamp(base + datetime.timedelta(hours=night_start)).tz_localize(
            tz, ambiguous=True, nonexistent='shift_forward')
        z = pd.Timestamp(base + datetime.timedelta(hours=night_start + 12)).tz_localize(
            tz, ambiguous=True, nonexistent='shift_forward')
        a, z = max(a, t_lo), min(z, t_hi)
        if a < z:
            spans.append((a, z))
        d += datetime.timedelta(days=1)
    return spans


def plot_hypnogram(orig_df, hypnogram_values=None, hypnogram_colors=None, fontsize=12, fig=None, night_start=22):
    """
    Creates a Matplotlib figure of the hypnogram from the annotations. Time must be in a time-zone aware format.

    Parameters
    ----------
    orig_df : pd.DataFrame
        Hypnogram with ``annotation`` and timezone-aware ``start``/``end`` columns, in any
        row order (it is sorted by ``start``). Arousal rows (``'Arousal'``, or the legacy
        ``'Arrousal'``) are drawn as events, not as states. The frame is not modified.
    hypnogram_values : dict
        dict of a y-axis values for each hypnogram state
    hypnogram_colors : dict
        dict of color hex codes for each hypnogram state
    fontsize : int
        Fontsize
    fig : figure
        Already existing figure object.
    night_start : int
        Hour at which the night begins. Nights ``[night_start, night_start + 12 h]`` are
        shaded for every calendar date in the plotted range, on the **wall clock of the
        timezone of the first** ``start`` (UTC data is shaded at ``night_start`` UTC: convert
        it to the recording site's timezone first, e.g. ``df.start.dt.tz_convert(...)``).

    Returns
    -------

    Raises
    ------
    ImportError
        If matplotlib is not installed (optional extra ``brainmaze-eeg[plot]``).
    TypeError
        If ``start``/``end`` are not timezone-aware datetimes.
    ValueError
        If the hypnogram has no sleep-state epochs, or a label has no y value or colour.

    Notes
    -----
    .. note:: **Changed in 2.0.1:**
       Works with pandas 2 (it used ``DataFrame.append``, removed in pandas 2, and
       called the ``datetime`` module instead of ``datetime.datetime``, so it always
       raised). It no longer adds ``state_id``/``state_color`` columns to the input
       frame, sorts the rows, and handles a one-epoch hypnogram. Night shading is drawn
       per calendar date over the plotted range (it was anchored to each ``day`` group's
       first epoch, so a recording starting after midnight, or a state lasting over
       several days, lost its night shading); a ``day`` column is no longer used.
       Arousals are recognised as ``'Arousal'`` (the brainmaze-utils label, which was drawn
       as a sleep state) and ``'Arrousal'`` (which raised ``KeyError`` with the default
       maps). Unknown labels raise ``ValueError`` naming them.
    """
    plt = _pyplot()
    import matplotlib.dates as mdates

    _hypnogram_values = {
        'AWAKE': 6,
        'Arousal': 5,
        'SLP': 4.5,
        'REM': 4,
        'N1': 3,
        'N2': 2,
        'N3': 1,
    }

    _hypnogram_colors = {
        'AWAKE': '#e7b233',
        'Arousal': '#d44b05',
        'SLP': '#3500d3',
        'REM': '#3500d3',
        'N1': '#2bc7c4',  # 2b7cc7
        'N2': '#2b5dc7',
        'N3': '#000000',
    }

    if isinstance(hypnogram_colors, type(None)):
        hypnogram_colors = _hypnogram_colors

    if isinstance(hypnogram_values, type(None)):
        hypnogram_values = _hypnogram_values

    where = 'plot_hypnogram'
    if len(orig_df) == 0:
        raise ValueError('plot_hypnogram: the hypnogram is empty.')
    secs = {}
    for col in ('start', 'end'):
        try:
            secs[col], kind = _to_seconds(orig_df[col], col, where)
        except TypeError as e:
            raise TypeError(f'plot_hypnogram: "{col}" must hold timezone-aware datetimes ({e})') from e
        if kind != 'datetime':
            raise TypeError(f'plot_hypnogram: "{col}" must hold timezone-aware datetimes '
                            '(convert numeric timestamps first).')

    # sorted copy: the caller's frame is not modified
    orig_df = orig_df.iloc[np.argsort(secs['start'], kind='mergesort')].reset_index(drop=True)
    is_arousal = orig_df['annotation'].isin(AROUSAL_TAGS)
    if is_arousal.any():
        arousal_key = next((k for k in AROUSAL_TAGS if k in hypnogram_values), None)
        if arousal_key is not None:
            orig_df['annotation'] = orig_df['annotation'].where(~is_arousal, arousal_key)
    for name, ref in (('hypnogram_values', hypnogram_values), ('hypnogram_colors', hypnogram_colors)):
        unknown = sorted(set(orig_df['annotation']) - set(ref), key=str)
        if unknown:
            raise ValueError(f'plot_hypnogram: no entry in {name} for label(s) {unknown}.')
    orig_df['state_id'] = orig_df['annotation'].map(hypnogram_values)
    orig_df['state_color'] = orig_df['annotation'].map(hypnogram_colors)
    df_arrousals = orig_df.loc[is_arousal].reset_index(drop=True)
    df = orig_df.loc[~is_arousal].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError('plot_hypnogram: the hypnogram has no sleep-state epochs (only arousals).')
    # if 2 consecutive epochs have the same state and touch in time, merge them
    merged_rows = []
    for _, row in df.iterrows():
        if merged_rows and merged_rows[-1]['state_id'] == row.state_id and merged_rows[-1]['end'] == row.start:
            merged_rows[-1]['end'] = row.end
        else:
            merged_rows.append(row.to_dict())
    df = pd.DataFrame(merged_rows, columns=df.columns)

    x_start = np.array([pd.Timestamp(t).to_pydatetime() for t in df['start']], dtype=object)
    x_end = np.array([pd.Timestamp(t).to_pydatetime() for t in df['end']], dtype=object)
    t_lo = pd.Timestamp(min(x_start))
    t_hi = pd.Timestamp(max(x_end))
    tz = getattr(orig_df['start'].dtype, 'tz', None) or pd.Timestamp(orig_df['start'].iloc[0]).tzinfo

    if not fig:
        plt.figure(dpi=200)
    plt.xlim(t_lo.to_pydatetime(), t_hi.to_pydatetime())

    # set background color for nights (calendar dates in the data's timezone)
    for a, z in _night_spans(t_lo.tz_convert(tz), t_hi.tz_convert(tz), tz, night_start):
        plt.axvspan(a.to_pydatetime(), z.to_pydatetime(), facecolor='gray', alpha=0.3)

    # plot columns
    for idx, row in enumerate(df.iterrows()):
        val = row[1]['state_id']
        clr = row[1]['state_color']

        plt.fill_between(
            [x_start[idx], x_end[idx]],
            [val, val],
            color=clr,
            alpha=0.5,
            linewidth=0
        )

    for idx in range(df.__len__() - 1):
        val0 = df.state_id[idx]
        val1 = df.state_id[idx + 1]
        start0 = df.start[idx]
        start1 = df.start[idx + 1]
        end0 = df.end[idx]
        end1 = df.end[idx + 1]

        if val0 == val1:
            if end0 == start1:
                x = [start0, start1]
                y = [val0, val1]
            else:
                x = [start0, end0]
                y = [val0, val0]
        else:
            if end0 == start1:
                x = [start0, end0, start1]
                y = [val0, val0, val1]
            else:
                x = [start0, end0]
                y = [val0, val0]

        plt.plot(x, y, color='black', alpha=1, linewidth=1)

    x = [df.start.iloc[-1], df.end.iloc[-1]]
    y = [df.state_id.iloc[-1], df.state_id.iloc[-1]]
    plt.plot(x, y, color='black', alpha=1, linewidth=1)

    # plot arousals
    for row in df_arrousals.iterrows():
        val = row[1].state_id
        clr = row[1].state_color
        a = pd.Timestamp(row[1].start).to_pydatetime()
        z = pd.Timestamp(row[1].end).to_pydatetime()
        plt.fill_between(
            [a, a, z, z],
            [0, val, val, 0],
            color=clr,
            alpha=1,
            linewidth=1
        )

    # format y ticks
    plt.yticks(list(hypnogram_values.values()), hypnogram_values.keys())
    for ticklabel in plt.gca().get_yticklabels():
        clr = hypnogram_colors[ticklabel.get_text()]
        ticklabel.set_color(clr)
        # ticklabel.set_fontsize(fontsize)

    # plot y grid
    for idx, key in enumerate(hypnogram_values.keys()):
        clr = hypnogram_colors[key]
        val = hypnogram_values[key]
        plt.plot([t_lo.to_pydatetime(), t_hi.to_pydatetime()], [val, val], color=clr, linewidth=0.7, alpha=0.7, linestyle=':')

    # format x_ticks
    plt.gcf().autofmt_xdate()
    formatter = mdates.DateFormatter("%H:%M", tz=tz)
    plt.gcf().get_axes()[0].xaxis.set_major_formatter(formatter)

    # plot hour x grid
    plt.grid(True, axis='x', alpha=1, linewidth=0.5, linestyle=':')

    # axes labels
    plt.xlabel('\n Time [' + t_lo.tz_convert(tz).strftime('%d.%m.%Y') + ' - ' + t_hi.tz_convert(tz).strftime('%d.%m.%Y') + ']',
               fontsize=fontsize)
    plt.ylabel('Sleep state', fontsize=fontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)
