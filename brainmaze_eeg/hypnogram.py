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

**Input conventions** (since 3.0.0):

A hypnogram is a :class:`pandas.DataFrame` with one row per scored epoch and the columns
``annotation`` (sleep-state label), ``start`` and ``end``, optionally ``duration``.

* ``start``/``end`` are either **timezone-aware** datetimes (``datetime.datetime``,
  :class:`pandas.Timestamp` or a ``datetime64[ns, tz]`` column) or **numeric POSIX
  timestamps in seconds**. Naive datetimes raise ``TypeError`` (elapsed time across a DST
  change would be wrong; localize them first). Numeric values whose magnitude exceeds
  ``1e11`` (the year 5138 in seconds) raise ``ValueError``, because they are almost
  certainly milliseconds or finer. Numbers are not checked further: *relative* times in
  milliseconds (small numbers) without a ``duration`` column pass as seconds, so pass
  seconds.
* ``duration``, if present, is numeric seconds or a timedelta column (e.g.
  ``df.end - df.start``) and must equal ``end - start`` (to 1 ms), otherwise ``ValueError``
  is raised. This also catches millisecond timestamps with durations in seconds. Every
  function that validates the input (all functions that read ``start``/``end``) uses
  ``end - start``. :func:`get_time_by_key`, :func:`get_stage_times`,
  :func:`get_stage_times_dataset` and :func:`valid_dataset_index_by_duration` need only
  ``annotation`` and ``duration`` and sum the ``duration`` column as given (not
  validated; a timedelta column gives a :class:`pandas.Timedelta`).
* Rows may come in any order; functions sort by ``start`` (stable) and never modify the
  caller's frame.
* Labels: ``'WAKE'`` (the label written by ``brainmaze_utils.annotations.load_NSRR``) is
  an alias of ``'AWAKE'`` everywhere in this module, in the data and in tag arguments.
  Functions that return a hypnogram (the ``fill_*``/``correct_*`` functions and
  :func:`do_median_filtration`) return it as ``'AWAKE'``.
* Arousals are events, not states: rows labelled ``'Arousal'`` (the label written by
  :mod:`brainmaze_utils`) or ``'Arrousal'`` (the legacy spelling of older data) are removed
  before scoring and never relabelled.
* After arousals are removed, epochs must not overlap (``ValueError`` otherwise) in the
  scoring functions (:func:`get_fell_asleep_time`, :func:`get_awakening_time`,
  :func:`get_rem_latency`, :func:`score_night`) and in :func:`get_hypnogram_datarate`,
  :func:`is_sleep_complete`, and in :func:`get_number_of_awakenings` and
  :func:`get_transition_counts` when the frame has ``start``/``end`` columns.
* The scoring functions first merge touching epochs with the same label into **bouts**
  (one row per continuous period), so a hypnogram tiled into 30-s epochs and the same
  hypnogram with merged rows give the same results. Their time windows (sleep onset,
  awakening) count only the part of each bout that lies inside the window.
* **Unscored time** is (a) epochs labelled ``'UNKNOWN'`` or any other label that is
  neither a sleep stage (N1, N2, N3, REM) nor the awake tag, and (b) time gaps between
  epochs (no rows) longer than ``max_gap_s`` (default 1 s; this includes the time of a
  removed ``'Arousal'`` row that was tiled in as an epoch). The scoring functions
  (:func:`get_fell_asleep_time`, :func:`get_awakening_time`, :func:`get_rem_latency`,
  :func:`score_night`) compute each result with all unscored time scored as AWAKE, and
  again as each sleep stage (N1, N2, N3, REM). If all agree, the result does not depend
  on the unscored time and is returned without a warning. If they differ, the result is
  ambiguous and is never returned silently: with ``on_unscored='warn'`` (default) a
  :class:`UserWarning` names the unscored spans and the alternative results, and
  ``NaT``/``NaN`` is returned for that result only; ``on_unscored='raise'`` raises
  ``ValueError``. New in 3.0.0.

  Limitations: each scoring relabels *all* unscored time with one state. A result that
  changes only when different unscored spans get different states, or when the state
  changes in the middle of a span, is not detected. Gaps up to ``max_gap_s`` are ignored
  (counted neither as awake nor as sleep) and may still change a result; pass
  ``max_gap_s=0`` to check every gap. Time before the first and after the last epoch is
  outside the recording, not unscored. Overlaps are stricter than gaps: epochs that
  overlap by more than 1e-6 s raise ``ValueError`` (``epochs overlap``), so a hypnogram
  whose ``start`` times jitter by a sample while ``end = start + 30`` must be fixed (e.g.
  ``end`` = the next ``start``) before scoring. With unscored time each scoring function
  runs once per scoring (five times); a recording with tens of thousands of bouts and no
  qualifying awakening can then take a minute or more in :func:`get_awakening_time`.
* **Unscored time in practice: expect NaN.** Because every unscored span may be any
  state, real nights with unscored time often get ``NaN`` (and a warning) for the **REM
  latencies**, ``n_awakenings`` and ``n_complete_sleep_cycles``, while the sleep onset,
  the awakening and the stage times usually stay computed. Typical causes:

  - a leading unscored segment before lights-off (e.g. NSRR ``Unscored`` epochs at the
    start of the recording): scored as REM it would be the first REM, so both REM
    latencies are ambiguous. Any unscored time before the first REM does the same.
  - on a night **without REM**, unscored time anywhere (even after the awakening): scored
    as REM it would create a REM, so the latencies are ambiguous.
  - an unscored epoch or a gap longer than ``max_gap_s`` inside the sleep: it may be an
    awakening (``n_awakenings``) or a REM bout (``n_complete_sleep_cycles``).

  Remedies (they bring every field of the demo night back): (1) **drop leading and
  trailing unscored rows**, because time outside the recording is not unscored (here:
  the REM latencies come back); (2) **relabel spans whose state is known**, e.g.
  ``df.loc[df.annotation == 'UNKNOWN', 'annotation'] = 'AWAKE'`` if they are known to be
  wake; (3) **raise** ``max_gap_s`` for short gaps known to be harmless (e.g.
  ``max_gap_s=120`` for epochs dropped by a loader). Only relabel what you know: a
  relabelled span gives the same numbers as if it had been scored that way.
* **NSRR data:** ``brainmaze_utils.annotations.load_NSRR`` (brainmaze-utils 3.0.0) raises
  ``KeyError`` on the NSRR stages ``'Unscored|9'`` and ``'Movement|6'``. Map both to
  ``'UNKNOWN'`` (unscored) before scoring; do not drop the rows (dropping leaves a gap,
  which is treated the same way, but only above ``max_gap_s``). A fix in the loader is a
  brainmaze-utils follow-up.
"""


import numpy as np
import pandas as pd
import datetime
import warnings
from copy import deepcopy
from tqdm import tqdm

# merge_annotations and filter_by_key are no longer used here; they stay importable from
# this module for backward compatibility.
from brainmaze_utils.annotations import merge_annotations, filter_by_key, create_day_indexes  # noqa: F401


AROUSAL_TAGS = ('Arousal', 'Arrousal')
"""Labels treated as arousal events: ``'Arousal'`` (brainmaze-utils) and the legacy
spelling ``'Arrousal'``."""

LABEL_ALIASES = {'WAKE': 'AWAKE'}
"""Alternative labels and the label they stand for: ``'WAKE'`` (written by
``brainmaze_utils.annotations.load_NSRR``) means ``'AWAKE'``."""

_N1 = 'N1'
_SLEEP_STAGES = ('N1', 'N2', 'N3', 'REM')
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


def _canon(label):
    """The canonical label (``'WAKE'`` -> ``'AWAKE'``); other labels unchanged."""
    try:
        return LABEL_ALIASES.get(label, label)
    except TypeError:  # unhashable
        return label


def _canon_tags(tags):
    return [_canon(t) for t in _tag_list(tags)]


def _canon_series(ann):
    """Annotation column with aliases replaced (keeps the index and dtype)."""
    ann = pd.Series(ann)
    if isinstance(ann.dtype, pd.CategoricalDtype):
        ann = ann.astype(object)
    for alias, label in LABEL_ALIASES.items():
        ann = ann.where(ann != alias, label)
    return ann


def _duration_seconds(values, where):
    """A ``duration`` column as float seconds: numeric seconds or timedeltas."""
    d = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_timedelta64_dtype(d.dtype):
        return d.dt.total_seconds().to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(d.dtype) and not pd.api.types.is_bool_dtype(d.dtype):
        return d.to_numpy(dtype=float)
    vals = list(d)
    if vals and all(isinstance(v, (datetime.timedelta, np.timedelta64)) for v in vals):
        return pd.to_timedelta(pd.Series(vals)).dt.total_seconds().to_numpy(dtype=float)
    if all(_is_real_number(v) for v in vals):
        return np.array(vals, dtype=float)
    raise TypeError(f'{where}: "duration" must hold numbers (seconds) or timedeltas, '
                    f'got dtype {d.dtype}.')


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
    h['annotation'] = _canon_series(h['annotation'])
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
        d = _duration_seconds(h['duration'], where)
        bad = ~(np.abs(d - dur) <= _DURATION_TOL)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f'{where}: "duration" does not equal end - start for {int(bad.sum())} epoch(s) '
                f'(first: duration={h["duration"].iloc[i]!r}, read as {float(d[i])!r} s; end - start='
                f'{float(dur[i])!r} s). Numeric durations and numeric timestamps must be in seconds.')
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


def _in_window(s, e, lo, hi):
    """Seconds of each bout ``[s, e)`` that lie inside the window ``[lo, hi)``."""
    return np.clip(np.minimum(e, hi) - np.maximum(s, lo), 0.0, None)


_ON_UNSCORED = ('warn', 'raise')
_MAX_GAP_S = 1.0  # s, default: time gaps (no epochs) longer than this are unscored time
_GAP = '(no epochs)'  # label of an inserted gap row, used in messages only
_MAX_SPANS_SHOWN = 5


def _check_on_unscored(on_unscored, where):
    if not isinstance(on_unscored, str) or on_unscored not in _ON_UNSCORED:
        raise ValueError(f"{where}: on_unscored must be one of {_ON_UNSCORED}, got {on_unscored!r}.")


def _check_max_gap(max_gap_s, where):
    if not _is_real_number(max_gap_s) or not max_gap_s >= 0:
        raise ValueError(f'{where}: max_gap_s must be a number of seconds >= 0, got {max_gap_s!r}.')
    return float(max_gap_s)


def _missing(b):
    """``NaT`` (datetime hypnogram) or ``NaN`` (numeric hypnogram)."""
    return float('nan') if _is_real_number(b['start'].iloc[0]) else pd.NaT


class _Fail:
    """Outcome "not found" (the ``ValueError`` a function would raise) under one scoring of
    the unscored time; all failures compare equal to each other and unequal to values."""

    def __init__(self, exc):
        self.exc = exc


def _same(values):
    """True if all outcomes agree: numbers equal to ``_TOL`` (NaN equals NaN), or all
    ``_Fail``."""
    first = values[0]
    for v in values[1:]:
        if isinstance(first, _Fail) or isinstance(v, _Fail):
            if not (isinstance(first, _Fail) and isinstance(v, _Fail)):
                return False
        elif not (abs(first - v) <= _TOL or (np.isnan(first) and np.isnan(v))):
            return False
    return True


def _with_gaps(b, max_gap_s):
    """Bouts ``b`` plus one ``_GAP`` row for every time gap between consecutive bouts longer
    than ``max_gap_s`` (columns annotation/start/end/_s/_e/duration only)."""
    cols = ['annotation', 'start', 'end', '_s', '_e', 'duration']
    if len(b) < 2:
        return b[cols]
    s = b['_s'].to_numpy()
    e = b['_e'].to_numpy()
    idx = np.flatnonzero(s[1:] - e[:-1] > max_gap_s + _TOL)
    if idx.size == 0:
        return b[cols]
    g = pd.DataFrame({'annotation': [_GAP] * idx.size,
                      'start': b['end'].iloc[idx].reset_index(drop=True),
                      'end': b['start'].iloc[idx + 1].reset_index(drop=True),
                      '_s': e[idx], '_e': s[idx + 1]})
    g['duration'] = g['_e'] - g['_s']
    return (pd.concat([b[cols], g], ignore_index=True)
            .sort_values('_s', kind='mergesort').reset_index(drop=True))


def _resolutions(b, awake_tag, sleep_tags, max_gap_s):
    """The ways of scoring the unscored time of the bouts ``b``.

    Unscored time = bouts whose label is neither ``awake_tag`` nor a sleep stage (N1, N2,
    N3, REM, ``sleep_tags``), and time gaps longer than ``max_gap_s``. Returns
    ``(variants, spans)``: ``variants`` is a list of ``(label, bouts)`` with all unscored
    time relabelled ``label`` (``awake_tag`` and each sleep stage in turn) and merged into
    bouts; ``[(None, b)]`` when there is no unscored time. ``spans`` lists the unscored
    spans as ``(labels, start, end)``."""
    a = _with_gaps(b, max_gap_s)
    stages = list(dict.fromkeys(list(_SLEEP_STAGES) + list(sleep_tags)))
    unscored = ~a['annotation'].isin([awake_tag] + stages).to_numpy()
    if not unscored.any():
        return [(None, b)], []
    idx = np.flatnonzero(unscored)
    first = idx[np.r_[True, np.diff(idx) > 1]]
    last = idx[np.r_[np.diff(idx) > 1, True]]
    spans = [(list(dict.fromkeys(a['annotation'].iloc[i:j + 1])), a['start'].iloc[i], a['end'].iloc[j])
             for i, j in zip(first, last)]
    variants = []
    for label in [awake_tag] + stages:
        ann = a['annotation'].to_numpy(dtype=object, copy=True)
        ann[unscored] = label
        variants.append((label, _merge_bouts(a.assign(annotation=ann))))
    return variants, spans


def _span_text(labels, s, e):
    epochs = [label for label in labels if label != _GAP]
    what = [f'{", ".join(repr(label) for label in epochs)} epochs'] if epochs else []
    if len(epochs) < len(labels):
        what.append('a gap (no epochs)')
    return f'{" and ".join(what)} from {s} to {e}'


def _spans_text(spans):
    parts = [_span_text(*span) for span in spans[:_MAX_SPANS_SHOWN]]
    if len(spans) > _MAX_SPANS_SHOWN:
        parts.append(f'and {len(spans) - _MAX_SPANS_SHOWN} more')
    return '; '.join(parts)


def _value_text(value):
    """A result for a message: a timedelta as signed ``[-]HH:MM:SS`` (``str`` would show
    -19 min as ``-1 days +23:41:00``), anything else as ``str``."""
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        return _format_hms(_delta_seconds(value))
    return str(value)


_REMEDY = (' To get a value: score the span(s) whose state is known (e.g. relabel them), drop '
           'unscored epochs at the start or end of the recording (time outside the recording is '
           'not unscored), or raise max_gap_s for short gaps that are known to be harmless.')


def _ambiguous(where, on_unscored, spans, what, shown, stacklevel):
    """Report that ``what`` depends on how the unscored ``spans`` are scored: raise
    ``ValueError`` or warn. ``shown`` is ``[(label, text), ...]`` (the result with all
    unscored time scored as ``label``). ``stacklevel`` of the warning: 3 when called from a
    public function."""
    alt = {}
    for label, text in shown:
        alt.setdefault(text, []).append(str(label))
    alt = '; '.join(f'as {"/".join(labels)}: {text}' for text, labels in alt.items())
    msg = (f'{where}: the {what} depends on how the unscored time is scored (unscored time: '
           f'{_spans_text(spans)}). Scoring all of it {alt}.')
    if on_unscored == 'raise':
        raise ValueError(msg + _REMEDY + " Or pass on_unscored='warn' to get NaT/NaN.")
    warnings.warn(msg + ' Returning NaT/NaN for it (on_unscored=\'raise\' makes this an error).'
                  + _REMEDY, UserWarning, stacklevel=stacklevel)


def _decide(where, on_unscored, spans, what, results, missing, stacklevel=4):
    """``results``: ``[(label, value_s or _Fail, value)]``, one per scoring of the unscored
    time. Agreement -> the value (or the ``ValueError`` if all failed); disagreement ->
    ``_ambiguous`` and ``missing``."""
    if _same([r[1] for r in results]):
        if isinstance(results[0][1], _Fail):
            raise results[0][1].exc
        return results[0][2]
    _ambiguous(where, on_unscored, spans, what,
               [(lab, 'none found' if isinstance(v, _Fail) else _value_text(obj)) for lab, v, obj in results],
               stacklevel)
    return missing


def _onset_index(b, t_sleep_check, t_awake_threshold, awake_tag, sleep_tags, where):
    """Index of the sleep-onset bout of fully scored bouts ``b``; ``ValueError`` if none."""
    ann = b['annotation'].to_numpy()
    s = b['_s'].to_numpy()
    e = b['_e'].to_numpy()
    sleep = np.isin(ann, sleep_tags)
    awake = ann == awake_tag
    candidates = ([0] if sleep[0] else []) + list(np.flatnonzero(awake[:-1] & sleep[1:]) + 1)
    window = t_sleep_check * 60.0
    limit = t_awake_threshold * 60.0
    for k in candidates:
        if _in_window(s, e, s[k], s[k] + window)[awake].sum() < limit:
            return int(k)
    raise ValueError(
        f'{where}: no sleep onset found: no transition into {sleep_tags} is followed by less '
        f'than {t_awake_threshold} min of {awake_tag!r} within {t_sleep_check} min.')


def _awakening_candidates(ann, awake_tag, sleep_tags):
    """Indices of the ``awake_tag`` bouts that follow a ``sleep_tags`` bout directly or
    through N1 bouts only (N2 -> N1 -> AWAKE)."""
    out = []
    for k in np.flatnonzero(ann == awake_tag):
        j = k - 1
        while j >= 0 and ann[j] == _N1 and _N1 not in sleep_tags:
            j -= 1
        if j >= 0 and ann[j] in sleep_tags:
            out.append(k)
    return np.array(out, dtype=int)


def _awakening_index(b, t_awake_threshold, t_sleep_threshold, awake_tag, sleep_tags, after_s, where):
    """Awakening of fully scored bouts ``b``: ``(time, seconds)``; ``ValueError`` if none."""
    ann = b['annotation'].to_numpy()
    s = b['_s'].to_numpy()
    e = b['_e'].to_numpy()
    asleep = np.isin(ann, list(dict.fromkeys(list(_SLEEP_STAGES) + sleep_tags)))
    awake = ann == awake_tag
    candidates = _awakening_candidates(ann, awake_tag, sleep_tags)
    if after_s is not None:
        candidates = candidates[s[candidates] >= after_s - _TOL]
    need_awake = t_awake_threshold * 60.0
    max_sleep = t_sleep_threshold * 60.0
    window = need_awake + max_sleep
    for k in candidates:
        inside = _in_window(s, e, s[k], s[k] + window)
        if inside[awake].sum() >= need_awake - _TOL and inside[asleep].sum() <= max_sleep + _TOL:
            return b['start'].iloc[k], s[k]
    if awake[-1]:
        if candidates.size == 0:
            raise ValueError(f'{where}: no awakening found: the hypnogram ends {awake_tag!r} but has no '
                             f'transition from {sleep_tags} (directly or through {_N1!r}) to '
                             f'{awake_tag!r}' + ('' if after_s is None else ' after the given time') + '.')
        k = candidates[-1]
        return b['start'].iloc[k], s[k]
    return b['end'].iloc[-1], float(b['_e'].iloc[-1])


def _rem_latency_s(b, onset, rem_tag, awake_tag):
    """REM latencies of fully scored bouts ``b`` with the onset bout index ``onset`` (or a
    ``_Fail``): ``{key: (seconds or _Fail, value)}``, or ``None`` without REM."""
    ann = b['annotation'].to_numpy()
    rem = np.flatnonzero(ann == rem_tag)
    if rem.size == 0:
        return None
    k_rem = rem[0]
    first_rem = b['start'].iloc[k_rem]
    before = np.flatnonzero((ann == awake_tag) & (b['_e'].to_numpy() <= b['_s'].iloc[k_rem] + _TOL))
    out = {}
    if isinstance(onset, _Fail):
        out['fall_asleep'] = (onset, None)
    else:
        out['fall_asleep'] = (b['_s'].iloc[k_rem] - b['_s'].iloc[onset], first_rem - b['start'].iloc[onset])
    if isinstance(onset, _Fail):
        # No sleep onset with this scoring: a fully scored night like it raises, so there is
        # no `last_awake` value either (it must not "agree" with the other scorings).
        out['last_awake'] = out['fall_asleep']
    elif before.size:
        j = before[-1]
        out['last_awake'] = (b['_s'].iloc[k_rem] - b['_e'].iloc[j], first_rem - b['end'].iloc[j])
    else:
        out['last_awake'] = out['fall_asleep']
    return out


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
    .. note:: **Changed in 3.0.0:**
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


def get_fell_asleep_time(df, t_sleep_check=60, t_awake_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N1', 'N2', 'N3'],
                        on_unscored='warn', max_gap_s=_MAX_GAP_S):
    """
    Determine when the subject fell asleep based on hypnogram data.

    Touching epochs with the same label are merged into bouts first (module docstring).
    Candidates are the first bout, if it is a sleep stage, and every sleep-stage bout that
    directly follows an ``awake_tag`` bout, in time order. The sleep onset is the first
    candidate for which less than ``t_awake_threshold`` minutes of ``awake_tag`` lie in the
    window ``[candidate start, candidate start + t_sleep_check)``. Only the part of each
    bout inside the window counts (a bout reaching past the window end is clipped to it).
    Unscored time (module docstring) is checked: the onset is returned only if it is the
    same with all unscored time scored as AWAKE and as each sleep stage.

    Example (defaults): AWAKE 30 min, N2 55, AWAKE 15, N2 200: the window after the N2 at
    30 min holds 5 min of AWAKE (the AWAKE bout from 85 to 100 min is clipped at 90), so
    the onset is at 30 min.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    t_sleep_check : int, optional
        Time window in minutes to check for sustained sleep. Default is 60.
    t_awake_threshold : int, optional
        The awake time in minutes inside the window must be below this. Default is 10.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE' (``'WAKE'`` is the same state).
    sleep_cycle_tags : list, optional
        List of tags indicating sleep states. Default is ['REM', 'N1', 'N2', 'N3'].
    on_unscored : {'warn', 'raise'}, optional
        What to do when the onset depends on how the unscored time (epochs labelled neither
        a sleep stage nor ``awake_tag``, gaps longer than ``max_gap_s``) is scored.
        ``'warn'`` (default) emits a :class:`UserWarning` naming the unscored spans and the
        alternative onsets and returns ``NaT``/``NaN``; ``'raise'`` raises ``ValueError``.
        New in 3.0.0.
    max_gap_s : float, optional
        Time gaps between epochs longer than this (seconds) are unscored time; shorter gaps
        are ignored. Default 1.0. New in 3.0.0.

    Returns
    -------
    datetime or float
        Start of the sleep-onset bout, in the type of the ``start`` column; ``NaT``
        (datetime input) or ``NaN`` (numeric input) if the onset is ambiguous because of
        unscored epochs and ``on_unscored='warn'``.

    Raises
    ------
    ValueError
        If no candidate qualifies (with every scoring of the unscored time), the hypnogram
        is empty, or epochs overlap; or the onset depends on the unscored time and
        ``on_unscored='raise'``.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       If no candidate qualified, v1.0.0 returned the start of the recording, which looks
       like a valid onset; it now raises ``ValueError``. Rows labelled ``'Arousal'`` are
       removed like ``'Arrousal'`` (only the latter was, so an ``'Arousal'`` epoch between
       AWAKE and sleep hid the onset). Epochs are sorted and merged into bouts, so tiled and
       merged hypnograms agree. The window counts only the part of each bout inside it
       (v1.0.0 counted every epoch that started inside it in full, so on merged bouts an
       AWAKE bout starting 5 min before the window end counted with its whole length).
       ``awake_tag`` is compared for equality (it was a substring test); ``'WAKE'`` is
       read as ``'AWAKE'``. Numeric timestamps (seconds) work. Unscored epochs and gaps no
       longer shift the onset silently (``on_unscored``, ``max_gap_s``).
    """
    where = 'get_fell_asleep_time'
    _check_on_unscored(on_unscored, where)
    max_gap_s = _check_max_gap(max_gap_s, where)
    b = _bouts(df, where)
    if len(b) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    awake_tag, sleep_tags = _canon(awake_tag), _canon_tags(sleep_cycle_tags)
    variants, spans = _resolutions(b, awake_tag, sleep_tags, max_gap_s)
    results = []
    for label, v in variants:
        try:
            k = _onset_index(v, t_sleep_check, t_awake_threshold, awake_tag, sleep_tags, where)
            results.append((label, v['_s'].iloc[k], v['start'].iloc[k]))
        except ValueError as exc:
            results.append((label, _Fail(exc), None))
    return _decide(where, on_unscored, spans, 'sleep onset', results, _missing(b))


def get_awakening_time(df, t_awake_threshold=90, t_sleep_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N1', 'N2', 'N3'], after=None,
                       on_unscored='warn', max_gap_s=_MAX_GAP_S):
    """
    Determine when the subject woke up based on hypnogram data.

    Touching epochs with the same label are merged into bouts first (module docstring).

    - **Candidates** are the ``awake_tag`` bouts that follow a ``sleep_cycle_tags`` bout,
      either directly or through ``'N1'`` bouts only (N2 -> N1 -> AWAKE is a candidate even
      if ``'N1'`` is not in ``sleep_cycle_tags``), and start at or after ``after`` if given.
    - **Sustained wake:** a candidate qualifies if, in the window ``[candidate start,
      candidate start + t_awake_threshold + t_sleep_threshold)``, at least
      ``t_awake_threshold`` minutes are ``awake_tag`` and at most ``t_sleep_threshold``
      minutes are sleep (N1, N2, N3, REM and ``sleep_cycle_tags``). Only the part of each
      bout inside the window counts.
    - The awakening is the start of the **first** qualifying candidate.
    - If none qualifies: when the hypnogram ends with ``awake_tag``, the start of the
      **last** candidate (the final transition from sleep to wake; ``ValueError`` if there
      is none); otherwise the end of the recording (the subject was still asleep when it
      ended).
    - **Unscored time** (module docstring) is checked: the awakening is returned only if
      it is the same with all unscored time scored as AWAKE and as each sleep stage
      (``on_unscored``).
    - ``after=None`` searches the whole recording, so a sustained wake after a short nap
      before the night's sleep is found first; :func:`score_night` passes the sleep onset.

    **What the time means.** The primary rule (first sustained wake of
    ``t_awake_threshold`` minutes) is a proxy for the end of the night, roughly when the
    subject got up (lights on); sleep after it (e.g. a morning nap) is not part of the
    night. It is *not* the AASM final awakening, which is the end of the last sleep epoch
    before lights on: the two agree when the subject stays awake after the last sleep
    epoch. The fallback (no sustained wake, the night ends awake: the last sleep-to-wake
    transition) is the AASM definition.

    Example (defaults): AWAKE 30 min, N2 60, REM 20, AWAKE 5, N2 120, N1 5, AWAKE 120 gives
    the start of the last AWAKE bout (at 240 min): the brief AWAKE at 110 min holds 95 min
    of sleep in its window.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    t_awake_threshold : int, optional
        Minimum sustained awake time in minutes to be considered awakened. Default is 90.
    t_sleep_threshold : int, optional
        Maximum sleep time in minutes allowed in the awakening window. Default is 10.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE' (``'WAKE'`` is the same state).
    sleep_cycle_tags : list, optional
        List of tags indicating sleep states. Default is ['REM', 'N1', 'N2', 'N3'].
    after : datetime or float, optional
        Only consider awakenings starting at or after this time. It must be of the same
        kind as ``start`` (a timezone-aware datetime for datetime input, a number of
        seconds for numeric input), otherwise ``TypeError`` is raised.
        :func:`score_night` passes the sleep onset. New in 3.0.0.
    on_unscored : {'warn', 'raise'}, optional
        What to do when the awakening depends on how the unscored time is scored (e.g.
        ``... N2, UNKNOWN, AWAKE``: the wake may have begun anywhere in the unknown span).
        ``'warn'`` (default) emits a :class:`UserWarning` naming the unscored spans and the
        alternative times and returns ``NaT``/``NaN``; ``'raise'`` raises ``ValueError``.
        New in 3.0.0.
    max_gap_s : float, optional
        Time gaps between epochs longer than this (seconds) are unscored time; shorter gaps
        are ignored. Default 1.0. New in 3.0.0.

    Returns
    -------
    datetime or float
        Time of the awakening, in the type of the ``start`` column; ``NaT`` (datetime
        input) or ``NaN`` (numeric input) if it is ambiguous because of unscored epochs
        and ``on_unscored='warn'``.

    Raises
    ------
    TypeError
        If ``after`` is not of the same kind as ``start``, or the input is invalid
        (module docstring).
    ValueError
        If the hypnogram is empty or invalid, or it ends awake without a candidate (with
        every scoring of the unscored time); or the awakening depends on the unscored time
        and ``on_unscored='raise'``.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       ``t_sleep_threshold`` (minutes) was compared with the sleep time in seconds, so
       10 meant 10 s: one minute of N2 in the window rejected an awakening. The default
       ``sleep_cycle_tags`` now include ``'N1'``, and a path through N1 bouts counts in
       any case: v1.0.0 missed an awakening reached through N1 (N2 -> N1 -> AWAKE, the
       usual way to wake up) and then returned an earlier brief awakening, which
       truncated the sleep period. The window counts only the part of each bout inside it
       and is ``t_awake_threshold + t_sleep_threshold`` long (v1.0.0 counted the
       ``t_awake_threshold`` window's epochs in full). A first epoch in a sleep stage is no
       longer a candidate (it returned the start of that sleep epoch), ``'Arousal'`` rows
       are removed like ``'Arrousal'``, ``'WAKE'`` is read as ``'AWAKE'``, epochs are
       sorted and merged into bouts, and ``ValueError`` replaces an ``IndexError`` when no
       transition exists. Numeric timestamps (seconds) work. A night whose awakening
       depends on unscored epochs or gaps (e.g. ``N2, UNKNOWN, AWAKE``, or a night ending
       ``N2, UNKNOWN``) used to return a silently shifted awakening (the transition before
       the unknown span, the end of the recording, or an error); it now warns and returns
       ``NaT``/``NaN`` (``on_unscored``, ``max_gap_s``).
    """
    where = 'get_awakening_time'
    _check_on_unscored(on_unscored, where)
    max_gap_s = _check_max_gap(max_gap_s, where)
    h = _hypnogram(df, where)
    after_s = None
    if after is not None:
        after_s, after_kind = _to_seconds([after], 'after', where)
        after_s = after_s[0]
        if len(h):
            kind = _to_seconds(h['start'].iloc[:1], 'start', where)[1]
            if after_kind != kind:
                raise TypeError(f'{where}: "after" is a {after_kind} time but "start" is {kind}; '
                                'pass "after" in the same kind as the hypnogram times.')
    b = _merge_bouts(h)
    if len(b) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    awake_tag, sleep_tags = _canon(awake_tag), _canon_tags(sleep_cycle_tags)
    variants, spans = _resolutions(b, awake_tag, sleep_tags, max_gap_s)
    results = []
    for label, v in variants:
        try:
            t, t_s = _awakening_index(v, t_awake_threshold, t_sleep_threshold, awake_tag, sleep_tags,
                                      after_s, where)
            results.append((label, t_s, t))
        except ValueError as exc:
            results.append((label, _Fail(exc), None))
    return _decide(where, on_unscored, spans, 'awakening', results, _missing(b))


def is_sleep_complete(df, awake_tag='AWAKE'):
    """
    Check if the sleep session is complete (starts and ends with awake state).

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE' (``'WAKE'`` is the same state).

    Returns
    -------
    bool
        True if the earliest and the latest epoch (arousals excluded) are ``awake_tag``.

    Raises
    ------
    TypeError
        On naive datetimes or other invalid time columns (module docstring).
    ValueError
        If the hypnogram is empty, epochs overlap (after removing arousals), or
        ``duration`` does not equal ``end - start``.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       Uses the earliest and latest epoch by ``start`` (it used the first and last rows),
       ignores arousal rows, reads ``'WAKE'`` as ``'AWAKE'``, and raises ``ValueError`` on
       an empty hypnogram. The input is now validated like everywhere else in the module
       (v1.0.0 ignored the times): naive datetimes raise ``TypeError``; overlapping states
       (e.g. a frame that also holds channel annotations) and a ``duration`` that differs
       from ``end - start`` raise ``ValueError``.
    """
    where = 'is_sleep_complete'
    h = _hypnogram(df, where)
    if len(h) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    awake_tag = _canon(awake_tag)
    return bool(h['annotation'].iloc[0] == awake_tag == h['annotation'].iloc[-1])


def get_rem_latency(df, rem_tag='REM', awake_tag='AWAKE', on_unscored='warn', max_gap_s=_MAX_GAP_S):
    """
    Calculate REM sleep latency (time from falling asleep or last awake to first REM).

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``).
    rem_tag : str, optional
        Tag for REM sleep state. Default is 'REM'.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE' (``'WAKE'`` is the same state).
    on_unscored : {'warn', 'raise'}, optional
        Each latency is returned only if it is the same with all unscored time (module
        docstring) scored as AWAKE and as each sleep stage; unscored time before the first
        REM may hide an earlier REM, an earlier or later sleep onset, or a later wake. If a
        latency differs, ``'warn'`` (default) warns and gives ``NaT``/``NaN`` for it;
        ``'raise'`` raises ``ValueError``. New in 3.0.0.

        **On real nights this is common:** any unscored time before the first REM (even a
        leading unscored segment before lights-off), or, on a night without REM, unscored
        time anywhere, makes both latencies ``NaT``/``NaN``, because that time may have
        been REM. To get the latencies, drop leading/trailing unscored rows, relabel the
        spans whose state is known, or raise ``max_gap_s`` for harmless gaps (module
        docstring, "Unscored time in practice").
    max_gap_s : float, optional
        Time gaps between epochs longer than this (seconds) are unscored time; shorter gaps
        are ignored. Default 1.0. New in 3.0.0.

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
        If there is no ``rem_tag`` epoch or no sleep onset (with every scoring of the
        unscored time); or a latency depends on the unscored time and
        ``on_unscored='raise'``.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       Sorted, validated input; arousal rows (both spellings) removed; ``'WAKE'`` read
       as ``'AWAKE'``; ``ValueError`` instead of ``IndexError`` without REM. On a fully
       scored night the latencies are defined as before, from the sleep onset of
       :func:`get_fell_asleep_time` (whose onset window changed in 3.0.0, see there). With
       unscored time (module docstring) a latency that depends on it is no longer returned:
       it is ``NaT``/``NaN`` with a warning (``on_unscored='warn'``), or ``ValueError`` is
       raised (``on_unscored='raise'``); see ``on_unscored`` for how often this happens.
    """
    where = 'get_rem_latency'
    _check_on_unscored(on_unscored, where)
    max_gap_s = _check_max_gap(max_gap_s, where)
    b = _bouts(df, where)
    if len(b) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    rem_tag, awake_tag = _canon(rem_tag), _canon(awake_tag)
    sleep_tags = ['REM', 'N1', 'N2', 'N3']
    variants, spans = _resolutions(b, awake_tag, list(dict.fromkeys(sleep_tags + [rem_tag])), max_gap_s)
    per_variant = []
    for label, v in variants:
        try:
            onset = _onset_index(v, 60, 10, awake_tag, sleep_tags, where)
        except ValueError as exc:
            onset = _Fail(exc)
        per_variant.append((label, onset, _rem_latency_s(v, onset, rem_tag, awake_tag)))
    if all(isinstance(o, _Fail) for _, o, _ in per_variant):
        raise per_variant[0][1].exc
    if all(lat is None for _, _, lat in per_variant):
        raise ValueError(f'{where}: the hypnogram has no {rem_tag!r} epoch.')
    nan = _missing(b)
    out, ambiguous = {}, []
    for key in ('last_awake', 'fall_asleep'):
        values = [(label, _Fail(ValueError()) if lat is None else lat[key][0],
                   None if lat is None else lat[key][1]) for label, _, lat in per_variant]
        if _same([x[1] for x in values]) and not isinstance(values[0][1], _Fail):
            out[key] = values[0][2]
        else:
            out[key] = nan
            ambiguous.append((key, values))
    if ambiguous:
        what = 'result for ' + ', '.join(key for key, _ in ambiguous)
        shown = [(label, ', '.join(f'{key}: ' + ('none' if isinstance(vals[i][1], _Fail) else _value_text(vals[i][2]))
                                   for key, vals in ambiguous))
                 for i, (label, _, _) in enumerate(per_variant)]
        _ambiguous(where, on_unscored, spans, what, shown, stacklevel=3)
    return {'last_awake': out['last_awake'], 'fall_asleep': out['fall_asleep']}


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
    .. note:: **Changed in 3.0.0:**
       v1.0.0 always raised under pandas >= 2 (it used ``DataFrame.append``). Its results
       under pandas < 2 differ from 3.0.0 as follows:

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
    tags = _canon_tags(tags)
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
    awake_tag, n1_tag = _canon(awake_tag), _canon(n1_tag)
    sleep_tags = set(_canon_tags(sleep_tags))
    for a in ann:
        if a in sleep_tags:
            sleep_happened = True
        if a == awake_tag:
            awake_happened = True
        elif a != n1_tag or a in sleep_tags:
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
    epoch after a ``sleep_tags`` epoch; the next awakening needs another ``sleep_tags``
    epoch first. With the default ``sleep_tags`` (N2, N3, REM) ``n1_tag`` epochs neither
    arm nor break the count: ``N2, N1, AWAKE`` is one awakening, and so is
    ``N2, AWAKE, N1, AWAKE`` (the N1/AWAKE alternation of drowsy wake is not counted
    again). This differs from :func:`get_fell_asleep_time` and :func:`get_awakening_time`,
    where N1 is sleep (as in AASM). Pass ``sleep_tags=['N1', 'N2', 'N3', 'REM']`` to count
    every transition from any sleep stage, N1 included, into ``awake_tag``. Other labels
    (e.g. ``'UNKNOWN'``) between sleep and ``awake_tag`` do not break the count;
    :func:`score_night` checks them (``on_unscored``).

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with an ``annotation`` column. With ``start``/``end`` columns the input
        is validated and sorted by ``start``; without them the row order is used.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE' (``'WAKE'`` is the same state).
    n1_tag : str, optional
        Tag for N1 sleep stage. Default is 'N1'.
    sleep_tags : list, optional
        List of tags for sleep states. Default is ['N2', 'N3', 'REM'].

    Returns
    -------
    int
        Number of awakenings.

    Raises
    ------
    TypeError
        With ``start``/``end`` columns: on naive datetimes or other invalid time columns
        (module docstring).
    ValueError
        With ``start``/``end`` columns: if epochs overlap (after removing arousals) or
        ``duration`` does not equal ``end - start``.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       With ``n1_tag`` in ``sleep_tags``, ``AWAKE, N1`` no longer counts an awakening at
       the N1 epoch (falling asleep is not an awakening). Sorted
       by ``start`` when available (it used the row order and label-based
       indexing, which broke on a non-default index), and ``'WAKE'`` is read as
       ``'AWAKE'``. Arousal rows (both spellings) are ignored; this does not change the
       count (an arousal never reset the state machine between sleep and AWAKE). With
       ``start``/``end`` columns the input is now validated (v1.0.0 ignored the times):
       naive datetimes raise ``TypeError``; overlapping states (e.g. a frame that also
       holds channel annotations) and a ``duration`` that differs from ``end - start``
       raise ``ValueError``.
    """
    where = 'get_number_of_awakenings'
    if 'start' in df.columns and 'end' in df.columns:
        ann = _hypnogram(df, where)['annotation'].to_numpy()
    else:
        ann = _canon_series(df['annotation'])
        ann = ann.loc[~ann.isin(AROUSAL_TAGS)].to_numpy()
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
        Total duration in the unit of the ``duration`` column (seconds by convention; a
        timedelta column gives a :class:`pandas.Timedelta`). The ``duration`` column is
        summed as given, not checked against ``start``/``end``. ``'WAKE'`` and
        ``'AWAKE'`` are the same state, in ``df`` and in ``key``.
    """
    ann = _canon_series(df['annotation'])
    if isinstance(key, (list, tuple)):
        value = 0
        for single_key in key:
            value += (df.duration[(ann == _canon(single_key)).to_numpy()]).sum()
        return value
    else:
        return (df.duration[(ann == _canon(key)).to_numpy()]).sum()


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


_NIGHT_SLEEP_TAGS = ['REM', 'N1', 'N2', 'N3']
_NIGHT_FIELDS = ('fell_asleep_time', 'rem_latency_fell_asleep', 'rem_latency_last_awake', 'awakening_time',
                 'n_complete_sleep_cycles', 'n_awakenings', 'n1_sleep_time', 'n2_sleep_time',
                 'n3_sleep_time', 'rem_sleep_time', 'awake_sleep_time')


def _score_resolved(v, b, where):
    """score_night fields of the fully scored bouts ``v`` (one scoring of the unscored time
    of the original bouts ``b``): ``{field: (comparable number or _Fail, value)}``. Stage
    times count the epochs of ``b`` as scored (unscored time in no stage); the counts use
    ``v``."""
    out = {}
    try:
        k = _onset_index(v, 60, 10, 'AWAKE', _NIGHT_SLEEP_TAGS, where)
        fell_s = float(v['_s'].iloc[k])
        out['fell_asleep_time'] = (fell_s, v['start'].iloc[k])
    except ValueError as exc:
        k = fell_s = _Fail(exc)
        out['fell_asleep_time'] = (k, None)
    lat = _rem_latency_s(v, k, 'REM', 'AWAKE')
    for key, lat_key in (('rem_latency_fell_asleep', 'fall_asleep'), ('rem_latency_last_awake', 'last_awake')):
        x = float('nan') if lat is None else lat[lat_key][0]
        x = x if isinstance(x, _Fail) else float(x)
        out[key] = (x, x)
    if isinstance(fell_s, _Fail):
        awake = fell_s
    else:
        try:
            t, awake_s = _awakening_index(v, 90, 10, 'AWAKE', _NIGHT_SLEEP_TAGS, fell_s, where)
            awake = None
            out['awakening_time'] = (float(awake_s), t)
        except ValueError as exc:
            awake = _Fail(exc)
    if awake is not None:   # no sleep period
        for key in _NIGHT_FIELDS:
            out.setdefault(key, (awake, None))
        return out
    period = (v['_s'] >= fell_s - _TOL) & (v['_s'] < awake_s - _TOL)
    sleep_v = v.loc[period].reset_index(drop=True)
    n = get_number_of_sleep_stages(sleep_v, tags='REM', delay=30)
    out['n_complete_sleep_cycles'] = (n, n)
    n = _count_awakenings(sleep_v['annotation'].to_numpy(), 'AWAKE', 'N1', ['N2', 'N3', 'REM'])
    out['n_awakenings'] = (n, n)
    sleep_b = b.loc[(b['_s'] >= fell_s - _TOL) & (b['_s'] < awake_s - _TOL)]
    for key, stage in (('n1_sleep_time', 'N1'), ('n2_sleep_time', 'N2'), ('n3_sleep_time', 'N3'),
                       ('rem_sleep_time', 'REM'), ('awake_sleep_time', 'AWAKE')):
        x = float(sleep_b['duration'].to_numpy()[(sleep_b['annotation'] == stage).to_numpy()].sum())
        out[key] = (x, x)
    return out


def score_night(df, plot=False, on_unscored='warn', max_gap_s=_MAX_GAP_S):
    """
    Compute comprehensive sleep metrics for a night's hypnogram.

    The hypnogram is validated, sorted, arousal rows (``'Arousal'``/``'Arrousal'``) are
    removed, ``'WAKE'`` is read as ``'AWAKE'``, and touching epochs with the same label are
    merged into bouts (module docstring). Then, with the defaults of the individual functions:

    - ``fell_asleep_time``: :func:`get_fell_asleep_time`.
    - ``awakening_time``: :func:`get_awakening_time` with ``after=fell_asleep_time``.
    - The **sleep period** is made of the bouts starting at or after ``fell_asleep_time``
      and before ``awakening_time``; the counts and stage times below are taken over it.
    - **Unscored time** (``'UNKNOWN'`` or any label that is neither a sleep stage nor
      AWAKE, and gaps longer than ``max_gap_s``; module docstring): the whole score is
      computed with all unscored time scored as AWAKE, and as each of N1, N2, N3 and REM
      (the awakening is searched after the onset found with the same scoring). A field
      that is the same in all of them is returned; a field that differs is ambiguous and
      is ``NaT``/``NaN`` with ``on_unscored='warn'`` (default; one :class:`UserWarning`
      names the fields, the unscored spans and the alternatives), or ``ValueError`` is
      raised with ``on_unscored='raise'``. Other fields stay computed. For example an
      ``UNKNOWN`` epoch inside the sleep usually leaves the onset, the awakening and the
      stage times unchanged but makes ``n_awakenings`` ambiguous (it may have been an
      awakening), and ``... N2, UNKNOWN, AWAKE`` makes ``awakening_time`` ambiguous.
    - **Expect NaN on real nights with unscored time.** Both REM latencies,
      ``n_awakenings`` and ``n_complete_sleep_cycles`` will often be ``NaN``: a leading
      unscored segment before lights-off (or any unscored time before the first REM)
      could be the first REM; on a night without REM, unscored time even after the
      awakening could be a REM; an unscored epoch or gap inside the sleep could be an
      awakening or a REM bout. Remedies: **drop leading/trailing unscored rows** (time
      outside the recording is not unscored), **relabel spans whose state is known**, and
      **raise** ``max_gap_s`` for short harmless gaps. On the demo night (5 min
      ``UNKNOWN`` at the start, 12.5 min ``UNKNOWN`` in the sleep, one 2-min gap) the
      four fields are NaN; trimming the leading segment brings back the latencies, and
      relabelling the inner ``UNKNOWN`` plus ``max_gap_s=120`` brings back all fields. See
      the module docstring, "Unscored time in practice".
    - The **stage times** count the scored epochs only: unscored time inside the sleep
      period is in none of them (so they do not depend on how it is scored; compare
      :func:`get_hypnogram_datarate`). ``sleep_complete`` reads the first and last epoch
      as scored.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram with ``annotation``, ``start``, ``end`` (optionally ``duration``):
        timezone-aware datetimes or numeric POSIX timestamps in seconds.
    plot : bool, optional
        If True, plot the hypnogram with sleep onset and awakening markers. Default is False.
        Needs matplotlib (optional extra ``brainmaze-eeg[plot]``) and datetime input.
    on_unscored : {'warn', 'raise'}, optional
        See above. New in 3.0.0.
    max_gap_s : float, optional
        Time gaps between epochs longer than this (seconds) are unscored time; shorter gaps
        are ignored. Default 1.0. New in 3.0.0.

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
        - n_awakenings: :func:`get_number_of_awakenings` (defaults) over the sleep period:
          awakenings after N2, N3 or REM; N1/AWAKE alternation is not counted again (unlike
          the onset and the awakening, where N1 is sleep)
        - n1_sleep_time, n2_sleep_time, n3_sleep_time, rem_sleep_time, awake_sleep_time:
          scored time in each stage within the sleep period, in seconds

        An ambiguous field (see above) is ``NaT`` (datetime input) or ``NaN`` (numeric
        input) for ``fell_asleep_time``/``awakening_time``, and ``NaN`` (a float) for the
        others.

    Raises
    ------
    ValueError
        If the hypnogram is empty, invalid (module docstring), or no sleep onset or
        awakening can be determined (with every scoring of the unscored time); or a field
        is ambiguous and ``on_unscored='raise'``.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       v1.0.0 raised with brainmaze-utils 3 (its ``merge_annotations`` accepts only
       numeric timestamps, while the rest of the function needed datetimes). Now datetime
       and numeric input both work. Numbers that v1.0.0 would have got wrong are fixed:
       REM latencies used ``timedelta.seconds`` (a latency of -23 min became 85020 s; now
       negative values stay negative), the awakening ignored ``t_sleep_threshold`` units
       (see :func:`get_awakening_time`), it missed a final awakening reached through N1
       (N2 -> N1 -> AWAKE) and then cut the sleep period at an earlier brief awakening,
       and it could be found before the sleep onset (now only after it). The onset and
       awakening windows count only the part of each bout inside them. ``'Arousal'``
       rows were not removed, ``'WAKE'`` (NSRR) was not recognised as ``'AWAKE'``, and a
       night without REM raised ``IndexError`` (now the latencies are NaN). Results that
       depend on unscored epochs or gaps are no longer returned silently (see
       ``on_unscored``).
    """
    where = 'score_night'
    _check_on_unscored(on_unscored, where)
    max_gap_s = _check_max_gap(max_gap_s, where)
    h = _hypnogram(df, where)
    if len(h) == 0:
        raise ValueError(f'{where}: the hypnogram is empty (after removing arousals).')
    b = _merge_bouts(h)
    sleep_complete = bool(b['annotation'].iloc[0] == 'AWAKE' == b['annotation'].iloc[-1])

    variants, spans = _resolutions(b, 'AWAKE', _NIGHT_SLEEP_TAGS, max_gap_s)
    nights = [(label, _score_resolved(v, b, where)) for label, v in variants]
    for key in ('fell_asleep_time', 'awakening_time'):   # not found however scored: raise
        if all(isinstance(n[key][0], _Fail) for _, n in nights):
            raise nights[0][1][key][0].exc
    nan, missing = float('nan'), _missing(b)
    score, ambiguous = {'sleep_complete': sleep_complete}, []
    for key in _NIGHT_FIELDS:
        values = [n[key][0] for _, n in nights]
        if _same(values) and not isinstance(values[0], _Fail):
            score[key] = nights[0][1][key][1]
        else:
            score[key] = missing if key in ('fell_asleep_time', 'awakening_time') else nan
            ambiguous.append(key)
    if ambiguous:
        shown = [(label, ', '.join(f'{key}: ' + ('none' if isinstance(n[key][0], _Fail) else _value_text(n[key][1]))
                                   for key in ambiguous)) for label, n in nights]
        _ambiguous(where, on_unscored, spans, 'result for ' + ', '.join(ambiguous), shown, stacklevel=3)

    if plot == True:
        plot_hypnogram(df)
        plt = _pyplot()
        marks = [t for t in (score['fell_asleep_time'], score['awakening_time']) if not pd.isna(t)]
        if marks:
            plt.stem(marks, [7] * len(marks), linefmt='r', markerfmt='or', basefmt='r')

    return {key: score[key] for key in ('sleep_complete',) + _NIGHT_FIELDS}


def _format_hms(seconds):
    """``[-]HH:MM:SS`` of a number of seconds (keeps the sign, hours may exceed 24)."""
    if seconds is None or not np.isfinite(seconds):
        return 'n/a'
    sign = '-' if seconds < 0 else ''
    hours, remainder = divmod(int(round(abs(seconds))), 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{sign}{hours:02}:{minutes:02}:{secs:02}'


def _format_clock(t):
    if pd.isna(t):
        return 'n/a'
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
    .. note:: **Changed in 3.0.0:**
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
        print('Absolute: {0}  Relative: {1:0.3f}'.format(int(value) if np.isfinite(value) else 'n/a', rel(value)))


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

    Raises
    ------
    TypeError
        With ``start``/``end`` columns: on naive datetimes or other invalid time columns
        (module docstring).
    ValueError
        With ``start``/``end`` columns: if epochs overlap (after removing arousals, unless
        they are listed in ``states``) or ``duration`` does not equal ``end - start``; or if
        ``states`` lists the same state twice.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       Rows are sorted by ``start`` (it used label-based indexing on the row order, which
       broke on a non-default index), and arousal rows (``'Arousal'``/``'Arrousal'``) are
       ignored unless listed in ``states`` (an arousal between two epochs hid their
       transition). ``'WAKE'`` is read as ``'AWAKE'`` in ``hyp`` and in ``states`` (the
       matrix keeps the order and names of ``states``). With ``start``/``end`` columns the
       input is now validated (v1.0.0 ignored the times): naive datetimes raise
       ``TypeError``; overlapping states (e.g. a frame that also holds channel
       annotations) and a ``duration`` that differs from ``end - start`` raise
       ``ValueError``.
    """
    where = 'get_transition_counts'
    states = np.array(states)
    canon_states = _canon_tags(list(states))
    if len(set(canon_states)) != len(canon_states):
        raise ValueError(f'{where}: states {list(states)} name the same state twice '
                         f"('WAKE' is an alias of 'AWAKE').")
    drop_arousals = not np.isin(AROUSAL_TAGS, states).any()
    if 'start' in hyp.columns and 'end' in hyp.columns:
        ann = _hypnogram(hyp, where, drop_arousals=drop_arousals)['annotation'].to_numpy()
    else:
        ann = _canon_series(hyp['annotation'])
        if drop_arousals:
            ann = ann.loc[~ann.isin(AROUSAL_TAGS)]
        ann = ann.to_numpy()
    index = {s: i for i, s in enumerate(canon_states)}
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
        column, seconds by convention). A stage that does not occur has duration 0, so a
        threshold of 0 accepts a hypnogram without that stage.

    Returns
    -------
    list
        Indices of hypnograms in which every listed stage lasts at least its threshold.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       v1.0.0 compared the total duration of the **whole** hypnogram (all stages) with
       each stage's threshold, so e.g. ``{'REM': 3600}`` accepted any night longer than
       1 h that contained any REM. v1.0.0 also rejected a hypnogram in which a listed stage
       does not occur, whatever its threshold; now an absent stage has duration 0, so it
       passes a threshold of 0 (and fails any positive one).
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
    have the same label takes that label. Arousal rows (``'Arousal'``/``'Arrousal'``) are
    events, not stages: they are returned unchanged and are skipped when looking for
    neighbours.

    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation' and 'start'/'end' (or 'duration') columns.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the hypnogram, ``'WAKE'`` returned as ``'AWAKE'``. The input is
        not modified.

    Notes
    -----
    .. note:: **Changed in 3.0.0:**
       It assigned through chained indexing (``df.iloc[k]['annotation'] = ...``), which
       does not write into the frame, so it returned the input unchanged. It now returns a
       filtered copy, in ``start`` order, in which arousal rows are unchanged and are not
       neighbours (a legacy 30-s ``'Arrousal'`` epoch between two N2 epochs stays an
       arousal).
    """
    where = 'do_median_filtration'
    if 'start' in df.columns and 'end' in df.columns:
        h = _correction_input(df, where).drop(columns=['_s', '_e'])
        dur_all = h['duration'].to_numpy(dtype=float)
    else:
        h = df.copy().reset_index(drop=True)
        h['annotation'] = _canon_series(h['annotation'])
        dur_all = _duration_seconds(h['duration'], where)
    ann_all = h['annotation'].to_numpy(dtype=object).copy()
    stages = np.flatnonzero(~h['annotation'].isin(AROUSAL_TAGS).to_numpy())
    ann = ann_all[stages]
    dur = dur_all[stages]
    for k in range(1, len(ann) - 1):
        if ann[k - 1] == ann[k + 1] and abs(dur[k] - 30) <= _DURATION_TOL:
            ann[k] = ann[k - 1]
    ann_all[stages] = ann
    h['annotation'] = ann_all
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
    .. note:: **Changed in 3.0.0:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
    """
    h = _correction_input(df, 'fill_same_voids')
    rows = []
    current_state = _canon(initial_state)
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
    current_state = _canon(initial_state)
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
    .. note:: **Changed in 3.0.0:**
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
    .. note:: **Changed in 3.0.0:**
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
    .. note:: **Changed in 3.0.0:**
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
    .. note:: **Changed in 3.0.0:**
       Rows are processed in ``start`` order (input may be unsorted), datetime input works
       (comparing a timedelta with a number of seconds raised ``TypeError``), the input is
       validated (module docstring) and ``duration`` is returned as ``end - start`` in
       seconds.
    """
    h = _correction_input(df, 'correct_rem')
    rows = []
    current_state = _canon(initial_state)
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
    .. note:: **Changed in 3.0.0:**
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
        dict of a y-axis values for each hypnogram state. The default covers ``'AWAKE'``
        (``'WAKE'`` is drawn as ``'AWAKE'`` unless the dict has a ``'WAKE'`` key),
        ``'Arousal'``, ``'SLP'`` and ``'N'`` (inserted by :func:`fill_sleep_voids` and
        :func:`fill_nonrem_voids`), ``'REM'``, ``'N1'``, ``'N2'``, ``'N3'`` and
        ``'UNKNOWN'`` (CyberPSG's unscored label, drawn at 0).
    hypnogram_colors : dict
        dict of color hex codes for each hypnogram state (default: the same labels)
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
    .. note:: **Changed in 3.0.0:**
       Works with pandas 2 (it used ``DataFrame.append``, removed in pandas 2, and
       called the ``datetime`` module instead of ``datetime.datetime``, so it always
       raised). It no longer adds ``state_id``/``state_color`` columns to the input
       frame, sorts the rows, and handles a one-epoch hypnogram. Night shading is drawn
       per calendar date over the plotted range (it was anchored to each ``day`` group's
       first epoch, so a recording starting after midnight, or a state lasting over
       several days, lost its night shading); a ``day`` column is no longer used.
       Arousals are recognised as ``'Arousal'`` (the brainmaze-utils label, which was drawn
       as a sleep state) and ``'Arrousal'`` (which raised ``KeyError`` with the default
       maps). The default maps also cover ``'N'``, ``'UNKNOWN'`` and ``'WAKE'``, so the
       output of :func:`correct_hypnogram` and of the brainmaze-utils loaders can be
       plotted. Other labels missing from the maps raise ``ValueError`` naming them.
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
        'N': 1.5,
        'N3': 1,
        'UNKNOWN': 0,
    }

    _hypnogram_colors = {
        'AWAKE': '#e7b233',
        'Arousal': '#d44b05',
        'SLP': '#3500d3',
        'REM': '#3500d3',
        'N1': '#2bc7c4',  # 2b7cc7
        'N2': '#2b5dc7',
        'N': '#1a3a80',
        'N3': '#000000',
        'UNKNOWN': '#9e9e9e',
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
    if 'WAKE' not in hypnogram_values:
        orig_df['annotation'] = _canon_series(orig_df['annotation'])
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
