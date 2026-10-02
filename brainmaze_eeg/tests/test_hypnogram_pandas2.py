"""Hypnogram helpers on pandas 2 (``DataFrame.append`` was removed in pandas 2.0).

The plotting test runs matplotlib with the non-interactive Agg backend and is skipped
when matplotlib (optional extra ``[plot]``) is not installed.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from brainmaze_eeg import hypnogram as H

TZ = dt.timezone(dt.timedelta(hours=-5))
T0 = dt.datetime(2024, 1, 1, 22, 0, tzinfo=TZ)

# 30 min AWAKE, 4 cycles of N1 N2 N3 N2 REM (10 min each), 2 h AWAKE
BLOCKS = [('AWAKE', 30)] + [(s, 10) for _ in range(4) for s in ('N1', 'N2', 'N3', 'N2', 'REM')] + [('AWAKE', 120)]


def night(epoch_s=30):
    """Tiled hypnogram (30-s epochs), timezone-aware datetimes."""
    rows, t = [], T0
    for lab, minutes in BLOCKS:
        for _ in range(minutes * 60 // epoch_s):
            rows.append(dict(annotation=lab, start=t, end=t + dt.timedelta(seconds=epoch_s), duration=float(epoch_s)))
            t += dt.timedelta(seconds=epoch_s)
    return pd.DataFrame(rows)


def to_numeric(df):
    df = df.copy()
    df['start'] = [x.timestamp() for x in df.start]
    df['end'] = [x.timestamp() for x in df.end]
    return df


def epochs(spec):
    """[(label, start offset in minutes, duration in minutes)] -> hypnogram."""
    return pd.DataFrame([
        dict(annotation=lab, start=T0 + dt.timedelta(minutes=s), end=T0 + dt.timedelta(minutes=s + d), duration=d * 60.0)
        for lab, s, d in spec
    ])


# ---------------------------------------------------------------- get_number_of_sleep_stages

@pytest.mark.parametrize('numeric', [False, True])
def test_number_of_sleep_stages_counts_rem_cycles(numeric):
    df = night()
    if numeric:
        df = to_numeric(df)
    assert H.get_number_of_sleep_stages(df, 'REM', delay=30) == 4
    # REM periods start every 50 min. The delay is measured from the end of the first
    # epoch of the last counted period (49.5 min to the next period), so with 60 min
    # periods 1 and 3 count, and with 200 min only the first.
    assert H.get_number_of_sleep_stages(df, 'REM', delay=60) == 2
    assert H.get_number_of_sleep_stages(df, 'REM', delay=200) == 1
    assert H.get_number_of_sleep_stages(df, 'N3', delay=30) == 4
    assert H.get_number_of_sleep_stages(df, 'N4', delay=30) == 0


def test_number_of_sleep_stages_delay_is_measured_from_the_counted_epoch():
    # Same rule as before 2.0.1: compare with the end of the epoch that started the
    # previous occurrence.
    df = epochs([('REM', 0, 10), ('REM', 15, 10), ('REM', 45, 10)])
    assert H.get_number_of_sleep_stages(df, 'REM', delay=30) == 2   # 15 < 10+30 ; 45 >= 10+30
    assert H.get_number_of_sleep_stages(df, 'REM', delay=36) == 1   # 45 - 10 = 35 < 36


def test_number_of_sleep_stages_several_tags():
    # Before 2.0.1 several tags were AND-ed (an epoch had to equal all of them): always 0.
    df = epochs([('N3', 0, 5), ('REM', 60, 5), ('N2', 120, 5), ('N3', 180, 5)])
    assert H.get_number_of_sleep_stages(df, ['REM', 'N3'], delay=30) == 3
    assert H.get_number_of_sleep_stages(df, ['REM'], delay=30) == 1


def test_number_of_sleep_stages_keeps_days_of_the_gap():
    # 1 day + 1 min apart: timedelta.seconds (the old code) saw 60 s < 30 min and
    # counted 1.
    df = epochs([('REM', 0, 5), ('REM', 5 + 24 * 60 + 1, 5)])
    assert H.get_number_of_sleep_stages(df, 'REM', delay=30) == 2


# ---------------------------------------------------------------- plot_hypnogram

@pytest.fixture
def plt():
    pytest.importorskip('matplotlib')
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    yield plt
    plt.close('all')


def _segments(ax):
    from matplotlib.collections import PolyCollection
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


@pytest.mark.parametrize('with_day', [False, True])
def test_plot_hypnogram_draws_merged_segments(plt, with_day):
    import matplotlib.dates as mdates
    from brainmaze_utils.annotations import create_day_indexes

    df = night()
    if with_day:
        df = create_day_indexes(df)
    before = df.copy()

    H.plot_hypnogram(df)

    ax = plt.gca()
    # one filled column per run of identical touching epochs: 1 + 4 * 5 + 1
    assert len(_segments(ax)) == 22
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(mdates.date2num(df.start.iloc[0]))
    assert hi == pytest.approx(mdates.date2num(df.end.iloc[-1]))
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ['AWAKE', 'Arousal', 'SLP', 'REM', 'N1', 'N2', 'N3']
    # the night shading (22:00 + 12 h) is drawn once for the single day
    spans = [p for p in ax.patches if p.get_facecolor()[:3] == pytest.approx((0.5, 0.5, 0.5), abs=0.01)]
    assert len(spans) == 1
    # the caller's frame is not modified
    pd.testing.assert_frame_equal(df, before)
    plt.gcf().canvas.draw()  # renders without error


def test_plot_hypnogram_one_epoch(plt):
    H.plot_hypnogram(epochs([('N2', 0, 10)]))
    assert len(_segments(plt.gca())) == 1


def test_plot_hypnogram_rejects_numeric_and_empty(plt):
    with pytest.raises(TypeError, match='timezone-aware'):
        H.plot_hypnogram(to_numeric(night()))
    with pytest.raises(ValueError, match='empty'):
        H.plot_hypnogram(night().iloc[:0])
