"""Correct numbers (or an error) from the hypnogram scoring functions (3.0.0).

Expected values are computed by hand in the comments. ``tile`` builds a hypnogram of 30-s
epochs, ``bouts`` the same hypnogram with one row per bout; both must give the same result.
"""
import contextlib
import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest
from dateutil import tz as dtz

from brainmaze_eeg import hypnogram as H

CT = dtz.gettz('America/Chicago')
T0 = dt.datetime(2024, 1, 1, 22, 0, tzinfo=CT)  # CST, no DST change in January


def bouts(spec, t0=T0):
    """[(label, minutes), ...] back to back -> one row per bout."""
    rows, t = [], t0
    for lab, minutes in spec:
        e = t + dt.timedelta(minutes=minutes)
        rows.append(dict(annotation=lab, start=t, end=e, duration=minutes * 60.0))
        t = e
    return pd.DataFrame(rows)


def tile(spec, t0=T0, epoch_s=30):
    """Same as ``bouts`` but tiled into ``epoch_s`` epochs."""
    rows, t = [], t0
    for lab, minutes in spec:
        for _ in range(int(round(minutes * 60 / epoch_s))):
            e = t + dt.timedelta(seconds=epoch_s)
            rows.append(dict(annotation=lab, start=t, end=e, duration=float(epoch_s)))
            t = e
    return pd.DataFrame(rows)


def numeric(df):
    df = df.copy()
    df['start'] = [x.timestamp() for x in df.start]
    df['end'] = [x.timestamp() for x in df.end]
    return df


def shuffled(df):
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def at(minutes):
    return T0 + dt.timedelta(minutes=minutes)


BUILDERS = [pytest.param(bouts, id='bouts'), pytest.param(tile, id='tiled'),
            pytest.param(lambda s: shuffled(tile(s)), id='tiled-shuffled'),
            pytest.param(lambda s: numeric(tile(s)), id='tiled-numeric')]


def _t(df, minutes):
    """Expected time in the representation of df's start column."""
    t = at(minutes)
    return t.timestamp() if isinstance(df['start'].iloc[0], (float, np.floating)) else t


# ---------------------------------------------------------------- R1/R2/R3: number of stages

@pytest.mark.parametrize('build', BUILDERS)
@pytest.mark.parametrize('spec, delay, expected', [
    ([('REM', 40)], 30, 1),                                 # one bout (v1.0.0 on epochs: 2)
    ([('REM', 90)], 30, 1),                                 # one bout (v1.0.0 on epochs: 3)
    ([('REM', 40), ('N2', 20), ('REM', 40)], 30, 1),        # 2nd starts 20 min after 1st ends
    ([('REM', 40), ('N2', 30), ('REM', 40)], 30, 2),        # exactly `delay` -> counts
    ([('REM', 40), ('N2', 29.5), ('REM', 40)], 30, 1),      # 30 s short
    ([('REM', 45), ('N2', 60), ('REM', 10)], 30, 2),
    # bouts at 0-10, 15-25, 45-55: the uncounted 2nd bout does not move the reference
    ([('REM', 10), ('N2', 5), ('REM', 10), ('N2', 20), ('REM', 10)], 30, 2),
    ([('REM', 10), ('N2', 5), ('REM', 10), ('N2', 20), ('REM', 10)], 36, 1),
    ([('N2', 30)], 30, 0),
])
def test_number_of_sleep_stages_counts_bouts(build, spec, delay, expected):
    assert H.get_number_of_sleep_stages(build(spec), 'REM', delay=delay) == expected


def test_number_of_sleep_stages_unsorted_and_overlapping():
    df = bouts([('REM', 5), ('N2', 55), ('REM', 5), ('N2', 55), ('REM', 5)])
    assert H.get_number_of_sleep_stages(df, 'REM', 30) == 3
    assert H.get_number_of_sleep_stages(df.iloc[[4, 0, 2, 1, 3]], 'REM', 30) == 3
    # overlapping REM epochs (two scorers) form one bout
    ov = pd.DataFrame([dict(annotation='REM', start=at(0), end=at(30)),
                       dict(annotation='REM', start=at(10), end=at(15))])
    assert H.get_number_of_sleep_stages(ov, 'REM', 0) == 1


def test_number_of_sleep_stages_long_delays_and_gaps():
    two_h = bouts([('REM', 5), ('N2', 115), ('REM', 5)])
    assert H.get_number_of_sleep_stages(two_h, 'REM', delay=1440) == 1   # v1.0.0: 2
    assert H.get_number_of_sleep_stages(two_h, 'REM', delay=1500) == 1
    day = bouts([('REM', 5), ('N2', 1440 + 40), ('REM', 5)])
    assert H.get_number_of_sleep_stages(day, 'REM', delay=1440) == 2
    assert H.get_number_of_sleep_stages(day, 'REM', delay=1481) == 1
    with pytest.raises(ValueError, match='delay'):
        H.get_number_of_sleep_stages(two_h, 'REM', delay=-1)


def test_number_of_sleep_stages_several_tags_form_one_class():
    df = bouts([('N3', 10), ('REM', 10), ('N2', 40), ('N3', 10)])
    # N3+REM touching -> one bout (0-20); N3 at 60 is 40 min later
    assert H.get_number_of_sleep_stages(df, ['REM', 'N3'], 30) == 2
    assert H.get_number_of_sleep_stages(df, ['REM', 'N3'], 0) == 2


# ---------------------------------------------------------------- input validation (R3)

def test_millisecond_timestamps_raise():
    df = numeric(tile([('REM', 10), ('N2', 40), ('REM', 10)]))
    ms = df.assign(start=df.start * 1000, end=df.end * 1000)
    with pytest.raises(ValueError, match='milliseconds'):
        H.get_number_of_sleep_stages(ms, 'REM', 30)
    # relative ms timestamps cannot be told apart by size, but the duration (s) disagrees
    rel = df.assign(start=(df.start - df.start[0]) * 1000, end=(df.end - df.start[0]) * 1000)
    with pytest.raises(ValueError, match='duration'):
        H.get_number_of_sleep_stages(rel, 'REM', 30)
    with pytest.raises(ValueError, match='milliseconds'):
        H.score_night(ms)


def test_naive_and_mixed_times_raise():
    df = tile([('AWAKE', 30), ('N2', 60)])
    naive = df.assign(start=[t.replace(tzinfo=None) for t in df.start],
                      end=[t.replace(tzinfo=None) for t in df.end])
    for f in (H.get_number_of_sleep_stages, H.get_hypnogram_datarate, H.score_night):
        with pytest.raises(TypeError, match='naive'):
            f(naive)
    mixed = df.assign(start=[t.timestamp() for t in df.start])
    with pytest.raises(TypeError):
        H.get_hypnogram_datarate(mixed)


def test_overlapping_states_raise_in_scoring():
    df = bouts([('AWAKE', 30), ('N2', 60), ('REM', 20), ('AWAKE', 100)])
    df.loc[len(df)] = dict(annotation='N3', start=at(40), end=at(50), duration=600.0)
    for f in (H.score_night, H.get_fell_asleep_time, H.get_hypnogram_datarate):
        with pytest.raises(ValueError, match='overlap'):
            f(df)


# ---------------------------------------------------------------- R7: data rate

@pytest.mark.parametrize('conv', [lambda d: d, numeric, shuffled], ids=['datetime', 'numeric', 'shuffled'])
def test_datarate_over_24h(conv):
    assert H.get_hypnogram_datarate(conv(tile([('N2', 25 * 60)]))) == pytest.approx(1.0)   # v1.0.0: 25.0
    gappy = tile([('N2', 25 * 60)])
    gappy = pd.concat([gappy.iloc[: len(gappy) // 2], gappy.iloc[-1:]])
    # 12.5 h + 30 s scored over a 25 h span
    assert H.get_hypnogram_datarate(conv(gappy)) == pytest.approx((12.5 * 3600 + 30) / (25 * 3600))


def test_datarate_ignores_arousal_events():
    df = bouts([('N2', 60)])
    df.loc[1] = dict(annotation='Arousal', start=at(10), end=at(11), duration=60.0)
    assert H.get_hypnogram_datarate(df) == pytest.approx(1.0)


# ---------------------------------------------------------------- R8/R9: onset and awakening

@pytest.mark.parametrize('build', BUILDERS)
def test_awakening_sleep_threshold_is_minutes(build):
    # sleep ends 03:30 (t=330), then AWAKE 40, N2 1 (<= 10 min), AWAKE 120 -> 03:30
    df = build([('AWAKE', 30), ('N2', 300), ('AWAKE', 40), ('N2', 1), ('AWAKE', 120)])
    assert H.get_awakening_time(df) == _t(df, 330)     # v1.0.0 on bouts: 04:11 (t=371)
    # 11 min of N2 exceeds the threshold -> next candidate at t=330+40+11
    df = build([('AWAKE', 30), ('N2', 300), ('AWAKE', 40), ('N2', 11), ('AWAKE', 120)])
    assert H.get_awakening_time(df) == _t(df, 381)


@pytest.mark.parametrize('label', ['Arousal', 'Arrousal'])
def test_arousal_labels_are_removed(label):
    # an arousal epoch between AWAKE and N2 used to hide the onset (fallback: 22:00)
    df = tile([('AWAKE', 30), (label, 0.5), ('N2', 300), ('AWAKE', 100)])
    assert H.get_fell_asleep_time(df) == at(30.5)
    # arousal as an overlay event (overlapping N2) is removed before the overlap check
    ov = bouts([('AWAKE', 30), ('N2', 300), ('AWAKE', 100)])
    ov.loc[3] = dict(annotation=label, start=at(100), end=at(100.25), duration=15.0)
    assert H.get_fell_asleep_time(ov) == at(30)
    assert H.score_night(ov)['n2_sleep_time'] == 300 * 60


def test_fell_asleep_without_onset_raises():
    # every sleep bout is followed by >= 10 min awake within 60 min: no onset
    df = bouts([('AWAKE', 30), ('N2', 5), ('AWAKE', 30), ('N2', 5), ('AWAKE', 30)])
    with pytest.raises(ValueError, match='no sleep onset'):   # v1.0.0: returned 22:00
        H.get_fell_asleep_time(df)
    with pytest.raises(ValueError, match='no sleep onset'):
        H.score_night(df)


def test_awakening_is_searched_after_onset():
    # nap 30-35, 100 min awake, sleep 135-435, awake: the nap's awakening (t=35) qualifies
    # on its own, but score_night must use the one after the sleep onset (t=135)
    df = bouts([('AWAKE', 30), ('N2', 5), ('AWAKE', 100), ('N2', 300), ('AWAKE', 100)])
    assert H.get_fell_asleep_time(df) == at(135)
    assert H.get_awakening_time(df) == at(35)
    assert H.get_awakening_time(df, after=at(135)) == at(435)
    assert H.score_night(df)['awakening_time'] == at(435)
    assert H.score_night(df)['n2_sleep_time'] == 300 * 60


def test_awakening_without_transition_raises_and_still_asleep_returns_end():
    with pytest.raises(ValueError, match='no awakening'):
        H.get_awakening_time(bouts([('AWAKE', 30), ('UNKNOWN', 10), ('AWAKE', 100)]))
    with pytest.raises(ValueError, match='no awakening'):   # N1 not a sleep tag, no N2 before it
        H.get_awakening_time(bouts([('AWAKE', 30), ('N1', 10), ('AWAKE', 100)]),
                             sleep_cycle_tags=['REM', 'N2', 'N3'])
    # N1 is a sleep stage by default (3.0.0): N1 -> AWAKE is a transition
    assert H.get_awakening_time(bouts([('AWAKE', 30), ('N1', 10), ('AWAKE', 100)])) == at(40)
    assert H.get_awakening_time(bouts([('AWAKE', 30), ('N2', 300)])) == at(330)


# ---------------------------------------------------------------- other functions

def test_is_sleep_complete_and_awakenings_use_time_order():
    df = bouts([('AWAKE', 30), ('N2', 60), ('AWAKE', 10), ('N2', 60), ('AWAKE', 30)])
    rev = df.iloc[::-1]   # reversed rows, non-default index
    assert H.is_sleep_complete(rev) is True
    assert H.get_number_of_awakenings(rev) == 2
    with_arousal = tile([('N2', 10), ('Arousal', 0.5), ('AWAKE', 10), ('N2', 10)])
    assert H.get_number_of_awakenings(with_arousal) == 1   # v1.0.0: 0
    with pytest.raises(ValueError, match='empty'):
        H.is_sleep_complete(df.iloc[:0])


def test_transition_counts_time_order_and_arousals():
    df = bouts([('AWAKE', 10), ('N1', 10), ('N2', 10), ('REM', 10)])
    expected = H.get_transition_counts(df)
    assert expected.sum() == 3 and expected[0, 1] == expected[1, 2] == expected[2, 4] == 1
    np.testing.assert_array_equal(H.get_transition_counts(df.iloc[[3, 1, 0, 2]]), expected)
    np.testing.assert_array_equal(H.get_transition_counts(df.set_index(pd.Index([7, 8, 9, 10]))), expected)
    arousal = tile([('N2', 1), ('Arousal', 0.5), ('N2', 1)])
    assert H.get_transition_counts(arousal)[2, 2] == 3   # N2->N2 across the removed arousal


def test_valid_dataset_index_by_duration_uses_stage_time():
    a = bouts([('N2', 120), ('REM', 10)])
    b = bouts([('N2', 60), ('REM', 70)])
    # v1.0.0 compared the whole hypnogram (130 min) with each threshold -> [0, 1]
    assert H.valid_dataset_index_by_duration([a, b], {'REM': 3600}) == [1]
    assert H.valid_dataset_index_by_duration([a, b], {'N3': 1}) == []
    assert H.valid_dataset_index_by_duration([a, b], {'N2': 3600, 'REM': 600}) == [0, 1]


def test_median_filtration_replaces_isolated_epochs():
    df = tile([('N2', 5), ('REM', 0.5), ('N2', 5), ('REM', 1), ('N2', 1)])
    out = H.do_median_filtration(df)
    assert (out.annotation == 'N2').sum() == 23     # 22 + the single REM epoch
    assert (out.annotation == 'REM').sum() == 2     # the 1-min REM is kept
    assert (df.annotation == 'REM').sum() == 3      # input unchanged


@pytest.mark.parametrize('conv', [lambda d: d, numeric], ids=['datetime', 'numeric'])
def test_fill_voids_durations_and_datetimes(conv):
    df = conv(pd.DataFrame([
        dict(annotation='AWAKE', start=at(0), end=at(10), duration=600.0),
        dict(annotation='REM', start=at(12), end=at(42), duration=1800.0),   # 2-min void
    ]))
    out = H.fill_wakerem_voids(df, time_threshold=5 * 60)
    assert list(out.annotation) == ['AWAKE', 'AWAKE', 'REM']
    assert list(out.duration) == [600.0, 120.0, 1800.0]   # v1.0.0: void got 1800
    same = conv(pd.DataFrame([dict(annotation='N2', start=at(0), end=at(10), duration=600.0),
                              dict(annotation='N2', start=at(11), end=at(20), duration=540.0)]))
    filled = H.fill_same_voids(same, time_threshold=120)
    assert list(filled.duration) == [600.0, 600.0]
    corrected = H.correct_hypnogram(conv(bouts([('AWAKE', 30), ('REM', 10), ('N2', 30)])))
    assert list(corrected.annotation) == ['AWAKE', 'AWAKE', 'N2']   # REM right after 30 min awake


# ---------------------------------------------------------------- R10: score_night end to end

# Two nights and the day between them (minutes from 22:00 Jan 1, America/Chicago).
NIGHTS = [
    ('AWAKE', 30),   # 0-30
    ('N1', 5),       # 30-35
    ('REM', 3),      # 35-38    early REM, before the sustained sleep onset
    ('AWAKE', 20),   # 38-58
    ('N1', 2),       # 58-60    sleep onset (22:58)
    ('N2', 60),      # 60-120
    ('N3', 30),      # 120-150
    ('REM', 20),     # 150-170  REM bout 1 (counted)
    ('AWAKE', 5),    # 170-175  awakening 1
    ('N2', 40),      # 175-215
    ('REM', 15),     # 215-230  REM bout 2: 45 min after bout 1 ended -> counted
    ('N2', 10),      # 230-240
    ('REM', 10),     # 240-250  REM bout 3: 10 min after bout 2 ended -> not counted
    ('N2', 60),      # 250-310
    ('AWAKE', 1130), # 310-1440 final awakening 03:10 Jan 2, awake all day
    ('N2', 120),     # 1440-1560 second night (22:00 Jan 2)
    ('REM', 20),     # 1560-1580
    ('AWAKE', 60),   # 1580-1640
]
EXPECTED = {
    # onset: the AWAKE->N1 at 30 has 20 min AWAKE (38-58) in its 60-min window -> rejected;
    # the AWAKE->N1 at 58 has none -> 22:58.
    'fell_asleep_time': 58,
    # sleep->AWAKE after onset: at 170 only 5 min awake in 90 min; at 310 the AWAKE bout
    # lasts 1130 min -> 03:10.
    'awakening_time': 310,
    'sleep_complete': True,
    'rem_latency_fell_asleep': (35 - 58) * 60.0,  # -1380 s (v1.0.0: .seconds -> 85020)
    'rem_latency_last_awake': (35 - 30) * 60.0,   # first REM at 35, last AWAKE ended at 30
    'n_complete_sleep_cycles': 2,
    'n_awakenings': 1,
    # sleep period = bouts starting in [58, 310)
    'n1_sleep_time': 2 * 60.0,
    'n2_sleep_time': (60 + 40 + 10 + 60) * 60.0,
    'n3_sleep_time': 30 * 60.0,
    'rem_sleep_time': (20 + 15 + 10) * 60.0,
    'awake_sleep_time': 5 * 60.0,
}


def _with_arousal_events(df):
    """Add overlay arousal events (both spellings) inside N2; they must be ignored."""
    ev = pd.DataFrame([dict(annotation='Arousal', start=at(90), end=at(90.25), duration=15.0),
                       dict(annotation='Arrousal', start=at(200), end=at(200.5), duration=30.0)])
    return pd.concat([df, ev], ignore_index=True)


@pytest.mark.parametrize('build', BUILDERS + [
    pytest.param(lambda s: shuffled(_with_arousal_events(tile(s))), id='tiled-shuffled-arousals'),
    pytest.param(lambda s: tile(s).assign(start=lambda d: pd.to_datetime(d.start, utc=True).dt.tz_convert('America/Chicago'),
                                          end=lambda d: pd.to_datetime(d.end, utc=True).dt.tz_convert('America/Chicago')),
                 id='tiled-datetime64tz'),
])
def test_score_night_two_nights(build):
    df = build(NIGHTS)
    before = df.copy()
    score = H.score_night(df)
    expected = dict(EXPECTED)
    expected['fell_asleep_time'] = _t(df, EXPECTED['fell_asleep_time'])
    expected['awakening_time'] = _t(df, EXPECTED['awakening_time'])
    assert set(score) == set(expected)
    for key, value in expected.items():
        assert score[key] == value, key
    pd.testing.assert_frame_equal(df, before)


def test_score_night_without_rem_gives_nan_latency():
    score = H.score_night(bouts([('AWAKE', 30), ('N2', 300), ('AWAKE', 100)]))
    assert np.isnan(score['rem_latency_fell_asleep']) and np.isnan(score['rem_latency_last_awake'])
    assert score['n_complete_sleep_cycles'] == 0


def test_rem_latency_keeps_sign():
    lat = H.get_rem_latency(tile(NIGHTS))
    assert lat['fall_asleep'] == dt.timedelta(minutes=-23)
    assert lat['last_awake'] == dt.timedelta(minutes=5)
    assert H.get_rem_latency(numeric(tile(NIGHTS)))['fall_asleep'] == -1380.0
    with pytest.raises(ValueError, match='REM'):
        H.get_rem_latency(bouts([('AWAKE', 30), ('N2', 300)]))


def test_print_sleep_score(capsys):
    H.print_sleep_score(H.score_night(tile(NIGHTS)))
    out = capsys.readouterr().out
    assert 'Falling asleep:  22:58:00' in out
    assert 'Awakening:  03:10:00' in out
    assert 'Sleep period (onset to awakening): 04:12:00' in out
    assert 'Rem latency - fall_asleep: -00:23:00' in out
    assert 'Rem latency - last_awake: 00:05:00' in out
    # relative REM time = 45 min / 252 min
    assert 'Absolute: 2700  Relative: {:0.3f}'.format(45 / 252) in out
    H.print_sleep_score(H.score_night(numeric(tile(NIGHTS))))
    assert '-00:23:00' in capsys.readouterr().out


def test_print_sleep_score_over_a_day():
    score = dict(H.score_night(bouts([('AWAKE', 30), ('N2', 300), ('AWAKE', 100)])),
                 awakening_time=at(30 + 25 * 60))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        H.print_sleep_score(score)
    assert 'Sleep period (onset to awakening): 25:00:00' in buf.getvalue()
    assert 'Rem latency - fall_asleep: n/a' in buf.getvalue()


# ---------------------------------------------------------------- R4/R5/R9: plot

@pytest.fixture
def plt():
    pytest.importorskip('matplotlib')
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    yield plt
    plt.close('all')


def _night_spans(ax, tzinfo):
    import matplotlib.dates as mdates
    out = []
    for p in ax.patches:
        if p.get_facecolor()[:3] == pytest.approx((0.5, 0.5, 0.5), abs=0.01):
            xy = np.atleast_2d(np.asarray(p.get_xy()))
            x0, x1 = (p.get_x(), p.get_x() + p.get_width()) if hasattr(p, 'get_width') else (xy[:, 0].min(), xy[:, 0].max())
            out.append((mdates.num2date(x0, tz=tzinfo).strftime('%m-%d %H:%M'),
                        mdates.num2date(x1, tz=tzinfo).strftime('%m-%d %H:%M')))
    return sorted(out)


def _segments(ax):
    from matplotlib.collections import PolyCollection
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_plot_night_shading_after_midnight(plt):
    # 01:00-07:50 Jan 2: the night of Jan 1 (22:00 -> 10:00) covers it (was shaded at Jan 2 22:00)
    df = tile([('N2', 60), ('REM', 20)] * 5, t0=dt.datetime(2024, 1, 2, 1, tzinfo=CT))
    H.plot_hypnogram(df)
    assert _night_spans(plt.gca(), CT) == [('01-02 01:00', '01-02 07:40')]


def test_plot_night_shading_long_state_over_days(plt):
    # N2 22:00 Jan 1 -> 06:00, AWAKE 06:00 Jan 2 -> 10:00 Jan 3 (merged into one row), N2 -> 15:00
    df = tile([('N2', 8 * 60), ('AWAKE', 28 * 60), ('N2', 300)])
    H.plot_hypnogram(df)
    assert _night_spans(plt.gca(), CT) == [('01-01 22:00', '01-02 10:00'), ('01-02 22:00', '01-03 10:00')]


def test_plot_night_shading_uses_data_timezone(plt):
    df = tile([('N2', 60)] * 4)   # 22:00-02:00 CST = 04:00-08:00 UTC
    utc = df.assign(start=[t.astimezone(dt.timezone.utc) for t in df.start],
                    end=[t.astimezone(dt.timezone.utc) for t in df.end])
    H.plot_hypnogram(utc)
    # shaded at 22:00 UTC (wall clock of the data); 04:00-08:00 UTC lies in 22:00 Jan 1 -> 10:00 Jan 2 UTC
    assert _night_spans(plt.gca(), dt.timezone.utc) == [('01-02 04:00', '01-02 08:00')]


def test_plot_unsorted_equals_sorted(plt):
    df = tile([('AWAKE', 30)] + [('N2', 60), ('REM', 20)] * 4 + [('AWAKE', 60)])
    H.plot_hypnogram(df)
    n_sorted = len(_segments(plt.gca()))
    plt.close('all')
    H.plot_hypnogram(shuffled(df))
    assert n_sorted == 10 and len(_segments(plt.gca())) == n_sorted   # was 819


@pytest.mark.parametrize('label', ['Arousal', 'Arrousal'])
def test_plot_arousal_is_an_event_not_a_state(plt, label):
    df = tile([('N2', 30), ('REM', 10)])
    df.loc[len(df)] = dict(annotation=label, start=at(5), end=at(5.5), duration=30.0)   # overlay
    H.plot_hypnogram(df)
    # N2, REM and the arousal event; the arousal does not split N2 or enter the step line
    assert len(_segments(plt.gca())) == 3
    plt.close('all')
    legacy_maps = dict(hypnogram_values={'AWAKE': 6, 'Arrousal': 5, 'REM': 4, 'N1': 3, 'N2': 2, 'N3': 1},
                       hypnogram_colors={'AWAKE': 'y', 'Arrousal': 'r', 'REM': 'b', 'N1': 'c', 'N2': 'g', 'N3': 'k'})
    H.plot_hypnogram(df, **legacy_maps)
    assert len(_segments(plt.gca())) == 3


def test_plot_unknown_label_raises(plt):
    df = tile([('N2', 30), ('IED', 1)])
    with pytest.raises(ValueError, match='IED'):
        H.plot_hypnogram(df)


def test_score_night_plot(plt):
    score = H.score_night(tile(NIGHTS), plot=True)
    assert score['n_complete_sleep_cycles'] == 2
    plt.gcf().canvas.draw()


# ---------------------------------------------------------------- round 3 (V1-V8)

# V1: the final awakening reached through N1. Minutes from 22:00:
# AWAKE 0-30, N2 30-90, REM 90-110, AWAKE 110-115, N2 115-235, N1 235-240, AWAKE 240-360.
N1_EXIT = [('AWAKE', 30), ('N2', 60), ('REM', 20), ('AWAKE', 5), ('N2', 120), ('N1', 5), ('AWAKE', 120)]


@pytest.mark.parametrize('build', BUILDERS)
def test_final_awakening_through_n1(build):
    df = build(N1_EXIT)
    # the AWAKE at 110 holds 95 min of sleep in its window -> rejected; the AWAKE at 240
    # (N2 -> N1 -> AWAKE) has 100 min awake -> 02:00. 6589abb: 23:50 (t=110).
    assert H.get_awakening_time(df) == _t(df, 240)
    # the v1.0.0 default tags (no N1) still find it: N1 bouts are passed through
    assert H.get_awakening_time(df, sleep_cycle_tags=['REM', 'N2', 'N3']) == _t(df, 240)
    score = H.score_night(df)
    assert score['fell_asleep_time'] == _t(df, 30)
    assert score['awakening_time'] == _t(df, 240)
    assert score['n2_sleep_time'] == (60 + 120) * 60.0     # 10800 s (6589abb: 3600)
    assert score['n1_sleep_time'] == 5 * 60.0
    assert score['rem_sleep_time'] == 20 * 60.0
    assert score['awake_sleep_time'] == 5 * 60.0
    assert score['n_awakenings'] == 1                      # 6589abb: 0


@pytest.mark.parametrize('build', BUILDERS)
def test_final_awakening_through_n1_fallback_and_rem(build):
    # the night ends N2 -> N1 -> AWAKE 60 (shorter than 90 min): no candidate qualifies,
    # the night ends awake -> the last sleep->wake transition (t=340), not the brief
    # awakening at t=230 (6589abb).
    df = build([('AWAKE', 30), ('N2', 200), ('AWAKE', 5), ('N2', 100), ('N1', 5), ('AWAKE', 60)])
    assert H.get_awakening_time(df) == _t(df, 340)
    assert H.score_night(df)['n2_sleep_time'] == 300 * 60.0
    # REM -> N1 -> N1 (two bouts: a gap of 30 s splits them) -> AWAKE
    df = build([('AWAKE', 30), ('N2', 200), ('REM', 20), ('N1', 3), ('AWAKE', 100)])
    assert H.get_awakening_time(df, sleep_cycle_tags=['REM', 'N2', 'N3']) == _t(df, 253)


@pytest.mark.parametrize('build', BUILDERS)
def test_onset_window_clips_bouts(build):
    # V7: AWAKE 0-30, N2 30-85, AWAKE 85-100, N2 100-300. Only 85-90 (5 min) of the AWAKE
    # bout lies in the 60-min window after 30 -> onset 22:30 (6589abb: 23:40).
    df = build([('AWAKE', 30), ('N2', 55), ('AWAKE', 15), ('N2', 200)])
    assert H.get_fell_asleep_time(df) == _t(df, 30)
    # 59 min of N2, then AWAKE: 1 min of AWAKE in the window -> onset (6589abb: raised)
    df = build([('N2', 59), ('AWAKE', 120)])
    assert H.get_fell_asleep_time(df) == _t(df, 0)
    # 11 min of AWAKE inside the window still rejects the candidate
    df = build([('AWAKE', 30), ('N2', 49), ('AWAKE', 15), ('N2', 200)])
    assert H.get_fell_asleep_time(df) == _t(df, 94)


def test_awakening_window_clips_bouts():
    # AWAKE 0-30, N2 30-330, AWAKE 330-360, gap (unscored) 360-410, AWAKE 410-470.
    # The 100-min window after 330 holds 30 + 20 = 50 min of AWAKE -> not sustained; the
    # night ends awake and the AWAKE at 410 follows a gap, not sleep -> the last candidate.
    # 6589abb counted the AWAKE bout starting at 410 in full (30 + 60 = 90) -> also 330, but
    # via the window; check the window directly with a later sleep candidate:
    df = pd.concat([bouts([('AWAKE', 30), ('N2', 300), ('AWAKE', 30)]),
                    bouts([('AWAKE', 60), ('N2', 30), ('AWAKE', 100)], t0=at(410))], ignore_index=True)
    # candidates: 330 (50 min awake in its window) and 500 (100 min awake) -> 500
    assert H.get_awakening_time(df) == at(500)


def test_after_must_match_the_time_kind():
    df = bouts([('AWAKE', 30), ('N2', 60), ('AWAKE', 100)])
    num = numeric(df)
    # V2: a datetime `after` on numeric times filtered out every candidate (-> end of the
    # recording); a number on datetime times was ignored. Both raise now.
    with pytest.raises(TypeError, match='after'):
        H.get_awakening_time(num, after=pd.Timestamp(at(10)))
    with pytest.raises(TypeError, match='after'):
        H.get_awakening_time(df, after=at(10).timestamp())
    with pytest.raises(TypeError, match='naive'):
        H.get_awakening_time(df, after=pd.Timestamp('2024-01-01 22:10'))
    assert H.get_awakening_time(num, after=at(10).timestamp()) == at(90).timestamp()
    assert H.get_awakening_time(df, after=pd.Timestamp(at(10))) == at(90)
    assert H.get_awakening_time(df, after=at(10).astimezone(dt.timezone.utc)) == at(90)


# V3: 'WAKE' (brainmaze_utils load_NSRR) is 'AWAKE'. Relative seconds from 0:
# AWAKE 0-30, N1 30-35, N2 35-95, N3 95-125, REM 125-145, AWAKE 145-150, N2 150-240,
# N1 240-245, AWAKE 245-365 (minutes).
NSRR_NIGHT = [('AWAKE', 30), ('N1', 5), ('N2', 60), ('N3', 30), ('REM', 20), ('AWAKE', 5),
              ('N2', 90), ('N1', 5), ('AWAKE', 120)]
NSRR_EXPECTED = {
    'sleep_complete': True,
    'fell_asleep_time': 30 * 60.0,
    'awakening_time': 245 * 60.0,           # N2 -> N1 -> AWAKE at 245
    'rem_latency_fell_asleep': 95 * 60.0,   # REM at 125 - onset 30
    'rem_latency_last_awake': 95 * 60.0,    # REM at 125 - end of the AWAKE at 30
    'n_complete_sleep_cycles': 1,
    'n_awakenings': 1,                      # REM -> AWAKE at 145
    'n1_sleep_time': 10 * 60.0,
    'n2_sleep_time': 150 * 60.0,
    'n3_sleep_time': 30 * 60.0,
    'rem_sleep_time': 20 * 60.0,
    'awake_sleep_time': 5 * 60.0,
}


def _nsrr_file(path, spec):
    concept = {'AWAKE': 'Wake|0', 'N1': 'Stage 1 sleep|1', 'N2': 'Stage 2 sleep|2',
               'N3': 'Stage 3 sleep|3', 'REM': 'REM sleep|5'}
    events, t = [], 0.0
    for lab, minutes in spec:
        events.append(f'<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>{concept[lab]}'
                      f'</EventConcept><Start>{t}</Start><Duration>{minutes * 60.0}</Duration></ScoredEvent>')
        t += minutes * 60
    path.write_text('<?xml version="1.0" encoding="UTF-8"?><PSGAnnotation><SoftwareVersion>Compumedics'
                    '</SoftwareVersion><EpochLength>30</EpochLength><ScoredEvents>' + ''.join(events)
                    + '</ScoredEvents></PSGAnnotation>')
    return str(path)


def test_wake_alias_with_load_nsrr(tmp_path):
    from brainmaze_utils.annotations import load_NSRR
    ns = load_NSRR(_nsrr_file(tmp_path / 'night.xml', NSRR_NIGHT))
    assert 'WAKE' in set(ns.annotation) and 'AWAKE' not in set(ns.annotation)
    before = ns.copy()
    # 6589abb: no sleep onset (ValueError); awakening = end of the recording; is_sleep_complete
    # False; 0 awakenings; no W transitions.
    assert H.score_night(ns) == NSRR_EXPECTED
    assert H.get_fell_asleep_time(ns) == 30 * 60.0
    assert H.get_awakening_time(ns) == 245 * 60.0
    assert H.is_sleep_complete(ns) is True
    assert H.get_number_of_awakenings(ns) == 2
    assert H.get_number_of_awakenings(ns[['annotation']]) == 2
    assert H.get_rem_latency(ns) == {'last_awake': 95 * 60.0, 'fall_asleep': 95 * 60.0}
    assert H.get_time_by_key(ns, 'AWAKE') == 155 * 60.0
    assert H.get_time_by_key(ns, ['WAKE', 'N1']) == 165 * 60.0
    assert H.valid_dataset_index_by_duration([ns], {'AWAKE': 155 * 60}) == [0]
    m = H.get_transition_counts(ns)
    awake, n1, n2, n3, rem = range(5)
    assert m[awake, n1] == 1 and m[rem, awake] == 1 and m[n1, awake] == 1 and m[awake, n2] == 1
    assert m.sum() == len(NSRR_NIGHT) - 1
    np.testing.assert_array_equal(H.get_transition_counts(ns, ['WAKE', 'N1', 'N2', 'N3', 'REM']), m)
    with pytest.raises(ValueError, match='twice'):
        H.get_transition_counts(ns, ['WAKE', 'AWAKE', 'N2'])
    # 'WAKE' as a tag argument is the same as 'AWAKE'
    same = ns.assign(annotation=ns.annotation.replace({'WAKE': 'AWAKE'}))
    assert H.score_night(same) == NSRR_EXPECTED
    assert H.get_awakening_time(same, awake_tag='WAKE') == 245 * 60.0
    assert H.is_sleep_complete(same, awake_tag='WAKE') is True
    # functions returning a hypnogram return 'AWAKE'
    assert 'WAKE' not in set(H.correct_hypnogram(ns).annotation)
    assert 'WAKE' not in set(H.do_median_filtration(ns).annotation)
    pd.testing.assert_frame_equal(ns, before)


def test_wake_alias_in_tiled_epochs_and_corrections():
    df = tile([('WAKE', 30), ('REM', 10), ('N2', 70), ('AWAKE', 100)])
    # WAKE and AWAKE epochs are one state; REM right after 30 min of WAKE -> AWAKE
    assert list(H.correct_hypnogram(df).annotation.unique()) == ['AWAKE', 'N2']
    assert H.get_fell_asleep_time(df) == at(30)
    assert H.get_awakening_time(df) == at(110)
    assert H.is_sleep_complete(df) is True


# V4: a timedelta duration column (df.end - df.start)
@pytest.mark.parametrize('to_td', [lambda d: d.end - d.start,
                                   lambda d: [x.to_pytimedelta() for x in pd.to_datetime(d.end, utc=True) - pd.to_datetime(d.start, utc=True)]],
                         ids=['timedelta64', 'python-timedelta'])
def test_timedelta_duration_is_accepted(to_td):
    df = tile(NIGHTS)
    td = df.assign(duration=to_td(df))
    assert H.score_night(td) == H.score_night(df)
    assert H.is_sleep_complete(td) is True
    assert H.get_number_of_awakenings(td) == H.get_number_of_awakenings(df)
    np.testing.assert_array_equal(H.get_transition_counts(td), H.get_transition_counts(df))
    assert H.get_number_of_sleep_stages(td) == H.get_number_of_sleep_stages(df)
    assert H.get_hypnogram_datarate(td) == 1.0
    small = bouts(NIGHTS)   # the correction functions loop per row; keep it small
    pd.testing.assert_frame_equal(H.correct_hypnogram(small.assign(duration=to_td(small))),
                                  H.correct_hypnogram(small))
    # without start/end the timedelta durations are read in seconds (the column is kept)
    assert list(H.do_median_filtration(td.drop(columns=['start', 'end'])).annotation) == \
        list(H.do_median_filtration(df.drop(columns=['start', 'end'])).annotation)


def test_duration_errors_name_the_problem():
    df = bouts([('AWAKE', 30), ('N2', 60)])
    wrong = df.assign(duration=pd.to_timedelta([30, 61], unit='min'))
    with pytest.raises(ValueError, match=r'read as 3660\.0 s'):
        H.is_sleep_complete(wrong)
    with pytest.raises(TypeError, match='numbers .seconds. or timedeltas'):
        H.score_night(df.assign(duration=['30 min', '60 min']))


# V6: arousal rows are events for the median filter
def test_median_filtration_leaves_arousals_alone():
    for label in ('Arousal', 'Arrousal'):
        df = tile([('N2', 1), (label, 0.5), ('N2', 1)])
        out = H.do_median_filtration(df)
        assert list(out.annotation) == ['N2', 'N2', label, 'N2', 'N2']   # 6589abb: all N2
    # an overlapping arousal event (starts inside the REM epoch) is not a neighbour
    df = tile([('N2', 1), ('REM', 0.5), ('N2', 1)])
    df.loc[len(df)] = dict(annotation='Arousal', start=at(1) + dt.timedelta(seconds=10),
                           end=at(1) + dt.timedelta(seconds=20), duration=10.0)
    out = H.do_median_filtration(df)
    assert list(out.annotation) == ['N2', 'N2', 'N2', 'Arousal', 'N2', 'N2']  # 6589abb: REM kept


# V8: the default plot maps cover the module's own output and CyberPSG/NSRR labels
def test_plot_default_maps_cover_n_unknown_and_wake(plt):
    df = bouts([('AWAKE', 30), ('N2', 20), ('N3', 20), ('N2', 20), ('AWAKE', 30)])
    gap = df.copy()
    gap.loc[2, 'start'] = at(51)              # 1-min void between N2 and N3 -> 'N'
    gap.loc[2, 'duration'] = 19 * 60.0
    corrected = H.correct_hypnogram(gap)
    assert 'N' in set(corrected.annotation)
    H.plot_hypnogram(corrected)               # 6589abb: ValueError ['N']
    plt.close('all')
    H.plot_hypnogram(tile([('WAKE', 30), ('N2', 30), ('UNKNOWN', 5), ('N2', 30)]))
    labels = [t.get_text() for t in plt.gca().get_yticklabels()]
    assert {'N', 'UNKNOWN', 'AWAKE'} <= set(labels) and 'WAKE' not in labels


# ---------------------------------------------------------------- unscored epochs (on_unscored)

import warnings  # noqa: E402

NIGHT_UNSCORED_END = [('AWAKE', 30), ('N2', 120), ('REM', 30), ('N2', 60), ('UNKNOWN', 10), ('AWAKE', 100)]
NIGHT_UNSCORED_START = [('AWAKE', 30), ('UNKNOWN', 10), ('N2', 200), ('REM', 30), ('AWAKE', 100)]
CLEAN_NIGHT = [('AWAKE', 30), ('N2', 120), ('REM', 30), ('N2', 60), ('AWAKE', 100)]


@pytest.mark.parametrize('build', [bouts, tile])
def test_awakening_unscored_before_final_wake(build):
    df = build(NIGHT_UNSCORED_END)
    with pytest.warns(UserWarning, match=r"UNKNOWN.*final wake"):
        assert H.get_awakening_time(df) is pd.NaT
    with pytest.warns(UserWarning, match='ambiguous'):
        assert H.get_awakening_time(df, on_unscored='warn') is pd.NaT
    with pytest.raises(ValueError, match=r"UNKNOWN.*last sleep epoch"):
        H.get_awakening_time(df, on_unscored='raise')
    with pytest.raises(ValueError, match='on_unscored'):
        H.get_awakening_time(df, on_unscored='ignore')


def test_awakening_unscored_message_names_span_and_numeric_gives_nan():
    df = bouts(NIGHT_UNSCORED_END)
    with pytest.warns(UserWarning) as rec:
        H.get_awakening_time(df)
    msg = str(rec[0].message)
    assert str(at(240)) in msg and str(at(250)) in msg      # span 240-250 min
    num = df.assign(start=df.start.map(lambda t: t.timestamp()), end=df.end.map(lambda t: t.timestamp()))
    with pytest.warns(UserWarning):
        assert np.isnan(H.get_awakening_time(num))


def test_awakening_unscored_even_with_an_earlier_brief_awakening():
    # earlier AWAKE (5 min) is not a sustained wake; the old fallback returned its start
    df = bouts([('AWAKE', 30), ('N2', 60), ('AWAKE', 5), ('N2', 60), ('UNKNOWN', 10), ('AWAKE', 100)])
    with pytest.warns(UserWarning):
        assert H.get_awakening_time(df) is pd.NaT


def test_clean_and_unscored_elsewhere_is_unchanged_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert H.get_awakening_time(bouts(CLEAN_NIGHT)) == at(240)
        assert H.get_fell_asleep_time(bouts(CLEAN_NIGHT)) == at(30)
        # unscored span inside the sleep, not before the wake / after the last sleep
        mid = bouts([('AWAKE', 30), ('N2', 120), ('UNKNOWN', 10), ('N2', 60), ('AWAKE', 100)])
        assert H.get_awakening_time(mid) == at(220)
        # unscored after the final wake, or no wake after the unscored tail
        assert H.get_awakening_time(bouts(CLEAN_NIGHT + [('UNKNOWN', 10)])) == at(240)
        assert H.get_awakening_time(bouts([('AWAKE', 30), ('N2', 120), ('UNKNOWN', 10)])) == at(160)
        sc = H.score_night(bouts(CLEAN_NIGHT), on_unscored='raise')
        assert sc['awakening_time'] == at(240) and sc['n2_sleep_time'] == 180 * 60


@pytest.mark.parametrize('build', [bouts, tile])
def test_fell_asleep_unscored_between_wake_and_sleep(build):
    df = build(NIGHT_UNSCORED_START)
    with pytest.warns(UserWarning, match=r"UNKNOWN.*wake and the first sleep"):
        assert H.get_fell_asleep_time(df) is pd.NaT
    with pytest.raises(ValueError, match='first sleep'):
        H.get_fell_asleep_time(df, on_unscored='raise')
    with pytest.warns(UserWarning):
        r = H.get_rem_latency(df)
    assert r['fall_asleep'] is pd.NaT
    assert r['last_awake'] == dt.timedelta(minutes=210)    
    with pytest.raises(ValueError):
        H.get_rem_latency(df, on_unscored='raise')


def test_score_night_with_unscored_before_final_wake_has_nan_fields():
    df = bouts(NIGHT_UNSCORED_END)
    with pytest.warns(UserWarning, match='final wake'):
        sc = H.score_night(df)
    assert sc['awakening_time'] is pd.NaT
    for k in ('n_complete_sleep_cycles', 'n_awakenings', 'n1_sleep_time', 'n2_sleep_time',
              'n3_sleep_time', 'rem_sleep_time', 'awake_sleep_time'):
        assert np.isnan(sc[k]), k
    # independent of the awakening
    assert sc['sleep_complete'] is True
    assert sc['fell_asleep_time'] == at(30)
    assert sc['rem_latency_fell_asleep'] == 120 * 60.0
    assert sc['rem_latency_last_awake'] == 120 * 60.0
    with pytest.raises(ValueError, match='last sleep epoch'):
        H.score_night(df, on_unscored='raise')
    H.print_sleep_score(sc)      # prints n/a, does not crash


def test_score_night_with_unscored_before_sleep_has_nan_fields():
    df = bouts(NIGHT_UNSCORED_START)
    with pytest.warns(UserWarning, match='first sleep'):
        sc = H.score_night(df)
    assert sc['fell_asleep_time'] is pd.NaT
    assert np.isnan(sc['rem_latency_fell_asleep']) and np.isnan(sc['n2_sleep_time'])
    assert sc['awakening_time'] == at(270)       # independent of the onset
    with pytest.raises(ValueError):
        H.score_night(df, on_unscored='raise')
