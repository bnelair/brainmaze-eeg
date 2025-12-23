# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy
from tqdm import tqdm

from brainmaze_utils.annotations import merge_annotations, filter_by_key




"""
Tools for analyzing hypnograms such as number of cycles, hypnogram time etc.
"""


def get_hypnogram_datarate(df):
    """
    Calculate the data rate of a hypnogram.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'duration', 'start', and 'end' columns.
    
    Returns
    -------
    float
        Data rate as the ratio of total duration to time span.
    """
    return ((df['duration'].sum()) / (df.iloc[-1]['end'] - df.iloc[0]['start']).seconds)


def get_fell_asleep_time(df, t_sleep_check=60, t_awake_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N1', 'N2', 'N3']):
    """
    Determine when the subject fell asleep based on hypnogram data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    t_sleep_check : int, optional
        Time window in minutes to check for sustained sleep. Default is 60.
    t_awake_threshold : int, optional
        Maximum awake time in minutes allowed within t_sleep_check window. Default is 10.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.
    sleep_cycle_tags : list, optional
        List of tags indicating sleep states. Default is ['REM', 'N1', 'N2', 'N3'].
    
    Returns
    -------
    datetime
        Timestamp when the subject fell asleep.
    """
    df = filter_by_key(df, 'annotation', 'Arrousal')
    # parameters
    t_sleep = datetime.timedelta(minutes=t_sleep_check) # interval since 1st asleep checked
    t_awake = datetime.timedelta(minutes=t_awake_threshold) # length of all awake cycles during the interval defined by t_sleep since the beginning of the hypnogram

    # get first asleep
    awake_to_sleep_changes = np.where(
        np.array([df.annotation[0] in sleep_cycle_tags] +
                 [(df.annotation[k-1] in awake_tag) and (df.annotation[k] in sleep_cycle_tags) for k in range(1, df.__len__())
                  ]))[0]

    fell_asleep_time = df.iloc[0].start
    for idx in awake_to_sleep_changes:
        start = df.start[idx]
        all_starts = df.start
        t_sleep_window_df = df.loc[(start <= all_starts) & (all_starts < (start + t_sleep))]
        time_awake = t_sleep_window_df.duration[t_sleep_window_df.annotation == awake_tag].sum()
        if time_awake < t_awake.seconds:
            fell_asleep_time = start
            break
    return fell_asleep_time


def get_awakening_time(df, t_awake_threshold=90, t_sleep_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N2', 'N3']):
    """
    Determine when the subject woke up based on hypnogram data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    t_awake_threshold : int, optional
        Minimum sustained awake time in minutes to be considered awakened. Default is 90.
    t_sleep_threshold : int, optional
        Maximum sleep time in minutes allowed in the awakening window. Default is 10.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.
    sleep_cycle_tags : list, optional
        List of tags indicating sleep states. Default is ['REM', 'N2', 'N3'].
    
    Returns
    -------
    datetime
        Timestamp when the subject woke up.
    """
    df = filter_by_key(df, 'annotation', 'Arrousal')
    t_awake = datetime.timedelta(minutes=t_awake_threshold)
    t_sleep = datetime.timedelta(minutes=t_sleep_threshold)

    sleep_to_awake_changes = np.where(
        np.array([df.annotation[0] in sleep_cycle_tags] +
                 [(df.annotation[k-1] in sleep_cycle_tags) and (df.annotation[k] in awake_tag) for k in range(1, df.__len__())]
                 ))[0]

    if df.iloc[-1].annotation == awake_tag:
        last_awake_time = df.iloc[sleep_to_awake_changes[-1]].start
    else:
        last_awake_time = df.iloc[-1].end

    for idx in sleep_to_awake_changes:
        start = df.start[idx]
        all_starts = df.start
        t_awake_window_df = df.loc[(start <= all_starts) & (all_starts < (start + t_awake))]
        time_awake = t_awake_window_df.duration[t_awake_window_df.annotation == awake_tag].sum()
        time_asleep = get_time_by_key(t_awake_window_df, ['N1', 'N2', 'N3', 'REM'])
        if time_awake >= t_awake.seconds and time_asleep <= t_sleep_threshold:
            last_awake_time = start
            break
    return last_awake_time


def is_sleep_complete(df, awake_tag='AWAKE'):
    """
    Check if the sleep session is complete (starts and ends with awake state).
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation' column.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.
    
    Returns
    -------
    bool
        True if sleep session starts and ends with awake state, False otherwise.
    """
    return df.iloc[0].annotation == awake_tag == df.iloc[-1].annotation


def get_rem_latency(df, rem_tag='REM', awake_tag='AWAKE'):
    """
    Calculate REM sleep latency (time from falling asleep or last awake to first REM).
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', and 'end' columns.
    rem_tag : str, optional
        Tag for REM sleep state. Default is 'REM'.
    awake_tag : str, optional
        Tag for awake state. Default is 'AWAKE'.
    
    Returns
    -------
    dict
        Dictionary with 'last_awake' and 'fall_asleep' latencies as timedelta objects.
    """
    first_rem_start = df.loc[df.annotation == rem_tag].reset_index(drop=True).iloc[0].start
    fall_asleep_start = get_fell_asleep_time(df)
    df_rem = df.loc[(df.end <= first_rem_start) & (df.annotation == awake_tag)].reset_index(drop=True)
    if df_rem.__len__() == 0:
        last_awake_end = fall_asleep_start
    else:
        last_awake_end = df_rem.iloc[-1].end

    return {'last_awake': first_rem_start - last_awake_end, 'fall_asleep': first_rem_start - fall_asleep_start}


def get_number_of_sleep_stages(df, tags ='REM', delay=30):
    """
    Count the number of sleep stage occurrences with minimum separation delay.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', and 'end' columns.
    tags : str or list, optional
        Sleep stage tag(s) to count. Default is 'REM'.
    delay : int, optional
        Minimum time in minutes between stage occurrences. Default is 30.
    
    Returns
    -------
    int
        Number of sleep stage occurrences.
    """
    if isinstance(tags, str):
        tags = [tags]

    delay = datetime.timedelta(minutes=delay)
    bool_idxes = np.ones(df.__len__(), dtype=bool)
    for tag in tags:
        bool_idxes = (bool_idxes) & (df.annotation == tag)

    df = df.loc[bool_idxes].reset_index(drop=True)

    stage_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):
        if idx == 0:
            stage_df = stage_df.append(row[1], ignore_index=True)
        else:
            if (row[1].start - stage_df.iloc[-1].end).seconds >= delay.seconds:
                stage_df = stage_df.append(row[1], ignore_index=True)

    return (stage_df.annotation == tag).sum()




def get_number_of_awakenings(df, awake_tag='AWAKE', n1_tag='N1', sleep_tags=['N2', 'N3', 'REM']):
    """
    Count the number of awakenings after sleep onset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation' column.
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
    """
    awake_bool = df.annotation == awake_tag
    n1_bool = df.annotation == n1_tag
    sleep_bool = np.zeros_like(n1_bool, dtype=bool)
    for tag in sleep_tags:
        sleep_bool = (sleep_bool) | (df.annotation == tag)

    awake_n1_bool = (awake_bool) | (n1_bool)

    n_awakenings = 0
    sleep_happened = False
    awake_happened = False
    for k in range(sleep_bool.shape[0]):
        if sleep_bool[k] == True:
            sleep_happened = True

        if awake_bool[k] == True:
            awake_happened = True
        else:
            if not awake_n1_bool[k] == True:
                awake_happened = False

        if sleep_happened == True and awake_happened == True:
            n_awakenings += 1
            sleep_happened = False
            awake_happened = False
    return n_awakenings


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
        Total duration in seconds.
    """
    if isinstance(key, list):
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
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation', 'start', 'end', and 'duration' columns.
    plot : bool, optional
        If True, plot the hypnogram with sleep onset and awakening markers. Default is False.
    
    Returns
    -------
    dict
        Dictionary containing sleep metrics including:
        
        - sleep_complete: whether sleep session is complete
        - fell_asleep_time: time of sleep onset
        - rem_latency_fell_asleep: REM latency from sleep onset (seconds)
        - rem_latency_last_awake: REM latency from last awake (seconds)
        - awakening_time: time of final awakening
        - n_complete_sleep_cycles: number of complete sleep cycles
        - n_awakenings: number of awakenings
        - n1_sleep_time, n2_sleep_time, n3_sleep_time, rem_sleep_time, awake_sleep_time: 
          durations in seconds for each stage
    """
    df = filter_by_key(df, 'annotation', 'Arrousal')
    df = merge_annotations(df)

    fell_asleep_time = get_fell_asleep_time(df)
    awakening_time = get_awakening_time(df)
    sleep_complete = is_sleep_complete(df)

    sleep_df = df.loc[(df.start >= fell_asleep_time) & (df.start < awakening_time)].reset_index(drop=True)

    n_complete_sleep_cycles = get_number_of_sleep_stages(sleep_df, tags='REM', delay=30)
    n_awakenings = get_number_of_awakenings(sleep_df)
    rem_latency = get_rem_latency(df)


    n1_sleep_time = get_time_by_key(sleep_df, 'N1')
    n2_sleep_time = get_time_by_key(sleep_df, 'N2')
    n3_sleep_time = get_time_by_key(sleep_df, 'N3')
    rem_sleep_time = get_time_by_key(sleep_df, 'REM')
    awake_sleep_time = get_time_by_key(sleep_df, 'AWAKE')


    if plot == True:
        plot_hypnogram(df)
        plt.stem([fell_asleep_time, awakening_time], [7, 7], linefmt='r', markerfmt='or', basefmt='r')

    return {
        'sleep_complete': sleep_complete,
        'fell_asleep_time': fell_asleep_time,
        'rem_latency_fell_asleep': rem_latency['fall_asleep'].seconds,
        'rem_latency_last_awake': rem_latency['last_awake'].seconds,
        'awakening_time': awakening_time,
        'n_complete_sleep_cycles': n_complete_sleep_cycles,
        'n_awakenings': n_awakenings,
        'n1_sleep_time': n1_sleep_time,
        'n2_sleep_time': n2_sleep_time,
        'n3_sleep_time': n3_sleep_time,
        'rem_sleep_time': rem_sleep_time,
        'awake_sleep_time': awake_sleep_time
    }


def print_sleep_score(score):
    """
    Print a formatted summary of sleep metrics.
    
    Parameters
    ----------
    score : dict
        Sleep score dictionary from score_night function.
    """
    total_sleep_time =(score['awakening_time'] - score['fell_asleep_time']).seconds
    hours, remainder = divmod(total_sleep_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    rem_lat1 = score['rem_latency_last_awake'].seconds
    hrs_rem1, remainder = divmod(rem_lat1, 3600)
    mins_rem1, secs_rem1 = divmod(remainder, 60)

    rem_lat2 = score['rem_latency_fell_asleep'].seconds
    hrs_rem2, remainder = divmod(rem_lat2, 3600)
    mins_rem2, secs_rem2 = divmod(remainder, 60)

    non_REM = score['n3_sleep_time'] + score['n2_sleep_time'] + score['n1_sleep_time']

    print('Sleep Complete: ', score['sleep_complete'])
    print('Falling asleep: ', score['fell_asleep_time'].strftime('%H:%M:%S'))
    print('Awakening: ', score['awakening_time'].strftime('%H:%M:%S'))
    print('Total Sleep Time: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
    print('Rem latency - last_awake: {:02}:{:02}:{:02}'.format(int(hrs_rem1), int(mins_rem1), int(secs_rem1)))
    print('Rem latency - fall_asleep: {:02}:{:02}:{:02}'.format(int(hrs_rem2), int(mins_rem2), int(secs_rem2)))

    print('Number of hypnogram cycles: ', score['n_complete_sleep_cycles'])
    print('Number of awakenings', score['n_awakenings'])
    print()
    print('Sleep-time non-REM')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(non_REM),  non_REM/total_sleep_time))
    print('Sleep-time REM')
    print()
    print('Sleep-time REM')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['rem_sleep_time']),  score['rem_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time awake')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['awake_sleep_time']),  score['awake_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N1')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n1_sleep_time']),  score['n1_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N2')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n2_sleep_time']),  score['n2_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N3')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n3_sleep_time']),  score['n3_sleep_time']/total_sleep_time))


def get_transition_counts(hyp, states=['AWAKE', 'N1', 'N2', 'N3', 'REM']):
    """
    Count transitions between sleep stages.
    
    Parameters
    ----------
    hyp : pd.DataFrame
        Hypnogram dataframe with 'annotation' column.
    states : list, optional
        List of sleep states to include. Default is ['AWAKE', 'N1', 'N2', 'N3', 'REM'].
    
    Returns
    -------
    np.ndarray
        Matrix of transition counts where [i,j] is count of transitions from state i to state j.
    """
    states = np.array(states)
    matrix = np.zeros((states.__len__(), states.__len__()))
    for k in range(hyp.__len__()-1):
        s1 =  hyp['annotation'][k]
        s2 =  hyp['annotation'][k+1]
        if s1 in states and s2 in states:
            idx1 = np.where(states == hyp['annotation'][k])[0][0]
            idx2 = np.where(states == hyp['annotation'][k+1])[0][0]
            matrix[idx1, idx2] += 1
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
        Mean transition matrix and standard deviation matrix.
    """
    ms = []
    for hyp in hypnograms:
        m = get_transition_counts(hyp, states)
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
        Dictionary mapping stage names to minimum duration thresholds.
    
    Returns
    -------
    list
        Indices of hypnograms meeting all duration requirements.
    """
    valid_hypnograms = []
    for idx, hyp in enumerate(hypnograms):
        if sum([hyp['duration'].sum() >= v for k, v in filt_dict.items() if k in hyp['annotation'].unique()]) == filt_dict.keys().__len__():
            valid_hypnograms += [idx]
    return valid_hypnograms




def do_median_filtration(df):
    """
    Apply median filtering to remove isolated 30-second stage annotations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hypnogram dataframe with 'annotation' and 'duration' columns.
    
    Returns
    -------
    pd.DataFrame
        Filtered hypnogram dataframe.
    """
    for k in range(1, df.__len__() - 1):
        if df.iloc[k - 1].annotation == df.iloc[k + 1].annotation and df.iloc[k].duration == 30:
            df.iloc[k]['annotation'] = df.iloc[k - 1].annotation
    return df


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
    """
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state == crow['annotation']:
            crow['start'] = last_annotation_end
            crow['duration'] = crow['end'] - crow['start']

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


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
    """
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state == 'AWAKE' and crow['annotation'] == 'REM':
            vrow = deepcopy(crow)
            vrow['annotation'] = 'AWAKE'
            vrow['start'] = last_annotation_end
            vrow['end'] = crow['start']
            vrow['duration'] = crow['end'] - crow['start']
            new_df += [vrow]

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


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
    """
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state in ('N2', 'N3', 'N') and crow['annotation'] in (
        'N2', 'N3', 'N'):
            vrow = deepcopy(crow)
            vrow['annotation'] = 'N'
            vrow['start'] = last_annotation_end
            vrow['end'] = crow['start']
            vrow['duration'] = crow['end'] - crow['start']
            new_df += [vrow]

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


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
    """
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state != 'AWAKE' and crow['annotation'] != 'AWAKE':
            vrow = deepcopy(crow)
            vrow['annotation'] = 'SLP'
            vrow['start'] = last_annotation_end
            vrow['end'] = crow['start']
            vrow['duration'] = crow['end'] - crow['start']
            new_df += [vrow]

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


def correct_rem(df, time_threshold=5*60, initial_state='AWAKE'):
    """
    Correct REM annotations that occur shortly after prolonged awake periods.
    
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
    """
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void <= 60 and current_state == 'AWAKE' and crow[
            'annotation'] == 'REM' and current_state_duration >= time_threshold:
            crow['annotation'] = 'AWAKE'

        if crow['annotation'] == 'AWAKE':
            current_state_duration += crow['duration']
        else:
            current_state_duration = 0
        current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


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
    """
    new_df = deepcopy(df)
    new_df = fill_same_voids(new_df, time_threshold=time_threshold)
    new_df = fill_wakerem_voids(new_df, time_threshold=time_threshold)
    new_df = fill_nonrem_voids(new_df, time_threshold=time_threshold)
    new_df = fill_sleep_voids(new_df, time_threshold=time_threshold)
    new_df = correct_rem(new_df, time_threshold=time_threshold)
    # new_df = do_median_filtration(new_df)
    return new_df



def plot_hypnogram(orig_df, hypnogram_values=None, hypnogram_colors=None, fontsize=12, fig=None, night_start=22):
    """
    Creates a Matplotlib figure of spectrogram from the annotations. Time must be in a time-zone aware format.

    Parameters
    ----------
    orig_df : annotations
    hypnogram_values : dict
        dict of a y-axis values for each hypnogram state
    hypnogram_colors : dict
        dict of color hex codes for each hypnogram state
    fontsize : int
        Fontsize
    fig : figure
        Already existing figure object.
    night_start : int
        A hour when does the night begin

    Returns
    -------

    """
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

    def set_hypnogram_properties(x, ref_dict):
        """Map hypnogram annotation to property value from reference dictionary."""
        return ref_dict[x.annotation]

    orig_df['state_id'] = orig_df.apply(lambda x: set_hypnogram_properties(x, hypnogram_values), axis=1)
    orig_df['state_color'] = orig_df.apply(lambda x: set_hypnogram_properties(x, hypnogram_colors), axis=1)
    df_arrousals = orig_df.loc[orig_df.annotation == 'Arrousal'].reset_index(drop=True)
    df = orig_df.loc[orig_df.annotation != 'Arrousal'].reset_index(drop=True)
    new_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):  # if 2 cons. states are same, merges them
        appbl = True
        if idx > 0:
            if new_df.iloc[-1].state_id == row[1].state_id and new_df.iloc[-1].end == row[1].start:
                appbl = False

        if appbl == True:
            new_df = new_df.append(row[1], ignore_index=True)
        else:
            new_df.loc[new_df.__len__() - 1, 'end'] = row[1].end
    df = new_df

    x_start = np.array(df['start'])
    x_end = np.array(df['end'])
    try:
        for k, time_sample in enumerate(x_start): x_start[k] = time_sample.to_pydatetime()
        for k, time_sample in enumerate(x_end): x_end[k] = time_sample.to_pydatetime()
    except:
        for k, time_sample in enumerate(x_start): x_start[k] = time_sample
        for k, time_sample in enumerate(x_end): x_end[k] = time_sample

    if not fig:
        plt.figure(dpi=200)
    plt.xlim(x_start[0], x_end[-1])
    """
    # set background color for days
    for idx, day_id in enumerate(np.unique(df.day)):
        if idx % 2 == 0:
            background_color = 'gray'
            background_alpha = 0.1
        else:
            background_color = 'gray'
            background_alpha = 0.3

        day_start = x_start[df.day == day_id][0]
        day_end = x_end[df.day == day_id][-1]
        plt.axvspan(day_start, day_end, facecolor=background_color, alpha=background_alpha)
    """

    # set background color for nights
    for idx, day_id in enumerate(np.unique(df.day)):
        background_color = 'gray'
        background_alpha = 0.3
        day_start = x_start[df.day == day_id][0]
        night_start_ = datetime(
            year=day_start.year,
            month=day_start.month,
            day=day_start.day,
            hour=night_start,
            tzinfo=day_start.tzinfo
        )
        night_end_ = night_start_ + datetime.timedelta(hours=12)
        plt.axvspan(night_start_, night_end_, facecolor=background_color, alpha=background_alpha)

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

    x = [start1, end1]
    y = [val1, val1]
    plt.plot(x, y, color='black', alpha=1, linewidth=1)

    # plot arrousals
    for row in df_arrousals.iterrows():
        val = row[1].state_id
        clr = row[1].state_color
        plt.fill_between(
            # plt.plot(
            [row[1].start.to_pydatetime(), row[1].start.to_pydatetime(row[1].start), row[1].end.to_pydatetime(),
             row[1].end.to_pydatetime(row[1].end)],
            [0, val, val, 0],
            color=clr,
            alpha=1,
            linewidth=1
        )

    # format y ticks
    plt.yticks(list(hypnogram_values.values()), hypnogram_values.keys())
    for ticklabel in plt.gca().get_yticklabels():
        clr = hypnogram_colors[ticklabel._text]
        ticklabel.set_color(clr)
        # ticklabel.set_fontsize(fontsize)

    # plot y grid
    for idx, key in enumerate(hypnogram_values.keys()):
        clr = hypnogram_colors[key]
        val = hypnogram_values[key]
        plt.plot([x_start[0], x_end[-1]], [val, val], color=clr, linewidth=0.7, alpha=0.7, linestyle=':')

    # format x_ticks
    plt.gcf().autofmt_xdate()
    # formatter = mdates.DateFormatter("%H:%M") #mdates.DateFormatter("%H:%M", tz=tz.tzlocal())
    formatter = mdates.DateFormatter("%H:%M", tz=df.start[0].tzinfo)
    plt.gcf().get_axes()[0].xaxis.set_major_formatter(formatter)

    # plot hour x grid
    plt.grid(True, axis='x', alpha=1, linewidth=0.5, linestyle=':')

    # axes labels
    # plt.title('Days  ' + df.start[0].strftime('%d.%m') +'-' + df.iloc[-1].end.strftime('%d.%m'))
    plt.xlabel('\n Time [' + df.start[0].strftime('%d.%m.%Y') + ' - ' + df.iloc[-1].end.strftime('%d.%m.%Y') + ']',
               fontsize=fontsize)
    plt.ylabel('Sleep state', fontsize=fontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)




