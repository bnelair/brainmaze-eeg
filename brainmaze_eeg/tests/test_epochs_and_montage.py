"""Epochs, the cluster-permutation test, ERSP/ITC, selection statistics and montages,
against known answers."""

import numpy as np
import pytest

from brainmaze_eeg.epochs import cluster_permutation_test, epoch_average, ersp_itc, extract_epochs
from brainmaze_eeg.montage import (apply_montage, bipolar, common_average, double_banana,
                                   nearest_neighbour, referential, region_average)
from brainmaze_eeg.selection_stats import selection_delta, selection_stats

FS = 250.0


def test_epochs_drop_events_past_the_edges_or_over_a_gap():
    x = np.arange(1000, dtype=float)
    x[600] = np.nan
    epochs, times, kept = extract_epochs(x, FS, [0.1, 1.0, 2.4, 3.9], pre_s=0.2, post_s=0.2)
    assert kept.tolist() == [False, True, False, False]        # edge, ok, gap, edge
    assert epochs.shape == (1, 100) and times[0] == pytest.approx(-0.2)
    assert epochs[0, 50] == 250                                  # the event sample sits at t = 0


def test_the_average_of_a_locked_response_recovers_it():
    rng = np.random.default_rng(0)
    n_t = 200
    shape = np.exp(-((np.arange(n_t) - 100) / 10) ** 2)
    epochs = shape + rng.normal(0, 1, (400, n_t))
    mean, sem = epoch_average(epochs)
    assert np.max(np.abs(mean - shape)) < 5 * sem.max()


def test_the_cluster_test_finds_a_response_and_not_noise():
    rng = np.random.default_rng(1)
    n_t = 120
    signal = np.zeros(n_t)
    signal[50:70] = 1.0
    t_obs, clusters = cluster_permutation_test(signal + rng.normal(0, 1, (30, n_t)), n_permutations=300, seed=0)
    hits = [c for c in clusters if c.p < 0.05]
    assert len(hits) == 1 and hits[0].start >= 45 and hits[0].stop <= 75
    _, clusters = cluster_permutation_test(rng.normal(0, 1, (30, n_t)), n_permutations=300, seed=0)
    assert not [c for c in clusters if c.p < 0.05]
    _, two = cluster_permutation_test(signal + rng.normal(0, 1, (25, n_t)), rng.normal(0, 1, (25, n_t)),
                                      n_permutations=300, seed=0)
    assert [c for c in two if c.p < 0.05]


def test_itc_is_one_for_a_locked_phase_and_small_for_a_random_one():
    rng = np.random.default_rng(2)
    t = np.arange(int(2 * FS)) / FS
    locked = np.array([np.sin(2 * np.pi * 10 * t) + 0.1 * rng.normal(size=t.size) for _ in range(60)])
    loose = np.array([np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi)) for _ in range(60)])
    _, itc = ersp_itc(locked, FS, [10.0])
    assert np.nanmean(itc) > 0.95
    _, itc = ersp_itc(loose, FS, [10.0])
    assert np.nanmean(itc) < 0.3


def test_ersp_shows_a_power_increase_after_the_event():
    rng = np.random.default_rng(3)
    t = np.arange(int(3 * FS)) / FS - 1.0
    burst = (t > 0.5) & (t < 1.5)
    epochs = np.array([rng.normal(0, 1, t.size) + 4 * burst * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 6))
                       for _ in range(40)])
    ersp, _ = ersp_itc(epochs, FS, [20.0], baseline=(t > -0.8) & (t < -0.1))
    assert np.nanmean(ersp[0, (t > 0.7) & (t < 1.3)]) > 6       # dB above baseline


def test_selection_statistics():
    x = np.array([[1.0, -1.0, 3.0, np.nan], [np.nan] * 4])
    s = selection_stats(x)
    assert s["min"][0] == -1 and s["max"][0] == 3 and s["p2p"][0] == 4 and s["n"].tolist() == [3, 0]
    assert s["mean"][0] == pytest.approx(1.0) and s["rms"][0] == pytest.approx(np.sqrt(11 / 3))
    assert s["sd"][0] == pytest.approx(2.0) and np.isnan(s["mean"][1])
    d = selection_delta([[0.0, 5.0, 2.0]], fs=2.0)
    assert d["dA"][0] == 2.0 and d["dt"] == 1.0 and d["inv_dt"] == 1.0


def test_montages():
    ch = ["LA1", "LA2", "LA3", "LA5", "RH1", "RH2", "ECG"]
    names = [d.name for d in bipolar(ch)]
    assert names == ["LA1-LA2", "LA2-LA3", "RH1-RH2"]          # LA3-LA5 is not bridged
    assert len(referential(ch)) == len(ch)
    data = np.vstack([np.full(4, i, dtype=float) for i in range(len(ch))])
    car = apply_montage(data, ch, common_average(ch, exclude=["ECG"]))
    assert car.shape == (len(ch), 4) and car[0, 0] == pytest.approx(0 - np.mean(range(6)))
    scalp = ["Fp1", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz"]
    assert [d.name for d in double_banana(scalp)] == ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fz-Cz", "Cz-Pz"]
    region = region_average(ch, {"LA1": "amygdala", "LA2": "amygdala", "RH1": "hippocampus"})
    assert [d.name for d in region] == ["LA1-amygdala", "LA2-amygdala"]   # a region of one is left out
    nn = nearest_neighbour(["A", "B", "C"], {"A": (0, 0, 0), "B": (1, 0, 0), "C": (5, 0, 0)})
    assert [d.minus for d in nn] == [("B",), ("A",), ("B",)]
    with pytest.raises(ValueError, match="not in the data"):
        apply_montage(data, ch, bipolar(["X1", "X2"]))
