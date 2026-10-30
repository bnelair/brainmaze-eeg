"""Epochs, the cluster-permutation test, ERSP/ITC, selection statistics and montages,
against known answers. Bounds are tight enough that an off-by-one at an epoch edge, a
missing +1 in a permutation p-value, a dropped tail, a mis-scaled wavelet or a reversed
derivation fails them; every random draw is seeded."""
import time

import numpy as np
import pytest
from scipy import stats

from brainmaze_eeg.epochs import (_morlet, cluster_permutation_test, epoch_average, ersp_itc, extract_epochs,
                                  itc_p_value)
from brainmaze_eeg.montage import (apply_montage, bipolar, common_average, double_banana, nearest_neighbour,
                                   referential, region_average)
from brainmaze_eeg.selection_stats import selection_delta, selection_stats

FS = 250.0


def test_epochs_keep_exactly_the_events_whose_window_fits_and_is_finite():
    x = np.arange(1000, dtype=float)
    x[600] = np.nan
    pre, post = 50, 50                                          # samples (0.2 s each side)
    events = np.array([pre, 1000 - post, pre - 1, 1000 - post + 1, 250, 2.4 * FS, np.nan]) / FS
    epochs, times, kept = extract_epochs(x, FS, events, pre_s=0.2, post_s=0.2)
    assert kept.tolist() == [True, True, False, False, True, False, False]
    assert epochs.shape == (3, 100) and times[0] == pytest.approx(-0.2) and times[pre] == 0
    assert [e[pre] for e in epochs] == [50, 950, 250]          # the event sample sits at t = 0


def test_the_sem_is_sigma_over_root_n():
    rng = np.random.default_rng(0)
    epochs = rng.normal(0, 2.0, (400, 300))
    mean, sem = epoch_average(epochs)
    assert np.mean(sem) == pytest.approx(2.0 / np.sqrt(400), rel=0.03)
    assert np.mean(np.abs(mean) < 3 * sem) > 0.99


def _response(sign=1.0):
    signal = np.zeros(120)
    signal[50:70] = sign * 1.0
    return signal


def test_the_cluster_test_finds_a_response_of_either_sign_and_not_noise():
    rng = np.random.default_rng(1)
    for sign in (1.0, -1.0):
        t_obs, clusters = cluster_permutation_test(_response(sign) + rng.normal(0, 1, (30, 120)),
                                                   n_permutations=300, seed=0)
        hits = [c for c in clusters if c.p < 0.05]
        assert len(hits) == 1 and 45 <= hits[0].start and hits[0].stop <= 75 and np.sign(hits[0].stat) == sign
    _, two = cluster_permutation_test(_response() + rng.normal(0, 1, (25, 120)), rng.normal(0, 1, (25, 120)),
                                      n_permutations=300, seed=0)
    assert [c for c in two if c.p < 0.05]


def test_the_cluster_tests_false_positive_rate_is_at_most_alpha():
    false = 0
    for seed in range(60):
        rng = np.random.default_rng(1000 + seed)
        _, clusters = cluster_permutation_test(rng.normal(0, 1, (20, 80)), n_permutations=199, seed=seed)
        false += any(c.p < 0.05 for c in clusters)
    assert false / 60 <= 0.1


def test_a_permutation_p_value_is_never_below_one_over_n_plus_one():
    rng = np.random.default_rng(2)
    _, clusters = cluster_permutation_test(_response() * 5 + rng.normal(0, 1, (30, 120)), n_permutations=19, seed=0)
    assert min(c.p for c in clusters) == pytest.approx(1 / 20)


def test_the_cluster_test_refuses_missing_data():
    a = np.random.default_rng(3).normal(size=(10, 50))
    a[3, 20] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cluster_permutation_test(a, n_permutations=10)


def test_the_default_threshold_is_the_two_sided_critical_t():
    rng = np.random.default_rng(4)
    a = _response(0.6) + rng.normal(0, 1, (30, 120))
    explicit = cluster_permutation_test(a, n_permutations=50, seed=0,
                                        threshold=stats.t.ppf(0.975, 29))[1]
    assert cluster_permutation_test(a, n_permutations=50, seed=0)[1] == explicit


def test_the_morlet_is_unit_energy_and_centred():
    w = _morlet(FS, 10.0, 7.0)
    assert w.size % 2 == 1 and np.sum(np.abs(w) ** 2) == pytest.approx(1.0)
    assert np.argmax(np.abs(w)) == w.size // 2


def test_ersp_has_the_wavelets_frequency_resolution_and_a_flat_noise_floor():
    t = np.arange(int(4 * FS)) / FS
    tone = np.array([np.sin(2 * np.pi * 20 * t + ph) for ph in np.linspace(0, 6, 10)])
    f1, cycles = 24.0, 7.0
    ersp, _ = ersp_itc(tone, FS, [20.0, f1], n_cycles=cycles)
    # The f1 wavelet's gain at 20 Hz is exp(-df^2 / (2 sigma_f^2)) in amplitude, sigma_f = f1 / cycles.
    # Unit energy makes white noise flat across frequencies, so a TONE's power also scales with
    # the wavelet's duration, i.e. with 1/f: hence the 20/f1.
    sigma_f = f1 / cycles
    expected_db = 10 * np.log10(np.exp(-((f1 - 20.0) ** 2) / sigma_f ** 2) * 20.0 / f1)
    assert np.nanmean(ersp[1]) - np.nanmean(ersp[0]) == pytest.approx(expected_db, abs=0.5)
    noise = np.random.default_rng(5).normal(0, 2.0, (40, int(4 * FS)))
    floor, _ = ersp_itc(noise, FS, [5.0, 20.0, 60.0])
    np.testing.assert_allclose(np.nanmean(floor, axis=1), 10 * np.log10(4.0), atol=0.5)


def test_ersp_edges_the_wavelet_does_not_cover_are_missing():
    e = np.random.default_rng(6).normal(size=(5, 500))
    ersp, itc = ersp_itc(e, FS, [10.0], n_cycles=7)
    half = _morlet(FS, 10.0, 7).size // 2
    assert np.isnan(ersp[0, :half]).all() and np.isnan(ersp[0, -half:]).all()
    assert np.isfinite(ersp[0, half:-half]).all() and np.isfinite(itc[0, half:-half]).all()


def test_ersp_shows_a_power_increase_after_the_event():
    rng = np.random.default_rng(7)
    t = np.arange(int(3 * FS)) / FS - 1.0
    burst = (t > 0.5) & (t < 1.5)
    epochs = np.array([rng.normal(0, 1, t.size) + 4 * burst * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 6))
                       for _ in range(40)])
    ersp, _ = ersp_itc(epochs, FS, [20.0], baseline=(t > -0.8) & (t < -0.1))
    assert np.nanmean(ersp[0, (t > 0.7) & (t < 1.3)]) > 6       # dB above baseline


def test_itc_is_one_for_a_locked_phase_and_judged_by_rayleigh_for_a_random_one():
    rng = np.random.default_rng(8)
    t = np.arange(int(2 * FS)) / FS
    n = 30
    locked = np.array([np.sin(2 * np.pi * 10 * t) + 0.1 * rng.normal(size=t.size) for _ in range(n)])
    loose = np.array([np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi)) for _ in range(n)])
    _, itc = ersp_itc(locked, FS, [10.0])
    assert np.nanmean(itc) > 0.95 and np.nanmax(itc_p_value(itc, n)) < 1e-6
    _, itc = ersp_itc(loose, FS, [10.0])
    assert np.nanmean(itc) < 0.89 / np.sqrt(n) * 2.5           # biased up, never 0 -- hence Rayleigh
    phases = rng.uniform(0, 2 * np.pi, (2000, 20))
    null_itc = np.abs(np.exp(1j * phases).mean(axis=1))
    assert np.mean(itc_p_value(null_itc, 20) < 0.05) == pytest.approx(0.05, abs=0.015)


def test_selection_statistics():
    x = np.array([[1.0, -1.0, 3.0, np.nan], [np.nan] * 4])
    s = selection_stats(x)
    assert s["min"][0] == -1 and s["max"][0] == 3 and s["p2p"][0] == 4 and s["n"].tolist() == [3, 0]
    assert s["mean"][0] == pytest.approx(1.0) and s["rms"][0] == pytest.approx(np.sqrt(11 / 3))
    assert s["sd"][0] == pytest.approx(2.0) and np.isnan(s["mean"][1])
    d = selection_delta([[0.0, 5.0, 2.0]], fs=2.0)
    assert d["dA"][0] == 2.0 and d["dt"] == 1.0 and d["inv_dt"] == 1.0


def test_bipolar_pairs_adjacent_contacts_with_the_right_sign_and_never_bridges():
    ch = ["LA1", "LA2", "LA3", "LA5", "RH1", "RH2", "ECG"]
    assert [d.name for d in bipolar(ch)] == ["LA1-LA2", "LA2-LA3", "RH1-RH2"]
    data = np.array([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [0.0]])
    np.testing.assert_allclose(apply_montage(data, ch, bipolar(ch))[:, 0], [1 - 4, 4 - 9, 25 - 36])
    assert len(referential(ch)) == len(ch)


def test_common_average_includes_itself_and_leaves_excluded_channels_out():
    ch = ["A", "B", "C", "ECG"]
    car = common_average(ch, exclude=["ECG"])
    assert [d.name for d in car] == ["A-avg", "B-avg", "C-avg"]
    data = np.random.default_rng(9).normal(size=(4, 50))
    out = apply_montage(data, ch, car)
    np.testing.assert_allclose(out.sum(axis=0), 0, atol=1e-12)


def test_region_average_subtracts_the_other_contacts_of_the_region():
    ch = ["LA1", "LA2", "LA3", "RH1"]
    region = region_average(ch, {"LA1": "amygdala", "LA2": "amygdala", "LA3": "amygdala", "RH1": "hippo"})
    assert [d.name for d in region] == ["LA1-amygdala", "LA2-amygdala", "LA3-amygdala"]
    assert region[0].minus == ("LA2", "LA3")                     # a region of one is left out
    pair = region_average(["A", "B"], {"A": "r", "B": "r"})
    np.testing.assert_allclose(apply_montage(np.array([[5.0], [2.0]]), ["A", "B"], pair)[:, 0], [3.0, -3.0])


def test_double_banana_and_nearest_neighbour():
    scalp = ["Fp1", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz"]
    assert [d.name for d in double_banana(scalp)] == ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fz-Cz", "Cz-Pz"]
    nn = nearest_neighbour(["A", "B", "C"], {"A": (0, 0, 0), "B": (1, 0, 0), "C": (5, 0, 0)})
    assert [d.minus for d in nn] == [("B",), ("A",), ("B",)]
    with pytest.raises(ValueError, match="not in the data"):
        apply_montage(np.zeros((1, 2)), ["A"], bipolar(["X1", "X2"]))


def test_a_common_reference_is_computed_once():
    ch = [f"C{i}" for i in range(128)]
    data = np.random.default_rng(10).normal(size=(128, 30000))
    start = time.perf_counter()
    apply_montage(data, ch, common_average(ch))
    assert time.perf_counter() - start < 2.0                    # was ~16 s at 10x the samples
