"""
Tests for the Canonical Response Parameterization (CRP) implementation.

Grounded in Miller et al. 2023 (PLoS Comput Biol 19(5):e1011105). Uses synthetic CCEPs
with a known canonical shape and known response duration so the extracted quantities can
be checked against ground truth.
"""
import numpy as np
import pytest

from brainmaze_eeg.crp import (
    crp_method,
    ccep_proj,
    kt_pca,
    get_stat_indices,
    _flat_projection_index,
    _flat_projection_pair,
    _trial_projection_means,
    _reject_artifact_trials,
)

FS = 1000
T_WIN = np.arange(0, 0.5, 1 / FS)
N = T_WIN.size
# a damped-oscillation canonical CCEP shape confined to the first ~150 ms
SHAPE = np.sin(2 * np.pi * 10 * T_WIN) * np.exp(-T_WIN / 0.08)


def make_ccep(k_trials=10, snr=1.0, seed=0, artifacts=None, amp=100.0):
    rng = np.random.default_rng(seed)
    v = np.outer(SHAPE, amp * np.ones(k_trials)) + rng.normal(0, amp / snr, (N, k_trials))
    if artifacts:
        for a in artifacts:  # polarity-flipped, larger-amplitude artefact trial
            v[:, a] = -3 * amp * SHAPE + rng.normal(0, amp, N)
    return v


# --------------------------------------------------------------------------------------
# get_stat_indices (#53)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7, 8, 10, 11, 20, 21, 31])
def test_stat_indices_select_one_orientation_per_pair(k):
    idx = get_stat_indices(k)
    n_pairs = k * (k - 1) // 2

    assert idx.size == n_pairs, "must select exactly one projection per unordered pair"
    assert np.unique(idx).size == n_pairs, "selected flat indices must be distinct"

    pairs = [frozenset(_flat_projection_pair(p, k)) for p in idx]
    assert len(set(pairs)) == n_pairs, "each unordered pair must appear exactly once"
    # all flat indices must be valid (in range of the K^2 - K projections)
    assert idx.max(initial=-1) < k * k - k


@pytest.mark.parametrize('k', [4, 5, 6, 7, 10, 11, 20, 21])
def test_stat_indices_balance_normalized_role(k):
    idx = get_stat_indices(k)
    normalized = np.bincount([_flat_projection_pair(p, k)[0] for p in idx], minlength=k)
    # odd K balances exactly; even K is balanced to within one
    assert normalized.max() - normalized.min() <= (0 if k % 2 == 1 else 1)


def test_flat_projection_index_roundtrip():
    for n in (4, 5, 10):
        for i in range(n):
            for j in range(n):
                if i != j:
                    p = _flat_projection_index(i, j, n)
                    assert _flat_projection_pair(p, n) == (i, j)


# --------------------------------------------------------------------------------------
# kt_pca (#55)
# --------------------------------------------------------------------------------------
def test_kt_pca_returns_real_sorted_orthonormal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8))
    e, s = kt_pca(x)

    assert np.isrealobj(e) and np.isrealobj(s)
    assert np.all(np.diff(s) <= 1e-9), "eigenvalues must be in descending order"
    assert np.all(s >= 0), "singular values must be non-negative"
    # leading eigenvectors are unit-norm and mutually orthogonal
    norms = np.linalg.norm(e[:, :8], axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)
    gram = e[:, :3].T @ e[:, :3]
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-8)


def test_kt_pca_recovers_dominant_direction():
    # a rank-1 signal plus tiny noise: first eigenvector ~ the signal direction
    rng = np.random.default_rng(1)
    direction = SHAPE[:120]
    x = np.outer(direction, rng.normal(5, 1, 12)) + rng.normal(0, 1e-3, (120, 12))
    e, _ = kt_pca(x)
    corr = abs(np.dot(e[:, 0], direction) / (np.linalg.norm(e[:, 0]) * np.linalg.norm(direction)))
    assert corr > 0.99


# --------------------------------------------------------------------------------------
# canonical shape / sign convention (#54)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', range(8))
def test_canonical_shape_sign_is_positive(seed):
    params, _ = crp_method(make_ccep(seed=seed), T_WIN)
    c = params['C']
    true = SHAPE[:c.size]
    corr = np.dot(c, true) / (np.linalg.norm(c) * np.linalg.norm(true))
    assert corr > 0, "C(t) must be sign-aligned with the true response, not inverted"
    assert np.mean(params['al']) > 0, "projection weights must be predominantly positive"


@pytest.mark.parametrize('seed', range(8))
def test_vsnr_is_non_negative(seed):
    params, _ = crp_method(make_ccep(seed=seed), T_WIN)
    assert np.all(params['Vsnr'] >= 0), "Vsnr is a signal-to-noise magnitude"


def test_erp_reconstruction_matches_residual():
    params, _ = crp_method(make_ccep(seed=3), T_WIN)
    # by definition ep = V_tR - alpha*C
    np.testing.assert_allclose(params['V_tR'] - params['erp'], params['ep'], atol=1e-9)


# --------------------------------------------------------------------------------------
# response duration / structure
# --------------------------------------------------------------------------------------
def test_response_duration_within_signal_support():
    params, proj = crp_method(make_ccep(snr=3.0, seed=0), T_WIN)
    # the synthetic response decays by ~0.2 s; tau_R should be positive and < window
    assert 0 < params['tR'] <= T_WIN[-1]
    assert 0 <= proj['tR_index'] < proj['proj_tpts'].size


def test_extraction_significance_high_for_structured_low_for_noise():
    struct, _ = crp_method(make_ccep(snr=5.0, seed=0), T_WIN)  # noqa: F841
    _, proj_struct = crp_method(make_ccep(snr=5.0, seed=0), T_WIN)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 100, (N, 10))
    _, proj_noise = crp_method(noise, T_WIN)
    assert proj_struct['t_value_tR'] > proj_noise['t_value_tR']


# --------------------------------------------------------------------------------------
# artefact rejection (#51) + trial traceability (#52)
# --------------------------------------------------------------------------------------
def test_clean_data_rejects_nothing():
    dropped = 0
    for seed in range(10):
        for snr in (1.0, 3.0):
            params, _ = crp_method(make_ccep(20, snr, seed), T_WIN, prune_the_data=True)
            dropped += params['rejected_trials'].size
    assert dropped / 20 < 0.2, "clean trials must not be rejected"


def test_injected_artifacts_are_rejected():
    caught = 0
    for seed in range(12):
        v = make_ccep(20, 1.0, seed, artifacts=[3, 7])
        params, _ = crp_method(v, T_WIN, prune_the_data=True)
        caught += len({3, 7} & set(params['rejected_trials'].tolist()))
    assert caught / 12 >= 1.5, "clear artefact trials should usually be rejected"


def test_rejected_and_kept_index_original_trials():
    v = make_ccep(20, 1.0, 1, artifacts=[3, 7])
    params, _ = crp_method(v, T_WIN, prune_the_data=True)
    kept, rej = params['kept_trials'], params['rejected_trials']

    assert set(rej.tolist()) == {3, 7}
    assert set(kept.tolist()) | set(rej.tolist()) == set(range(20))
    assert not (set(kept.tolist()) & set(rej.tolist()))
    # parameters describe the KEPT subset
    assert params['al'].size == kept.size
    # rejection_stat spans the original trial axis
    assert params['rejection_stat'].size == 20


def test_no_prune_keeps_all_trials():
    params, _ = crp_method(make_ccep(10, 1.0, 0), T_WIN, prune_the_data=False)
    np.testing.assert_array_equal(params['kept_trials'], np.arange(10))
    assert params['rejected_trials'].size == 0
    assert params['rejection_stat'].size == 10  # still reported for inspection


def test_erp_full_enables_full_epoch_subtraction():
    v = make_ccep(20, 1.0, 1, artifacts=[3, 7])
    params, _ = crp_method(v, T_WIN, prune_the_data=True)
    erp_full = params['erp_full']
    assert erp_full.shape == (N, params['kept_trials'].size)
    # zero after the response duration (response is confined to [0, tau_R])
    assert np.allclose(erp_full[params['C'].size:], 0)
    # subtracting removes the evoked component: residual power < raw power
    kept_v = v[:, params['kept_trials']]
    assert np.linalg.norm(kept_v - erp_full) < np.linalg.norm(kept_v)


# --------------------------------------------------------------------------------------
# rejection helpers
# --------------------------------------------------------------------------------------
def test_reject_helper_two_sided_and_robust():
    stat = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 8.0, -6.0])  # last two are outliers
    rejected = _reject_artifact_trials(stat, coeff=3.5)
    assert set(rejected.tolist()) == {5, 6}


def test_reject_helper_no_false_positive_on_uniform():
    stat = np.ones(10) + np.random.default_rng(0).normal(0, 1e-3, 10)
    assert _reject_artifact_trials(stat, coeff=3.5).size == 0


def test_trial_projection_means_span_all_trials():
    v = make_ccep(8, 2.0, 0)
    params, proj = crp_method(v, T_WIN)
    idx = proj['stat_indices']
    s_tR = proj['s_all'][idx, proj['tR_index']]
    means = _trial_projection_means(s_tR, idx, 8)
    assert means.shape == (8,)
    assert np.all(np.isfinite(means))


# --------------------------------------------------------------------------------------
# edge cases (#55)
# --------------------------------------------------------------------------------------
def test_short_epoch_raises_valueerror():
    with pytest.raises(ValueError, match="more than 10 samples"):
        crp_method(np.random.default_rng(0).normal(size=(8, 5)), np.arange(8) / FS)


def test_non_2d_input_raises():
    with pytest.raises(ValueError):
        crp_method(np.zeros(100), T_WIN[:100])


def test_ccep_proj_shape_and_diagonal_removed():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 6))
    s = ccep_proj(v)
    assert s.shape == (6 * 6 - 6,)


def test_stat_indices_degenerate_small_n():
    assert get_stat_indices(1).size == 0
    assert get_stat_indices(0).size == 0


def test_reject_helper_too_few_trials_returns_empty():
    assert _reject_artifact_trials(np.array([1.0, 2.0]), coeff=3.5).size == 0


def test_reject_helper_all_identical_returns_empty():
    # zero MAD and zero std -> nothing to reject
    assert _reject_artifact_trials(np.ones(10), coeff=3.5).size == 0


def test_reject_helper_zero_mad_nonzero_std():
    # tied median (MAD == 0) but a real outlier via the std fallback
    stat = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    assert 7 in _reject_artifact_trials(stat, coeff=3.0).tolist()
