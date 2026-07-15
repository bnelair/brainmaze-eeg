
import numpy as np
from scipy.stats import ttest_1samp


def crp_method(v, t_win, prune_the_data=False, rejection_coeff=3.5, min_trials=5):
    """
    Canonical Response Parameterization (CRP) of single-pulse stimulation responses.

    Python implementation of the method described in:

        Miller, K. J., Müller, K.-R., Valencia, G. O., Huang, H., Gregg, N. M.,
        Worrell, G. A., & Hermes, D. (2023). *Canonical Response Parameterization:
        Quantifying the structure of responses to single-pulse intracranial electrical
        brain stimulation.* PLoS Computational Biology, 19(5), e1011105.
        https://doi.org/10.1371/journal.pcbi.1011105

    The method takes a set of ``K`` single-trial voltage traces of an evoked response
    (e.g. a cortico-cortical evoked potential, CCEP), and:

    1. Builds a temporal profile of the semi-normalized pairwise cross-projection
       magnitude ``S(t)`` and takes its peak as the **response duration** ``tau_R``.
    2. Extracts a **canonical response shape** ``C(t)`` over ``[0, tau_R]`` via linear
       kernel PCA (the 1st principal component of the truncated trial matrix).
    3. Parameterizes each trial ``k`` as ``V_k(t) = alpha_k * C(t) + eps_k(t)``,
       yielding per-trial projection weight ``alpha_k``, residual ``eps_k`` and derived
       quantities (SNR, explained variance).
    4. Optionally rejects artefactual trials as outliers of the per-trial
       sub-distributions of ``S(tau_R)`` (Miller et al. 2023, "extraction significance"
       section).

    Parameters
    ----------
    v : np.ndarray
        Voltage matrix of shape ``(n_samples, n_trials)``. Each column is one trial;
        trials must be baseline-corrected (the method is sensitive to baseline offsets).
    t_win : np.ndarray
        Time vector of shape ``(n_samples,)`` in seconds, aligned with the rows of ``v``.
    prune_the_data : bool, optional
        If True, run one pass of artefact-trial rejection and re-parameterize on the
        surviving trials. Default False.
    rejection_coeff : float, optional
        Robust (median/MAD) z-score threshold for the two-sided artefact-trial
        rejection. Larger is more permissive. Default 3.5.
    min_trials : int, optional
        Minimum number of trials required to attempt rejection, and minimum number that
        must remain after rejection. Default 5.

    Returns
    -------
    crp_parameters : dict
        - ``V_tR`` : truncated voltage matrix ``(T_R, K)`` over ``[0, tau_R]``.
        - ``C`` : canonical response shape ``C(t)``, unit-norm vector of length ``T_R``.
          Sign-normalized so the response is positively represented across trials.
        - ``al`` : projection weights ``alpha_k`` (length ``K``), in the same units as
          ``v`` times ``sqrt(#samples)``.
        - ``al_p`` : ``alpha_k`` normalized by ``sqrt(T_R)`` (paper's ``alpha'_k``, uV).
        - ``ep`` : residual ``eps_k(t)`` after removing ``alpha_k * C(t)``, shape
          ``(T_R, K)``.
        - ``erp`` : fitted evoked response ``alpha_k * C(t)``, shape ``(T_R, K)`` -- the
          part of each trial captured by the canonical shape (over the response window).
        - ``erp_full`` : ``erp`` embedded in a full-length ``(n_samples, K)`` array, zero
          after ``tau_R``. Subtract from ``v`` to remove the evoked response across the
          whole epoch.
        - ``Vsnr`` : per-trial signal-to-noise magnitude ``|alpha_k| / ||eps_k||``.
        - ``expl_var`` : per-trial fraction of variance explained by ``C(t)``.
        - ``tR`` : response duration ``tau_R`` in seconds.
        - ``epep_root`` : ``sqrt(diag(ep.T @ ep))``, per-trial residual norm.
        - ``avg_trace_tR`` / ``std_trace_tR`` : mean / std trace over ``[0, tau_R]``.
        - ``parms_times`` : time vector over ``[0, tau_R]``.
        - ``kept_trials`` : indices (into the **original** trial axis) of retained trials.
        - ``rejected_trials`` : indices (into the original trial axis) of rejected trials.
        - ``rejection_stat`` : per-original-trial anomaly statistic used for rejection
          (mean of the trial's ``S(tau_R)`` sub-distribution; ``NaN`` if not computed).
    crp_projections : dict
        - ``proj_tpts`` : profile time points (seconds).
        - ``s_all`` : all cross-projection magnitudes, shape ``(K^2 - K, n_tpts)``.
        - ``mean_proj_profile`` / ``var_proj_profile`` : mean / variance of ``S`` vs time.
        - ``tR_index`` : index into ``proj_tpts`` of the response duration.
        - ``avg_trace_input`` / ``std_trace_input`` : mean / std of the full input traces.
        - ``stat_indices`` : the non-overlapping half-selection of ``s_all`` rows used for
          significance and rejection.
        - ``t_value_tR`` / ``p_value_tR`` : extraction significance at ``tau_R``.
        - ``t_value_full`` / ``p_value_full`` : extraction significance at the full window.

    Notes
    -----
    The sign of ``C`` (and hence ``al`` / ``al_p``) is fixed so that the projection
    weights are predominantly positive; the product ``alpha_k * C(t)`` (``erp``) is
    sign-invariant regardless.

    Raises
    ------
    ValueError
        If ``v`` has 10 or fewer samples (the projection profile cannot be formed).

    Example
    -------
    Remove the evoked response from every trial across the full epoch, and inspect which
    trials were rejected as artefacts::

        params, proj = crp_method(v, t_win, prune_the_data=True)
        clean = v[:, params['kept_trials']] - params['erp_full']
        print('rejected trials:', params['rejected_trials'])
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 2:
        raise ValueError("'v' must be a 2D array of shape (n_samples, n_trials).")

    n_samples, n_trials = v.shape

    # Initial housekeeping
    sampling_rate = 1 / np.mean(np.diff(t_win))  # Get sampling rate

    # Calculate sets of normalized single stimulation cross-projection magnitudes.
    t_step = 5  # Timestep between timepoints (in samples)
    proj_tpts = np.arange(10, n_samples, t_step)  # Profile timepoints (in samples)
    if proj_tpts.size == 0:
        raise ValueError(
            "'v' must have more than 10 samples to build the projection profile; "
            f"got n_samples={n_samples}."
        )

    n_proj = n_trials ** 2 - n_trials  # number of off-diagonal cross-projections
    s_all = np.zeros((n_proj, proj_tpts.size))  # Projection weights per duration
    m = np.zeros(proj_tpts.size)   # Mean projection magnitudes
    v2 = np.zeros(proj_tpts.size)  # Variance of projection magnitudes
    for col, k in enumerate(proj_tpts):  # Different data lengths
        # Get projection magnitudes for this duration
        s = ccep_proj(v[:k, :])
        # Change units from uV*sqrt(samples) to uV*sqrt(seconds)
        s = s / np.sqrt(sampling_rate)
        m[col] = np.mean(s)
        v2[col] = np.var(s)
        s_all[:, col] = s
    tt = int(np.argmax(m))  # index into proj_tpts of the response duration

    # Parameterize trials over the response duration
    v_t_r = v[:proj_tpts[tt], :]  # Reduced-length voltage matrix (to response duration)
    e_t_r, _ = kt_pca(v_t_r)  # Linear kernel-trick PCA to capture structure
    c = e_t_r[:, 0]  # 1st PC, canonical shape C(t)

    al = np.dot(c, v_t_r)  # Alpha coefficient weights for C into V
    # Fix the (otherwise arbitrary) eigenvector sign so the canonical response is
    # positively represented across trials. alpha_k * C(t) is sign-invariant either way.
    if np.sum(al) < 0:
        c = -c
        al = -al
    ep = v_t_r - np.outer(c, al)  # Residual epsilon after removal of the CCEP form

    # Fitted evoked response (ERP) per trial, and full-epoch embedding for subtraction.
    erp = np.outer(c, al)  # (T_R, K) over the response window
    erp_full = np.zeros_like(v)
    erp_full[:proj_tpts[tt], :] = erp

    # ----- projections output -----
    crp_projections = {
        'proj_tpts': t_win[proj_tpts], 's_all': s_all,
        'mean_proj_profile': m, 'var_proj_profile': v2, 'tR_index': tt,
        'avg_trace_input': np.mean(v, axis=1), 'std_trace_input': np.std(v, axis=1),
    }

    # Significance statistics - only non-overlapping projections are used, so that each
    # trial is the normalized one half of the time and each pair is counted once.
    stat_indices = get_stat_indices(n_trials)
    crp_projections['stat_indices'] = stat_indices

    s_tR = s_all[stat_indices, tt]
    s_full = s_all[stat_indices, -1]
    crp_projections['t_value_tR'] = np.mean(s_tR) / (np.std(s_tR) / np.sqrt(len(s_tR)))
    _, crp_projections['p_value_tR'] = ttest_1samp(s_tR, 0, alternative='greater')
    crp_projections['t_value_full'] = np.mean(s_full) / (np.std(s_full) / np.sqrt(len(s_full)))
    _, crp_projections['p_value_full'] = ttest_1samp(s_full, 0, alternative='greater')

    # ----- parameterization output -----
    epep_root = np.sqrt(np.diag(ep.T @ ep))
    denom_snr = epep_root.copy()
    denom_snr[denom_snr == 0] = 1  # avoid divide-by-zero
    denom_var = np.diag(v_t_r.T @ v_t_r).copy()
    denom_var[denom_var == 0] = 1

    crp_parameters = {
        'V_tR': v_t_r, 'al': al, 'C': c, 'ep': ep, 'erp': erp, 'erp_full': erp_full,
        'tR': t_win[proj_tpts[tt]], 'parms_times': t_win[:proj_tpts[tt]],
        'avg_trace_tR': np.mean(v_t_r, axis=1), 'std_trace_tR': np.std(v_t_r, axis=1),
        'al_p': al / (len(c) ** 0.5), 'epep_root': epep_root,
        # "signal-to-noise" per trial, reported as a non-negative magnitude
        'Vsnr': np.abs(al) / denom_snr,
        'expl_var': 1 - np.diag(ep.T @ ep) / denom_var,
    }

    # ----- artefact-trial rejection (Miller et al. 2023) -----
    # Per-trial anomaly statistic: mean of the trial's sub-distribution of S(tau_R),
    # taken over the same non-overlapping half-selection used for significance.
    rejection_stat = _trial_projection_means(s_tR, stat_indices, n_trials)
    reject_local = _reject_artifact_trials(rejection_stat, rejection_coeff)

    if (prune_the_data and n_trials >= min_trials
            and reject_local.size > 0
            and (n_trials - reject_local.size) >= min_trials):
        kept_idx = np.setdiff1d(np.arange(n_trials), reject_local)
        # Re-run CRP on the surviving trials only (single pass; no further pruning).
        crp_parameters, crp_projections = crp_method(
            v[:, kept_idx], t_win, prune_the_data=False,
            rejection_coeff=rejection_coeff, min_trials=min_trials)
        # Map trial bookkeeping back onto the ORIGINAL trial axis.
        crp_parameters['kept_trials'] = kept_idx
        crp_parameters['rejected_trials'] = reject_local
        crp_parameters['rejection_stat'] = rejection_stat
        return crp_parameters, crp_projections

    crp_parameters['kept_trials'] = np.arange(n_trials)
    crp_parameters['rejected_trials'] = np.array([], dtype=int)
    crp_parameters['rejection_stat'] = rejection_stat

    return crp_parameters, crp_projections


def ccep_proj(V):
    """
    Semi-normalized pairwise cross-projections between trials.

    Each trial is L2-normalized and projected onto every other (raw) trial; the
    self-projections (diagonal) are discarded. For ``K`` trials this returns the
    ``K^2 - K`` off-diagonal magnitudes, flattened in row-major order so that element
    ``i * (K - 1) + (j or j - 1)`` is the projection of the normalized trial ``i`` onto
    the raw trial ``j``.

    :param V: Trial matrix of shape ``(n_samples, K)`` (each column is a trial).
    :type V: numpy.ndarray
    :return: Off-diagonal cross-projection magnitudes, shape ``(K^2 - K,)``.
    :rtype: numpy.ndarray
    """
    # Normalize (L2 norm) each trial
    denominator = np.sqrt(np.sum(V ** 2, axis=0))[np.newaxis, :]
    denominator[denominator == 0] = 1  # avoid divide-by-zero
    V0 = V / denominator
    V0[np.isnan(V0)] = 0  # guard any residual divide-by-zero in normalization

    # Internal projections (semi-normalized: one side normalized, one side raw)
    P = np.dot(V0.T, V)

    # Keep only off-diagonal elements (drop self-projections)
    p0 = P.copy()
    np.fill_diagonal(p0, np.nan)
    S0 = np.reshape(p0, (1, -1))
    S0 = S0[~np.isnan(S0)]
    return S0


def kt_pca(X):
    """
    Linear kernel PCA ("kernel trick") of ``X``.

    Implements the trick from Schölkopf et al., ICANN 1998, needed when the number of
    timepoints ``T`` greatly exceeds the number of trials ``N``: the eigenvectors of the
    ``T x T`` matrix ``X @ X.T`` are recovered from the eigendecomposition of the much
    smaller ``N x N`` matrix ``X.T @ X``.

    ``X.T @ X`` is symmetric positive-semidefinite, so ``np.linalg.eigh`` is used: it
    returns real, ascending eigenvalues and orthonormal eigenvectors (no complex
    round-off, no manual re-orthogonalization).

    :param X: Data matrix of shape ``(T, N)``.
    :return: ``(E, S)`` -- eigenvectors of ``X @ X.T`` (columns) and eigenvalues, both in
        descending order of eigenvalue.
    """
    # Symmetric eigendecomposition of the (small) N x N Gram matrix.
    S2, F = np.linalg.eigh(X.T @ X)  # ascending, real, orthonormal
    order = np.argsort(S2)[::-1]     # descending
    S2 = S2[order]
    F = F[:, order]

    # Eigenvalues of X.T@X (== nonzero eigenvalues of X@X.T) are >= 0; clip round-off.
    S = np.sqrt(np.clip(S2, 0, None))

    ES = X @ F  # Kernel trick: eigenvectors of X @ X.T (up to scaling)
    denominator = np.ones((X.shape[0], 1)) @ S.reshape(1, -1)
    denominator[denominator == 0] = 1  # avoid divide-by-zero
    E = ES / denominator  # unit-normalized eigenvectors

    return E, S


def get_stat_indices(N):
    """
    Non-overlapping half-selection of the cross-projection magnitudes for statistics.

    ``ccep_proj`` returns both orientations of every trial pair (``i`` normalized onto
    ``j``, and ``j`` normalized onto ``i``). Using both double-counts each interaction and
    inflates significance (Miller et al. 2023). This function selects **one orientation
    per unordered pair** -- exactly ``N * (N - 1) / 2`` of the ``N^2 - N`` projections --
    balancing which trial plays the normalized role so that each trial is the normalized
    one in as close to half of its pairs as possible (exactly half when ``N`` is odd).

    :param N: Number of trials.
    :return: Sorted flat indices into ``ccep_proj``'s output, shape ``(N * (N - 1) / 2,)``.
    """
    if N < 2:
        return np.array([], dtype=int)

    # Orient each unordered pair {a, b} by its circular distance (a regular-tournament
    # construction): this gives every trial the normalized role exactly (N-1)/2 times for
    # odd N, and balanced to within one for even N.
    selected = []
    for a in range(N):
        for b in range(a + 1, N):
            d = b - a  # 1 .. N-1
            if d < N / 2:
                i, j = a, b
            elif d > N / 2:
                i, j = b, a
            else:  # d == N/2 (only when N is even): split ties by parity to balance
                i, j = (a, b) if a % 2 == 0 else (b, a)
            selected.append(_flat_projection_index(i, j, N))

    return np.sort(np.array(selected, dtype=int))


def _flat_projection_index(i, j, N):
    """Flat index of projection (normalized ``i``, raw ``j``) in ``ccep_proj``'s output."""
    return i * (N - 1) + (j if j < i else j - 1)


def _flat_projection_pair(p, N):
    """Inverse of :func:`_flat_projection_index`: flat index ``p`` -> ``(i, j)``."""
    i, r = divmod(p, N - 1)
    j = r if r < i else r + 1
    return i, j


def _trial_projection_means(s_selected, stat_indices, N):
    """
    Mean of each trial's sub-distribution of the selected cross-projection magnitudes.

    For each trial ``k``, averages the selected ``S`` values in which ``k`` participates
    (either as the normalized or the raw trial). This is the per-trial statistic used to
    flag anomalous (artefactual) trials.

    :param s_selected: ``S`` values at the selected flat indices, shape ``(len(stat_indices),)``.
    :param stat_indices: Flat indices returned by :func:`get_stat_indices`.
    :param N: Number of trials.
    :return: Per-trial mean sub-distribution value, shape ``(N,)`` (``NaN`` if a trial
        appears in no selected pair).
    """
    sums = np.zeros(N)
    counts = np.zeros(N)
    for p, val in zip(stat_indices, s_selected):
        i, j = _flat_projection_pair(p, N)
        sums[i] += val
        sums[j] += val
        counts[i] += 1
        counts[j] += 1
    out = np.full(N, np.nan)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out


def _reject_artifact_trials(trial_stat, coeff):
    """
    Two-sided robust outlier detection on the per-trial anomaly statistic.

    Trials whose statistic lies more than ``coeff`` robust z-scores (median / scaled MAD)
    from the population median are flagged as artefactual, in either direction: a large
    artefact can project anomalously high (as the raw trial) or low (as the normalized
    trial).

    :param trial_stat: Per-trial statistic, shape ``(N,)``.
    :param coeff: Robust z-score threshold.
    :return: Indices of trials to reject.
    """
    valid = np.isfinite(trial_stat)
    if valid.sum() < 3:
        return np.array([], dtype=int)

    med = np.median(trial_stat[valid])
    mad = np.median(np.abs(trial_stat[valid] - med)) * 1.4826
    if mad == 0:  # degenerate spread -> fall back to std, else reject nothing
        mad = np.std(trial_stat[valid])
        if mad == 0:
            return np.array([], dtype=int)

    z = np.full(trial_stat.shape, 0.0)
    z[valid] = np.abs(trial_stat[valid] - med) / mad
    return np.where(z > coeff)[0]
