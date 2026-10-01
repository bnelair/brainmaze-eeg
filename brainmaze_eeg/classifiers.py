# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import warnings

import numpy as np
import pandas as pd
import scipy.signal as signal

from tqdm import tqdm
from copy import deepcopy, copy

from scipy.stats import gaussian_kde
from scipy.special import logsumexp
from scipy.signal import filtfilt
from scipy.signal.windows import gaussian
from scipy.optimize import differential_evolution
from scipy.linalg import norm
from scipy.stats._multivariate import  multivariate_normal_frozen

from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.svm import SVR, LinearSVC
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score, roc_curve
from sklearn.naive_bayes import GaussianNB

# from best.annotations.utils import time_to_timestamp, time_to_utc, merge_annotations
# from best.feature import augment_features, balance_classes
# from best.signal import unify_sampling_frequency, get_datarate, buffer
# from best.stats import kl_divergence_nonparametric
# from best.vector import scale, translate, get_mutual_vectors
# from best.modules import multivariate_normal_
# from best.modules import ZScoreModule, PCAModule
# from best.feature_extraction.FeatureExtractor import SleepSpectralFeatureExtractor
# from best.feature_extraction.SpectralFeatures import mean_bands, mean_frequency, relative_bands

from brainmaze_eeg.features.feature_extraction import SleepSpectralFeatureExtractor
from brainmaze_eeg.features.spectral_features import mean_bands, mean_frequency, relative_bands
from brainmaze_eeg.features.utils import augment_features, balance_classes
from brainmaze_eeg.scikit_modules import PCAModule, ZScoreModule
from brainmaze_utils.signal import unify_sampling_frequency, get_datarate
from brainmaze_utils.annotations import merge_annotations
from brainmaze_utils.stat import kl_divergence_nonparametric
from brainmaze_utils.vector import scale, translate

def _scores_to_annotations(scores, start_time, segm_size):
    """
    Turn per-epoch class scores into a merged annotation table.

    Parameters
    ----------
    scores : pd.DataFrame, shape (n_epochs, n_classes)
        One row per epoch, one column per class (any monotone score; the arg-max column
        is the label).
    start_time : np.ndarray, shape (n_epochs,), float, seconds
        Epoch start times relative to the start of the signal.
    segm_size : float, seconds
        Epoch length. ``end = start + segm_size`` and ``duration = segm_size`` before merging.

    Returns
    -------
    pd.DataFrame with columns ``['annotation', 'start', 'end', 'duration']``
        Consecutive epochs with the same label and touching bounds are merged.
        ``start`` / ``end`` / ``duration`` are in seconds relative to the signal start.
        Rows whose scores are all NaN (out-of-distribution epochs, see
        ``log_lik_floor``) are labelled ``UNKNOWN_LABEL`` (``'UNKNOWN'``).

    Notes
    -----
    Up to v1.0.0 ``predict_signal`` built this table with a hard-coded ``end = start + 30``
    and ``duration = 30`` (wrong for any ``segm_size != 30``), and passed it through
    ``time_to_utc -> merge_annotations -> time_to_timestamp``. ``merge_annotations``
    requires numeric ``start``/``end`` and rejects the datetimes that ``time_to_utc``
    produces, so ``predict_signal`` raised ``TypeError`` for every model. The UTC round
    trip was an identity on numeric seconds, so it is dropped (issue #57).
    """
    start_time = np.asarray(start_time, dtype=float)
    df = pd.DataFrame({
        'annotation': _labels_from_scores(scores),
        'start': start_time,
        'end': start_time + segm_size,
        'duration': float(segm_size),
    })
    if df.shape[0] == 0:
        return df
    df = merge_annotations(df)
    return df[['annotation', 'start', 'end', 'duration']]


def _smooth_scores(scores, window):
    """
    Zero-phase smoothing of each score column across consecutive rows (epochs).

    ``scores`` rows must be consecutive epochs in time order. ``filtfilt`` needs more than
    ``padlen`` samples; up to v1.0.0 the default ``padlen = 3 * len(window)`` made
    ``scores()`` raise ``ValueError`` for <= 9 epochs (default window). The pad length is
    now capped at ``n_epochs - 1`` (identical output for longer inputs).
    """
    n = scores.shape[0]
    window = np.asarray(window, dtype=float)
    if n == 0:
        return scores
    if window.size == 1:
        # single-tap window (``window_smooth_n=1``) = no smoothing; scipy < 1.15
        # ``filtfilt`` rejects filters of length 1
        for key in scores.keys():
            scores[key] = np.asarray(scores[key], dtype=float) * (window[0] ** 2)
        return scores
    padlen = min(3 * len(window), n - 1)
    for key in scores.keys():
        scores[key] = filtfilt(window, 1, np.asarray(scores[key], dtype=float), padlen=padlen)
    return scores


def _pairwise_differences(x):
    """
    All ordered pairwise difference vectors ``x[i] - x[j]`` for ``i != j``.

    Local replacement for ``brainmaze_utils.vector.get_mutual_vectors``, which (a) raises
    ``UnboundLocalError`` when called without labels (brainmaze_utils <= 1.0.x) and (b)
    includes the self-pairs ``i == j`` (zero vectors, whose unit direction is 0/0 = NaN).

    Parameters
    ----------
    x : np.ndarray, shape (n, d)

    Returns
    -------
    diff : np.ndarray, shape (n * (n - 1), d)
        Row ordering is ``i``-major: all ``j != i`` for ``i = 0``, then ``i = 1``, ...
    i_idx, j_idx : np.ndarray of int, shape (n * (n - 1),)
        The ``i`` and ``j`` of each row.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    i_idx, j_idx = np.nonzero(~np.eye(n, dtype=bool))
    return x[i_idx] - x[j_idx], i_idx, j_idx


def _norm_direction(diff):
    """
    Represent difference vectors as ``[||v||, v / ||v||]``.

    Parameters
    ----------
    diff : np.ndarray, shape (m, d)

    Returns
    -------
    np.ndarray, shape (m, d + 1)
        Column 0 is the Euclidean length, columns 1.. the unit direction. Rows with zero
        length (duplicate samples) get direction 0 instead of NaN.
    """
    length = norm(diff, axis=1)
    safe = np.where(length > 0, length, 1.0)
    return np.concatenate((length.reshape(-1, 1), diff / safe.reshape(-1, 1)), axis=1)


def _normalise_log_likelihood(log_lik):
    """
    Row-wise softmax of log-likelihoods: ``p[k, c] = L[k, c] / sum_c' L[k, c']`` computed
    stably as ``exp(log L - max log L)``.

    Up to v1.0.0 the KDE-family ``scores()`` divided raw pdfs, which underflow to 0 for
    every class for a sample far from all training data (0 / 0 = NaN -> ``predict``
    raised "Encountered all NA values"). Wherever the old computation did not underflow
    the result is identical up to floating-point round-off.
    """
    m = log_lik.max(axis=1)
    m = np.where(np.isfinite(m), m, 0.0)
    lik = np.exp(log_lik - m[:, None])
    tot = lik.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return lik / tot


#: Label given to epochs that cannot be classified: out-of-distribution epochs (best
#: state log-likelihood below the model's floor, see ``log_lik_floor``). Same string the
#: rest of the package uses for unscored epochs (e.g. ``SleepClassifierWrapper.train``
#: drops ``'UNKNOWN'`` training labels).
UNKNOWN_LABEL = 'UNKNOWN'

# default floor = 0.1 % quantile of the training epochs' (leave-one-out) best-state
# log-likelihood minus 30 nats; see ``_EpochClassifierMixin`` and the PR #69 notes.
_DEFAULT_LOG_LIK_QUANTILE = 0.001
_DEFAULT_LOG_LIK_MARGIN = 30.0

# maximum relative RMS error of the resampling self-test (see ``_resample_epochs``)
_RESAMPLE_SELF_TEST_TOL = 1e-3


def _labels_from_scores(scores):
    """
    Arg-max label per row of a score frame; rows that are entirely NaN get
    ``UNKNOWN_LABEL``. Returns np.ndarray of object, shape (n_rows,).
    """
    cols = np.asarray(list(scores.columns), dtype=object)
    arr = np.asarray(scores.to_numpy(dtype=float)).reshape(scores.shape[0], len(cols))
    out = np.full(arr.shape[0], UNKNOWN_LABEL, dtype=object)
    valid = ~np.all(np.isnan(arr), axis=1) if arr.shape[1] else np.zeros(arr.shape[0], bool)
    if valid.any():
        out[valid] = cols[np.nanargmax(arr[valid], axis=1)]
    return out


def _consecutive_runs(n, flagged, start_time=None, segm_size=None):
    """
    Split rows ``0..n-1`` into runs of consecutive, usable epochs.

    A run ends at a flagged row (excluded from every run) and, if ``start_time`` is given,
    wherever two neighbouring rows are not adjacent in time
    (``start_time[k] - start_time[k-1] != segm_size``, i.e. epochs were skipped between
    them). Returns a list of np.ndarray of row indices.
    """
    flagged = np.asarray(flagged, dtype=bool)
    if start_time is not None:
        st = np.asarray(start_time, dtype=float)
        if st.shape[0] != n:
            raise ValueError(f'start_time has {st.shape[0]} entries for {n} epochs')
        tol = 1e-6 * max(float(segm_size), 1.0)
        contiguous = np.r_[False, np.abs(np.diff(st) - float(segm_size)) <= tol]
    else:
        contiguous = np.r_[False, np.ones(max(n - 1, 0), dtype=bool)]
    runs, cur = [], []
    for k in range(n):
        if flagged[k]:
            if cur:
                runs.append(np.array(cur))
            cur = []
            continue
        if cur and not contiguous[k]:
            runs.append(np.array(cur))
            cur = []
        cur.append(k)
    if cur:
        runs.append(np.array(cur))
    return runs


def _utils_version():
    try:
        from importlib.metadata import version
        return version('brainmaze-utils')
    except Exception:  # pragma: no cover - metadata missing (e.g. source checkout)
        return 'unknown'


def _resampling_probe(t, f_ref):
    return (np.sin(2 * np.pi * 0.02 * f_ref * t)
            + 0.5 * np.sin(2 * np.pi * 0.07 * f_ref * t + 1.0)
            + 0.25 * np.sin(2 * np.pi * 0.15 * f_ref * t + 2.0))


def _resampling_self_test(fs, fs_new, n):
    """
    Resample a known 3-sine probe (0.02, 0.07 and 0.15 x ``min(fs, fs_new)``, i.e. well
    inside every anti-alias passband) of ``n`` samples from ``fs`` to ``fs_new`` with
    ``brainmaze_utils.signal.unify_sampling_frequency`` and raise ``ValueError`` unless
    the output is finite and its RMS error, excluding up to 1 s at each edge, is
    <= ``_RESAMPLE_SELF_TEST_TOL`` (1e-3) of the probe's RMS.

    Why: brainmaze_utils 2.0.0 designs a 16th-order Butterworth in transfer-function form,
    which is numerically unstable near the input Nyquist or at very low normalised
    cutoffs. Resampling to 200 Hz it raises below ~134 Hz, returns all-NaN at 134-138 Hz,
    returns values of 1e72-1e230 at 139-141 Hz and distorts the signal at 142-145 Hz
    (rel. error 0.29 at 142 Hz) and at >= ~1600 Hz (0.36 at 2048 Hz, all-NaN >= 3000 Hz).
    brainmaze_utils 2.1.0 (PR #28) fixes this (error <= 1e-6 at every rate tested).
    """
    f_ref = min(float(fs), float(fs_new))
    t = np.arange(int(n)) / float(fs)
    what = (f'resampling {float(fs):g} Hz -> {float(fs_new):g} Hz with brainmaze_utils '
            f'{_utils_version()}')
    hint = ('brainmaze_utils >= 2.1.0 resamples correctly at every rate; with older versions '
            'resample the data to the model rate yourself before calling this method.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y, _ = unify_sampling_frequency([_resampling_probe(t, f_ref)], [float(fs)], fs_new=fs_new)
    except Exception as e:
        raise ValueError(f'{what} fails ({type(e).__name__}: {e}). {hint}') from e
    y = np.asarray(y[0], dtype=float)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError(f'{what} returns non-finite values for a finite test signal. {hint}')
    expected = _resampling_probe(np.arange(y.size) / float(fs_new), f_ref)
    edge = int(min(float(fs_new), y.size // 4))
    core = slice(edge, y.size - edge)
    err = np.sqrt(np.mean((y[core] - expected[core]) ** 2)) / np.sqrt(np.mean(expected[core] ** 2))
    if not err <= _RESAMPLE_SELF_TEST_TOL:
        raise ValueError(f'{what} distorts a test signal (relative RMS error {err:.2g} > '
                         f'{_RESAMPLE_SELF_TEST_TOL:g}). {hint}')


def _resample_epochs(list_of_signals, fsamp_list, fs_new, owner='classifier'):
    """
    ``unify_sampling_frequency`` to ``fs_new`` with sanity checks (review R1 of PR #69).

    For every distinct (input rate, epoch length) that needs resampling the path is first
    checked with :func:`_resampling_self_test`. After resampling, every resampled epoch
    must (a) contain no +-inf, (b) keep at least half of its input's finite fraction
    (brainmaze_utils >= 2.1.0 re-applies NaN gaps, slightly widened; 2.0.0 fills them)
    and (c) not grow in RMS (about its mean) by more than 10x. Any violation raises
    ``ValueError``: a resampler failure must never turn into confident sleep labels.

    Returns ``(list of np.ndarray, fs_new)``.
    """
    data = [np.array(s, dtype=float, copy=True) for s in list_of_signals]
    fsamp = np.asarray(fsamp_list, dtype=float).reshape(-1)
    if fsamp.shape[0] != len(data):
        raise ValueError(f'{owner}: {len(data)} signals but {fsamp.shape[0]} sampling rates')
    need = np.flatnonzero(fsamp != float(fs_new))
    if need.size == 0:
        return data, fs_new
    for fs_k, n_k in sorted({(float(fsamp[k]), int(data[k].shape[-1])) for k in need}):
        _resampling_self_test(fs_k, fs_new, n_k)
    out, fs = unify_sampling_frequency([d.copy() for d in data], sampling_frequency=list(fsamp), fs_new=fs_new)
    out = [np.asarray(o, dtype=float) for o in out]
    for k in need:
        xin, xout = data[k], out[k]
        bad = None
        fin_in, fin_out = np.isfinite(xin), np.isfinite(xout)
        if np.any(np.isinf(xout)):
            bad = 'contains +-inf'
        elif fin_in.any() and fin_out.mean() < 0.5 * fin_in.mean():
            bad = (f'finite fraction dropped from {fin_in.mean():.3f} to {fin_out.mean():.3f}')
        elif fin_in.any() and fin_out.any():
            rms_in = np.sqrt(np.mean((xin[fin_in] - xin[fin_in].mean()) ** 2))
            rms_out = np.sqrt(np.mean((xout[fin_out] - xout[fin_out].mean()) ** 2))
            tiny = 1e-9 * max(np.max(np.abs(xin[fin_in])), 1e-300)
            if rms_out > 10.0 * rms_in + tiny:
                bad = f'RMS grew from {rms_in:.3g} to {rms_out:.3g}'
        if bad is not None:
            raise ValueError(f'{owner}: resampling epoch {k} from {fsamp[k]:g} Hz to {float(fs_new):g} Hz '
                             f'with brainmaze_utils {_utils_version()}: output {bad}. '
                             'brainmaze_utils >= 2.1.0 fixes the known resampling failures.')
    return out, fs


def _loo_logpdf_kde(kde, X):
    """
    Exact leave-one-out log density of each of a ``gaussian_kde``'s own (unweighted)
    training points. ``X`` is (n, d), the points the KDE was fitted on, in order.
    ``log p_{-i}(x_i) = logsumexp_{j != i}(-0.5 * maha(x_i, x_j)) - 0.5 log det(2 pi S) - log(n - 1)``
    with ``S = kde.covariance`` (the kernel covariance). Chunked, O(n^2 d).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 2:
        return np.full(n, -np.inf)
    L = np.linalg.cholesky(np.atleast_2d(kde.covariance))
    Z = np.linalg.solve(L, X.T).T
    c = -0.5 * (d * np.log(2 * np.pi)) - np.log(np.diag(L)).sum() - np.log(n - 1)
    sq = np.sum(Z ** 2, axis=1)
    out = np.empty(n)
    step = max(1, int(2e7 // max(n, 1)))
    for a in range(0, n, step):
        b = min(n, a + step)
        D = sq[a:b, None] + sq[None, :] - 2.0 * Z[a:b] @ Z.T
        np.maximum(D, 0.0, out=D)
        D[np.arange(b - a), np.arange(a, b)] = np.inf
        out[a:b] = logsumexp(-0.5 * D, axis=1) + c
    return out


def _resolve_log_lik_floor(spec, train_max_log_lik, quantile, margin):
    """``'auto'`` -> ``quantile(train_max_log_lik, quantile) - margin``; float -> itself;
    ``None`` -> ``-inf`` (disabled)."""
    if spec is None:
        return -np.inf
    if isinstance(spec, str):
        if spec != 'auto':
            raise ValueError(f"log_lik_floor must be 'auto', None or a float, got {spec!r}")
        finite = np.asarray(train_max_log_lik, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return -np.inf
        return float(np.quantile(finite, quantile) - margin)
    return float(spec)


class _EpochClassifierMixin:
    """
    Shared signal-level API of the epoch classifiers: epoching, skipping of low-datarate
    epochs, out-of-distribution flagging, labels and annotation tables.

    Out-of-distribution (OOD) epochs
    --------------------------------
    The class probabilities are a softmax over the states' log-likelihoods, so an epoch
    far from *every* trained state still gets probability ~1 for whichever state's density
    decays slowest. Such epochs (saturation, disconnection, heavy artefacts) are flagged
    instead: if an epoch's best-state log-likelihood ``max_s log p(x | s)`` is below
    ``self.log_lik_floor_``, its probabilities are NaN and its label is
    ``UNKNOWN_LABEL`` (``'UNKNOWN'``). Constructor parameters:

    ``log_lik_floor`` : ``'auto'`` (default), float or ``None``
        ``'auto'``: set in ``fit`` to ``quantile(train, log_lik_quantile) - log_lik_margin``
        where ``train`` is the best-state log-likelihood of every training epoch
        (leave-one-out for KDE densities, so it is not inflated by the epoch's own kernel).
        A float is used as is; ``None`` disables flagging.
    ``log_lik_quantile`` : float, default 0.001
    ``log_lik_margin`` : float, nats, default 30
        30 nats = a density about 1e13 times lower than the 0.1 %-quantile training epoch.
        Evidence (PR #69, through this pipeline, ~170 and ~700 training epochs, ~19 600
        held-out in-distribution epochs per run): false-flag rate 0 for Gaussian
        features (all models) and for synthetic raw EEG (1255 held-out epochs);
        1e-4 to 7e-4 for heavy-tailed Student-t5 features (some held-out epochs are
        farther from the training data than any training epoch). Artefact rows
        (features + 1e3) were flagged in 100 % of cases, flat/disconnected raw epochs
        too. Smaller margins flag more in-distribution epochs (Student-t features:
        about 1e-3 at 10 nats); larger ones miss moderate outliers.

    Limits: OOD is judged in the model's own feature space (after selection / PCA for the
    KDE family), so a change confined to discarded features or directions is invisible,
    and the spectral features are amplitude-invariant ratios, so artefacts that keep
    the spectral shape (e.g. clipping / saturation of a normal-looking epoch) are not
    flagged. A row about 7 SD from every state in that space is not flagged by default
    (lower ``log_lik_margin`` to be stricter).

    After ``fit``: ``log_lik_floor_`` (float, ``-inf`` if disabled) and
    ``train_max_log_lik_`` (np.ndarray, (n_train,)). After every ``scores`` call:
    ``max_log_lik_`` (np.ndarray, (n_epochs,)), the best-state log-likelihood of each
    scored row, in the same order. :meth:`max_log_likelihood` computes it directly.
    """

    def _log_lik_params(self):
        # getattr: models pickled before these parameters existed keep working (no floor)
        return (getattr(self, 'log_lik_floor', None),
                getattr(self, 'log_lik_quantile', _DEFAULT_LOG_LIK_QUANTILE),
                getattr(self, 'log_lik_margin', _DEFAULT_LOG_LIK_MARGIN))

    def _fit_log_lik_floor(self, train_max_log_lik):
        spec, q, m = self._log_lik_params()
        self.train_max_log_lik_ = np.asarray(train_max_log_lik, dtype=float)
        self.log_lik_floor_ = _resolve_log_lik_floor(spec, self.train_max_log_lik_, q, m)

    def _ood_mask(self, max_log_lik):
        """True where an epoch is out of distribution (below the floor) or non-finite."""
        floor = getattr(self, 'log_lik_floor_', -np.inf)
        max_log_lik = np.asarray(max_log_lik, dtype=float)
        return ~np.isfinite(max_log_lik) | (max_log_lik < floor)

    def predict(self, X, start_time=None):
        """
        Arg-max state per epoch, np.ndarray of str, shape (n_epochs,).
        Out-of-distribution epochs (all-NaN score rows) get ``UNKNOWN_LABEL``.
        ``start_time`` as in ``scores``.
        """
        return _labels_from_scores(self.scores(X, start_time=start_time))

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def preprocess_signal(self, signal, fs, datarate_threshold=0.85):
        """
        Cut ``signal`` into ``self.segm_size``-s epochs (an incomplete last epoch is
        dropped) and keep those whose fraction of non-NaN samples is
        ``>= datarate_threshold``.

        Returns ``(list of epochs, start_time, end_time)``; times in seconds from the first
        sample. All three are empty if the signal is shorter than one epoch or every epoch
        is below the threshold.
        """
        n_segm = int(round(fs * self.segm_size))
        signal = np.asarray(signal, dtype=float).reshape(-1)
        n_ep = signal.shape[0] // n_segm if n_segm > 0 else 0
        data = signal[:n_ep * n_segm].reshape(n_ep, n_segm)
        start_time = np.arange(n_ep, dtype=float) * self.segm_size
        end_time = start_time + self.segm_size
        if n_ep == 0:
            return [], start_time, end_time
        datarate = np.asarray(get_datarate(data))
        keep = datarate >= datarate_threshold
        return list(data[keep]), start_time[keep], end_time[keep]

    def _signal_scores(self, signal, fs, datarate_threshold):
        data, start_time, _ = self.preprocess_signal(signal, fs, datarate_threshold)
        if len(data) == 0:
            self.max_log_lik_ = np.zeros(0)
            return pd.DataFrame(np.zeros((0, len(self.STATES))), columns=list(self.STATES)), start_time
        x, _ = self.extract_features_bulk(data, [fs] * len(data))
        return self.scores(x, start_time=start_time), start_time

    def predict_signal(self, signal, fs, datarate_threshold=0.85):
        """
        Classify a continuous single-channel signal epoch by epoch.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples,)
            Raw signal; NaN marks missing data.
        fs : float, Hz
            Sampling rate of ``signal``. Epochs are resampled to ``self.fs`` (KDE family;
            checked, see ``_resample_epochs``) exactly as in ``extract_features_bulk``.
        datarate_threshold : float, 0-1
            Epochs with a smaller fraction of non-NaN samples are skipped.

        Returns
        -------
        pd.DataFrame with columns ``['annotation', 'start', 'end', 'duration']``
            Times in seconds relative to the first sample; epochs of ``self.segm_size``
            seconds, consecutive equal labels merged (see ``_scores_to_annotations``).
            Skipped (low-datarate) epochs leave a gap in the table. Out-of-distribution
            epochs are annotated ``'UNKNOWN'``. Smoothing (and the Markov filter of the
            causal models) runs separately on each run of consecutive scored epochs: it
            never crosses a skipped or out-of-distribution epoch. Empty table (same
            columns) if no epoch is usable.
        """
        scores, start_time = self._signal_scores(signal, fs, datarate_threshold)
        return _scores_to_annotations(scores, start_time, self.segm_size)

    def predict_signal_scores(self, signal, fs, datarate_threshold=0.85):
        """
        As :meth:`predict_signal` but returns the per-epoch score frame (one row per kept
        epoch, columns ``self.STATES``; NaN rows = out-of-distribution). The row's epoch
        start times are ``self.preprocess_signal(signal, fs, datarate_threshold)[1]``, and
        ``self.max_log_lik_`` holds each row's best-state log-likelihood.
        """
        return self._signal_scores(signal, fs, datarate_threshold)[0]


class _KDEFamilyMixin(_EpochClassifierMixin):
    """
    Density / scoring part shared by :class:`KDEBayesianModel` (and its subclasses) and
    :class:`KDEBayesianModelNC`. Requires ``transform``, ``segm_size``, ``fs``, ``WINDOW``
    and ``CAT_BIAS`` on the instance.
    """

    def extract_features_bulk(self, list_of_signals, fsamp_list, return_names=False):
        """
        Features of a list of epochs, after resampling each to ``self.fs`` with
        ``unify_sampling_frequency`` (checked by ``_resample_epochs``: raises
        ``ValueError`` instead of returning features of a corrupted resampled signal).

        Returns ``(features (n_epochs, n_features), fs)``, or ``(features,
        feature_names)`` if ``return_names``.
        """
        data, fs = _resample_epochs(list_of_signals, fsamp_list, self.fs, owner=type(self).__name__)
        x = []
        for k in tqdm(range(data.__len__())):
            x += [self.extract_features(data[k])]
        if return_names:
            _, feature_names = self.extract_features(data[k], return_names=True)
            return np.array(x), feature_names
        return np.array(x), fs

    def _make_density(self, X_state):
        """Density of one state, fitted on (n_state, d) transformed features."""
        return gaussian_kde(X_state.T)

    def _fit_kde(self, X, y):
        self.STATES = np.unique(y)
        self.KDE = []
        for state in self.STATES:
            X_ = X[y==state, :]
            self.KDE.append(self._make_density(X_))
        # training best-state log-likelihood; an epoch's own state uses the leave-one-out
        # density for KDEs (its own kernel would inflate it)
        ll = np.array(self._log_likelihood(X), dtype=float).reshape(X.shape[0], len(self.STATES))
        for idx, state in enumerate(self.STATES):
            if isinstance(self.KDE[idx], gaussian_kde):
                sel = np.asarray(y) == state
                ll[sel, idx] = _loo_logpdf_kde(self.KDE[idx], X[sel, :])
        self._fit_log_lik_floor(ll.max(axis=1))

    def _likelihood(self, X):
        """
        Class-conditional densities in the transformed feature space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_selected_features)
            Already transformed features (output of ``transform``).

        Returns
        -------
        pd.DataFrame, shape (n_samples, n_states)
            ``p(x | state)`` per column (raw pdf values, not normalised; may underflow to 0).
        """
        scores = {}
        for idx, kde in enumerate(self.KDE):
            scores[self.STATES[idx]] = np.atleast_1d(kde.pdf(X.T))
        scores = pd.DataFrame(scores)
        return scores

    def _log_likelihood(self, X):
        """Same as ``_likelihood`` but ``log p(x | state)`` (no underflow)."""
        # atleast_1d: scipy's multivariate normal returns a scalar for a single sample
        return pd.DataFrame({self.STATES[idx]: np.atleast_1d(kde.logpdf(X.T)) for idx, kde in enumerate(self.KDE)})

    def max_log_likelihood(self, X):
        """
        Best-state log-likelihood ``max_s log p(x | s)`` of each row of the raw feature
        matrix ``X`` (n_epochs, n_features); np.ndarray, shape (n_epochs,). Compare with
        ``self.log_lik_floor_`` / ``self.train_max_log_lik_``.
        """
        Xt = self.transform(X)
        ll = np.asarray(self._log_likelihood(Xt), dtype=float).reshape(Xt.shape[0], len(self.STATES))
        return ll.max(axis=1) if ll.shape[1] else np.full(Xt.shape[0], -np.inf)

    def _filter_run(self, scores, first_run):
        """Post-processing of one run of consecutive epochs (identity; the causal models
        apply the Markov filter here)."""
        return scores

    def scores(self, X, start_time=None):
        """
        Per-epoch posterior-like class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_features)
            Raw feature matrix (same columns as used in ``fit``), rows in time order.
        start_time : array-like, shape (n_epochs,), seconds, optional
            Start time of each row's epoch. If given, rows that are not exactly
            ``segm_size`` apart start a new run (e.g. low-datarate epochs were skipped
            between them). If omitted, all rows are treated as consecutive epochs.

        Returns
        -------
        pd.DataFrame, shape (n_epochs, n_states)
            Columns are ``self.STATES``. Steps: per-state log-likelihoods; rows whose best
            log-likelihood is below ``self.log_lik_floor_`` (out of distribution) or
            non-finite become all-NaN; the others are normalised across states (equal
            priors, log space); then, separately on each run of consecutive non-NaN
            epochs, zero-phase smoothing with ``self.WINDOW`` (``window_smooth_n=1``
            disables it), multiplication by ``cat_bias``, renormalisation (rows sum to 1)
            and, for the causal models, the Markov filter. ``self.max_log_lik_`` is set to
            each row's best-state log-likelihood.
        """
        return self._scores_impl(X, start_time=start_time, apply_floor=True, apply_filter=True)

    def _scores_impl(self, X, start_time=None, apply_floor=True, apply_filter=True):
        X = self.transform(X)
        n = X.shape[0]
        states = list(self.STATES)
        log_lik = np.asarray(self._log_likelihood(X), dtype=float).reshape(n, len(states))
        max_ll = log_lik.max(axis=1) if len(states) else np.full(n, -np.inf)
        self.max_log_lik_ = max_ll
        if apply_floor:
            flagged = self._ood_mask(max_ll)
        else:
            flagged = ~np.isfinite(max_ll)
        prob = _normalise_log_likelihood(log_lik)
        out = np.full((n, len(states)), np.nan)
        for i_run, run in enumerate(_consecutive_runs(n, flagged, start_time, self.segm_size)):
            sc = pd.DataFrame(prob[run], columns=states)
            sc = _smooth_scores(sc, self.WINDOW)
            for cat in self.CAT_BIAS.keys():
                if cat in sc.keys(): sc[cat] = sc[cat]*self.CAT_BIAS[cat]
            sc = sc.div(sc.sum(axis=1), axis=0)
            if apply_filter:
                sc = self._filter_run(sc, first_run=(i_run == 0))
            out[run] = sc[states].to_numpy(dtype=float)
        return pd.DataFrame(out, columns=states)


class multivariate_normal_(multivariate_normal_frozen):
    """
    Frozen multivariate normal fitted to data (sample mean and covariance).

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        Training data, **features in rows** (same convention as ``scipy.stats.gaussian_kde``).
    allow_singular, seed : see ``scipy.stats.multivariate_normal``.

    ``pdf`` / ``logpdf`` take the same (n_features, n_samples) layout and return shape
    (n_samples,).
    """
    def __init__(self, X, allow_singular=False, seed=None):
        X = X.copy().T
        cov = np.cov(X.T)
        mu = X.mean(axis=0)
        super().__init__(mu, cov, allow_singular, seed)

    def logpdf(self, X):
        # call the base implementation explicitly: depending on the scipy version the
        # frozen ``pdf`` is implemented via ``self.logpdf``, so delegating ``pdf`` to
        # ``super().pdf`` would transpose twice.
        return multivariate_normal_frozen.logpdf(self, np.asarray(X).T)

    def pdf(self, X):
        return np.exp(self.logpdf(X))


class KDEBayesianModel(_KDEFamilyMixin):
    """
    Single-channel sleep classifier: spectral features -> standardisation -> feature
    selection -> PCA -> z-score -> per-state density model -> equal-prior posterior,
    smoothed over epochs, with out-of-distribution epochs flagged ``'UNKNOWN'``.

    Pipeline fitted by ``fit(X, y)`` (``X``: raw feature matrix, e.g. from
    ``extract_features_bulk``):

    1. classes are balanced by duplicating minority-class rows (``balance_classes``);
       the balanced copy is used to fit every *transform* below, the unbalanced data
       to fit the densities;
    2. if ``standardize`` (default): ``StandardScaler`` (mean/std of the balanced
       training data, frozen). The raw features mix units (Hz, [0, 1] ratios, log10
       ratios) and steps 3-4 are not scale-invariant: without this step a feature's
       influence depends on its unit (PR #69 review R4: discriminative features x 0.1
       -> chance-level accuracy without any error). With it the whole model is
       invariant to per-feature affine rescaling of the input;
    3. ``RFECV`` with a linear ``SVR`` regressing the label-encoded class index
       (alphabetical order) selects features (``step=5``, ``>= 4`` features);
    4. ``PCAModule`` keeps the components explaining ``>= 98 %`` of variance;
    5. ``ZScoreModule`` (trainable): mean/std of the balanced training data, stored and
       re-used unchanged at prediction time (no test-set statistics);
    6. optional ``SelectFromModel`` (L1 ``LinearSVC``, ``max_features=4``) if
       ``Selector2`` (``ValueError`` if it keeps no feature);
    7. one ``gaussian_kde`` per state on the transformed unbalanced training data, and
       the out-of-distribution floor (see ``log_lik_floor``).

    Validation status (PR #69): every class is exercised end to end on synthetic data
    with known structure (feature-level Gaussian classes and synthetic sine + noise
    "EEG"), against a QDA / Gaussian naive Bayes reference. This checks the wiring and
    the numerics; it is not a validation of sleep-staging accuracy on real recordings.

    Parameters
    ----------
    fbands : list of [low, high], Hz
        Bands for the spectral features.
    segm_size : float, seconds
        Epoch length (also used for annotation bounds in ``predict_signal``).
    fs : float, Hz
        Sampling rate the feature extractor expects (signals are resampled to it in
        ``extract_features_bulk``).
    bands_to_erase : list of [low, high], Hz
        Spectral ranges ignored by the extractor (e.g. stimulation artefacts).
    filter_bands, filter_order :
        Kept for API compatibility, unused.
    nfft : int
        FFT length of the extractor.
    window_smooth_n, window_std :
        Gaussian smoothing window (taps, std in taps) applied to the scores across
        consecutive epochs. ``window_smooth_n=1`` disables smoothing.
    cat_bias : dict state -> float
        Multiplicative per-state bias applied to the smoothed scores.
    Selector2 : bool
        Enable step 6.
    n_jobs : int
        Parallel jobs for ``RFECV`` (default 10, as hard-coded up to v1.0.0).
    standardize : bool
        Enable step 2 (default True). ``False`` reproduces the v1.0.0 pipeline.
    log_lik_floor, log_lik_quantile, log_lik_margin :
        Out-of-distribution floor, see :class:`_EpochClassifierMixin` (default
        ``'auto'``, 0.001, 30 nats). ``log_lik_floor=None`` disables flagging.
    """
    __name__ = "KDEBayesianModel"
    def __init__(self, fbands=[[0.5, 5], # delta
                               [4, 9], # theta
                               [8, 14], # alpha
                               [11, 16], # spindle
                               [14, 20],
                               [20, 30]], segm_size=30, fs=200, bands_to_erase=[], filter_bands = True, nfft=12000,
                 window_smooth_n=3, window_std=1, cat_bias={'AWAKE': 1, 'N2': 1, 'N3': 1, 'REM': 1}, Selector2=True,
                 n_jobs=10, standardize=True, log_lik_floor='auto',
                 log_lik_quantile=_DEFAULT_LOG_LIK_QUANTILE, log_lik_margin=_DEFAULT_LOG_LIK_MARGIN):

        self.fbands = fbands
        self.n_jobs = n_jobs
        self.standardize = standardize
        self.log_lik_floor = log_lik_floor
        self.log_lik_quantile = log_lik_quantile
        self.log_lik_margin = log_lik_margin
        self.segm_size = segm_size
        self.fs = fs
        self.bands_to_erase = bands_to_erase
        self.filter_bands = filter_bands
        self.nfft=nfft
        self.filter_order = 100

        self.STATES = []
        self.KDE = []
        self.PipelineClustering = None
        self.FeatureSelector = None

        self.SELECTOR2 = Selector2

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            ignore_bands=self.bands_to_erase,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )




        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                mean_bands,
            ]



        self.FeatureExtractor = SleepSpectralFeatureExtractor(
            fs=self.fs,
            fbands=self.fbands,
            ignore_bands=self.bands_to_erase,
            segm_size=self.segm_size,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )

        self.FeatureExtractor._extraction_functions = \
            [
                mean_frequency,
                #self.FeatureExtractor.MedFreq,
                relative_bands,
                #self.FeatureExtractor.normalized_entropy,
                #self.FeatureExtractor.normalized_entropy_bands
            ]


        self.WINDOW = gaussian(window_smooth_n, window_std)
        self.WINDOW = self.WINDOW / self.WINDOW.sum()
        self.CAT_BIAS = cat_bias
        self.feature_names = None


    def extract_features(self, signal, return_names=False):
        if signal.ndim > 1:
            raise AssertionError('[INPUT ERROR]: Input data has to be of a dimension size 1 - raw signal')
        if signal.shape[0] != self.fs * self.segm_size:
            print('[INPUT WARNING]: input data is not a defined size fs*segm_size ' + str(self.fs*self.segm_size) + '; Signal of a size ' + str(signal.shape[0]) + ' found instead. Extracted features might be inaccurate.')


        ## Mean band-derived features - delta/beta ratio etc
        mean_bands, feature_names = self.FeatureExtractor_MeanBand(signal)
        mean_bands = np.concatenate(mean_bands)

        functions = [np.divide]
        symbols = ['/']
        mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
        for idx in range(functions.__len__()):
            mean_band_derived_features, mean_band_derived_names = augment_features(

                mean_band_derived_features.reshape(1, -1), feature_indexes=np.arange(mean_band_derived_features.shape[0]), operation=functions[idx], mutual=True,  operation_str=symbols[idx], feature_names=mean_band_derived_names

            )


        mean_band_derived_names = mean_band_derived_names[feature_names.__len__():]
        mean_band_derived_features = mean_band_derived_features[0, feature_names.__len__():]
        #mean_band_derived_names = mean_band_derived_names.squeeze()

        #features = np.log10(np.append(other_features, mean_band_derived_features))
        #feature_names = feature_names + mean_band_derived_names
        features = np.log10(mean_band_derived_features)
        #feature_names = mean_band_derived_names

        ## other features
        other_features, feature_names_other = self.FeatureExtractor(signal)
        other_features = np.concatenate(other_features)
        features = np.append(other_features, features)
        feature_names = list(feature_names_other) + list(mean_band_derived_names)


        self.feature_names = feature_names
        if return_names:
            return features, feature_names
        return features

    def fit(self, X, y):
        """
        Fit the transform pipeline and the per-state densities.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_features)
            Feature matrix (one row per epoch).
        y : np.ndarray of str, shape (n_epochs,)
            Labels; every state needs enough rows for a non-singular density
            (more rows than retained dimensions).
        """
        X, y = self._fit(X, y)
        self._fit_kde(X, y)

    def _fit(self, X, y):

        X = deepcopy(X)
        y = deepcopy(y)
        X_, y_ = balance_classes(X, y, std_factor=0.0)

        # step 2: standardise the raw features before the scale-dependent RFECV / PCA
        # (balanced training statistics, frozen; review R4 of PR #69)
        self.Scaler = preprocessing.StandardScaler() if getattr(self, 'standardize', False) else None
        if self.Scaler is not None:
            X_ = self.Scaler.fit_transform(X_)
            X = self.Scaler.transform(X)

        estimator = SVR(kernel="linear")
        self.SELECTOR = RFECV(estimator, step=5, verbose=True, min_features_to_select=4, n_jobs=self.n_jobs)
        self.PCA = PCAModule(var_threshold=0.98)
        self.ZScore = ZScoreModule(trainable=True, continuous_learning=False, multi_class=False)

        #self.UMAP = UMAP(n_neighbors=30, min_dist=1,
        #                 n_components=2)

        le = preprocessing.LabelEncoder()
        le.fit(y_)
        y__ = le.transform(y_)


        X_ = self.SELECTOR.fit_transform(X_, y__)
        X_ = self.PCA.fit_transform(X_)
        # z-score statistics come from the balanced training data and are then frozen.
        # Up to v1.0.0 the unbalanced X was passed to ``fit_transform`` here as well,
        # silently replacing the balanced fit with unbalanced statistics (issue #57).
        # Prediction output is unchanged by this fix: a per-feature affine change of the
        # space rescales every state's density by the same Jacobian, which cancels in
        # the normalised scores; ``SelectFromModel`` is fitted on X_ either way.
        X_ = self.ZScore.fit_transform(X_, y_)

        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
        X = self.ZScore.transform(X)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_, y_)
        if self.SELECTOR2:
            self.SELECTOR2 = SelectFromModel(lsvc, prefit=True, max_features=4)
            #X_ = self.SELECTOR2.transform(X_)
            with warnings.catch_warnings():
                # sklearn warns "No features were selected"; we raise below instead
                warnings.simplefilter('ignore', UserWarning)
                X = self.SELECTOR2.transform(X)
            if X.shape[1] == 0:
                raise ValueError(f'{type(self).__name__}.fit: the L1 LinearSVC of Selector2 kept no '
                                 'feature (all coefficients are 0), so no density can be fitted. '
                                 'Use Selector2=False or more / more separable training data.')

        #X = self.UMAP.fit_transform(X)
        return X, y


    def transform(self, X):
        if getattr(self, 'Scaler', None) is not None:
            X = self.Scaler.transform(X)
        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
        X = self.ZScore.transform(X)
        if self.SELECTOR2:
            X = self.SELECTOR2.transform(X)
        #X = self.UMAP.transform(X)
        return X

class KDEBayesianCausalModel(KDEBayesianModel):
    __name__ = "KDEBayesianCausalModel"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MarkovFilter = None

    def fit(self, X, y):
        super().fit(X, y)
        # training scores for the filter: no OOD flagging, no filter, rows consecutive
        scores = self._scores_impl(X, apply_floor=False, apply_filter=False)
        self.MarkovFilter = SleepStageProbabilityMarkovChainFilter()
        self.MarkovFilter.fit(scores, y)

    def _filter_run(self, scores, first_run):
        """
        Markov-chain filter (causal forward pass) over one run of consecutive epochs.

        The first run starts in ``'AWAKE'`` if that state was trained, otherwise in the
        first trained state (up to v1.0.0 it always started in ``'AWAKE'`` and raised
        ``IndexError`` if the training labels had no ``'AWAKE'``). A later run (after
        skipped or out-of-distribution epochs) starts in the arg-max state of its first
        epoch: the one-epoch transition prior is not applied across a gap of unknown
        length.
        """
        scores = scores.reset_index(drop=True)
        if first_run:
            state = 'AWAKE' if 'AWAKE' in self.MarkovFilter.STATES else self.MarkovFilter.STATES[0]
        else:
            state = scores.iloc[0].idxmax()
        return self.MarkovFilter.predict(scores, state)

    def _scores(self, X, start_time=None):
        """Scores without the Markov filter."""
        return self._scores_impl(X, start_time=start_time, apply_floor=True, apply_filter=False)


class MVGaussBayesianModel(KDEBayesianModel):
    """:class:`KDEBayesianModel` with one multivariate Gaussian (sample mean and
    covariance) per state instead of a KDE."""
    __name__ = "MVGaussBayesianModel"

    def _make_density(self, X_state):
        return multivariate_normal_(X_state.T)


class MVGaussBayesianCausalModel(MVGaussBayesianModel, KDEBayesianCausalModel):
    """:class:`MVGaussBayesianModel` densities + the Markov-chain filter of
    :class:`KDEBayesianCausalModel`. ``__name__`` was ``"MVGaussBayesianModel"`` up to
    v1.0.0 (copy-paste); it is now ``"MVGaussBayesianCausalModel"``."""
    __name__ = "MVGaussBayesianCausalModel"


class SleepStageProbabilityMarkovChainFilter:
    # canonical states and transition matrix; ``fit`` always starts from these, so a
    # refit is not limited to the states of a previous fit (Copilot, PR #69)
    _ALL_STATES = ('AWAKE', 'N1', 'N2', 'N3', 'REM')
    _ALL_TMAT = ((0.961, 0.038, 0.001, 0.000, 0.000),
                 (0.097, 0.215, 0.634, 0.000, 0.054),
                 (0.020, 0.001, 0.846, 0.060, 0.073),
                 (0.005, 0.001, 0.105, 0.880, 0.009),
                 (0.017, 0.003, 0.061, 0.000, 0.918))

    def __init__(self):
        self.STATES = np.array(self._ALL_STATES)
        self.removed_classes = []
        self.stability = np.ones(self.STATES.__len__())
        self._tmat_orig = np.array(self._ALL_TMAT, dtype=float)

        #self._tmat_orig = np.array(
        #    [
        #        [0.802, 0.19,  0.007, 0.,    0.002],
        #        [0.11,  0.625, 0.24,  0.,    0.026],
        #        [0.018, 0.046, 0.89,  0.036, 0.01 ],
        #        [0.007, 0.006, 0.171, 0.815, 0.001],
        #        [0.017, 0.043, 0.012, 0.,    0.927]
        #    ])


        self.tmat = self._tmat_orig.copy()




        #np.fill_diagonal(self.tmat, self.tmat.diagonal()*stability)
        #diag = self.tmat.diagonal()
        #self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)


        #diag = self.tmat.diagonal() * stability
        #self.tmat = self.tmat * (1-stability)
        #np.fill_diagonal(self.tmat, diag)
        self._get_vars()

    def fit(self, scores, y):
        """
        Adapt the transition matrix to the trained states and set per-state stability.

        Parameters
        ----------
        scores : pd.DataFrame, shape (n_epochs, n_states)
            Training-set class probabilities (columns = state names).
        y : array-like of str, shape (n_epochs,)
            Training labels. Must be a subset of ``AWAKE, N1, N2, N3, REM``; other
            labels raise ``ValueError`` (up to v1.0.0 they were silently ignored by the
            filter). Every ``fit`` starts again from all five states and the canonical
            transition matrix, then removes the states absent from ``y``.

        Notes
        -----
        Stability of a state = ``1 - t*``, where ``t*`` is the ROC threshold on that
        state's score maximising ``sqrt((1 - FPR)^2 + TPR^2)``. scikit-learn >= 1.3 puts
        ``+inf`` as the first ROC threshold; if that point won (a weak score), the
        stability became ``-inf`` and the whole transition matrix NaN. Only finite
        thresholds are now considered and the stability is clipped to [0, 1].
        """
        unknown = sorted(set(np.unique(np.asarray(y)).tolist()) - set(self._ALL_STATES))
        if unknown:
            raise ValueError(f'SleepStageProbabilityMarkovChainFilter: labels {unknown} are not '
                             f'in the supported states {list(self._ALL_STATES)}')
        self.STATES = np.array(self._ALL_STATES)
        self.removed_classes = []
        self._tmat_orig = np.array(self._ALL_TMAT, dtype=float)
        self.tmat = self._tmat_orig.copy()
        classes = np.unique(y)
        #class_certainty = dict([[state, X[state][y==state].median()] for state in classes])

        for cl in self.STATES:
            if cl not in classes:
                self.remove_class(cl)

        self._tmat_orig = self.tmat.copy()

        #self.correct_certainty(class_certainty)

        stability = []
        for state in self.STATES:
            fpr, tpr, thresholds = roc_curve(np.array(y)==state, np.array(scores[state]))
            #fpr, tpr, thresholds = precision_recall_curve(np.array(y)==state, np.array(scores[state]))
            finite = np.isfinite(thresholds)
            fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
            t = np.sqrt((1-fpr)**2 + tpr**2)
            #t = 2*(fpr*tpr) / (fpr+tpr)
            s = float(np.clip(1 - thresholds[t.argmax()], 0.0, 1.0))

            #fpr, tpr, thresholds = precision_recall_curve(np.array(y)==state, np.array(scores[state]))
            #t = np.sqrt(fpr**2 + tpr**2)
            #t = 2 * fpr * tpr / (fpr+tpr)
            #s = thresholds[t.argmax()]
            stability += [s]


            #precision_recall_curve
        #stability = self._optimize(scores, y)
        #stability = [0.51861717, 0.39611577, 1.09778543, 0.32874582]
        #stability = [0.12461134, 0.3359193,  0.18570281, 0.37578364]
        #stability = [0.10092481, 0.30177824, 0.21477364, 0.25693041]
        #stability = [0.06515025678829323, 0.3012126596534704, 0.18660074488436126, 0.2829450179411525]
        #stability = [0.75009757, 0.46948611, 0.53506914, 0.78013241]
        self.reset_probabilities()
        self.weight_probabilities(stability)
        self.stability = stability

    def fit_optimize(self, scores, y, Niter=200, popsize=10):
        self.fit(scores, y)

        stability = self._optimize(scores, y, Niter, popsize)
        self.reset_probabilities()
        self.weight_probabilities(stability)
        self.stability = stability


    def _get_vars(self):
        self.tmat = copy(self.tmat / self.tmat.sum(axis=1).reshape(-1, 1))
        self.change_prob = copy(1 - self.tmat.diagonal())
        self.prob_change_to = copy(self.tmat / (self.tmat.sum(axis=1) - self.tmat.diagonal()).reshape(-1, 1))
        self.prob_change_to[range(self.prob_change_to.shape[0]), range(self.prob_change_to.shape[1])] = 0

    def get_state_idx(self, state):
        return np.where(self.STATES == state)[0][0]

    def get_state_priors(self, state):
        idx = self.get_state_idx(state)
        priors = self.tmat[idx]
        return priors

    def get_state_change_prior(self, state):
        state_idx = self.get_state_idx(state)
        priors = self.get_state_priors(state)
        change_prior = 1 - priors[state_idx]
        return change_prior

    def get_prob_to_change(self, state, x_prob):
        idx = self.get_state_idx(state)
        prob_to_stay = x_prob[idx]
        return 1 - prob_to_stay

    def get_state_change_posterior(self, state, x_prob):
        prior = self.get_state_change_prior(state)
        prob = self.get_prob_to_change(state, x_prob)
        return prior * prob / (prior*prob + (1-prior)*(1-prob))


    def get_changing_state_priors(self, state):
        state_idx = copy(self.get_state_idx(state))
        priors = copy(self.get_state_priors(state))
        priors[state_idx] = 0
        priors = priors / priors.sum()
        return priors


    def get_changing_state_probabilities(self, state, x_prob):
        idx = self.get_state_idx(state)
        x_prob[idx] = 0
        return x_prob / x_prob.sum()


    def get_changing_state_posteriors(self, state, x_prob):
        priors = self.get_changing_state_priors(state)
        probs = self.get_changing_state_probabilities(state, x_prob)
        return probs*priors / np.sum(priors*probs)


    def correct_certainty(self, certainty: dict):
        for state in certainty.keys():
            idx = self.get_state_idx(state)
            cert = certainty[state]
            self.tmat[idx, idx] = cert * self.tmat[idx, idx]

        #diag = self.tmat.diagonal()
        #self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)
        self._get_vars()


    def remove_class(self, class_name):
        idx = self.get_state_idx(class_name)
        self.tmat = np.delete(np.delete(self.tmat, idx, axis=0), idx, axis=1)

        diag = self.tmat.diagonal()
        self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)
        np.fill_diagonal(self.tmat, diag)

        self.STATES = self.STATES[self.STATES != class_name]
        self._get_vars()
        self.removed_classes.append(class_name)


    def reset_probabilities(self):
        self.tmat = self._tmat_orig.copy()


    def weight_probabilities(self, stability: np.ndarray):
        #for idx in range(self.STATES.__len__()):
        #self.tmat[idx, idx] *= stability[idx]
        #diag = self.tmat.diagonal()
        #self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)

        for idx in range(self.STATES.__len__()):
            self.tmat[idx, idx] = stability[idx]

        diag = self.tmat.diagonal()
        self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)

        for idx in range(self.STATES.__len__()):
            self.tmat[idx, idx] = stability[idx]

        self._get_vars()


    def predict(self, scores, state='AWAKE'):
        ch_posts = []
        for k in range(scores.__len__()):
            p_likelihood_change = scores.iloc[k][self.STATES[self.STATES != state]].sum()
            p_prior_change = self.get_state_change_prior(state)
            p_post_change = (p_likelihood_change * p_prior_change) / ((p_likelihood_change * p_prior_change) + ((1-p_likelihood_change) * (1-p_prior_change)))
            ch_posts += [[p_post_change, state]]

            p_likelihood = scores.iloc[k][self.STATES[self.STATES != state]]
            p_prior = self.get_changing_state_priors(state)[self.STATES != state]
            p_post = p_likelihood*p_prior / sum(p_likelihood*p_prior)

            if p_post_change > 0.5:
                state = p_post.idxmax()
                scores.loc[k, state] = p_post_change
                scores.loc[k, self.STATES[self.STATES!=state]] = p_post*(1-p_post_change)
            else:
                scores.loc[k, state] = 1-p_post_change
                scores.loc[k, self.STATES[self.STATES!=state]] = p_post*p_post_change


        for k in np.where(np.isnan(scores))[0]:
            s = np.isnan(scores.iloc[k]).idxmax()
            scores.loc[k, s] = 1 - scores.iloc[k].sum()
        return scores


    def _optimize(self, scores, Y, Niter=100, popsize=20):
        print('Optimizing hypnogram stage stability using Differential Evolution')
        self.idx = -1
        self.best_kappa = -1
        self.best_stability = np.zeros(self.STATES.__len__())
        self.scores = scores
        self.Y = Y
        self.n_optimize=Niter

        def optimize_(args):
            self.idx += 1
            if self.idx < self.n_optimize:
                #args = [10**i for i in args]
                scores__ = deepcopy(self.scores)
                #self.reset_probabilities()
                self.weight_probabilities(args)
                sc_ = self.predict(scores__)
                y_ = np.array(sc_.idxmax(axis=1))
                kappa = cohen_kappa_score(self.Y, y_)
                sc = 1 - kappa
                #sc = np.sum([1-float(c) for c in list(get_classification_scores(self.Y, y_).values())])
                print(self.idx, kappa, args)

                if kappa > self.best_kappa:
                    self.best_kappa = kappa
                    self.best_stability = args
            else: sc = 0

            return sc




        bounds = [(0.2, 0.90)] * self.STATES.__len__()
        result = differential_evolution(optimize_, bounds, popsize=popsize, seed=1)
        result = self.best_stability

        del self.idx
        del self.best_kappa
        del self.best_stability
        del self.scores
        del self.Y

        return result


class KDEBayesianModelNC(_KDEFamilyMixin):
    """
    Same as :class:`KDEBayesianModel` but **without** the z-score step after PCA (NC = no
    centering/normalisation): features -> [StandardScaler] -> RFECV -> PCA ->
    [SelectFromModel] -> KDE. Parameters as for :class:`KDEBayesianModel`; the input
    standardisation (``standardize=True``, default) is the scale fix of review R4, not
    the omitted z-score (``standardize=False`` reproduces v1.0.0).

    ``__name__`` was ``"KDEBayesianModel"`` up to v1.0.0 (copy-paste); it is now
    ``"KDEBayesianModelNC"``.
    """
    __name__ = "KDEBayesianModelNC"
    def __init__(self, fbands=[
        [0.5, 5], # delta
        [4, 9], # theta
        [8, 14], # alpha
        [11, 16], # spindle
        [14, 20],
        [20, 30]
    ], segm_size=30, fs=200, bands_to_erase=[], filter_bands = True, filter_order=5001, nfft=12000,
                 window_smooth_n=3, window_std=1, cat_bias={'AWAKE': 1, 'N2': 1, 'N3': 1, 'REM': 1}, Selector2=True,
                 n_jobs=10, standardize=True, log_lik_floor='auto',
                 log_lik_quantile=_DEFAULT_LOG_LIK_QUANTILE, log_lik_margin=_DEFAULT_LOG_LIK_MARGIN):

        self.fbands = fbands
        self.n_jobs = n_jobs
        self.standardize = standardize
        self.log_lik_floor = log_lik_floor
        self.log_lik_quantile = log_lik_quantile
        self.log_lik_margin = log_lik_margin
        self.segm_size = segm_size
        self.fs = fs
        self.bands_to_erase = bands_to_erase
        self.filter_bands = filter_bands
        self.filter_order = filter_order
        self.nfft=nfft

        self.STATES = []
        self.KDE = []
        self.PipelineClustering = None
        self.FeatureSelector = None

        self.SELECTOR2 = Selector2

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            ignore_bands=self.bands_to_erase,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )



        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                mean_bands,
            ]



        self.FeatureExtractor = SleepSpectralFeatureExtractor(
            fs=self.fs,
            fbands=self.fbands,
            ignore_bands=self.bands_to_erase,
            segm_size=self.segm_size,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )
        self.FeatureExtractor._extraction_functions = \
            [
                mean_frequency,
                #self.FeatureExtractor.MedFreq,
                relative_bands,
                #self.FeatureExtractor.normalized_entropy,
                #self.FeatureExtractor.normalized_entropy_bands
            ]


        self.WINDOW = gaussian(window_smooth_n, window_std)
        self.WINDOW = self.WINDOW / self.WINDOW.sum()
        self.CAT_BIAS = cat_bias
        self.feature_names = None


    def extract_features(self, signal, return_names=False):
        if signal.ndim > 1:
            raise AssertionError('[INPUT ERROR]: Input data has to be of a dimension size 1 - raw signal')
        if signal.shape[0] != self.fs * self.segm_size:
            print('[INPUT WARNING]: input data is not a defined size fs*segm_size ' + str(self.fs*self.segm_size) + '; Signal of a size ' + str(signal.shape[0]) + ' found instead. Extracted features might be inaccurate.')


        ## Mean band-derived features - delta/beta ratio etc
        mean_bands, feature_names = self.FeatureExtractor_MeanBand(signal)
        mean_bands = np.concatenate(mean_bands)

        functions = [np.divide]
        symbols = ['/']
        mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
        for idx in range(functions.__len__()):
            mean_band_derived_features, mean_band_derived_names = augment_features(

                mean_band_derived_features.reshape(1, -1), feature_indexes=np.arange(mean_band_derived_features.shape[0]), operation=functions[idx], mutual=True,  operation_str=symbols[idx], feature_names=mean_band_derived_names

            )


        mean_band_derived_names = mean_band_derived_names[feature_names.__len__():]
        mean_band_derived_features = mean_band_derived_features[0, feature_names.__len__():]
        #mean_band_derived_names = mean_band_derived_names.squeeze()

        #features = np.log10(np.append(other_features, mean_band_derived_features))
        #feature_names = feature_names + mean_band_derived_names
        features = np.log10(mean_band_derived_features)
        #feature_names = mean_band_derived_names

        ## other features
        other_features, feature_names_other = self.FeatureExtractor(signal)
        other_features = np.concatenate(other_features)
        features = np.append(other_features, features)
        feature_names = list(feature_names_other) + list(mean_band_derived_names)


        self.feature_names = feature_names
        if return_names:
            return features, feature_names
        return features

    def fit(self, X, y):
        """
        Fit the transform pipeline and the per-state densities.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_features)
            Feature matrix (one row per epoch).
        y : np.ndarray of str, shape (n_epochs,)
            Labels; every state needs enough rows for a non-singular density
            (more rows than retained dimensions).
        """
        X, y = self._fit(X, y)
        self._fit_kde(X, y)

    def _fit(self, X, y):

        X = deepcopy(X)
        y = deepcopy(y)
        X_, y_ = balance_classes(X, y, std_factor=0.0)

        # step 2: standardise the raw features before the scale-dependent RFECV / PCA
        # (balanced training statistics, frozen; review R4 of PR #69)
        self.Scaler = preprocessing.StandardScaler() if getattr(self, 'standardize', False) else None
        if self.Scaler is not None:
            X_ = self.Scaler.fit_transform(X_)
            X = self.Scaler.transform(X)

        estimator = SVR(kernel="linear")
        self.SELECTOR = RFECV(estimator, step=5, verbose=True, min_features_to_select=4, n_jobs=self.n_jobs)
        self.PCA = PCAModule(var_threshold=0.98)
        #self.ZScore = ZScoreModule(trainable=True, continuous_learning=False, multi_class=False)

        #self.UMAP = UMAP(n_neighbors=30, min_dist=1,
        #                 n_components=2)

        le = preprocessing.LabelEncoder()
        le.fit(y_)
        y__ = le.transform(y_)


        X_ = self.SELECTOR.fit_transform(X_, y__)
        X_ = self.PCA.fit_transform(X_)
        #X_ = self.ZScore.fit_transform(X_, y)

        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
       # X = self.ZScore.fit_transform(X, y)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_, y_)
        if self.SELECTOR2:
            self.SELECTOR2 = SelectFromModel(lsvc, prefit=True, max_features=4)
            #X_ = self.SELECTOR2.transform(X_)
            with warnings.catch_warnings():
                # sklearn warns "No features were selected"; we raise below instead
                warnings.simplefilter('ignore', UserWarning)
                X = self.SELECTOR2.transform(X)
            if X.shape[1] == 0:
                raise ValueError(f'{type(self).__name__}.fit: the L1 LinearSVC of Selector2 kept no '
                                 'feature (all coefficients are 0), so no density can be fitted. '
                                 'Use Selector2=False or more / more separable training data.')

        #X = self.UMAP.fit_transform(X)
        return X, y


    def transform(self, X):
        if getattr(self, 'Scaler', None) is not None:
            X = self.Scaler.transform(X)
        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
        #X = self.ZScore.transform(X)
        if self.SELECTOR2:
            X = self.SELECTOR2.transform(X)
        #X = self.UMAP.transform(X)
        return X

class Mapper:
    """
    Co-registration of a feature space onto a template distribution.

    A *map* is a per-dimension affine transform ``x -> scale(x, s) + t`` (scaling about
    the data mean, then translation; ``t`` in ``SHIFT_RANGE``, ``s`` in ``SCALE_RANGE``)
    chosen to minimise a cost against the template:

    * ``fit_genetic``: sum over dimensions of the KL divergence between the 200-bin
      marginal histograms of the template and of the mapped data;
    * ``fit_genetic_likelihood``: ``1000 - sum`` of a fitted model's class likelihoods of
      the mapped data (``model._likelihood``);
    * ``fit_map``: random search (``self.N`` random transforms), returns all costs.

    Usage: ``create_template(x_ref[, y_ref])`` then ``map(x, name)`` (fits once per
    ``name``, cached in ``self.MAPS``). ``x`` arrays are (n_samples, n_dims) with the same
    ``n_dims`` as the template.

    The KL costs come from ``brainmaze_utils.stat.kl_divergence_nonparametric`` on the
    (n_dims, 200) histogram array: the sum of the per-dimension divergences. Every bin is
    floored at 1e-9 in ``get_probabilities``, so the ``inf`` that brainmaze_utils >= 2.1.0
    returns for ``q == 0, p > 0`` cannot occur. (An intermediate state of brainmaze_utils
    PR #28 averaged over dimensions instead, which would have rescaled the stored
    ``MAPS[name]['cost']`` but not the selected transform; the released behaviour keeps
    the sum, so costs are comparable across versions.)
    """
    def __init__(self):
        self.TEMPLATE = None
        self.TEMPLATE_NAME = None
        self.CLASSES = None
        self.TEMPLATE_CLASS = None
        self.MAPS = dict()
        self.NDIM = None

        self.N = 1000

        self.SHIFT_RANGE = np.array([-5, 5])
        self.SCALE_RANGE = np.array([0.9, 1.25])
        self.ROTATE_RANGE = np.array([-25, 25])

        self.X_RANGE = []

        self.COREGISTRATION_CLASS_BAN = ['N1']

    def create_template(self, x, y=None, name='Template'):
        self.NDIM = x.shape[1]
        self._get_template(x)
        if not isinstance(y, type(None)):
            self._get_template_class(x, y)

        self.TEMPLATE_NAME = name

    def _get_template(self, x):
        self.X_RANGE = [x.mean() - 20*x.std(), x.mean() + 20*x.std()]
        self.TEMPLATE = self.get_probabilities(x)

    def _get_template_class(self, x, y):
        y = np.array(y)
        self.CLASSES = np.sort(np.unique(y))
        self.TEMPLATE_CLASS = dict([(cl, self.get_probabilities(x[y == cl])) for cl in self.CLASSES])

    def _cost(self, x):
        px = self.get_probabilities(x)
        return kl_divergence_nonparametric(self.TEMPLATE, px)

    def _cost_class(self, x, y, bias={}):
        cost = 0
        classes = list(np.unique(y))
        for cl in self.CLASSES:
            if not cl in self.COREGISTRATION_CLASS_BAN:
                if cl in classes:
                    px = self.get_probabilities(x[y == cl])
                    c_ = kl_divergence_nonparametric(self.TEMPLATE_CLASS[cl], px)
                    if cl in bias.keys():
                        c_ *= bias[cl]
                    cost += c_
        return cost

    def _cost_semi_supervised(self, x, y, bias={}):
        xs_ = x[y!='']
        ys_ = y[y!='']

        return self._cost_class(xs_, ys_, bias=bias) + self._cost(x)

    def fit_map(self, x, y=None, bias={'REM': 2}):
        """
        Random search over ``self.N`` transforms (``np.random.seed(0)``, deterministic).

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_dims)
        y : np.ndarray of str, shape (n_samples,), optional
            Labels ('' = unlabelled). If given, classes are balanced and the
            semi-supervised cost (per-class KL to ``TEMPLATE_CLASS`` + global KL) is used;
            requires ``create_template(x, y)``.
        bias : dict class -> float
            Weight of each class's KL term.

        Returns
        -------
        costs : np.ndarray, shape (N,)
        transforms : list of dict ``{'translate': (n_dims,), 'scale': (n_dims,)}``

        Up to v1.0.0 this did ``y = deepcopy(x)``, so the labels were replaced by the
        feature matrix and the semi-supervised cost was computed on nonsense "labels"
        without any error (issue #57).
        """
        if not isinstance(y, type(None)):
            x = deepcopy(x)
            y = deepcopy(np.asarray(y))
            x, y = balance_classes(x, y)

        np.random.seed(0)
        transforms = []
        kl = []
        for k in tqdm(range(self.N)):
            x_ = deepcopy(x)
            tr = self._get_random_transform()
            x_ = self._transform(x_, tr)
            if isinstance(y, type(None)):
                c = self._cost(x_)
            else:
                #c = self._cost_class(x_, y)
                c = self._cost_semi_supervised(x_, y, bias=bias)

            kl += [c]
            transforms += [tr]
        return np.array(kl), transforms

    @staticmethod
    def _args_to_transform(args, ndim):
        """Split a differential-evolution parameter vector ``[t_0..t_{d-1}, s_0..s_{d-1}]``
        into ``{'translate': (d,), 'scale': (d,)}``."""
        args = np.asarray(args, dtype=float)
        return {'translate': args[:ndim].copy(), 'scale': args[ndim:2 * ndim].copy()}

    def fit_genetic(self, x, y=None, popsize=15, seed=None, **de_kwargs):
        """
        Differential-evolution search for the transform minimising the KL cost to the
        template (``_cost``).

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_dims)
        y : ignored
        popsize : int
            ``differential_evolution`` population multiplier.
        seed : int or None
            ``differential_evolution`` seed (None = not reproducible, as before).
        **de_kwargs :
            Passed to ``scipy.optimize.differential_evolution`` (e.g. ``maxiter``).

        Returns
        -------
        transform : dict ``{'translate': (n_dims,), 'scale': (n_dims,)}``
        cost : float
            KL cost of the mapped data.

        Up to v1.0.0 the parameter vector was split with hard-coded ``[0:3]`` / ``[3:6]``
        (assumes 3 dimensions) while the bounds had ``2 * n_dims`` entries: 2-D data
        raised ``IndexError`` and > 3-D data raised in ``scale`` (issue #57).
        """
        ndim = x.shape[1]

        def optimize(args):
            x_ = deepcopy(x)
            tr = self._args_to_transform(args, ndim)
            x_ = self._transform(x_, tr)
            c = self._cost(x_)
            return c

        bounds = [tuple(self.SHIFT_RANGE)] * ndim + [tuple(self.SCALE_RANGE)] * ndim
        result = differential_evolution(optimize, bounds, popsize=popsize, seed=seed, **de_kwargs)

        tr = self._args_to_transform(result.x, ndim)
        return tr, self._cost(self._transform(x, tr))

    def fit_genetic_likelihood(self, x, y=None, popsize=15, model=None, seed=None, **de_kwargs):
        """
        As :meth:`fit_genetic` but maximises ``model``'s summed class likelihood of the
        mapped data (cost ``1000 - sum(model._likelihood(x_mapped))``). ``model`` is a
        fitted KDE-family model and ``x`` must already be in that model's transformed
        feature space. Returns ``(transform, KL cost of the mapped data)``.
        """
        ndim = x.shape[1]

        def optimize(args):
            x_ = deepcopy(x)
            tr = self._args_to_transform(args, ndim)
            x_ = self._transform(x_, tr)
            c = model._likelihood(x_)
            c = 1000 - c.sum().sum()
            return c

        bounds = [tuple(self.SHIFT_RANGE)] * ndim + [tuple(self.SCALE_RANGE)] * ndim
        result = differential_evolution(optimize, bounds, popsize=popsize, seed=seed, **de_kwargs)

        tr = self._args_to_transform(result.x, ndim)
        return tr, self._cost(self._transform(x, tr))


    #def fit_map(self, x, name, y=None, model=None):
        #tr, cost = self.fit_genetic(x)
    #    tr, cost = self.fit_genetic_likelihood(x, model=model)
    #    self.MAPS[name] = {'transformation': tr, 'cost': cost}

    def map(self, x, name, model=None, popsize=15, seed=None, **de_kwargs):
        """
        Map ``x`` onto the template, fitting the map once per ``name``.

        If ``name`` is not in ``self.MAPS`` a transform is fitted with
        :meth:`fit_genetic_likelihood` when ``model`` is given, otherwise with
        :meth:`fit_genetic`, and stored as
        ``self.MAPS[name] = {'transformation': tr, 'cost': cost}`` (this restores the
        behaviour of the commented-out ``fit_map(x, name, y, model)`` this method was
        written against). Later calls with the same ``name`` reuse it.

        Up to v1.0.0 this called ``fit_map(x, name=..., model=...)``, which accepts
        neither keyword (``TypeError``) and does not store a map (issue #57).

        Returns
        -------
        np.ndarray, shape (n_samples, n_dims)
            The mapped copy of ``x``.
        """
        if not name in self.MAPS.keys():
            if model is None:
                tr, cost = self.fit_genetic(x, popsize=popsize, seed=seed, **de_kwargs)
            else:
                tr, cost = self.fit_genetic_likelihood(x, popsize=popsize, model=model, seed=seed, **de_kwargs)
            self.MAPS[name] = {'transformation': tr, 'cost': cost}

        x = self._transform(x, self.MAPS[name]['transformation'])
        return x


    def _transform(self, x, transform):
        x = deepcopy(x)
        x = scale(x, transform['scale'])
        x = translate(x, transform['translate'])
        # x = rotate(x, transform['rotate'])
        return x

    def _get_random_transform(self):
        transform = {
            'translate': np.random.rand(self.NDIM) * np.diff(self.SHIFT_RANGE) + self.SHIFT_RANGE[0],
            'scale': np.random.rand(self.NDIM) * np.diff(self.SCALE_RANGE) + self.SCALE_RANGE[0]
        }
        #if self.NDIM == 2:
        #    transform['rotate'] = np.random.rand() * np.diff(self.ROTATE_RANGE) + self.ROTATE_RANGE[0]
        #if self.NDIM == 3:
        #    transform['rotate'] = np.random.rand(3) * np.diff(self.ROTATE_RANGE) + self.ROTATE_RANGE[0]

        return transform

    def get_probabilities(self, x):
        rng = self.X_RANGE
        bins = 200
        ps = []
        for k in range(x.shape[1]):
            p, _ = np.histogram(x[:, k], bins, rng)
            ps += [p]

        ps = np.array(ps) / x.shape[0]
        ps[ps == 0] = 1e-9
        ps = ps / ps.sum(axis=1).reshape(-1, 1)
        return ps


class SleepStructureClassifier:
    """
    Transductive classifier based on the *structure* of a recording: the distribution of
    pairwise difference vectors between epochs.

    ``fit``: for every ordered pair of distinct training epochs ``(i, j)`` the difference
    ``v = x_i - x_j`` is represented as ``[||v||, v / ||v||]`` (shape ``d + 1``) and one
    ``gaussian_kde`` is fitted per state pair ``(s_i, s_j)``.

    ``scores(x)``: the same pairwise vectors are formed **among the epochs of ``x``**
    (e.g. one test recording; no labels are used). For epoch ``k``, the score of state
    ``s`` is ``sum_{i != k} sum_{s'} KDE[s'][s](x_i - x_k)``, i.e. how well the epoch's
    position relative to every other epoch matches state ``s`` (the other epoch's state
    marginalised), normalised over states.

    Cost: O(n^2) pairs for both fitting and scoring (``n`` = epochs); keep ``n`` modest
    (a few hundred).

    Parameters
    ----------
    states : list of str
        States the model may use (default ``['WAKE', 'N1', 'N2', 'N3', 'REM']``; note
        that the other classifiers in this module use ``'AWAKE'``). Every ``fit`` starts
        again from this list: states without training epochs are dropped for that fit
        with a warning (``self.STATES`` = the states actually fitted), and training labels
        that are not in the list raise ``ValueError``. Each fitted state pair needs enough
        pairs for a non-singular ``(d + 1)``-dimensional KDE (see ``fit``).
    """
    def __init__(self, states=['WAKE', 'N1', 'N2', 'N3', 'REM']):
        self._states_init = tuple(states)
        self.STATES = list(states)
        self.norml2 = None
        # up to v1.0.0 this was None and ``fit`` raised TypeError on the first
        # ``self.KDE[s1] = {}`` (issue #57)
        self.KDE = {}
        self.N = None

    def fit(self, x, y):
        """
        Parameters
        ----------
        x : np.ndarray, shape (n_epochs, n_features)
        y : array-like of str, shape (n_epochs,)

        Notes
        -----
        Up to v1.0.0 the pairs included ``i == j`` (zero vectors, direction 0/0 = NaN), so
        ``gaussian_kde`` raised on NaN input; self-pairs are now excluded.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        states_init = list(getattr(self, '_states_init', self.STATES))
        unknown = sorted(set(np.unique(y).tolist()) - set(states_init))
        if unknown:
            # up to round 1 of PR #69 these epochs were silently ignored (review R3)
            raise ValueError(f'SleepStructureClassifier.fit: labels {unknown} are not in states='
                             f'{states_init}; pass states=... including them (e.g. AWAKE vs WAKE) '
                             'or remove those epochs.')
        present = [s_ for s_ in states_init if np.sum(y == s_) > 0]
        missing = [s_ for s_ in states_init if s_ not in present]
        if missing:
            warnings.warn(f'SleepStructureClassifier.fit: no training samples for {missing}; '
                          f'these states are dropped for this fit.')
        if not present:
            raise ValueError('SleepStructureClassifier.fit: no training epochs')
        d = x.shape[1]
        diff, i_idx, j_idx = _pairwise_differences(x)
        rep_ = _norm_direction(diff)
        kde = {}
        for s1 in present:
            kde[s1] = {}
            for s2 in present:
                sel = (y[i_idx] == s1) & (y[j_idx] == s2)
                n_pairs = int(sel.sum())
                if n_pairs < d + 2:
                    raise ValueError(
                        f'SleepStructureClassifier.fit: state pair ({s1}, {s2}) has {n_pairs} epoch '
                        f'pairs; a KDE in d + 1 = {d + 1} dimensions needs at least {d + 2} '
                        f'(epochs per state: { {s_: int(np.sum(y == s_)) for s_ in present} }).')
                try:
                    kde[s1][s2] = gaussian_kde(rep_[sel, :].T)
                except (np.linalg.LinAlgError, ValueError) as e:
                    raise ValueError(f'SleepStructureClassifier.fit: the pair vectors of state pair '
                                     f'({s1}, {s2}) ({n_pairs} pairs, {d + 1} dims) have a singular '
                                     f'covariance ({e}); more / less degenerate epochs are needed.') from e
        self.STATES = present
        self.KDE = kde
        self.N = x.shape[1]
        return self

    def scores(self, x):
        """
        Parameters
        ----------
        x : np.ndarray, shape (n_epochs, n_features), n_epochs >= 2

        Returns
        -------
        pd.DataFrame, shape (n_epochs, n_states)
            Columns ``self.STATES``; rows sum to 1.

        Notes
        -----
        Up to v1.0.0 this (a) called ``get_mutual_vectors(x)`` without labels, which
        raises ``UnboundLocalError`` in brainmaze_utils, (b) evaluated the KDEs on raw
        difference vectors although they were fitted on ``[norm, direction]`` (dimension
        mismatch), and (c) aggregated with ``N = x.shape[1]`` (n_features) instead of the
        number of epochs, returning n_features rows. All three are fixed.
        """
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if n < 2:
            # one epoch has no pairs: the score would be 0/0 (review R6)
            raise ValueError(f'SleepStructureClassifier.scores: needs >= 2 epochs (scores are built '
                             f'from pairs of epochs), got {n}')
        diff, i_idx, j_idx = _pairwise_differences(x)
        rep_ = _norm_direction(diff)
        out = np.zeros((n, len(self.STATES)))
        for c, s2 in enumerate(self.STATES):
            # likelihood that epoch j (the subtracted one) is in state s2, partner i any state
            lik = np.zeros(rep_.shape[0])
            for s1 in self.STATES:
                lik += self.KDE[s1][s2].pdf(rep_.T)
            out[:, c] = np.bincount(j_idx, weights=lik, minlength=n)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = out / out.sum(axis=1, keepdims=True)
        return pd.DataFrame(out, columns=list(self.STATES))

    def predict(self, x):
        """Arg-max state per epoch, np.ndarray of str, shape (n_epochs,). An epoch whose
        densities all underflow (all-NaN score row) gets ``UNKNOWN_LABEL``."""
        return _labels_from_scores(self.scores(x))


class SleepClassifierWrapper:
    """
    One :class:`KDEBayesianModel` per stimulation frequency (0, 2, 7, 72.5 Hz) at 250 Hz.

    ``train(X, df, fs)``: ``X`` is an (n_epochs, n_samples) array of 30-s epochs sampled at
    ``fs`` Hz (resampled to 250 Hz); ``df`` has one row per epoch with columns
    ``annotation`` (str) and ``freq`` (stimulation frequency, Hz). Epochs with datarate
    <= 0.85, ``N1`` and ``UNKNOWN`` are dropped. Models: 0 Hz <- freq 0; 2 Hz <- freq 0 or
    2 (bands around 2 Hz harmonics erased); 7 Hz <- freq 0 or 7; 72.5 Hz <- freq 72.5. A
    model with no training epochs is skipped with a warning.

    ``predict_signal(X, fs, stim_freq)``: any ``fs``; returns the annotation table of the
    matching model. Training and prediction resample each 30-s epoch to 250 Hz the same way
    (``KDEBayesianModel.extract_features_bulk`` -> ``unify_sampling_frequency``, checked
    by ``_resample_epochs``). Up to round 1 of PR #69 ``predict_signal`` accepted only 250
    or 500 Hz and used a different anti-alias path (Butterworth 40 Hz + ``[::2]``) from
    ``train`` (review R9).

    Every ``train`` call starts from an empty ``MODEL`` dict: a model skipped for lack of
    epochs is absent, never a stale one from an earlier call.

    Parameters
    ----------
    n_jobs : int
        Passed to every model's ``RFECV`` (default 10, as before).
    """
    def __init__(self, n_jobs=10):
        self.MODEL = {}
        self.n_jobs = n_jobs

    def train(self, X, df, fs=250):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_samples)
            Epochs (30 s each).
        df : pd.DataFrame, n_epochs rows, columns ``annotation``, ``freq``
        fs : float, Hz
            Sampling rate of ``X`` (default 250, the models' rate). Up to v1.0.0 there was
            no ``fs`` and ``extract_features_bulk`` was called without its required
            ``fsamp_list`` (``TypeError``; issue #57); its ``(features, fs)`` tuple was
            also passed to ``fit`` as if it were the feature matrix.
        """
        # forget models of a previous ``train`` call (Copilot, PR #69)
        self.MODEL = {}
        datarate = get_datarate(X)
        df = df.loc[datarate > 0.85].reset_index(drop=True)
        X = X[datarate > 0.85]


        X = X[df.annotation != 'N1']
        df = df.loc[df.annotation != 'N1'].reset_index(drop=True)

        X = X[df.annotation != 'UNKNOWN']
        df = df.loc[df.annotation != 'UNKNOWN'].reset_index(drop=True)


        fs_downsample = 250
        fbands = [[0.5, 5],  # delta
                  [4, 9],  # theta
                  [8, 14],  # alpha
                  [11, 16],  # spindle
                  [14, 20],
                  [20, 30]]  # (beta3)

        bands_to_erase_2 = [[0, 0.5]] + [[k - 0.5, k + 0.5] for k in np.arange(2, 30, 2)]
        bands_to_erase_7 = [[0, 0.5], [6, 8], [13, 15], [20, 22], [27, 29]]


        #### Train 0 ####
        X0 = X[df.freq == 0]
        df0 = df.loc[df.freq == 0].reset_index(drop=True)
        model0 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=[], Selector2=False, n_jobs=self.n_jobs)
        if X0.shape[0] == 0:
            warnings.warn('SleepClassifierWrapper.train: no epochs for model 0; skipped')
        else:
            X0, _ = model0.extract_features_bulk(list(X0), [fs] * X0.shape[0])
            Y0 = df0.annotation.to_numpy()
            model0.fit(X0, Y0)
            self.MODEL[0] = model0

        #### Train 2 ####
        X2 = X[(df.freq == 2) | (df.freq == 0)]
        df2 = df.loc[(df.freq == 2) | (df.freq == 0)].reset_index(drop=True)
        model2 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=bands_to_erase_2, Selector2=False, n_jobs=self.n_jobs)
        if X2.shape[0] == 0:
            warnings.warn('SleepClassifierWrapper.train: no epochs for model 2; skipped')
        else:
            X2, _ = model2.extract_features_bulk(list(X2), [fs] * X2.shape[0])
            Y2 = df2.annotation.to_numpy()
            model2.fit(X2, Y2)
            self.MODEL[2] = model2

        #### Train 7 ####
        X7 = X[(df.freq == 7) | (df.freq == 0)]
        df7 = df.loc[(df.freq == 7) | (df.freq == 0)].reset_index(drop=True)
        model7 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=bands_to_erase_7, Selector2=False, n_jobs=self.n_jobs)
        if X7.shape[0] == 0:
            warnings.warn('SleepClassifierWrapper.train: no epochs for model 7; skipped')
        else:
            X7, _ = model7.extract_features_bulk(list(X7), [fs] * X7.shape[0])
            Y7 = df7.annotation.to_numpy()
            model7.fit(X7, Y7)
            self.MODEL[7] = model7

        #### Train 72.5 ####
        X725 = X[(df.freq == 72.5)]
        df725 = df.loc[(df.freq == 72.5)].reset_index(drop=True)
        model725 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=[], Selector2=False, n_jobs=self.n_jobs)
        if X725.shape[0] == 0:
            warnings.warn('SleepClassifierWrapper.train: no epochs for model 725; skipped')
        else:
            X725, _ = model725.extract_features_bulk(list(X725), [fs] * X725.shape[0])
            Y725 = df725.annotation.to_numpy()
            model725.fit(X725, Y725)
            self.MODEL[72.5] = model725

    def predict_signal(self, X, fs, stim_freq, datarate_threshold=0.85):
        """
        Classify a continuous signal with the model of ``stim_freq``.

        ``X`` : np.ndarray (n_samples,), NaN = missing; ``fs`` : any rate in Hz (epochs are
        resampled to 250 Hz exactly as in ``train``). Returns the model's
        ``predict_signal`` annotation table (``'UNKNOWN'`` = out-of-distribution epoch).
        ``ValueError`` if no model was trained for ``stim_freq``.
        """
        if stim_freq not in self.MODEL:
            raise ValueError(f'SleepClassifierWrapper: no trained model for stim_freq={stim_freq}; '
                             f'available: {sorted(self.MODEL)}')
        clf = self.MODEL[stim_freq]
        return clf.predict_signal(np.asarray(X, dtype=float), fs, datarate_threshold)


class MultiChannelMVGaussBayesClassifier(_EpochClassifierMixin):
    """
    Gaussian naive Bayes (``sklearn.naive_bayes.GaussianNB``) on a feature matrix.

    "Multi-channel" refers to the intended use: concatenate the per-channel feature
    vectors (``extract_features`` is single-channel) into one row per epoch. There is no
    feature selection, PCA or normalisation; ``transform`` is the identity. Unlike the
    KDE family, ``scores`` are not smoothed across epochs.

    Constructor parameters as for :class:`KDEBayesianModel` (``window_*``, ``cat_bias``,
    ``Selector2`` are accepted but unused). Out-of-distribution epochs are flagged as in
    the KDE family (``log_lik_floor`` etc., see :class:`_EpochClassifierMixin`), using the
    class-conditional log-likelihood of the naive-Bayes model (joint log-likelihood minus
    log prior; in-sample training values). ``__name__`` was ``"KDEBayesianModel"`` up to
    v1.0.0 (copy-paste); it is now ``"MultiChannelMVGaussBayesClassifier"``.
    """
    __name__ = "MultiChannelMVGaussBayesClassifier"

    def __init__(self, fbands=[[0.5, 5],  # delta
                               [4, 9],  # theta
                               [8, 14],  # alpha
                               [11, 16],  # spindle
                               [14, 20],
                               [20, 30]], segm_size=30, fs=200, bands_to_erase=[], filter_bands=True, nfft=12000,
                 window_smooth_n=3, window_std=1, cat_bias={'AWAKE': 1, 'N2': 1, 'N3': 1, 'REM': 1},
                 Selector2=True, log_lik_floor='auto', log_lik_quantile=_DEFAULT_LOG_LIK_QUANTILE,
                 log_lik_margin=_DEFAULT_LOG_LIK_MARGIN):

        self.fbands = fbands
        self.log_lik_floor = log_lik_floor
        self.log_lik_quantile = log_lik_quantile
        self.log_lik_margin = log_lik_margin
        self.segm_size = segm_size
        self.fs = fs
        self.bands_to_erase = bands_to_erase
        self.filter_bands = filter_bands
        self.nfft = nfft
        self.filter_order = 100

        self.STATES = []
        self.KDE = []
        self.PipelineClustering = None
        self.FeatureSelector = None

        self.SELECTOR2 = Selector2

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            ignore_bands=self.bands_to_erase,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )

        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                mean_bands,
            ]

        self.FeatureExtractor = SleepSpectralFeatureExtractor(
            fs=self.fs,
            fbands=self.fbands,
            ignore_bands=self.bands_to_erase,
            segm_size=self.segm_size,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )

        self.FeatureExtractor._extraction_functions = \
            [
                mean_frequency,
                relative_bands,
            ]

        self.WINDOW = gaussian(window_smooth_n, window_std)
        self.WINDOW = self.WINDOW / self.WINDOW.sum()
        self.CAT_BIAS = cat_bias
        self.feature_names = None

    def extract_features(self, signal, return_names=False):
        if signal.ndim > 1:
            raise AssertionError('[INPUT ERROR]: Input data has to be of a dimension size 1 - raw signal')
        if signal.shape[0] != self.fs * self.segm_size:
            print('[INPUT WARNING]: input data is not a defined size fs*segm_size ' + str(
                self.fs * self.segm_size) + '; Signal of a size ' + str(
                signal.shape[0]) + ' found instead. Extracted features might be inaccurate.')

        ## Mean band-derived features - delta/beta ratio etc
        mean_bands, feature_names = self.FeatureExtractor_MeanBand(signal)
        mean_bands = np.concatenate(mean_bands)

        functions = [np.divide]
        symbols = ['/']
        mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
        for idx in range(functions.__len__()):
            mean_band_derived_features, mean_band_derived_names = augment_features(

                mean_band_derived_features.reshape(1, -1),
                feature_indexes=np.arange(mean_band_derived_features.shape[0]), operation=functions[idx],
                mutual=True, operation_str=symbols[idx], feature_names=mean_band_derived_names

            )

        mean_band_derived_names = mean_band_derived_names[feature_names.__len__():]
        mean_band_derived_features = mean_band_derived_features[0, feature_names.__len__():]
        # mean_band_derived_names = mean_band_derived_names.squeeze()

        # features = np.log10(np.append(other_features, mean_band_derived_features))
        # feature_names = feature_names + mean_band_derived_names
        features = np.log10(mean_band_derived_features)
        # feature_names = mean_band_derived_names

        ## other features
        other_features, feature_names_other = self.FeatureExtractor(signal)
        other_features = np.concatenate(other_features)
        features = np.append(other_features, features)
        feature_names = list(feature_names_other) + list(mean_band_derived_names)

        self.feature_names = feature_names
        if return_names:
            return features, feature_names
        return features

    def extract_features_bulk(self, list_of_signals, fsamp_list, return_names=False):
        """
        Extract features from a list of epochs.

        Unlike :meth:`KDEBayesianModel.extract_features_bulk` this does **not** resample:
        every entry of ``fsamp_list`` must equal ``self.fs`` (Hz), otherwise
        ``ValueError`` (up to v1.0.0 other rates were silently processed as if sampled at
        ``self.fs``, mislabelling every frequency band).

        Returns ``(features (n_epochs, n_features), fs)``, or
        ``(features, feature_names)`` if ``return_names``.
        """
        data = list_of_signals
        if np.any(np.asarray(fsamp_list, dtype=float) != float(self.fs)):
            raise ValueError(f'MultiChannelMVGaussBayesClassifier.extract_features_bulk: all signals '
                             f'must be sampled at fs={self.fs} Hz (no resampling is done); got '
                             f'{sorted(set(np.asarray(fsamp_list, dtype=float).tolist()))}')
        x = []
        for k in tqdm(range(data.__len__())):
            x += [self.extract_features(data[k])]
        if return_names:
            _, feature_names = self.extract_features(data[k], return_names=True)
            return np.array(x), feature_names
        return np.array(x), fsamp_list[0]

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_features)
        y : array-like, shape (n_epochs,)
        """
        self.classifier = GaussianNB()
        self.classifier.fit(X, y)
        self.STATES = self.classifier.classes_
        self._fit_log_lik_floor(self.max_log_likelihood(X))
        return self

    def max_log_likelihood(self, X):
        """Best-class log-likelihood ``max_c log p(x | c)`` per row, shape (n_epochs,)."""
        X = np.asarray(X, dtype=float)
        jll = self.classifier.predict_joint_log_proba(X)
        return (jll - np.log(self.classifier.class_prior_)).max(axis=1)

    def transform(self, X):
        """Identity (this model has no feature transform); returns ``np.asarray(X)``."""
        return np.asarray(X)

    def scores(self, X, start_time=None):
        """
        Class probabilities.

        ``start_time`` is accepted for API symmetry and unused (no smoothing).

        Returns
        -------
        pd.DataFrame, shape (n_epochs, n_classes)
            Columns are ``self.classifier.classes_``; rows sum to 1, except
            out-of-distribution rows (best class log-likelihood below
            ``self.log_lik_floor_``), which are all-NaN. ``self.max_log_lik_`` is set.
            Up to v1.0.0 this returned an (n_classes, n_classes) frame whose cells held
            whole arrays, and ``predict`` raised ``ValueError`` (issue #57).
        """
        X = np.asarray(X, dtype=float)
        scr = np.array(self.classifier.predict_proba(X), dtype=float)
        self.max_log_lik_ = self.max_log_likelihood(X)
        scr[self._ood_mask(self.max_log_lik_)] = np.nan
        return pd.DataFrame(scr, columns=list(self.classifier.classes_))
