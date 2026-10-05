# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Sleep classification: train, predict, and out-of-distribution epochs (``'UNKNOWN'``)
===================================================================================

Trains :class:`brainmaze_eeg.classifiers.KDEBayesianModel` on the scored first half of the
6.8-h scalp recording in ``../eeg_wave_detection/patient_one_data.mat`` (Fz-Cz, 500 Hz,
30-s epochs with an expert hypnogram) and classifies the second half from the raw signal.

1. **Labels.** Hypnogram codes 0/2/3/5 = AWAKE/N2/N3/REM are used for training. N1 (code 1)
   is left out of training, as in :class:`~brainmaze_eeg.classifiers.SleepClassifierWrapper`
   (the models know AWAKE/N2/N3/REM); 9 = not scored and -1 = no data are ignored.
2. **Missing data.** The file stores amplifier dropouts as a constant; they are set to NaN
   from its ``data_present`` mask. ``predict_signal`` skips epochs with less than 85 %
   non-NaN samples (``datarate_threshold``): they leave a gap in the output table.
3. **Training.** ``extract_features_bulk`` (spectral features; epochs resampled to the
   model's 200 Hz) -> ``fit``. ``fit`` also sets the out-of-distribution floor
   ``log_lik_floor_`` ('auto': 0.1 % quantile of the training epochs' best-state
   log-likelihood minus 30 nats).
4. **Prediction.** ``predict_signal`` returns an annotation table (consecutive equal labels
   merged); ``predict_signal_scores`` the per-epoch class probabilities.
5. **UNKNOWN.** An epoch whose best-state log-likelihood ``max_log_lik_`` is below the
   floor gets NaN probabilities and the label ``'UNKNOWN'`` instead of a confident wrong
   label. Eight artificial epochs are written into copies of N2/N3/REM/AWAKE test epochs.
   The table shows, for each, the expert stage, the label of the **clean** epoch, and the
   label of the corrupted epoch, so you can see what is flagged, what silently changes the
   stage, and what changes nothing. **The floor is not an artifact detector**: it flags only
   epochs whose features are far from every trained state. Use a signal-quality / artifact
   detector before classification.
6. **Agreement.** Accuracy, balanced accuracy (mean recall of the four states) and Cohen's
   kappa on the clean test epochs. The set is imbalanced (N3 is rare), so look at the
   balanced accuracy and kappa, not only at the accuracy.

``fit`` and the predict calls show tqdm progress bars on stderr; the script hides them to
keep the output readable (the classes have no switch for them).

This is one subject, one channel, and a single train/test split: the agreement printed
below illustrates the API, it is not a validation of sleep-staging accuracy.

Run (about 1-2 min)::

    python train_predict_unknown.py [output_dir]

Writes ``hypnogram_unknown.png`` to ``output_dir`` (default ``./demo_output``) if
matplotlib is installed.
"""

import contextlib
import io
import os
import sys
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

from brainmaze_eeg.classifiers import UNKNOWN_LABEL, KDEBayesianModel

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                         'eeg_wave_detection', 'patient_one_data.mat')
SEGM = 30                                         # s, epoch length
STAGES = {0: 'AWAKE', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'}
TRAIN_STATES = ['AWAKE', 'N2', 'N3', 'REM']


def labels_from_scores(scores):
    """Arg-max state per row of a score frame; all-NaN rows (out of distribution) ->
    ``UNKNOWN_LABEL``."""
    p = scores.to_numpy(dtype=float)
    out = np.full(p.shape[0], UNKNOWN_LABEL, dtype=object)
    ok = ~np.isnan(p).all(axis=1)
    out[ok] = np.asarray(scores.columns)[np.nanargmax(p[ok], axis=1)]
    return out


def band_noise(rng, n, fs, lo, hi):
    """Gaussian noise band-passed to lo-hi Hz, unit standard deviation."""
    z = sosfiltfilt(butter(4, [lo, hi], 'bandpass', fs=fs, output='sos'), rng.normal(size=n))
    return z / z.std()


def _predict(model, signal, fs):
    """Annotation table, per-epoch scores and labels, ``max_log_lik_`` and epoch indices of a
    signal; tqdm bars (stderr) hidden, warnings printed."""
    with warnings.catch_warnings(record=True) as caught, \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter('always')
        table = model.predict_signal(signal, fs)                # annotation table
        scores = model.predict_signal_scores(signal, fs)        # per-epoch probabilities
    for w in caught:
        print(f'warning: {w.category.__name__}: {w.message}')
    max_ll = np.array(model.max_log_lik_)                       # per scored epoch
    start = model.preprocess_signal(signal, fs)[1]              # epoch start times (s)
    return labels_from_scores(scores), max_ll, (start // SEGM).astype(int), table, scores


def main(outdir='demo_output'):
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f'Missing demo data: {DATA_PATH}')
    d = loadmat(DATA_PATH)
    fs = int(d['fsamp'].ravel()[0])
    x = d['fzcz'].ravel().astype(np.float64)
    x[d['data_present'].ravel() == 0] = np.nan                 # dropouts -> NaN
    hyp = d['hypnogram'].ravel()

    ns = fs * SEGM
    n_ep = x.size // ns
    codes = np.round(hyp[ns // 2::ns][:n_ep]).astype(int)      # stage at each epoch centre
    stage = np.array([STAGES.get(c, '') for c in codes], dtype=object)
    epochs = x[:n_ep * ns].reshape(n_ep, ns)
    half = n_ep // 2
    print(f'{x.size / fs / 3600:.1f} h at {fs} Hz, {n_ep} epochs of {SEGM} s; '
          f'train: epochs 0-{half - 1}, test: epochs {half}-{n_ep - 1}')

    # -- 1.-3. train on the scored, complete epochs of the first half --------------------
    train = np.flatnonzero(np.isfinite(epochs[:half]).all(axis=1)
                           & np.isin(stage[:half], TRAIN_STATES))
    print('training epochs per state:',
          {s: int(np.sum(stage[train] == s)) for s in TRAIN_STATES})
    model = KDEBayesianModel(fs=200, segm_size=SEGM, n_jobs=1)
    with contextlib.redirect_stderr(io.StringIO()):            # hide the tqdm bar
        X, names = model.extract_features_bulk(list(epochs[train]), [fs] * train.size,
                                               return_names=True)
    with contextlib.redirect_stdout(io.StringIO()):            # RFECV prints every step
        model.fit(X, stage[train])
    print(f'{X.shape[1]} features; states {list(model.STATES)}; '
          f'log_lik_floor_ = {model.log_lik_floor_:.1f} '
          f'(lowest training epoch: {np.min(model.train_max_log_lik_):.1f})\n')

    # -- 4. predict the clean test half from the raw signal --------------------------------
    clean = x[half * ns:n_ep * ns].copy()
    pred_c, max_ll_c, k_c, table, scores = _predict(model, clean, fs)
    n_test = n_ep - half
    print(f'predict_signal: {len(table)} annotation rows; first 8:')
    print(table.head(8).to_string(index=False))
    print(f'\n{n_test} test epochs: {k_c.size} scored, {n_test - k_c.size} skipped '
          f'(< 85 % data present), {int(np.sum(pred_c == UNKNOWN_LABEL))} UNKNOWN')

    # agreement with the expert on the clean, scored test epochs
    truth = stage[half + k_c]
    ok = np.isin(truth, TRAIN_STATES)
    acc = np.mean(pred_c[ok] == truth[ok])
    bacc = balanced_accuracy_score(truth[ok], pred_c[ok])
    kappa = cohen_kappa_score(truth[ok], pred_c[ok])
    print(f'\nclean test epochs scored AWAKE/N2/N3/REM by the expert: {ok.sum()}; '
          f'{int(np.sum(pred_c[ok] == UNKNOWN_LABEL))} UNKNOWN')
    print(f'accuracy {acc:.1%}, balanced accuracy {bacc:.1%}, Cohen kappa {kappa:.2f}')
    print('confusion (rows: hypnogram, columns: predicted):')
    cols = TRAIN_STATES + [UNKNOWN_LABEL]
    print(f"{'':>8}" + ''.join(f'{c:>9}' for c in cols))
    for s_ in TRAIN_STATES:
        print(f'{s_:>8}' + ''.join(f'{int(np.sum((truth[ok] == s_) & (pred_c[ok] == c))):>9}'
                                  for c in cols))
    n1 = truth == 'N1'
    print(f'(N1 epochs, not trained: {n1.sum()}, predicted as '
          f'{ {c: int(np.sum(pred_c[n1] == c)) for c in cols if np.any(pred_c[n1] == c)} })')

    # -- 5. artificial epochs written into the test half ----------------------------------
    rng = np.random.default_rng(0)
    sd = np.nanstd(clean)
    t = np.arange(ns) / fs
    # target epochs: complete, scored, in the middle of a stage; N2/N3/REM mostly, so that a
    # changed label is a corrupted one
    full = np.isfinite(clean.reshape(n_test, ns)).all(axis=1)
    stage_t = stage[half:half + n_test]
    stable = np.array([k > 0 and k < n_test - 1 and len({stage_t[k - 1], stage_t[k],
                                                          stage_t[k + 1]}) == 1
                       for k in range(n_test)])
    pool = {s_: list(np.flatnonzero(full & stable & (stage_t == s_))) for s_ in TRAIN_STATES}
    spec = [   # (stage of the target epoch, description, function of the clean epoch)
        ('N2', 'EMG burst: 20-95 Hz noise (5x EEG sd) added',
         lambda e: e + 5 * sd * band_noise(rng, ns, fs, 20, 95)),
        ('REM', 'EMG only: 20-95 Hz noise (5x EEG sd) replaces it',
         lambda e: 5 * sd * band_noise(rng, ns, fs, 20, 95)),
        ('N3', '12 Hz sine, 500 uV, added',
         lambda e: e + 500 * np.sin(2 * np.pi * 12 * t)),
        ('REM', 'disconnected electrode: 0.5 uV white noise',
         lambda e: 0.5 * rng.normal(size=ns)),
        ('N2', '60 Hz line noise, 300 uV, added',
         lambda e: e + 300 * np.sin(2 * np.pi * 60 * t)),
        ('N2', 'clipping at +/-200 uV of a 20x amplified epoch',
         lambda e: np.clip(e * 20, -200, 200)),
        ('N3', 'slow drift only: 0.5-3 Hz noise (5x EEG sd)',
         lambda e: 5 * sd * band_noise(rng, ns, fs, 0.5, 3)),
        ('AWAKE', 'white noise with the EEG sd',
         lambda e: sd * rng.normal(size=ns)),
    ]
    injected, taken = {}, {s_: 0 for s_ in TRAIN_STATES}
    noisy = clean.copy()
    for s_, desc, fn in spec:
        k = int(pool[s_][taken[s_] * 8 + 3])                    # spread out, deterministic
        taken[s_] += 1
        injected[k] = (desc, s_)
        noisy[k * ns:(k + 1) * ns] = fn(clean[k * ns:(k + 1) * ns])
    pred, max_ll, k_test, _, _ = _predict(model, noisy, fs)

    print(f"\n{'injected artefact':<50} {'expert':>6} {'clean':>8} {'injected':>9} "
          f"{'clean ll':>9} {'injected ll':>12}   (ll = max_log_lik_; floor "
          f"{model.log_lik_floor_:.1f})")
    for k, (desc, s_) in injected.items():
        j = np.flatnonzero(k_test == k)[0]
        jc = np.flatnonzero(k_c == k)[0]
        print(f'{desc:<50} {s_:>6} {pred_c[jc]:>8} {pred[j]:>9} {max_ll_c[jc]:>9.1f} '
              f'{max_ll[j]:>12.1f}')

    _plot(stage[half:], k_test, pred, max_ll, model.log_lik_floor_, injected, outdir)
    return model, table, scores


def _plot(stage_test, k_test, pred, max_ll, floor, injected, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nmatplotlib not installed: no figure written')
        return
    os.makedirs(outdir, exist_ok=True)
    order = {UNKNOWN_LABEL: 5, 'AWAKE': 4, 'REM': 3, 'N1': 2, 'N2': 1, 'N3': 0}
    th = np.arange(stage_test.size) * SEGM / 3600
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 0.8]})
    hy = np.array([order.get(s, np.nan) for s in stage_test], dtype=float)
    axes[0].step(th, hy, where='post', color='0.2', lw=1)
    axes[0].set_title('hypnogram (expert), test half')
    pr = np.full(stage_test.size, np.nan)
    pr[k_test] = [order[p] for p in pred]
    axes[1].step(th, pr, where='post', color='tab:blue', lw=1)
    unk = k_test[pred == UNKNOWN_LABEL]
    axes[1].plot(unk * SEGM / 3600, np.full(unk.size, order[UNKNOWN_LABEL]), 'o',
                 color='tab:red', ms=5, label="'UNKNOWN'")
    axes[1].set_title('predicted (gaps: skipped epochs)')
    axes[1].legend(loc='lower right', fontsize=8)
    for ax in axes[:2]:
        ax.set_yticks(list(order.values()))
        ax.set_yticklabels(list(order.keys()))
        for k in injected:
            ax.axvline(k * SEGM / 3600, color='tab:orange', lw=0.6, alpha=0.6)
    axes[2].plot(k_test * SEGM / 3600, max_ll, '.', ms=3, color='0.3')
    axes[2].axhline(floor, color='tab:red', lw=1, label='log_lik_floor_')
    axes[2].set_yscale('symlog', linthresh=10)
    axes[2].set_ylim(1.5 * min(np.nanmin(max_ll), floor), 10)
    axes[2].set_ylabel('max_log_lik_ (symlog)')
    axes[2].set_xlabel('time from test start (h); orange lines: injected epochs')
    axes[2].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, 'hypnogram_unknown.png')
    fig.savefig(path, dpi=110)
    print(f'\nfigure: {path}')


if __name__ == '__main__':
    main(*sys.argv[1:2])
