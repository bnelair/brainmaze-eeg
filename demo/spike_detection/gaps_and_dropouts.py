# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Spike detection on data with gaps: NaN and constant-value dropouts
==================================================================

Real recordings have missing data. It is stored either as NaN or, very often, as a constant
(0, the last value, a fixed code). Neither is safe with a raw detector:

* **NaN/inf**: the raw detectors raise ``ValueError`` (they never silently return nothing);
* **constant runs**: the raw detectors accept them and fire **trains of false detections**
  at and around the steps into and out of the flat run.

:class:`~brainmaze_eeg.spikes.GapAwareSpikeDetector` wraps any detector object
(``JancaDetector``, ``BarkmeierDetector``, ...). It treats NaN/inf **and** constant runs of
at least ``flat_as_gap_s`` (0.1 s) as gaps, fills them, runs the detector, removes
detections inside or within ``edge_margin_s`` (0.2 s) of a gap, and reports per channel the
time in which a detection could have been reported (``info['valid_s']``). Normalise spike
rates by ``valid_s``, not by the record length.

Part 1 (synthetic, 4 channels, 1 kHz, planted spikes): zeros, a held last value, a fixed
code, a NaN gap and a 10-ms NaN gap; raw vs gap-aware, Janca and Barkmeier.

Part 2 (real data, if ``../eeg_wave_detection/patient_one_data.mat`` is present): 8 min of
the 6.8-h scalp recording with two dropouts (5 s and 120 s) that the file stores as the
constant 0.197 uV (its ``data_present`` mask is 0 there). The recording has 60 Hz mains,
hence ``powerline=60``.

Run::

    python gaps_and_dropouts.py [output_dir]

Writes ``gaps_synthetic.png`` and ``gaps_real_dropout.png`` to ``output_dir`` (default
``./demo_output``) if matplotlib is installed.
"""

import os
import sys
import warnings

import numpy as np

from brainmaze_eeg.spikes import (BarkmeierDetector, GapAwareSpikeDetector, JancaDetector,
                                  detect_spikes_janca)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import peak_indices, score, synth_ieeg  # noqa: E402

FS = 1000
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                         'eeg_wave_detection', 'patient_one_data.mat')


def synthetic_part(outdir):
    x, truth = synth_ieeg(fs=FS, dur=120.0, n_channels=4, spiking=(0, 1, 2, 3), seed=2)
    y = x.copy()
    s = lambda t: int(t * FS)                                  # noqa: E731  (s -> sample)
    # missing data stored as a constant (what many file formats / amplifiers do)
    y[0, s(30):s(40)] = 0.0                                    # zeros, 10 s
    y[1, s(60):s(75)] = y[1, s(60) - 1]                        # last value held, 15 s
    y[2, s(90):s(95)] = 500.0                                  # a fixed "missing" code, 5 s
    # missing data stored as NaN
    y[3, s(20):s(21)] = np.nan                                 # 1 s
    y[3, s(50):s(50) + 10] = np.nan                            # 10 ms (interpolated)
    removed = [np.array([[s(30), s(40)]]), np.array([[s(60), s(75)]]),
               np.array([[s(90), s(95)]]), np.array([[s(20), s(21)], [s(50), s(50) + 10]])]

    print('Part 1: synthetic, 4 channels x 120 s at 1 kHz\n')
    print('raw detector on data with NaN:')
    try:
        detect_spikes_janca(y, FS, powerline=60)
    except ValueError as e:
        print(f'  ValueError: {e}\n')

    print('raw detector on the constant-value dropouts only (NaN channels left out):')
    raw = detect_spikes_janca(y[:3], FS, powerline=60)
    for c in range(3):
        h, n, f = score(raw[c], truth[c], FS, exclude=removed[c])
        print(f'  ch {c}: {h}/{n} planted spikes found, {f} false detections')

    print('\nGapAwareSpikeDetector (all 4 channels, NaN and constant runs):')
    results = {}
    for name, det in (('Janca', JancaDetector(powerline=60)),
                      ('Barkmeier', BarkmeierDetector())):
        gad = GapAwareSpikeDetector(det)
        out, info = gad.detect(y, FS, return_info=True)
        results[name] = (out, info)
        print(f'  {gad!r}')
        print(f"  {'ch':>4} {'gaps (s)':<26} {'valid_s':>7} {'removed':>7} {'found':>7} "
              f"{'false':>5} {'rate/min':>8}")
        for c in range(y.shape[0]):
            idx = peak_indices(out[c])
            h, n, f = score(idx, truth[c], FS, exclude=removed[c])
            gaps = ', '.join(f'{a:.2f}-{b:.2f}' for a, b in info['gap_intervals_s'][c])
            rate = idx.size / (info['valid_s'][c] / 60)
            print(f"  {c:>4} {gaps:<26} {info['valid_s'][c]:>7.1f} {info['n_removed'][c]:>7} "
                  f"{f'{h}/{n}':>7} {f:>5} {rate:>8.1f}")
        print()
    print('Planted spikes inside a gap are not in the data and are not counted. "removed": '
          'detections dropped\nbecause they lie in a gap or within edge_margin_s (0.2 s) of '
          'one. rate/min = detections / valid_s.\n')

    _plot_synthetic(y, truth, removed, raw, results['Janca'][0], outdir)


def real_part(outdir):
    if not os.path.exists(DATA_PATH):
        print(f'Part 2 skipped: {DATA_PATH} not found')
        return
    from scipy.io import loadmat
    d = loadmat(DATA_PATH)
    fs = int(d['fsamp'].ravel()[0])
    a, b = 15430 * fs, 15910 * fs                     # 8 min around the 15610-15730 s dropout
    x = d['fzcz'].ravel()[a:b]
    present = d['data_present'].ravel()[a:b].astype(bool)
    print(f'Part 2: real scalp EEG (Fz-Cz, {fs} Hz), 8 min with two dropouts (5 s, 120 s) '
          f'stored as a constant ({np.unique(x[~present]).round(3).tolist()} uV)\n')

    raw = detect_spikes_janca(x, fs, powerline=60)             # 1-D in -> 1-D out
    gad = GapAwareSpikeDetector(JancaDetector(powerline=60))
    out, info = gad.detect(x, fs, return_info=True)

    # better still: mark the missing samples as NaN from the file's data_present mask
    x_nan = np.where(present, x, np.nan)
    out_nan, info_nan = gad.detect(x_nan, fs, return_info=True)

    # raw detections that the wrapper does not report, and how far they are from a dropout
    extra = np.setdiff1d(raw, out) / fs
    dropouts_s = np.array([[127.0, 132.0], [180.0, 300.0]])        # data_present == 0
    dist = np.min([np.maximum(0, np.maximum(g0 - extra, extra - g1)) for g0, g1 in dropouts_s],
                  axis=0)
    print(f'  raw detect_spikes_janca           : {raw.size:4d} detections; the {extra.size} not '
          f'kept by the wrapper are all within {dist.max():.1f} s of a dropout')
    print(f'  GapAwareSpikeDetector (flat runs) : {out.size:4d} detections, gaps '
          f"{np.round(info['gap_intervals_s'], 2).tolist()} s, valid {info['valid_s']:.1f} s")
    print(f'  GapAwareSpikeDetector (NaN mask)  : {out_nan.size:4d} detections, gaps '
          f"{np.round(info_nan['gap_intervals_s'], 2).tolist()} s, valid {info_nan['valid_s']:.1f} s")
    print('\n  Both give the same gaps: the constant-run rule (flat_as_gap_s=0.1) finds the '
          'dropouts without\n  the mask, including the 0.2-0.6 s flat stretch (0.04 uV) that '
          'precedes each one in this file.')
    _plot_real(x, present, fs, raw, out, outdir)


def _pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print('matplotlib not installed: no figure written')
        return None


def _plot_synthetic(y, truth, removed, raw, gap_aware, outdir):
    plt = _pyplot()
    if plt is None:
        return
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = np.arange(y.shape[1]) / FS
    for ax, c, (lo, hi) in ((axes[0], 0, (25, 45)), (axes[1], 1, (55, 80))):
        k = (t >= lo) & (t < hi)
        ax.plot(t[k] - lo, y[c, k], lw=0.5, color='0.3')
        planted = truth[c][~np.any([(truth[c] >= g0) & (truth[c] < g1)
                                    for g0, g1 in removed[c]], axis=0)]
        for idx, mk, col, lab, yy in ((planted, '|', 'tab:green', 'planted', 260),
                                      (raw[c], 'v', 'tab:red', 'raw Janca', 330),
                                      (gap_aware[c], '^', 'tab:blue', 'gap-aware Janca', 400)):
            idx = idx[(idx >= lo * FS) & (idx < hi * FS)]
            ax.plot(idx / FS - lo, np.full(idx.size, yy), mk, color=col, ms=7, label=lab)
        ax.set_title(f'channel {c}: ' + ('zeros 30-40 s' if c == 0 else 'last value held 60-75 s'))
        ax.set_ylabel('uV')
        ax.legend(loc='lower right', ncol=3, fontsize=8)
    axes[1].set_xlabel('time from window start (s)')
    fig.tight_layout()
    path = os.path.join(outdir, 'gaps_synthetic.png')
    fig.savefig(path, dpi=110)
    print(f'figure: {path}\n')


def _plot_real(x, present, fs, raw, gap_aware, outdir):
    plt = _pyplot()
    if plt is None:
        return
    os.makedirs(outdir, exist_ok=True)
    t = np.arange(x.size) / fs
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, x, lw=0.4, color='0.3')
    ax.fill_between(t, -300, 300, where=~present, color='tab:red', alpha=0.1,
                    label='data_present == 0')
    ax.plot(raw / fs, np.full(raw.size, 220), 'v', color='tab:red', ms=5, label='raw Janca')
    ax.plot(gap_aware / fs, np.full(gap_aware.size, 260), '^', color='tab:blue', ms=5,
            label='gap-aware Janca')
    ax.set_ylim(-300, 300)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('uV')
    ax.set_title('Real recording: constant-value dropouts (15430-15910 s of patient_one_data.mat)')
    ax.legend(loc='lower left', ncol=3, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, 'gaps_real_dropout.png')
    fig.savefig(path, dpi=110)
    print(f'\nfigure: {path}')


def main(outdir='demo_output'):
    with warnings.catch_warnings():
        warnings.simplefilter('always')                       # show every warning once
        synthetic_part(outdir)
        real_part(outdir)


if __name__ == '__main__':
    main(*sys.argv[1:2])
