# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Janca detector, ``'ripple'`` preset
===================================

Spikes and ripples use **one implementation**, :func:`~brainmaze_eeg.spikes.detect_spikes_janca`;
a preset only supplies default parameter values (:data:`~brainmaze_eeg.spikes.JANCA_PRESETS`),
and any parameter can be overridden on top of it:

* ``preset='spike'`` (default): band 10-60 Hz, analysis at ~200 Hz (eeg_forge values);
* ``preset='ripple'``: band 80-250 Hz, analysis at ~1000 Hz, every other value as
  ``'spike'`` (untuned). Needs ``fs > 500`` Hz.

**The ripple preset has not been validated on real ripples.** This demo only shows what it
does on 2 min of synthetic data at 2 kHz: 1/f background with ripple-like bursts
(100-200 Hz sine under a ~60 ms envelope, 70-120 uV) and, separately, IED-like spikes.
Note that real sharp spikes contain high frequencies too and can trigger a ripple-band
detector ("false ripples"); the smooth synthetic spikes here do not.

Run::

    python ripple_preset.py [output_dir]

Writes ``ripple_preset.png`` to ``output_dir`` (default ``./demo_output``) if matplotlib is
installed.
"""

import os
import sys

import numpy as np

from brainmaze_eeg.spikes import (GapAwareSpikeDetector, JancaDetector, detect_spikes_janca,
                                  janca_params)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ied, pink_noise, random_times, ripple, score  # noqa: E402

FS = 2000
DUR = 120.0


def make_signal(seed=3):
    rng = np.random.default_rng(seed)
    n = int(DUR * FS)
    x = pink_noise(n, FS, 30.0, rng) + 3.0 * rng.normal(size=n)
    t_rip = random_times(rng, 40, DUR, 1.0)
    rip = []
    for t in t_rip:
        w = ripple(FS, rng.uniform(100, 200), rng.uniform(70, 120))
        i = int(t * FS) - w.size // 2
        x[i:i + w.size] += w
        rip.append(int(t * FS))
    t_sp = random_times(rng, 20, DUR, 1.0)
    t_sp = t_sp[np.min(np.abs(t_sp[:, None] - t_rip[None, :]), axis=1) > 1.0]   # apart
    sp = []
    for t in t_sp:
        w = ied(FS, 250.0)
        i = int(t * FS) - int(0.05 * FS)
        x[i:i + w.size] += w
        sp.append(int(t * FS))
    return x, np.asarray(rip), np.asarray(sp)


def main(outdir='demo_output'):
    print('resolved parameters of the ripple preset (janca_params):')
    for k, v in janca_params('ripple', powerline=60).items():
        print(f'  {k:<16} {v}')

    x, rip, sp = make_signal()
    print(f'\nsynthetic: {DUR:.0f} s at {FS} Hz, {rip.size} ripple bursts, {sp.size} spikes\n')

    print(f"{'preset':<8} {'detections':>10} {'ripples found':>14} {'spikes found':>13}")
    det = {}
    for preset in ('spike', 'ripple'):
        det[preset] = detect_spikes_janca(x, FS, preset=preset, powerline=60)
        hr = score(det[preset], rip, FS)[0]
        hs = score(det[preset], sp, FS)[0]
        print(f'{preset:<8} {det[preset].size:>10} {f"{hr}/{rip.size}":>14} {f"{hs}/{sp.size}":>13}')

    # overrides on top of a preset; the same objects work in the gap-aware wrapper
    stricter = detect_spikes_janca(x, FS, preset='ripple', powerline=60, threshold=5.0)
    print(f"\npreset='ripple', threshold=5.0 (override): {stricter.size} detections")
    gad = GapAwareSpikeDetector(JancaDetector('ripple', powerline=60))
    y = x.copy()
    y[int(30 * FS):int(32 * FS)] = np.nan
    out, info = gad.detect(y, FS, return_info=True)
    print(f'gap-aware ripple detector with a 2-s NaN gap: {out.size} detections, '
          f'valid {info["valid_s"]:.1f} s of {DUR:.0f} s')

    _, details = detect_spikes_janca(x, FS, preset='ripple', powerline=60, return_details=True)
    _plot(x, rip, sp, det, details, outdir)


def _plot(x, rip, sp, det, details, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nmatplotlib not installed: no figure written')
        return
    os.makedirs(outdir, exist_ok=True)
    # 2-s window around the first ripple, and the ripple-band envelope / threshold
    c = rip[0] / FS
    lo, hi = c - 1.0, c + 1.0
    t = np.arange(x.size) / FS
    k = (t >= lo) & (t < hi)
    fa = details['fs_analysis']
    env = np.ravel(details['envelope'])
    thr = np.ravel(details['threshold'])
    ta = np.arange(env.size) / fa
    ka = (ta >= lo) & (ta < hi)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(t[k], x[k], lw=0.5, color='0.3')
    for idx, mk, col, lab, yy in ((rip, '|', 'tab:green', 'planted ripple', 250),
                                  (sp, '|', 'tab:purple', 'planted spike', 250),
                                  (det['ripple'], 'v', 'tab:blue', "preset='ripple'", 300),
                                  (det['spike'], 'v', 'tab:orange', "preset='spike'", 340)):
        idx = idx[(idx >= lo * FS) & (idx < hi * FS)]
        axes[0].plot(idx / FS, np.full(idx.size, yy), mk, color=col, ms=8, label=lab)
    axes[0].set_ylabel('uV')
    axes[0].legend(loc='lower right', ncol=2, fontsize=8)
    axes[0].set_title('Ripple preset on synthetic data (not validated on real ripples)')
    axes[1].plot(ta[ka], env[ka], lw=0.8, label='80-250 Hz envelope')
    axes[1].plot(ta[ka], thr[ka], lw=1.2, color='tab:red', label='threshold')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('envelope')
    axes[1].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, 'ripple_preset.png')
    fig.savefig(path, dpi=110)
    print(f'\nfigure: {path}')


if __name__ == '__main__':
    main(*sys.argv[1:2])
