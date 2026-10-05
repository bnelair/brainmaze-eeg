# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Raw spike detectors: Janca (``'spike'`` preset) and Barkmeier
=============================================================

Runs the two raw detectors of :mod:`brainmaze_eeg.spikes` on 2 min of synthetic 8-channel
iEEG (1 kHz, 1/f background, 60 Hz hum). Channels 0-3 carry IED-like transients at known
times, channels 4-7 are background only. Every detection is scored against the planted
spikes (hit = within 50 ms).

* :func:`~brainmaze_eeg.spikes.detect_spikes_janca` (Janca et al. 2015, eeg_forge
  formulation): band-pass 10-60 Hz, power-line notch, Hilbert envelope, threshold from a
  sliding log-normal model of the envelope. Channels are independent. Returns one array of
  sample indices per channel. ``powerline=60`` for North-American data (the default, 50,
  is the reference's).
* :func:`~brainmaze_eeg.spikes.detect_spikes_barkmeier` (Barkmeier et al. 2012): amplitude,
  slope and duration criteria on the two half-waves of each candidate, after scaling every
  one-minute block of the **whole montage** to a common amplitude. Pass all channels
  together. Returns one dict per detection (channel, ``peak_index``, half-wave metrics).

Both raw detectors need finite input (NaN/inf raise ``ValueError``); for data with gaps see
``gaps_and_dropouts.py``. The layout is ``(n_channels, n_samples)``.

Run::

    python raw_detectors.py [output_dir]

Writes ``raw_detectors.png`` to ``output_dir`` (default ``./demo_output``) if matplotlib is
installed.
"""

import os
import sys

import numpy as np

from brainmaze_eeg.spikes import detect_spikes_barkmeier, detect_spikes_janca

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import peak_indices, score, synth_ieeg  # noqa: E402

FS = 1000          # Hz
DUR = 120.0        # s


def main(outdir='demo_output'):
    x, truth = synth_ieeg(fs=FS, dur=DUR, n_channels=8, spiking=(0, 1, 2, 3), seed=1)
    print(f'synthetic iEEG: {x.shape[0]} channels x {x.shape[1] / FS:.0f} s at {FS} Hz; '
          f'planted spikes per channel: {[len(t) for t in truth]}\n')

    # -- Janca, 'spike' preset (the default) -------------------------------------------
    janca = detect_spikes_janca(x, FS, powerline=60)          # list: one index array / channel

    # -- Barkmeier: the whole montage at once -------------------------------------------
    bark = detect_spikes_barkmeier(x, FS)                     # list of dicts, all channels
    bark_per_ch = [[d for d in bark if d['channel'] == c] for c in range(x.shape[0])]

    print(f"{'channel':>7} {'planted':>7} | {'Janca hits':>10} {'false':>5} | "
          f"{'Barkmeier hits':>14} {'false':>5}")
    tot = np.zeros(5, dtype=int)
    for c in range(x.shape[0]):
        hj, n, fj = score(janca[c], truth[c], FS)
        hb, _, fb = score(peak_indices(bark_per_ch[c]), truth[c], FS)
        tot += (n, hj, fj, hb, fb)
        print(f'{c:>7} {n:>7} | {hj:>10} {fj:>5} | {hb:>14} {fb:>5}')
    print(f"{'total':>7} {tot[0]:>7} | {tot[1]:>10} {tot[2]:>5} | {tot[3]:>14} {tot[4]:>5}")

    d = bark[0]
    print('\nOne Barkmeier detection (amplitudes/slopes in the block-scaled domain):')
    print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v))
           for k, v in d.items()})

    _plot(x, truth, janca, bark_per_ch, outdir)
    return janca, bark


def _plot(x, truth, janca, bark_per_ch, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nmatplotlib not installed: no figure written')
        return
    os.makedirs(outdir, exist_ok=True)
    t0, t1 = 20.0, 30.0                                       # 10-s window
    a, b = int(t0 * FS), int(t1 * FS)
    t = np.arange(a, b) / FS
    fig, ax = plt.subplots(figsize=(11, 7))
    step = 250.0
    for c in range(x.shape[0]):
        off = -c * step
        ax.plot(t, x[c, a:b] + off, lw=0.6, color='0.3')
        for k, (idx, mk, col, lab) in enumerate((
                (truth[c], '|', 'tab:green', 'planted'),
                (janca[c], 'v', 'tab:blue', 'Janca'),
                (peak_indices(bark_per_ch[c]), '^', 'tab:orange', 'Barkmeier'))):
            idx = idx[(idx >= a) & (idx < b)]
            ax.plot(idx / FS, np.full(idx.size, off + 120 + 35 * k), mk, color=col, ms=7,
                    label=lab if c == 0 else None)
    ax.set_yticks([-c * step for c in range(x.shape[0])])
    ax.set_yticklabels([f'ch {c}' for c in range(x.shape[0])])
    ax.set_xlabel('time (s)')
    ax.set_title('Raw detectors on synthetic iEEG (channels 0-3 spiking, 4-7 background)')
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, 'raw_detectors.png')
    fig.savefig(path, dpi=110)
    print(f'\nfigure: {path}')


if __name__ == '__main__':
    main(*sys.argv[1:2])
