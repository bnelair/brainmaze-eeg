# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Slow-wave feature extraction demo
=================================

Reproduces the slow-wave morphology pipeline of Carvalho et al. 2024 using
:class:`brainmaze_eeg.features.wave_detector.WaveDetector`.

For each 30 s epoch of a single EEG channel (stored as ``fzcz`` in the demo file;
the study used Fz-(A1+A2)/2) it extracts, in two bands:

* slow oscillation (SO) : 0.5-0.9 Hz
* delta                 : 1.0-3.9 Hz

the mean and median **downslope** (zero-crossing -> negative trough, in uV/s) of slow
waves whose negative peak is at least 5 uV deep, in two configurations:

* default (``trough='refine'``, brainmaze-eeg's feature since v1.0.0): detection on the
  band-limited signal, downslope measured on a 0.5-35 Hz broadband trace
  (``measure_on=``);
* ``trough='paper'``: the Methods text of the study (0.5-35 Hz FIR + 50 ms moving
  average; zero crossings and the negative peak on that trace).

The original ``SlowWaveDetect`` source is not available, so numerical identity with the
published values is not verified (see the ``wave_detector`` module docstring, *Trough
placement*).

Reference
---------
Carvalho D.Z. et al. (2024), Brain Communications 6(5): fcae354.
https://doi.org/10.1093/braincomms/fcae354

Historical note
---------------
The published study used a standalone ``SlowWaveDetect`` routine. That routine was
folded into :class:`WaveDetector`; ``slope='downslope'`` + ``amplitude_threshold`` +
``measure_on`` give this package's feature for any band; ``trough='paper'`` follows the
paper's Methods text.

Run
---
    python example_one_file.py

Requires ``patient_one_data.mat`` (an ~6.8 h Fz recording at 500 Hz with a hypnogram)
in this directory. Output with brainmaze-eeg 2.0.0 (WaveDetector class version 2.1.0)::

    811 epochs, 325 NREM

      band  trough  downslope mean  median  wave rate (1/s)
        SO  refine           173.9    89.3            0.266
     delta  refine           239.3   174.6            1.403
        SO   paper           173.1   163.8            0.043
     delta   paper           195.3   162.8            1.343

(brainmaze-eeg 1.0.0 gave mean downslope SO 186.0 / delta 250.2 and rates 0.253 / 1.354
for the default configuration; the difference is the Butterworth filter that replaced
the ringing brick-wall FFT filter. ``median`` is the NREM mean of the per-epoch
``WAVE_SLOPE_MEDIAN``. The SO paper mode finds few waves: half-waves of 0.55-1 s are
rare on a 0.5-35 Hz trace.)
"""

import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import firwin, filtfilt

from brainmaze_utils.signal import buffer
from brainmaze_eeg.features.wave_detector import WaveDetector

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'patient_one_data.mat')

# hypnogram code -> stage; NREM stages contribute to slow-wave statistics
NREM_STAGES = {1, 2, 3}          # N1, N2, N3
SEGM_SIZE = 30                   # s, one epoch


def bandpass_fir(x, fs, lo, hi, numtaps=1999):
    """Zero-phase FIR band-pass, matching the study's broadband pre-filter."""
    taps = firwin(numtaps, [lo, hi], pass_zero='bandpass', fs=fs, window='hamming')
    return filtfilt(taps, 1.0, x)


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f'Missing demo data: {DATA_PATH}')

    data = loadmat(DATA_PATH)
    fzcz = data['fzcz'].ravel().astype(np.float64)          # uV
    fs = int(data['fsamp'].ravel()[0])
    hypnogram = data['hypnogram'].ravel()                   # per-sample stage code
    print(f'Loaded {fzcz.size / fs / 3600:.1f} h at {fs} Hz')

    # broadband trace the amplitudes/slopes are measured on (0.5-35 Hz)
    broadband = bandpass_fir(fzcz, fs, 0.5, 35.0)

    # two detectors, same interface, different band; both report the downslope on the
    # broadband trace and keep only waves with a >= 5 uV negative peak
    # (the paper mode builds its own 0.5-35 Hz + 50 ms trace from the raw signal)
    detectors = {}
    for trough in ('refine', 'paper'):
        for band, fband in (('SO', (0.5, 0.9)), ('delta', (1.0, 3.9))):
            detectors[(band, trough)] = WaveDetector(
                fs=fs, fband=fband, segm_size=SEGM_SIZE, slope='downslope',
                amplitude_threshold=5, trough=trough)

    features = {}
    for (band, trough), det in detectors.items():
        values, names = det(fzcz, measure_on=broadband if trough == 'refine' else None)
        features[(band, trough)] = dict(zip(names, values))

    # epoch-wise sleep stage (stage at the centre of each 30 s epoch)
    epochs = buffer(hypnogram, fs, segm_size=SEGM_SIZE)      # (n_epochs, epoch_samples)
    epoch_stage = np.ceil(epochs[:, epochs.shape[1] // 2]).astype(int)
    n = min(epoch_stage.size, features[('SO', 'refine')]['WAVE_SLOPE_MEAN'].size)
    epoch_stage = epoch_stage[:n]
    nrem = np.isin(epoch_stage, list(NREM_STAGES))

    print(f'{n} epochs, {int(nrem.sum())} NREM\n')
    print(f"{'band':>6} {'trough':>7} {'downslope mean':>15} {'median':>7} {'wave rate (1/s)':>16}")
    for (band, trough) in detectors:
        f = features[(band, trough)]
        stats = [np.nanmean(f[k][:n][nrem]) if nrem.any() else np.nan
                 for k in ('WAVE_SLOPE_MEAN', 'WAVE_SLOPE_MEDIAN', 'WAVE_RATE')]
        print(f'{band:>6} {trough:>7} {stats[0]:>15.1f} {stats[1]:>7.1f} {stats[2]:>16.3f}')

    return features, epoch_stage


if __name__ == '__main__':
    main()
