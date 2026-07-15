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

For each 30 s epoch of a single Fz-(A1+A2)/2 channel it extracts, in two bands:

* slow oscillation (SO) : 0.5-0.9 Hz
* delta                 : 1.0-3.9 Hz

the mean **downslope** (zero-crossing -> negative trough, in uV/s) of slow waves whose
negative peak is at least 5 uV deep. Detection runs on the band-limited signal; the
downslope amplitude is measured on a 0.5-35 Hz broadband trace (``measure_on=``), exactly
as in the study.

Reference
---------
Carvalho D.Z. et al. (2024), Brain Communications 6(5): fcae354.
https://doi.org/10.1093/braincomms/fcae354

Historical note
---------------
The published study used a standalone ``SlowWaveDetect`` routine. That routine was
folded into :class:`WaveDetector`; ``slope='downslope'`` + ``amplitude_threshold`` +
``measure_on`` are the same feature, now available for any band.

Run
---
    python example_one_file.py

Requires ``patient_one_data.mat`` (an ~6.8 h Fz recording at 500 Hz with a hypnogram)
in this directory.
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
    detectors = {
        'SO':    WaveDetector(fs=fs, fband=(0.5, 0.9), segm_size=SEGM_SIZE,
                              slope='downslope', amplitude_threshold=5),
        'delta': WaveDetector(fs=fs, fband=(1.0, 3.9), segm_size=SEGM_SIZE,
                              slope='downslope', amplitude_threshold=5),
    }

    features = {}
    for name, det in detectors.items():
        values, names = det(fzcz, measure_on=broadband)
        features[name] = dict(zip(names, values))

    # epoch-wise sleep stage (stage at the centre of each 30 s epoch)
    epochs = buffer(hypnogram, fs, segm_size=SEGM_SIZE)      # (n_epochs, epoch_samples)
    epoch_stage = np.ceil(epochs[:, epochs.shape[1] // 2]).astype(int)
    n = min(epoch_stage.size, features['SO']['WAVE_SLOPE_MEAN'].size)
    epoch_stage = epoch_stage[:n]
    nrem = np.isin(epoch_stage, list(NREM_STAGES))

    print(f'{n} epochs, {int(nrem.sum())} NREM\n')
    print(f"{'band':>6} {'mean downslope (uV/s)':>24} {'mean wave rate (1/s)':>22}")
    for name in detectors:
        slope = features[name]['WAVE_SLOPE_MEAN'][:n]
        rate = features[name]['WAVE_RATE'][:n]
        ms = np.nanmean(slope[nrem]) if nrem.any() else np.nan
        mr = np.nanmean(rate[nrem]) if nrem.any() else np.nan
        print(f'{name:>6} {ms:>24.1f} {mr:>22.3f}')

    return features, epoch_stage


if __name__ == '__main__':
    main()
