# Sleep classification demo

`train_predict_unknown.py` trains `brainmaze_eeg.classifiers.KDEBayesianModel` on the
scored first half of the 6.8-h recording in
[`../eeg_wave_detection/patient_one_data.mat`](../eeg_wave_detection/) and classifies the
second half from the raw signal. API reference: the
[Classifiers](https://bnelair.github.io/brainmaze-eeg/classifiers.html) page.

```bash
python train_predict_unknown.py            # ~1-2 min; figure in ./demo_output
```

## What it does

```python
from brainmaze_eeg.classifiers import KDEBayesianModel, UNKNOWN_LABEL

model = KDEBayesianModel(fs=200, segm_size=30, n_jobs=1)
X = model.extract_features_bulk(list_of_30s_epochs, [fs] * n_epochs)[0]  # resampled to 200 Hz
model.fit(X, labels)                       # 'AWAKE', 'N2', 'N3', 'REM'; never 'UNKNOWN'
table = model.predict_signal(signal, fs)   # annotation, start, end, duration (s)
scores = model.predict_signal_scores(signal, fs)  # per-epoch probabilities
model.max_log_lik_, model.log_lik_floor_   # per-epoch best-state log-likelihood, floor
```

- **Labels.** N1 is not trained (the models know AWAKE/N2/N3/REM, as in
  `SleepClassifierWrapper`). Training labels must not contain `'UNKNOWN'` (`fit` raises).
- **Missing data.** The file stores dropouts as a constant; they are set to NaN from its
  `data_present` mask. `predict_signal` skips epochs with less than 85 % non-NaN samples
  (`datarate_threshold`); they leave a gap in the table, and the smoothing does not cross
  them.
- **`'UNKNOWN'`.** An epoch whose best-state log-likelihood is below `log_lik_floor_` gets
  NaN probabilities and the label `'UNKNOWN'` instead of a confident label. With the
  default `log_lik_floor='auto'` the floor is the 0.1 % quantile of the training epochs'
  values minus 30 nats.

The script writes eight artificial epochs into copies of test epochs. **The floor is not an
artifact detector**: it catches epochs whose spectral features are far from every trained
state (EMG-like broadband noise, a strong rhythmic artefact), but a disconnected electrode,
line noise, clipping, slow drift and white noise all get a sleep stage here. Use a signal
quality / artifact detector before classification; a label other than `'UNKNOWN'` does not
mean the epoch is clean.

## Output (brainmaze-eeg 2.0.0, brainmaze-utils 3.0.0, numpy 2.5)

```
6.8 h at 500 Hz, 811 epochs of 30 s; train: epochs 0-404, test: epochs 405-810
training epochs per state: {'AWAKE': 179, 'N2': 87, 'N3': 12, 'REM': 50}
22 features; states ['AWAKE', 'N2', 'N3', 'REM']; log_lik_floor_ = -53.5 (lowest training epoch: -30.0)

predict_signal: 51 annotation rows; first 8:
annotation  start    end  duration
       REM    0.0   90.0      90.0
        N2   90.0  120.0      30.0
        N2  150.0  210.0      60.0
     AWAKE  210.0  570.0     360.0
        N2  570.0  600.0      30.0
   UNKNOWN  600.0  630.0      30.0
     AWAKE  630.0 1200.0     570.0
   UNKNOWN 1200.0 1230.0      30.0

406 test epochs: 399 scored, 7 skipped (< 85 % data present), 3 UNKNOWN

injected epoch                                        label max_log_lik_  (floor -53.5)
EMG burst: 20-95 Hz noise (5x EEG sd) added         UNKNOWN       -164.5
EMG only: 20-95 Hz noise (5x EEG sd) replaces it    UNKNOWN       -158.4
12 Hz sine, 500 uV, added                           UNKNOWN      -1100.4
disconnected electrode: 0.5 uV white noise              REM        -30.5
60 Hz line noise, 300 uV, added                       AWAKE         -2.1
clipping at +/-200 uV of a 20x amplified epoch        AWAKE         -2.0
slow drift only: 0.5-3 Hz noise (5x EEG sd)             REM        -25.3
white noise with the EEG sd                           AWAKE        -16.9

agreement with the hypnogram on 331 untouched AWAKE/N2/N3/REM test epochs: 85.2%
```

(The 120-150 s gap in the table is a skipped epoch: a 6-s dropout, 20 % of it, falls in it.
The 2-s dropout at 118-120 s leaves the previous epoch above the 85 % threshold.)
This is one subject, one channel and one train/test split: the agreement illustrates the
API, it is not a validation of sleep-staging accuracy.
