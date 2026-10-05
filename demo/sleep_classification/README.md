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

The script writes eight artificial epochs into copies of test epochs (N2, N3, REM and AWAKE
epochs) and prints, for each, the expert stage, the label of the **clean** epoch and the label
of the corrupted one. **The floor is not an artifact detector**: it catches epochs whose
spectral features are far from every trained state (EMG-like broadband noise, a strong
rhythmic artefact), and reading the table shows three different outcomes:

- flagged `UNKNOWN`: the two EMG epochs and the 500 uV sine;
- **silently mislabelled**: the slow drift turns an N3 epoch into REM, with a
  `max_log_lik_` of -25.3 that is well above the floor (-53.5);
- no change of the label: a disconnected electrode (REM stays REM, but the epoch is junk and
  only has a lower log-likelihood), clipping, white noise and 60 Hz line noise. The 300 uV
  60 Hz sine is invisible to the features (same label, same log-likelihood as the clean epoch),
  because 60 Hz is above the feature bands; that is harmless for staging, not a success of
  the floor.

Use a signal quality / artifact detector before classification; a label other than
`'UNKNOWN'` does not mean the epoch is clean. 0 of the clean test epochs are `UNKNOWN`, so the
floor raises no false alarm here.

The training and prediction calls show tqdm progress bars on stderr; the script hides them
(the classes have no switch for them). They appear when you call the methods yourself.

## Output (brainmaze-eeg 2.0.0, brainmaze-utils 3.0.0, numpy 2.5)

```
6.8 h at 500 Hz, 811 epochs of 30 s; train: epochs 0-404, test: epochs 405-810
training epochs per state: {'AWAKE': 179, 'N2': 87, 'N3': 12, 'REM': 50}
22 features; states ['AWAKE', 'N2', 'N3', 'REM']; log_lik_floor_ = -53.5 (lowest training epoch: -30.0)

predict_signal: 42 annotation rows; first 8:
annotation  start    end  duration
       REM    0.0   90.0      90.0
        N2   90.0  120.0      30.0
        N2  150.0  210.0      60.0
     AWAKE  210.0 1500.0    1290.0
        N2 1500.0 1590.0      90.0
     AWAKE 1590.0 1620.0      30.0
        N2 1620.0 1680.0      60.0
       REM 1680.0 1740.0      60.0

406 test epochs: 399 scored, 7 skipped (< 85 % data present), 0 UNKNOWN

clean test epochs scored AWAKE/N2/N3/REM by the expert: 338; 0 UNKNOWN
accuracy 86.1%, balanced accuracy 83.2%, Cohen kappa 0.81
confusion (rows: hypnogram, columns: predicted):
            AWAKE       N2       N3      REM  UNKNOWN
   AWAKE      105       14        0        1        0
      N2        4       70        2        6        0
      N3        2        9       28        4        0
     REM        1        2        2       88        0
(N1 epochs, not trained: 42, predicted as {'AWAKE': 23, 'N2': 16, 'REM': 3})

injected artefact                                  expert    clean  injected  clean ll  injected ll   (ll = max_log_lik_; floor -53.5)
EMG burst: 20-95 Hz noise (5x EEG sd) added            N2       N2   UNKNOWN       0.8       -160.9
EMG only: 20-95 Hz noise (5x EEG sd) replaces it      REM      REM   UNKNOWN      -0.8       -158.4
12 Hz sine, 500 uV, added                              N3       N2   UNKNOWN      -1.5      -1081.7
disconnected electrode: 0.5 uV white noise            REM      REM       REM      -1.7        -30.5
60 Hz line noise, 300 uV, added                        N2       N2        N2      -1.3         -1.2
clipping at +/-200 uV of a 20x amplified epoch         N2       N2        N2      -2.3         -3.0
slow drift only: 0.5-3 Hz noise (5x EEG sd)            N3       N3       REM       0.4        -25.3
white noise with the EEG sd                         AWAKE    AWAKE     AWAKE      -4.2        -16.9
```

(The 120-150 s gap in the table is a skipped epoch: a 6-s dropout, 20 % of it, falls in it.
The 2-s dropout at 118-120 s leaves the previous epoch above the 85 % threshold.)
Accuracy alone flatters an imbalanced set (N3 has only 12 training epochs; N3 recall is
28/43 = 65 %): the balanced accuracy (mean recall of the four states) and Cohen's kappa are
printed next to it. This is one subject, one channel and one train/test split: the numbers
illustrate the API, they are not a validation of sleep-staging accuracy.
