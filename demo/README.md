# brainmaze-eeg demos

Runnable examples for [brainmaze-eeg](https://github.com/bnelair/brainmaze-eeg) 2.0.0
(with brainmaze-utils 3.0.0). Documentation: <https://bnelair.github.io/brainmaze-eeg/>.

| folder | what it shows | data |
|---|---|---|
| [`eeg_wave_detection/`](eeg_wave_detection/) | slow-wave downslope features with `WaveDetector` (Carvalho et al. 2024), default and `trough='paper'` modes | `patient_one_data.mat` |
| [`spike_detection/`](spike_detection/) | Janca (`'spike'` preset) and Barkmeier detectors, `GapAwareSpikeDetector` on NaN gaps and constant-value dropouts, the `'ripple'` preset | synthetic, plus `patient_one_data.mat` |
| [`sleep_classification/`](sleep_classification/) | train / predict a sleep classifier, skipped epochs, out-of-distribution epochs labelled `'UNKNOWN'` | `patient_one_data.mat` |

## Setup

```bash
python -m venv venv && . venv/bin/activate
pip install "brainmaze-eeg>=2.0.0" matplotlib     # matplotlib only for the figures
git clone https://github.com/bnelair/brainmaze-eeg.git   # the demo folder and its data
cd brainmaze-eeg/demo
```

The scripts run without matplotlib (they then skip the figures). Each script takes an
optional output directory for its figures (default `./demo_output`).

```bash
python eeg_wave_detection/example_one_file.py              # ~1 min
python spike_detection/raw_detectors.py                    # seconds
python spike_detection/gaps_and_dropouts.py                # seconds
python spike_detection/ripple_preset.py                    # seconds
python sleep_classification/train_predict_unknown.py       # ~1-2 min
```

## Data

`eeg_wave_detection/patient_one_data.mat` (93 MB, MATLAB v5): a 6.8-h overnight scalp EEG
recording, one channel (`fzcz`, Fz-Cz, µV) at 500 Hz, with

- `hypnogram`: per-sample sleep stage, 0 = awake, 1 = N1, 2 = N2, 3 = N3, 5 = REM,
  9 = not scored, -1 = no data. The codes are stored as **floats with tiny offsets** (e.g.
  1.99992 for N2, -7.6e-05 for awake), so `hyp == 2` matches nothing: round before
  comparing (`np.round(hyp).astype(int)`);
- `data_present`: 1 where the amplifier recorded, 0 in dropouts. In this file the dropouts
  are stored as a **constant** (0.197 µV), not as NaN; set them to NaN with this mask (the
  spike and sleep demos show why);
- `fsamp`, `fs_hypno` (500 Hz), `patient_id`.

The spike demos also use seeded synthetic signals (`spike_detection/synthetic.py`) so that
each detection can be scored against known events.
