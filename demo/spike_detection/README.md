# Spike detection demos

Examples for `brainmaze_eeg.spikes` (brainmaze-eeg 2.0.0). API reference: the
[Spike detectors](https://bnelair.github.io/brainmaze-eeg/spikes.html) page; algorithms,
parameters and validation: the [detector README](../../brainmaze_eeg/spikes/README.md).

The package has two layers:

1. **raw detectors**, the algorithms only: `detect_spikes_janca` / `JancaDetector`
   (Janca et al. 2015, presets `'spike'` and `'ripple'`), `SpikeDetectorHilbert` (MATLAB
   v24 port) and `detect_spikes_barkmeier` / `BarkmeierDetector` (Barkmeier et al. 2012).
   Input `(n_channels, n_samples)` (or 1-D), finite: NaN/inf raise `ValueError`;
2. **`GapAwareSpikeDetector`**, which wraps a detector object for data with gaps.

| script | shows | runtime |
|---|---|---|
| `raw_detectors.py` | Janca and Barkmeier on 8-channel synthetic iEEG, scored against planted spikes | ~2 s |
| `gaps_and_dropouts.py` | NaN gaps and constant-value dropouts (zeros, held value, fixed code): raw vs gap-aware; a real dropout in `patient_one_data.mat` | ~5 s |
| `ripple_preset.py` | the same Janca implementation with `preset='ripple'` (80-250 Hz) on synthetic ripple bursts | ~2 s |

`synthetic.py` holds the seeded signal generators (1/f background, 60 Hz hum, IED-like
transients, ripple-like bursts) and the scoring helper (hit = within 50 ms of a planted
event). The signals are a test bench, not a physiological model; the numbers below show
that the API works as described, not detector performance on real data.

Run from this folder (figures go to `./demo_output`, or to the directory given as the
first argument):

```bash
python raw_detectors.py
python gaps_and_dropouts.py
python ripple_preset.py
```

## `raw_detectors.py`

```python
from brainmaze_eeg.spikes import detect_spikes_janca, detect_spikes_barkmeier
janca = detect_spikes_janca(x, fs, powerline=60)   # one array of sample indices per channel
bark = detect_spikes_barkmeier(x, fs)              # one dict per detection, whole montage
```

- `powerline`: the notch frequency. Set it to the **mains frequency of the recording**:
  check the spectrum (50 Hz in Europe, 60 Hz in the Americas). The default (50 Hz) is the
  reference implementation's. The demo recording has 60 Hz mains: its 60 Hz spectral peak is
  about 680 times the neighbouring spectrum, with harmonics at 120, 180 and 240 Hz, and there
  is no 50 Hz peak. With a 50 Hz notch the 60 Hz hum stays in the 10-60 Hz band, inflates
  the envelope and raises the threshold. On the whole night, with the dropouts set to NaN
  (`GapAwareSpikeDetector`), Janca reports **1309** events with `powerline=50` and **2069**
  with `powerline=60`. The 60 Hz result agrees with a run on hum-free data (60/120/180/240 Hz
  notched out beforehand): 2231 against 2241 events with the raw detector, and 2056 of the
  2069 gap-aware detections match within 50 ms. The count is not the argument; agreement
  with the hum-free reference is. These are envelope events, **not validated spikes**: the
  recording has no spike annotations, and 61 % of the detections fall in wake, so many are
  probably EMG, movement or other transients.
- Barkmeier scales each one-minute block of **the whole montage** to a common amplitude:
  pass all channels together, not one at a time.

Output (brainmaze-eeg 2.0.0, numpy 2.5):

```
channel planted | Janca hits false | Barkmeier hits false
      0      33 |         33     0 |             33     0
      1      35 |         31     0 |             33     0
      2      31 |         31     0 |             31     0
      3      34 |         32     0 |             34     0
      4       0 |          0     0 |              0     0
      5       0 |          0     0 |              0     1
      6       0 |          0     0 |              0     1
      7       0 |          0     0 |              0     3
  total     133 |        127     0 |            131     5
```

## `gaps_and_dropouts.py`

Missing data is stored either as NaN or as a constant (0, the last value, a code).

- NaN/inf: the raw detectors raise `ValueError` instead of silently returning nothing.
- Constant runs: the raw detectors accept them and fire **trains of false detections** at
  the steps into and out of the flat stretch (26-30 per 5-15 s dropout below).

`GapAwareSpikeDetector(detector)` treats NaN/inf and constant runs of at least
`flat_as_gap_s` (0.1 s) as gaps. It fills them (`'mirror'`, short gaps up to 20 ms linearly),
runs the detector, drops detections inside or within `edge_margin_s` (0.2 s) of a gap, and
returns per channel the time in which a detection could have been reported,
`info['valid_s']`. **Normalise rates by `valid_s`**, not by the record length.

```python
from brainmaze_eeg.spikes import GapAwareSpikeDetector, JancaDetector, BarkmeierDetector
det = GapAwareSpikeDetector(JancaDetector(powerline=60))      # or BarkmeierDetector()
spikes, info = det.detect(x_with_gaps, fs, return_info=True)
rate_per_min = [len(s) / (v / 60) for s, v in zip(spikes, info['valid_s'])]
```

Output, synthetic part (abridged):

```
raw detector on the constant-value dropouts only (NaN channels left out):
  ch 0: 30/31 planted spikes found, 26 false detections
  ch 1: 25/27 planted spikes found, 30 false detections
  ch 2: 30/31 planted spikes found, 27 false detections

GapAwareSpikeDetector (all 4 channels, NaN and constant runs):
  GapAwareSpikeDetector(JancaDetector('spike', powerline=60), ...)
    ch gaps (s)                   valid_s removed   found false rate/min
     0 30.00-40.00                  109.6       4   29/31     0     15.9
     1 60.00-75.00                  104.6       4   25/27     0     14.3
     2 90.00-95.00                  114.6       1   30/31     0     15.7
     3 20.00-21.00, 50.00-50.01     118.2       0   29/33     0     14.7
```

The real-data part takes 8 min of `../eeg_wave_detection/patient_one_data.mat` with two
amplifier dropouts (5 s and 120 s) that the file stores as the constant 0.197 µV:

```
  raw detect_spikes_janca           :  135 detections; the 117 not kept by the wrapper are all within 2.4 s of a dropout
  GapAwareSpikeDetector (flat runs) :   18 detections, gaps [[126.4, 132.0], [179.8, 300.0]] s, valid 353.4 s
  GapAwareSpikeDetector (NaN mask)  :   18 detections, gaps [[126.4, 132.0], [179.8, 300.0]] s, valid 353.4 s
```

## `ripple_preset.py`

Spikes and ripples use one implementation; a preset only sets defaults, and every
parameter can be overridden on top of it:

```python
detect_spikes_janca(x, fs, preset='ripple', powerline=60)                 # 80-250 Hz, ~1 kHz
detect_spikes_janca(x, fs, preset='ripple', powerline=60, threshold=5.0)  # override
GapAwareSpikeDetector(JancaDetector('ripple', powerline=60))              # with gaps
```

The ripple preset needs `fs > 500` Hz and **has not been validated on real ripples**; its
parameters other than band and analysis rate are the spike values, untuned. With them, a
burst must exceed about 5-6 times the median 80-250 Hz envelope: in this demo 70-120 µV
bursts on a 30 µV-rms 1/f background are found, 15-30 µV bursts are not. Real sharp
spikes have high-frequency content and can also trigger a ripple-band detector; the smooth
synthetic spikes here do not.

```
preset   detections  ripples found  spikes found
spike             8           0/30           8/8
ripple           30          30/30           0/8

preset='ripple', threshold=5.0 (override): 19 detections
gap-aware ripple detector with a 2-s NaN gap: 29 detections, valid 117.6 s of 120 s
```

## References

- Janca R. et al. (2015), Brain Topography 28(1):172-183. https://doi.org/10.1007/s10548-014-0379-1
- Barkmeier D.T. et al. (2012), Clinical Neurophysiology 123(6):1088-1095. https://doi.org/10.1016/j.clinph.2011.09.023
- Zijlmans M. et al. (2012), Annals of Neurology 71(2):169-178 (ripple band). https://doi.org/10.1002/ana.22548
