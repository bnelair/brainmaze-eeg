# Spike detectors (`brainmaze_eeg.spikes`)

Interictal epileptiform discharge (IED, "spike") detectors for scalp EEG and iEEG.

The package has **two layers**:

1. **Raw detectors**: the algorithms only. The input must be finite; NaN or ±inf raise a
   `ValueError`. Use them directly on clean data or inside your own pipeline.
2. **`GapAwareSpikeDetector`**: the easy path for signals with missing data, e.g. artifacts
   replaced by NaN. It wraps any detector object, then:
   - fills the gaps;
   - runs the detector;
   - removes detections in or near gaps;
   - reports the valid time per channel, so rates can be normalised by it.

   See [Gaps](#gaps-gapawarespikedetector).

```python
from brainmaze_eeg.spikes import (GapAwareSpikeDetector, JancaDetector, BarkmeierDetector,
                                  detect_spikes_janca)

# raw: finite (n_channels, n_samples) array -> sample indices per channel
spikes = detect_spikes_janca(x, fs, powerline=60)

# with gaps (NaN/inf anywhere, including whole channels)
det = GapAwareSpikeDetector(JancaDetector(powerline=60))        # or BarkmeierDetector()
spikes, info = det.detect(x_with_nans, fs, return_info=True)
rate_per_min = [len(s) / (v / 60) for s, v in zip(spikes, info['valid_s'])]
```

Raw detectors:

| detector | function / detector object | layout | output |
|---|---|---|---|
| Janca, eeg_forge formulation (**recommended**) | `detect_spikes_janca(x, fs, ...)` / `JancaDetector(...)` | `(n_samples,)` or `(n_channels, n_samples)` | sample indices per channel |
| Janca, MATLAB v24 port | `SpikeDetectorHilbert(...).run(d, fs)` (`.detect(x, fs)` for the wrapper, channels first) | `(n_samples,)` or **`(n_samples, n_channels)`** | v24 output dicts (positions, weights, discharges) |
| Barkmeier 2012 | `detect_spikes_barkmeier(sig, fs, ...)` / `BarkmeierDetector(...)` | `(n_samples,)` or `(n_channels, n_samples)` | list of dicts with half-wave metrics |

`SpikeDetectorHilbert` keeps MATLAB's `[samples, channels]` layout; the two functions use
the family convention `(n_channels, n_samples)`. Every 2-D detector raises `ValueError` when
an array has more channels than samples (a probable transposition), so a wrong layout is
not silently analysed.

**Which one?** Use `detect_spikes_janca` for spike rates and timing: it is fast, has no
segment buffering, and reproduces the eeg_forge reference exactly (see below). Use
`SpikeDetectorHilbert` only when you need v24's extra output (CDF/PDF weights, the
ambiguous `k2` class, multichannel discharge grouping) or comparability with MATLAB v24
results. Barkmeier is a different method (morphology thresholds on block-scaled
amplitudes); it is designed for a multichannel iEEG montage.

All amplitude-dependent steps are scale-invariant: Janca models the envelope relative to its
own background, Barkmeier rescales each block to a fixed median amplitude. The input unit
does not matter (µV and V give the same detections; tested).

## References

- Janca, R., Jezdik, P., Cmejla, R., Tomasek, M., Worrell, G.A., Stead, M., Wagenaar, J.,
  Jefferys, J.G.R., Krsek, P., Komarek, V., Jiruska, P., Marusic, P. (2015). *Detection of
  Interictal Epileptiform Discharges Using Signal Envelope Distribution Modelling:
  Application to Epileptic and Non-Epileptic Intracranial Recordings.* Brain Topography
  28(1), 172–183. https://doi.org/10.1007/s10548-014-0379-1
- Reference implementation of the Janca detector used for `detect_spikes_janca`:
  `spike_detection_Janca` in **eeg_forge** by xnejed07,
  https://gitlab.com/xnejed07/eeg_forge (`eeg_forge/detection/spike_detection_janca.py`;
  compared at commit `54c3704d`).
- Barkmeier, D.T., Shah, A.K., Flanagan, D., Atkinson, M.D., Agarwal, R., Fuerst, D.R.,
  Jafari-Khouzani, K., Loeb, J.A. (2012). *High inter-reviewer variability of spike
  detection on intracranial EEG addressed by an automated multi-channel algorithm.*
  Clinical Neurophysiology 123(6), 1088–1095. https://doi.org/10.1016/j.clinph.2011.09.023
  (PMC3277646)

## `detect_spikes_janca`

Per channel: band-pass → power-line band-stop → decimation → Hilbert envelope → sliding
log-normal model of the envelope → threshold `threshold * (mode + median)` → envelope
maxima above the threshold, at least `min_distance_s` apart. The full algorithm, with the
exact sliding-statistics definition of the reference, is in the module docstring.

| parameter | default | unit | source | meaning |
|---|---|---|---|---|
| `band` | `(10, 60)` | Hz | reference | band-pass edges (−6 dB zero-phase) |
| `filter_order` | `3` | – | reference | Butterworth prototype order of the band-pass |
| `powerline` | `50` | Hz | reference | band-stop centre; **use 60 in North America**; `None` = off |
| `notch_width` | `5` | Hz | reference | total band-stop width (±2.5 Hz, −6 dB zero-phase) |
| `notch_order` | `3` | – | reference | Butterworth prototype order of the band-stop |
| `notch_harmonics` | `1` | – | ours | band-stops at `k * powerline`, `k = 1..n`; any that don't fit below Nyquist are skipped with a warning |
| `target_fs` | `200` | Hz | reference | decimation target; `None` = analyse at the input rate |
| `decimation` | `'integer'` | – | reference / ours | `'integer'`: factor `floor(fs/target_fs)` only if `fs >= 2*target_fs` (500 → 250 Hz, 512 → 256, 2048 → 204.8, 256 → not decimated). `'exact'`: resample to exactly `target_fs` whenever `fs > target_fs` |
| `window_s` | `5` | s | reference (`w`) | sliding window of the envelope statistics |
| `threshold` | `3.65` | – | reference (`thr`), paper `k1` | threshold multiplier |
| `min_distance_s` | `0.1` | s | reference | minimum distance between detections |
| `eps_rel` | `1e-6` | – | ours | offset before the log, relative to the median envelope (the reference uses an absolute 1e-6) |
| `return_details` | `False` | – | ours | also return the envelope, threshold curve, filters and rates |

Parameters are validated. A band edge at or above the Nyquist frequency of the input or of
the analysis rate, a low edge ≥ the high edge, a non-positive order, window or threshold,
or an unknown option raises `ValueError` with a hint.

### Agreement with eeg_forge (`spike_detection_Janca`)

With default parameters the algorithm and every number are the reference's. Detection
indices are **identical**:

- on seeded synthetic iEEG (1/f background, mains hum, 25 IED-like transients, 60 s) at 200,
  250, 256, 500, 512, 1000, 2000, 2048, 5000 and 8000 Hz (2 seeds each), and at 10 kHz;
- on a real 6.8 h scalp recording (Fz–Cz, 500 Hz): 1494 vs 1494 detections.

That holds with numpy 1.26 and 2.5. The test-suite pins it with golden detection indices
produced by the reference (`tests/data/janca_eeg_forge_reference.npz`, generated by
`tests/data/make_janca_reference_fixtures.py`, which runs the reference from a local clone;
its source is not included here). Set `EEG_FORGE_PATH=/path/to/eeg_forge` to also compare
against the live reference in the tests.

Intentional differences (each a fix):

| reference behaviour | here | evidence |
|---|---|---|
| filters as `b, a` polynomials: the 50 Hz band-stop loses precision as fs grows (max response error 5e-5 at 5 kHz, 2e-3 at 8 kHz, 6e-2 at 16 kHz) and is **unstable at 32 kHz** (pole radius 1.0006) → NaN → no detections, silently | second-order sections; same response where `b, a` is accurate; stable up to 32 kHz | 16 kHz: 1 of 23 detections moved by one analysis sample; 32 kHz: reference 0, ours 17 (of 25 injected transients) |
| `log(envelope + 1e-6)`: data in volts have envelopes of 1e-5–1e-8, the constant dominates and the detector stops adapting | `eps = eps_rel * median(envelope)` | 6.8 h recording in volts: reference 600 detections, ours 1494 (same as in µV) |
| integer decimation only: analysis at 256 Hz for 512 Hz input, 204.8 Hz for 2048 Hz, none for 250–399 Hz | same by default; `decimation='exact'` resamples to exactly `target_fs` | 6.8 h recording at 500 Hz: exact 200 Hz gives 1466 detections, 96 % of them within 20 ms of the default (250 Hz) ones |
| 1-D only, no checks; a NaN anywhere → no detections for that channel, silently | `(n_channels, n_samples)`, validated parameters; NaN/inf raise `ValueError`; gaps handled by `GapAwareSpikeDetector` | tests |
| `int(0.1*fs) < 1` raises in `find_peaks` | distance clamped to 1 sample | – |

### Ripple / HFO band

The same envelope method can be run on an HFO band, e.g.
`detect_spikes_janca(x, fs, band=(80, 250), target_fs=1000)` (fs ≥ ~1 kHz; the band must be
below the Nyquist frequency of the analysis rate, otherwise a `ValueError` says so). The
filters of this configuration are verified like the default ones (−6 dB at 80 and 250 Hz at
1–32 kHz). The threshold and window were tuned for spikes, not ripples. Detection
performance on ripples has **not** been validated here.

## `SpikeDetectorHilbert` (MATLAB v24 port)

Pipeline: resample to `decimation` Hz (200; `0` = input rate) → power-line notch comb
(`main_hum_freq` 50 Hz and harmonics up to 1.1 × the band top) → 1 Hz high-pass → per
`buffering` segment: band-pass `bandwidth` (10–60 Hz) → Hilbert envelope → per-window
(`winsize` 5 s, `noverlap` 4 s) log-normal fit, smoothed and interpolated → threshold
`k1*(mode+median) - k3*(mean-mode)` → local maxima, poly-spike union → output. All
parameters are attributes settable in the constructor; see the class docstring.

Filters (zero-phase, second-order sections):

- **Band-pass, `f_type=1` (default): Chebyshev II**, minimum order for ≤ `cheb_rp` = 6 dB
  single-pass loss at 10 and 60 Hz and ≥ `cheb_rs` = 60 dB attenuation 5 Hz below / 10 Hz
  above (`cheb_transition_hz`), i.e. v24's normalised specification expressed in Hz.
  Measured zero-phase: −12.00 dB at 10 and 60 Hz, −6 dB at 10.8 and 58.7 Hz, 0 dB at
  15–50 Hz, ≤ −120 dB at 5 and 70 Hz.
  **Fixed defect:** the previous port passed the pass-band edge as `cheby2`'s `Wn`, which
  is the stop-band edge, and discarded `cheb2ord`'s `Wn`. The effective band shrank to
  about 18–48 Hz (−15 dB at 15 Hz, −8 dB at 50 Hz).
- `f_type=2`: Butterworth order 4 high-pass + low-pass (−6 dB zero-phase at the edges).
  `f_type=3`: FIR (`firwin`, fs/2 taps) (−12 dB zero-phase at the edges). As in v24,
  Chebyshev is replaced by Butterworth, with a warning, when `decimation` is neither 200
  nor 0.
- Hum notches: 2nd-order IIR, pole radius `1 - 0.015*200/fs`. That is v24's 0.985 at
  200 Hz, but it scales with fs so the width stays 0.955 Hz (−3 dB, single pass) at every
  rate. v24's fixed radius widened the notch to about 24 Hz at 5 kHz.
- High-pass: Butterworth order 2, 1 Hz (−6 dB zero-phase).

Resampling uses the exact rational ratio `decimation / fs`; fractional rates such as
204.8 Hz work. Not implemented: beta/mu rejection (`beta`) and the `ti_switch == 2` timing
mode.

On the 6.8 h recording v24 finds 2059 detections; 73 % of the eeg_forge detections lie
within 100 ms of a v24 detection. The two formulations differ in background estimation
(windowed MLE + interpolation vs. a sliding mean), filters, and peak selection, so they
are not expected to agree exactly.

## `detect_spikes_barkmeier`

Each step is marked [paper] or [ours] in the module docstring. In brief:

1. **[paper]** The record is processed in one-minute blocks (`block_s=60`; `None` = whole
   record). A trailing remainder shorter than half a block joins the previous block.
   **[ours]** Filtering runs once on the whole record, so there are no block-edge
   transients.
2. **[paper]** Artifact channels: in each block, a channel whose mean slope is more than
   `artifact_sd`=10 SD from the other channels' is excluded. **[ours]** The test is
   leave-one-out, because including the tested channel caps its z-score at √(n−1).
3. **[paper]** Candidates are maxima of the rectified 20–50 Hz signal above
   `mean + 4·SD` of that signal in the block.
4. **[paper]** Each block is scaled so that the median channel's mean rectified 1–35 Hz
   amplitude is 70 µV. A candidate is accepted if both half-waves of the 1–35 Hz signal
   have a total amplitude > 600 µV, each a slope > 7 µV/ms and each a duration > 10 ms.
5. **[ours]** Detections closer than `trough_search` (50 ms) are merged, keeping the largest.
   The optional `refractory` is applied after the merge.

| parameter | default | unit | source |
|---|---|---|---|
| `broad_band`, `broad_order` | `(1, 35)`, `2` | Hz, – | paper ("1–35 Hz, second order digital Butterworth") |
| `narrow_band`, `narrow_order` | `(20, 50)`, `2` | Hz, – | band: paper; order/type not in the paper text: ours (2nd-order Butterworth, like the broad band) |
| `block_s` | `60` | s | paper |
| `scale` | `70` | µV | paper |
| `std_coeff` | `4` | SD | paper |
| `thresholds` | `{'total_amp': 600, 'slope': 7000, 'half_dur': 0.010}` | µV, µV/s, s | paper (7 µV/ms = 7000 µV/s) |
| `artifact_sd` | `10` | SD | paper (`None` = off) |
| `trough_search` | `0.05` | s | ours |
| `refractory` | `0` | s | ours |
| `valid` | `None` | bool mask | ours: samples used for the block statistics (set by the wrapper to exclude filled gaps) |
| `return_info` | `False` | – | ours: per-block scale factors, artifact flags, thresholds, filters |

Filters: both are 2nd-order Butterworth band-passes applied zero-phase, so the response is
−6 dB at 1/35 Hz and 20/50 Hz. The broad band is −30 dB at 80 Hz.

Changes from the previous version:

- **Broad band.** It was 1–80 Hz while the docstring claimed "per the paper". On 1/f noise
  that gave 0.15–0.16 false detections/s; the paper's band gives 0.015–0.027/s.
  Sensitivity is unchanged for spikes ≥ 200 µV (8 channels, 30 µV 1/f background: 98/98
  for both). It is lower for small spikes (100 µV: 39 vs 50 of 98; 150 µV: 78 vs 86), where
  the old band also produced 4–10× more extra detections.
- **Blocks.** There used to be no one-minute blocks and no artifact rule.
- **NaN.** One NaN in one channel made the scaling factor NaN for every channel, so all
  thresholds silently applied to unscaled data. The raw detector now raises. Through
  `GapAwareSpikeDetector` the gap is filled and excluded from the block statistics, and
  the other channels' detections are unchanged (tested).
- **Refractory order.** The refractory period was applied before the merge. It now comes
  after.
- **Transposed input.** A transposed array was accepted silently. It now raises.

## Gaps: `GapAwareSpikeDetector`

```python
GapAwareSpikeDetector(detector, detector_kwargs=None, *, short_gap_s=0.1, fill='mirror',
                      edge_margin_s=0.1, seed=0, fill_kwargs=None).detect(x, fs,
                      return_info=False, return_mask=False)
```

Steps:

1. **Find gaps.** Gaps are runs of NaN or ±inf, found per channel on the original signal.
2. **Fill.** Gaps up to `short_gap_s` are interpolated linearly. Longer gaps get `fill`:
   - `'mirror'` (default): the neighbouring signal is mirrored into the gap from both sides
     and the two images are cross-faded.
   - `'pink'`: 1/f noise at the neighbours' robust amplitude and level, cross-faded into the
     mirrored signal at the edges. It is seeded per gap, so it is reproducible and
     independent between channels.
   - `'linear'`.

   Channels without any finite sample are left out, so they cannot bias cross-channel
   statistics such as Barkmeier's median scaling. They are flagged and give no detections.
3. **Detect.** The detector runs on the filled montage. Detectors with
   `accepts_valid = True` (Barkmeier) also receive the gap mask and keep filled samples out
   of their statistics.
4. **Remove.** Every detection inside a gap, or within `edge_margin_s` of one, is dropped.
5. **Report.** With `return_info=True`, per channel:
   - `gaps` (samples) and `gap_intervals_s`;
   - `all_nan`;
   - `n_removed`;
   - `valid_s`: record length minus the gaps widened by the margin, i.e. the time in which a
     detection could be reported. Normalise rates by this.
   - `valid_fraction`;
   - with `return_mask=True`, `gap_mask`.

On gap-free data the result is **identical** to the raw detector (tested for all three
detectors).

**Detector protocol.** Any object with `detect(x, fs)` plugs in:

- **Input:** `x`, finite, of shape `(n_channels, n_samples)`.
- **Output:** one entry per channel. Each entry is either an int array of sample indices,
  or a list of dicts with `'peak_index'`.
- **Optional:** `accepts_valid = True` makes the wrapper call `detect(x, fs, valid=mask)`.

So a future detector (e.g. a ripple preset `JancaDetector(band=(80, 250), target_fs=1000)`)
works without changes to the wrapper.

**Why `'mirror'` is the default.** Measured on 1 h of the real recording with 30 gaps per
length and IED-like transients injected 0.15–1.2 s outside both gap edges (Janca; see PR #67
for the measurement):

| long-gap fill | Janca threshold 0.1–3 s from a gap, vs gap-free (median / 90th pct) | sensitivity to the transients (gap-free: 0.85–0.93 at 60 µV, 0.98–1.0 at 120 µV) | extra detections within 3 s |
|---|---|---|---|
| mirror | ×1.00–1.02 / ×1.04–1.08 | −0.00 to −0.03 | 0–2 per 30 gaps |
| pink | ×1.02–1.07 / ×1.07–1.21 | down to −0.18 (2 s gaps, 60 µV) | 0–1 |
| linear | ×0.2–0.6 | ≈ 1.0 (the threshold collapses) | ~40 (0.5 s gaps) to ~710 (≥ 2 s gaps) |

**Helpers.** `brainmaze_eeg/spikes/_gaps.py` provides `find_gaps`, `gap_intervals`,
`fill_gaps`, `mask_in_gaps(det, gaps, fs, *, units, margin_s, end)` and `drop_in_gaps`.
Units are always explicit; a units mix-up raises instead of silently masking nothing.

These helpers are a thin stand-in with the same names and semantics as the final API of
`brainmaze_utils.gaps` (brainmaze-utils PR #26, not yet released). The stand-in has no
`'spectral'` fill; that fill matches the neighbours' spectrum and is the default there.

Follow-up: replace `_gaps.py` with `brainmaze_utils.gaps` once it is released, and
re-evaluate making `'spectral'` the default.

## Filter verification

`brainmaze_eeg/tests/test_spikes_filters.py` designs every filter from its parameters at
200, 250, 256, 500, 512, 1000, 2000, 2048, 5000, 8000 and 32000 Hz and checks:

- the zero-phase magnitude response at the edges (to 0.01 dB) and the edge frequencies
  themselves (to 0.01 Hz);
- pass-band and stop-band levels;
- stability (all poles inside the unit circle);
- zero phase (symmetric impulse response);
- the gain of pure tones through each detector's real processing path, decimation included.

Measured (zero-phase, all rates 200 Hz–32 kHz):

| filter | design | −6 dB edges (measured) | other checks |
|---|---|---|---|
| Janca band-pass | Butterworth 3, 10–60 Hz | 10.00 / 60.00 Hz | 0.000 dB at 24.5 Hz, ≤ −90 dB at 2 Hz |
| Janca band-stop | Butterworth 3, 47.5–52.5 Hz | 47.50 / 52.50 Hz | ≤ −190 dB at 50 Hz, > −0.01 dB at 40/60 Hz |
| Barkmeier narrow | Butterworth 2, 20–50 Hz | 20.00 / 50.00 Hz | −15 to −19 dB at 60 Hz |
| Barkmeier broad | Butterworth 2, 1–35 Hz | 1.00 / 35.00 Hz | −30 to −57 dB at 80 Hz |
| v24 band-pass | Chebyshev II, rp 6, rs 60 | −12.00 dB at 10/60 Hz (spec) | ≤ −120 dB at 5/70 Hz |
| v24 hum notch | IIR 2nd order | −3 dB width 0.955 Hz | 0 dB at DC |
| v24 high-pass | Butterworth 2, 1 Hz | 1.00 Hz | – |

The measurement scripts are kept outside the repository; the tests are the permanent record.
