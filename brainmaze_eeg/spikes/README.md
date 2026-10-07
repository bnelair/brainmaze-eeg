# Spike detectors (`brainmaze_eeg.spikes`)

Interictal epileptiform discharge (IED, "spike") detectors for scalp EEG and iEEG.

API reference: the [Spike detectors](https://bnelair.github.io/brainmaze-eeg/spikes.html)
documentation page. Runnable examples: [`demo/spike_detection`](../../demo/spike_detection/).

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
# the same implementation on the ripple band (preset; NOT validated on real ripples)
ripples = detect_spikes_janca(x, fs, preset='ripple', powerline=60)

# with gaps (NaN/inf anywhere, including whole channels)
det = GapAwareSpikeDetector(JancaDetector(powerline=60))        # or BarkmeierDetector()
spikes, info = det.detect(x_with_nans, fs, return_info=True)
rate_per_min = [len(s) / (v / 60) for s, v in zip(spikes, info['valid_s'])]
```

Raw detectors:

| detector | function / detector object | layout | output |
|---|---|---|---|
| Janca, eeg_forge formulation (**recommended**); presets `'spike'` (default) and `'ripple'` | `detect_spikes_janca(x, fs, preset=..., ...)` / `JancaDetector(preset, ...)` | `(n_samples,)` or `(n_channels, n_samples)` | sample indices per channel |
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
does not matter (µV and V give the same detections; tested). The one exception is the
optional Janca **reference baseline** (3.1.0): it is an absolute level, so the detection
input must have the baseline recording's unit and gain.

**Filter edges.** Every `band` parameter is the filter's **design edge**: the −3 dB point of
the single-pass Butterworth. All detectors apply their filters forward-backward (zero
phase), which squares the response, so the **realised response at a stated edge is
−6.02 dB** (the same convention as eeg_forge and MATLAB v24, which also run their designs
forward-backward). The v24 Chebyshev-II band-pass is specified by its single-pass loss
`cheb_rp` = 6 dB at the edges, i.e. −12 dB realised.

**Missing data must be NaN.** Many recordings store dropouts as a constant (0, the last
value, a fixed code), not as NaN. A raw detector fires trains of false detections at the
steps of such flat runs. Convert missing data to NaN (e.g. from a `data_present` mask), and
use `GapAwareSpikeDetector`, which by default also treats constant runs ≥ 0.1 s as gaps
(see [Gaps](#gaps-gapawarespikedetector)).

**Memory.** The Janca detectors process one channel at a time; through the wrapper, peak
memory for 64 ch × 1 h at 1024 Hz (float32, 0.94 GB) is 1.95 GB (was 12.4 GB). Barkmeier
needs the whole montage per block (≈ 3–4 × the float64 input). For very long recordings,
feed segments (e.g. 30–60 min) and concatenate.

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
- Ripple band of the `'ripple'` preset: Zijlmans, M., Jiruska, P., Zelmann, R., Leijten,
  F.S.S., Jefferys, J.G.R., Gotman, J. (2012). *High-frequency oscillations as a new
  biomarker in epilepsy.* Annals of Neurology 71(2), 169–178.
  https://doi.org/10.1002/ana.22548
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

Every parameter defaults to the value of the chosen `preset` (`'spike'` below) and can be
overridden individually (`preset='ripple', threshold=4`); `janca_params(preset, **overrides)`
returns the resolved values.

| parameter | default (`'spike'`) | unit | source | meaning |
|---|---|---|---|---|
| `preset` | `'spike'` | – | ours | named parameter set: `'spike'` or `'ripple'` (see [Presets](#presets-spike-and-ripple)) |
| `band` | `(10, 60)` | Hz | reference | band-pass design edges: −3 dB single pass, **−6.02 dB realised** (zero-phase) |
| `filter_order` | `3` | – | reference | Butterworth prototype order of the band-pass |
| `powerline` | `50` | Hz | reference | band-stop centre; **use 60 in North America**; `None` = off |
| `notch_width` | `5` | Hz | reference | total band-stop width (±2.5 Hz, −6 dB zero-phase) |
| `notch_order` | `3` | – | reference | Butterworth prototype order of the band-stop |
| `notch_harmonics` | `1` | – | ours | band-stops at `k * powerline`, `k = 1..n`; any that don't fit below Nyquist are skipped with a warning |
| `target_fs` | `200` | Hz | reference | decimation target; `None` = analyse at the input rate |
| `decimation` | `'integer'` | – | reference / ours | `'integer'`: factor `floor(fs/target_fs)` only if `fs >= 2*target_fs` (500 → 250 Hz, 512 → 256, 2048 → 204.8, 256 → not decimated). `'exact'`: rational resampling to `target_fs` (within 1e-6) whenever `fs > target_fs`; works for any rate (24414.0625, 511.99 Hz) |
| `window_s` | `5` | s | reference (`w`) | sliding window of the envelope statistics |
| `threshold` | `3.65` | – | reference (`thr`), paper `k1` | threshold multiplier |
| `min_distance_s` | `0.1` | s | reference | minimum distance between detections |
| `eps_rel` | `1e-6` | – | ours | offset before the log, relative to the median envelope (the reference uses an absolute 1e-6) |
| `return_details` | `False` | – | ours | also return the envelope, threshold curve, filters and rates |
| `baseline` | `None` | – | ours (3.1.0) | a `JancaBaseline` (reference statistics); `None` = the original local model. See [Reference baseline](#reference-baseline-jancabaseline-310) |
| `combine` | `'reference'` | – | ours (3.1.0) | with a baseline: `'reference'` (fixed threshold), `'min'` (lower of local and reference: more sensitive; the reference where the local threshold is undefined), `'max'` (higher: stricter; undefined where the local one is) |
| `broadcast_baseline` | `False` | – | ours (3.1.0) | allow a 1-channel baseline for every channel |
| `gap_aware_stats` | `False` | – | ours (3.1.0) | local statistics over valid samples only, O(n) for any window; see [Long windows and data drops](#long-windows-and-data-drops-gap_aware_stats-310) |
| `valid` | `None` | – | ours (3.1.0) | with `gap_aware_stats`: boolean mask of usable samples (default: all but constant runs ≥ 0.1 s) |
| `min_valid_fraction` | `0.5` | – | ours (3.1.0) | with `gap_aware_stats`: minimum valid fraction of a window, else no threshold (no detections) |
| `stats_margin_s` | `0.5` | s | ours (3.1.0, measured) | with `gap_aware_stats`: margin around invalid samples excluded from the statistics and from detection |
| `channel_names` | `None` | – | ours (3.1.0) | with a baseline: names of the rows of `x`; must equal the baseline's `channel_names` in order, if it has them (`ValueError`) |

**Validation.** Every parameter is checked for type, finiteness (NaN and ±inf are rejected;
`None` means "off" for `powerline` and `target_fs`) and range when it is resolved, i.e.
already when `JancaDetector(...)` is constructed. Checks that need `fs` run at call time: a
band edge at or above the Nyquist frequency of the input or of the analysis rate; a band
edge that the resampler's anti-alias filter attenuates by more than 0.1 dB (about
`band[1] > 0.858 × fs_analysis/2`; the realised edge is −6.02 to −6.12 dB up to that limit; e.g.
`band=(80, 99)` at 1000 → 200 Hz was −11.1 dB); a window shorter than 3 analysis samples;
a record too short for the zero-phase filters. Each raises `ValueError` with a hint. A
`window_s` longer than the analysis record is accepted with a `UserWarning`: the
background statistics then cover the whole (reflected) record instead of a local window
(planted spikes in 4–30 s records are still found; `SpikeDetectorHilbert` warns likewise
for `winsize`).

**Resampling.** `'integer'` keeps the reference's integer factor. `'exact'` uses
`up/down`, the smallest-denominator continued-fraction convergent of `target_fs/fs` within
a relative 1e-6 (exact for simple ratios such as 2/5 or 25/256; 71/8667 for TDT's
24414.0625 Hz → 200 Hz). The realised analysis rate `fs·up/down` is reported, and
detections are mapped back with that ratio, so the approximation causes no timing drift.

**Hilbert length.** If the analysis length has a prime factor > 1000, the FFT of the
Hilbert transform is padded to `scipy.fft.next_fast_len` (pocketfft is 3–11× slower on
such lengths). The padding changes the envelope near **both ends** of the record (more
than 1 % in the first ~0.2 s and the last ~0.05–0.2 s on band-passed noise), so only
detections there can differ (5 of ~1500 on the 6.8 h recording cut to a prime length, all
in its last 1.6 s), as for a record a few samples longer. All other lengths, including
every parity case, are transformed unpadded.

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

### Presets: `'spike'` and `'ripple'`

Spikes and ripples use **one implementation**; a preset only supplies the default values
(`JANCA_PRESETS`, read-only), and any parameter can be overridden on top of it.

| preset | `band` | `target_fs` | everything else | source |
|---|---|---|---|---|
| `'spike'` (default) | 10–60 Hz | 200 Hz | reference values (table above) | eeg_forge `spike_detection_Janca` (54c3704); Janca et al. 2015 |
| `'ripple'` | 80–250 Hz | 1000 Hz | the `'spike'` values, **untuned** | band: clinical ripple band (Zijlmans et al. 2012); analysis rate: ours (4 × the band top: after decimation the band top is at half the analysis Nyquist) |

```python
detect_spikes_janca(x, fs, preset='ripple')                         # 80-250 Hz at ~1 kHz
JancaDetector('ripple', notch_harmonics=5, powerline=50)            # also notch 100-250 Hz
GapAwareSpikeDetector(JancaDetector('ripple'))                      # with gaps
```

With the default `decimation='integer'` the ripple analysis rate is `fs / floor(fs/1000)`:
1000 Hz for 2–5 kHz input, 1024 Hz for 2048 Hz, and the input rate itself for 501–1999 Hz
(not decimated). At the input rate the 250 Hz edge can lie close to Nyquist (0.98 of it at
512 Hz, 0.998 at 501 Hz): the edges are still −6.02 dB, but there is almost no room above
the band. Input of 500 Hz or less is rejected (the 250 Hz edge must lie below Nyquist). Mains harmonics
inside 80–250 Hz are not notched by default (`notch_harmonics=1`).

**The ripple preset has not been validated on real ripples.** The test-suite verifies its
filters (−6.02 dB at 80 and 250 Hz at 1–32 kHz, by design and end to end through the
resampling with tones), its alias rejection (< −40 dB), its validation (rejects 500 Hz
input and analysis rates that put 250 Hz near the resampling Nyquist), and that it finds
synthetic 120 Hz bursts in 1/f noise. The threshold, window and minimum distance were
tuned for spikes; their suitability for ripples is unknown.

### Reference baseline: `JancaBaseline` (3.1.0)

The Janca threshold is `threshold × (mode + median)` of a log-normal model of the envelope
in a sliding window (5 s): `mode = exp(mu − sd²)`, `median = exp(mu)`, where `mu`, `sd` are
the mean and SD of `log(envelope)`. The model assumes that spikes are rare in the window. On a
channel that spikes **permanently** (several per second, e.g. after an injury) the window
learns the spikes as background, the threshold rises with the spike rate, and the detector
goes blind. A longer window does not help (5, 30 and 120 s give the same result: the
spiking is everywhere).

A `JancaBaseline` holds `mu` and `sd` per channel measured on a **reference** recording of
the same channels (quiet, or before the injury); the detector then uses that fixed
threshold, alone or combined with the local one. **The original algorithm stays the
default** (`baseline=None`): detection indices and threshold curves are bit-identical to
3.0.0 (66 golden arrays, numpy 1.24 and 2.5) and to eeg_forge (1494 = 1494 on the real
6.8 h recording).

**When to use it**

- channels with permanent or very frequent spiking (the local model under-detects there);
- comparing a recording with a baseline (pre-injury, pre-treatment) recording of the same
  electrodes: "how far above the old background are these events?".

**Evidence** (synthetic iEEG, 300 s at 500 Hz: 150 µV sharp-and-slow waves in pink
background of SD 30 µV, ~12 µV in the 10–60 Hz band; reference = 300 s of the same
background without spikes; `scratch/janca-baseline/dense_eval.py`, reproducing the prototype
`scratch/janca-dense/` exactly):

| spikes/s | local (original) | `combine='reference'` | `'min'` | `'max'` | false detections |
|---|---|---|---|---|---|
| 1 | 92.6 % | 97.0 % | 97.3 % | 92.3 % | 0 in every cell |
| 3 | 41.6 % | 96.9 % | 96.9 % | 41.6 % | 0 |
| 5 | 0.5 % | 96.2 % | 96.2 % | 0.5 % | 0 |

The test-suite checks this with fixed seeds (≥ 95 % with the reference, 0 false; the local
model ≤ 60 % at 3/s and ≤ 10 % at 5/s).

**Worked example**

```python
from brainmaze_eeg.spikes import JancaBaseline, JancaDetector, GapAwareSpikeDetector, detect_spikes_janca

# 1. measure the baseline once, on a quiet / pre-injury recording (n_channels, n_samples), µV.
#    NaN gaps and constant dropouts are excluded; 'segments' picks spike-free stretches.
base = JancaBaseline.from_signal(x_pre, fs, powerline=60, channel_names=names, units='uV',
                                 segments=[(600, 1800), (5400, 7200)])
print(base)                       # mu, sd, analysis rate, valid seconds per channel
print(base.envelope_levels(threshold=3.65))   # median / mode / threshold envelope in µV
base.save('pre_injury_baseline.json')

# 2. detect later recordings of the same channels against it (same signal path!)
base = JancaBaseline.load('pre_injury_baseline.json')
spikes = detect_spikes_janca(x_post, fs, powerline=60, baseline=base)              # fixed threshold
spikes = detect_spikes_janca(x_post, fs, powerline=60, baseline=base, combine='max')  # stricter
# with gaps: the wrapper keeps every channel aligned with its baseline row
det = GapAwareSpikeDetector(JancaDetector(powerline=60, baseline=base))
spikes, info = det.detect(x_post_with_nans, fs, return_info=True)
# a channel subset / another order: base.select(['LA1', 'LA3'])
```

A baseline can also be given directly: `JancaBaseline(mu=[...], sd=[...], fs=500)` (or
`fs_analysis=250`), e.g. from values reported elsewhere. `janca_threshold(mu, sd, k)` and
`base.envelope_levels()` translate between the log statistics and envelope levels in the
input unit (`exp(mu)` is about 1.06 × the SD of the band-passed background).

**What must match, what is free**

| must equal the baseline's (else `ValueError` naming every mismatch) | free |
|---|---|
| `band`, `filter_order`, `powerline`, `notch_width`, `notch_order`, `notch_harmonics` (notch settings ignored when both have `powerline=None`), `target_fs`, `decimation`, `eps_rel` (`SIGNAL_PATH_PARAMS`), and the analysis rate `fs_analysis` | `threshold` (applies to the baseline model too), `min_distance_s`, `window_s` (local model), `combine` |

The analysis rate follows from the input rate: with the default `decimation='integer'`, a
500 Hz baseline (analysis 250 Hz) does not fit a 1000 Hz recording (200 Hz); use
`decimation='exact'` for both to share one analysis rate across input rates (tested: a
512 Hz baseline on a 1000 Hz recording, ≥ 95 % sensitivity). With `'exact'` the analysis
rates are compared to a relative 2e-6 (each is `target_fs` to within 1e-6), so baselines
made at 511.99 Hz (analysis 200.00008 Hz) or 24414.0625 Hz (199.99982 Hz) fit a 1000 Hz
recording (200 Hz); with `'integer'` they must agree to 1e-9 (tested).

**Channels.** The baseline's channel count must equal the input's. A 1-channel baseline is
applied to every channel only with `broadcast_baseline=True` (explicit, because channels
usually differ in amplitude); a multichannel baseline is never broadcast. Use
`base.select(...)` (indices or names) to match a montage. Through `GapAwareSpikeDetector`
the detector receives the caller's channel indices (detector protocol `accepts_channels`),
so channels fed one at a time or all-missing channels left out stay aligned (tested).
Pass `channel_names=` (to `detect_spikes_janca`, or the full montage's names to
`JancaDetector`) to have the order checked against the baseline's names: a different order
or a different set raises `ValueError` (with a `base.select([...])` hint). `detect()` also
accepts `channels` as names; channel indices must be integers (floats and bools raise).

**How the baseline is measured** (`from_signal`)

1. Missing data is **excluded**, never filled: NaN/inf, and constant runs ≥ `flat_as_gap_s`
   (0.1 s, the wrapper's rule). The values inside a dropout cannot influence the result
   (tested with NaN, inf, 0 and 1e6 dropouts: identical baselines).
2. Every valid contiguous run is filtered, resampled and enveloped on its own, exactly as
   the detector does.
3. `stats_margin_s` = 0.5 s is dropped at both ends of every run (also the record ends), so
   filter/Hilbert transients at the edges stay out. Runs too short for the zero-phase
   filters or the two margins are skipped.
4. `mu`, `sd` = mean and population SD of `log(e + eps_rel·median(e))` over all pooled
   samples (the local model's definition, with the whole valid reference as the window).
   `robust=True`: median and 1.4826·MAD.
5. The valid time per channel is stored (`valid_s`); less than `min_valid_s` (60 s) warns,
   none raises. `x_ref` may be a list of segments of any lengths (pooled; a list is always
   segments, never channels).

Why 0.5 s (`scratch/janca-baseline/margin_probe2.out`): runs of 5–20 s enveloped separately
vs the envelope of the uninterrupted signal (pink noise; 500, 5000 Hz spike, 2 kHz ripple):

| margin | 0 s | 0.1 s | 0.25 s | 0.5 s | 1 s |
|---|---|---|---|---|---|
| threshold ratio (5 s runs, 500 Hz) | 1.0000 | 1.0001 | 1.0001 | 1.0000 | 1.0000 |
| mean \|Δe\| / median(e) (5 s runs, 500 Hz) | 1.36 % | 0.45 % | 0.27 % | 0.18 % | 0.10 % |

The statistics are insensitive to the edges already at 0.1 s on clean noise; 0.5 s adds a
margin for DC steps and amplifier recovery after a dropout (tested: runs at ±5000 µV
offsets give the threshold of the offset-free data within 1 %).

`robust=True` is a different statistic, not a drop-in (`scratch/janca-baseline/robust_probe.out`):

| | mean / SD (default) | median / 1.4826·MAD |
|---|---|---|
| threshold on clean background | 61.3 µV | 70.0 µV (+14 %: the log of a noise envelope is skewed) |
| threshold inflation, reference with 1 / 3 spikes/s | ×1.10 / ×1.44 | ×1.08 / ×1.31 |

**Limitations**

- **Slow amplitude drift is not followed.** A fixed baseline does not track changes of the
  background amplitude over hours to days (electrode impedance, gain, sleep/wake,
  medication). A background that grows makes a reference baseline over-detect, one that
  shrinks makes it under-detect. Refresh the baseline from a recent quiet stretch, or use
  `combine='max'` (never more sensitive than the local model: guards against drift up) or
  `combine='min'` (never less sensitive: keeps detecting where the local model goes blind).
- **Unit, gain, montage and channel order must be the baseline's.** Unit, gain and montage
  are not recorded in the signal and cannot be checked; the channel order is checked when
  both sides have names (`channel_names=`, see Channels). Two safety nets (`UserWarning`):
  1. a channel's envelope level (`level_ratio` = exp(mean log-envelope − `mu`)) differs from
     the baseline's by more than ×10 (`BASELINE_LEVEL_WARN_RATIO`; e.g. V vs µV);
  2. the **median over channels** of the background level (`background_ratio`: the 0.1
     quantile of the log-envelope vs the baseline model's, `mu + ndtri(0.1)·sd`) differs
     by more than ×3 either way (`MONTAGE_LEVEL_WARN_RATIO`; gain or montage mismatch).
     A low quantile measures the background between the spikes, so dense spiking moves it
     much less than `level_ratio`, and the median ignores a minority of spiking or broken
     channels. Through `GapAwareSpikeDetector` the detector sees one channel at a time, so
     there the median is over that channel.

  `return_details` reports both ratios (~1 when everything matches). Tuning of check 2
  (`scratch/janca-baseline/r2/r3_probe.py`, `.out`; 300 s at 500 Hz, two spike generators,
  mean/SD and robust baselines):

  | case | `level_ratio` | `background_ratio` | warns |
  |---|---|---|---|
  | same gain | 0.86–1.13 | 0.77–1.11 | no |
  | gain ×2 / ×2.5 | 1.7–2.3 / 2.1–2.8 | 1.6–2.2 / 1.9–2.8 | no |
  | gain ×3 | 2.6–3.4 | 2.3–3.3 | only where ≥ 3 (one of four settings) |
  | gain ×0.3 | 0.26–0.33 | 0.23–0.32 | yes |
  | dense spiking on every channel: 1–10/s of 150 µV, 1–3/s of 600 µV (in 30 µV background), 5/s of 600 µV with the reviewer's spike shape | 1.2–4.8 | 0.94–2.86 | no |
  | discharges covering most of the record (5/s of 600 µV sharp-and-slow waves, 10/s of 600 µV) | 6.4–13.7 | 2.9–9.9 | 5 of 6 |
  | 4-channel montage in reversed order (amplitudes 10–100 µV) | 0.1–9.6 | median 1.0–1.2 | ×10 per channel only |

  So check 2 catches a gain or montage mismatch from about ×3 (×2.7–3.8 depending on the
  background shape and the baseline statistic) and is not triggered by dense spiking up to
  `level_ratio` ×4.4; it does fire when discharges cover most of the record (then a fixed
  threshold from a quiet baseline should be checked anyway). On the real 6.8 h Fz-Cz
  recording, baselines from one hour applied to another warned in 3 of 60 hour pairs, all
  between hours whose own thresholds differ ×2.7–2.9 (a change of the background level). A channel
  permutation without names is only caught by check 1 when amplitudes differ ×10.
- **A reference with spikes or artifacts** raises the baseline threshold (×1.44 with
  3 spikes/s). Choose quiet data (`segments=`), or `robust=True` (partly).
- **Combined with `gap_aware_stats`**: where the local threshold is undefined (window
  coverage below `min_valid_fraction`) `'min'` uses the reference threshold (`np.fmin`:
  never less sensitive than `'reference'`), `'max'` stays undefined (no detections: never
  more sensitive than the local model). Samples excluded from the statistics (invalid, or
  within `stats_margin_s` of an invalid sample) are undefined in every mode, `'reference'`
  included: no detections on a dropout or its edges (tested).

### Long windows and data drops: `gap_aware_stats` (3.1.0)

A long local window (e.g. `window_s=600`–3600) smooths the background model, but in long
recordings it nearly always contains dropouts. With the original statistics a dropout
stored as a constant (or filled by the wrapper) enters the model: the background drops
and the detector fires everywhere. `gap_aware_stats=True` computes the sliding `mu`/`sd`
over **valid samples only**:

- `valid` (bool, shape of `x`) marks usable samples; by default all except constant runs
  ≥ 0.1 s. `GapAwareSpikeDetector` passes its gap mask (detector protocol
  `accepts_valid`), so filled samples never enter the statistics.
- Analysis samples within `stats_margin_s` (0.5 s) of an invalid sample are left out too.
- Samples left out of the statistics are never detected (threshold NaN there): the filter
  transient of a step into or out of a dropout is not a spike. Without this, zero dropouts
  in a signal with a DC offset (DC-coupled amplifier, raw ADC counts) gave 30–107 false
  detections at the dropout edges with windows ≥ 30 s (review of PR #75, R1); now 0 (tested
  with offsets of 500 and 5000 µV, windows 30 / 60 / 600 s).
- A window needs at least `min_valid_fraction` (0.5) of its samples valid; elsewhere the
  threshold is NaN and nothing is detected (`details['threshold']`, `details['stats_valid']`).
- Same definition as the reference otherwise (window centred, ends reflected, `sd` around
  the per-sample local mean); without invalid samples the threshold equals the original to
  1e-10 and the detections are identical (all parity fixtures; tested). Checked against a
  brute-force evaluation, including NaN exactly where the coverage is too low.
- Cumulative sums: O(n) time and memory for any window.

24 h at 500 Hz, 0.1 spikes/s (`synth_ieeg`), 50 dropouts of 1–600 s stored as 0
(`scratch/janca-baseline/bench24h.py`):

| call | `window_s` | detections | spikes found | false | time | peak RSS |
|---|---|---|---|---|---|---|
| original | 5 | 5734 | 4530 | 1230 (at the dropouts) | 58 s | 2.2 GB |
| original | 3600 | 385 563 | 6793 | 378 805 | 66 s | 2.2 GB |
| `gap_aware_stats=True` | 5 | 4490 | 4515 | 0 | 66 s | 2.3 GB |
| `gap_aware_stats=True` | 3600 | 4496 | 4521 | 0 | 65 s | 2.3 GB |
| `GapAwareSpikeDetector(JancaDetector(window_s=3600, gap_aware_stats=True))` | 3600 | 4496 | 4521 | 0 | 26 s | 3.0 GB |

(7056 injected spikes outside the dropouts, amplitudes 80–300 µV, so ~64 % are found by any
setting; "spikes found" counts injected spikes with a detection within 50 ms, close pairs
can share one. The signal alone takes 0.86 GB.) The sliding statistics themselves take
2.8 s for 21.6 M analysis samples with a 3600 s window and 2.8 s with a 5 s window. Most of
the time is filtering: through long runs of exact zeros the IIR filters decay into subnormal
numbers, which is ~30× slower on x86 (`denormal_probe.out`); the wrapper fills the dropouts
first, hence its 26 s. This affects the original detector equally and is unchanged here.

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
  Measured zero-phase at the default 200 Hz analysis rate: −12.00 dB at 10 and 60 Hz,
  −6 dB at 10.78 and 58.49 Hz (10.80 / 58.72 Hz at 5 kHz: the minimum-order design
  differs slightly with the rate), 0 dB at 15–50 Hz, ≤ −120 dB at 5 and 70 Hz.
  **Fixed defect:** the previous port passed the pass-band edge as `cheby2`'s `Wn`, which
  is the stop-band edge, and discarded `cheb2ord`'s `Wn`. The effective band shrank to
  about 18–48 Hz (−15 dB at 15 Hz, −8 dB at 50 Hz).
- `f_type=2`: Butterworth order 4 high-pass + low-pass (−6 dB zero-phase at the edges).
  `f_type=3`: FIR (`firwin`, fs/2 taps) (−12 dB zero-phase at the edges).
  **Change from v24:** v24 replaced Chebyshev by Butterworth whenever `decimation` was not
  200 Hz, because its Chebyshev spec was normalised to 200 Hz. The spec here is in Hz and
  verified at 200 Hz–32 kHz, so `f_type` is used as given at every rate (pass `f_type=2` to
  reproduce v24 at other rates).
- Hum notches: 2nd-order IIR, pole radius `1 - 0.015*200/fs`. That is v24's 0.985 at
  200 Hz, but it scales with fs so the width stays 0.955 Hz (−3 dB, single pass) at every
  rate. v24's fixed radius widened the notch to about 24 Hz at 5 kHz.
- High-pass: Butterworth order 2, 1 Hz (−6 dB zero-phase).

Resampling uses the rational ratio `decimation / fs` (smallest-denominator convergent
within 1e-6, as `decimation='exact'` above), so 204.8, 511.99 and 24414.0625 Hz input all
work; filters, windows and positions use the realised rate. The band must lie below the
**input** Nyquist too (upsampling 100 Hz data to 200 Hz cannot create 50–60 Hz content),
and the resampler may lose at most 0.1 dB at `bandwidth[1]`. All parameters are validated
at construction and again by `run` (attributes changed later are checked). Not
implemented: beta/mu rejection (`beta`) and the `ti_switch == 2` timing mode.

**Ambiguous spikes (`k2`).** `0 < k2 <= k1`, as in v24 ("k1 >= k2"). A maximum above the
`k2` threshold but not above `k1` is reported with `con` = 0.5 only if an obvious
detection, on any channel, lies within the preceding 10 ms. v24 tests the single sample
10 ms earlier; we read that as a typo for the window. `k2 = k1` (default) disables the
class. An earlier version enforced `k2 >= k1`, the inverse of v24, so the class could never
fire (verification V2). With `k2 < k1` a channel's result depends on the other channels:
`channel_independent` is then False, and `GapAwareSpikeDetector` passes the whole montage
(tested: wrapper = raw, ambiguous detections included).

**Ambiguous class is not comparable to v24 (verification W2).** The reading of the window
decides how many ambiguous detections come out, and MATLAB v23/v25 (same help and code as
v24, which we could not obtain) use the literal single sample. On the real 15-channel
iEEG (256 Hz, 1 h; `k2` 3.0 / 2.5) the window `[i - 10 ms, i]` gives 1908 / 3142 ambiguous
detections on record 0 and 288 / 576 on record 1, against 32 / 65 and 75 / 174 for the
literal single sample (4-60x fewer); a symmetric +-10 ms window gives 2146 / 3674 and
430 / 873 (12-50 % more than ours). Restricting the confirming detection to other channels
changes nothing. So with `k2 < k1` the output is **not** comparable to v24/v23/v25 and
counts are several times larger than the literal reading. Defaults (`k2 == k1`) are
unaffected and identical to the previous version. The behaviour is unchanged here; a
symmetric-window or literal-sample switch is a possible follow-up.

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
2. **[paper] Artifact channels: OFF by default (opt-in).** The paper excludes, per block,
   a channel whose mean slope is more than 10 SD from the channels' mean. No formulation
   we tested keeps every real detection *and* removes realistic artifacts (table below), so
   by default no channel is excluded (`artifact_sd=None`, `artifact_ratio=None`). Two
   formulations are opt-in; both can be combined:
   - `artifact_sd=10` (**spatial robust rule**): centre = median slope, spread =
     `max(1.4826·MAD, artifact_rel_floor·median)`, `artifact_rel_floor` = 0.2. It flags any
     channel above **3× the median channel's slope, whatever the cause**. Use it only
     for montages of similar contacts.
   - `artifact_ratio=3` (**self-referenced rule**, ours): each channel's slope is divided
     by its own median over all blocks; a channel-block is flagged when this ratio is
     more than `artifact_ratio` times the montage's median ratio in that block. It is safe
     for heterogeneous montages, but needs ≥ 3 blocks and misses an artifact that is
     present in most blocks.

   Both rules also remove strong IED bursts confined to a few minutes (2000 µV at 3/s:
   541 → 4 at 256 Hz). A `UserWarning` names the excluded channels.

   **Rule comparison**:
   - Setup: 1/f background, 10 × 60 s blocks, 256 Hz (1000 Hz in brackets where it
     differs).
   - Literal, leave-one-out and self-referenced (`ratio 3`) rules: applied post hoc to
     the detector's per-block slopes.
   - "spatial (sd 10)" and "ratio 3": also run through the detector itself, with the same
     counts.
   - Probes: `brainmaze-work/scratch/eeg-spikes/r3/v1_rules.py`, `v1_final.py`,
     `v1_real_seizure.py`, `v1_burst.py`, `v1_lvfa.py`.

   | case | no rule (default) | paper literal | leave-one-out (round 1) | spatial, sd 10 (round 2) | ratio 3 |
   |---|---|---|---|---|---|
   | (a) 8 equal ch, 300 µV IEDs 1/s on one: kept / 598 | 595 (598) | 595 | **120** (598) | 595 | 595 |
   | (a) same, 1000 µV at 3/s: kept / 1794 (spatial: setup-specific, see note below) | 1619 (1684) | 1619 | **0** (0) | 1619 (**can drop**, see note) | 1619 |
   | (b) noise, 3–32 ch, false flags | 0 | 0 | **3–12 %** at 3–4 ch | 0 | 0 |
   | (c) 12 ch 1× + 4 ch at 3.5 / 4 / 6×: large ch flagged | 0 | 0 | 0 | **100 %** | 0 |
   | (c) spiking large ch: kept (no rule: 221 / 181 / 145) | = | = | = | **0 / 0 / 0** | = |
   | (d) white noise 3× rms in 3 of 10 blocks: flagged; FPs left (no rule: 44 (61)) | 0; 44 | 0; 44 | 100 %; 0 | 100 %; 0 | 100 %; 0 |
   | (d) same, whole record: flagged | 0 | 0 | 100 % | 100 % | **0** |
   | (d) 60 Hz pick-up 300 µV in 3 of 10 blocks: flagged (it causes no FP anyway) | 0 | 0 | 100 % | 100 % | 100 % |
   | (d) slow drift 1500 µV rms / EMG bursts / electrode pops / flat + jumps, 3 of 10 blocks: flagged (FPs with no rule: 0 / 64 / 18 / 39) | 0 | 0 | 33 / 100 / 0 / 100 % | 0 | 0 |
   | strong burst 2000 µV at 3/s in 3 of 10 blocks: kept / 540 | 541 | 541 | **4** | **4** | **4** |
   | real 15-ch iEEG, 2 × 1 h (contact slopes 0.14–4.1× median): flags | 0 | 0 | 0 | 0 | 0 |

   **The 3× limit of the spatial rule depends on the sampling rate (verification W1).** The
   slope ratio of a given discharge to the background depends on the sampling rate and
   bandwidth, so "no loss" in row (a) holds only for that setup. In a homogeneous 8-channel
   montage at 256 Hz, a steady train of 1000 µV IEDs at 3/s on one channel reached a slope
   ratio of 2.92-3.03 (limit 3): `artifact_sd=10` flagged it in 6 of 10 blocks and kept
   510 of 1235 detections (`ratio 3` and no rule: 1235). At 512 and 1000 Hz the same
   discharge gave ratios 1.9 and 1.4 and nothing was lost. The opposite also occurs (the
   2000 µV 3/s burst is dropped at 256 Hz but kept at 1000 Hz, ratio 5.4 vs 2.0). So
   frequent large spikes alone can exceed the 3× limit; do not enable `artifact_sd` on
   data where the spiking channel's IEDs are large and frequent relative to the montage.

   The mean slope of a realistic artifact (drift, EMG, pops, flat stretches with jumps)
   is 0.03–1.9× normal, i.e. inside the range of real spiking channels (1.1–2.3×). A
   slope rule can only catch broadband noise and mains pick-up, and only leave-one-out,
   which fails (a) and (b), catches more. This is why the rule is off by default.
3. **[paper]** Candidates are maxima of the rectified 20–50 Hz signal above a threshold of
   4 SD. **[ours: interpretation]** The paper's wording ("four standard deviations of the
   channel mean amplitude") is ambiguous; here it is `mean + 4·SD` of the *rectified*
   narrow-band signal in the block (on Gaussian noise ≈ 3.2 SD of the narrow-band signal).
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
| `artifact_sd` | `None` (off) | SD | paper value 10; opt-in spatial robust rule, see step 2 |
| `artifact_rel_floor` | `0.2` | – | ours: floor of the spatial rule's spread relative to the median slope (flag above 3× the median) |
| `artifact_ratio` | `None` (off) | – | ours: opt-in self-referenced rule (e.g. 3), see step 2 |
| `trough_search` | `0.05` | s | ours |
| `refractory` | `0` | s | ours |
| `valid` | `None` | bool mask | ours: samples used for the block statistics (set by the wrapper to exclude filled gaps) |
| `return_info` | `False` | – | ours: per-block scale factors, artifact flags, thresholds, filters |

Filters: both are 2nd-order Butterworth band-passes. The paper only says "second order
digital Butterworth"; applying them zero-phase is **[ours]**, and it doubles the
attenuation: the response is −6 dB (not −3 dB) at 1/35 Hz and 20/50 Hz. The broad band is
−30 dB at 80 Hz.

All parameters are validated (type, finiteness, range; e.g. `scale` > 0, `block_s` ≥ 1 s
or `None`, thresholds ≥ 0), also at `BarkmeierDetector(...)` construction.

Changes from the previous version:

- **Broad band.** It was 1–80 Hz while the docstring claimed "per the paper". On 1/f noise
  that gave 0.15–0.16 false detections/s; the paper's band gives 0.015–0.027/s.
  Sensitivity is unchanged for spikes ≥ 200 µV (8 channels, 30 µV 1/f background: 98/98
  for both). It is lower for small spikes (100 µV: 39 vs 50 of 98; 150 µV: 78 vs 86), where
  the old band also produced 4–10× more extra detections.
- **Blocks.** There used to be no one-minute blocks and no artifact rule (now opt-in, see step 2).
- **NaN.** One NaN in one channel made the scaling factor NaN for every channel, so all
  thresholds silently applied to unscaled data. The raw detector now raises. Through
  `GapAwareSpikeDetector` the gap is filled and excluded from the block statistics, and
  the other channels' detections are unchanged (tested).
- **Refractory order.** The refractory period was applied before the merge. It now comes
  after.
- **Transposed input.** A transposed array was accepted silently. It now raises.

## Gaps: `GapAwareSpikeDetector`

```python
GapAwareSpikeDetector(detector, detector_kwargs=None, *, short_gap_s=0.02, fill='mirror',
                      edge_margin_s=0.2, flat_as_gap_s=0.1, seed=0,
                      fill_kwargs=None).detect(x, fs, return_info=False, return_mask=False)
```

Steps:

1. **Find gaps.** Gaps are runs of NaN or ±inf, and runs of exactly equal consecutive
   samples lasting ≥ `flat_as_gap_s` (0.1 s; `None` disables), found per channel on the
   original signal.
2. **Fill.** Gaps up to `short_gap_s` (0.02 s) are interpolated linearly. Longer gaps get
   `fill` (always passed explicitly to the fill function, so a different default there,
   e.g. `brainmaze_utils.gaps`' `'spectral'`, cannot change it):
   - `'mirror'` (default): the neighbouring signal is mirrored into the gap from both sides
     and the two images are cross-faded.
   - `'pink'`: 1/f noise at the neighbours' robust amplitude and level, cross-faded into the
     mirrored signal at the edges. It is seeded per gap, so it is reproducible and
     independent between channels.
   - `'linear'`.

   Channels without any usable sample (all NaN/inf, or constant throughout) are left out,
   so they cannot bias cross-channel statistics such as Barkmeier's median scaling. They
   are flagged and give no detections.
3. **Detect.** The detector runs on the filled montage. Detectors with
   `accepts_valid = True` (Barkmeier; `JancaDetector(gap_aware_stats=True)`) also receive
   the gap mask and keep filled samples out of their statistics. Detectors with
   `channel_independent = True` (Janca, v24) are fed one channel at a time, which bounds
   memory; `JancaDetector(baseline=...)` also receives the channel indices
   (`accepts_channels`) so that each channel uses its own baseline row.
4. **Remove.** Every detection inside a gap, or within `edge_margin_s` (0.2 s) of one, is
   dropped, for short (interpolated) gaps as well as long ones.
5. **Report.** With `return_info=True`, per channel:
   - `gaps` (samples) and `gap_intervals_s`; `flat_runs` (the constant runs among them);
   - `all_nan`;
   - `n_removed`;
   - `valid_s`: record length minus the gaps widened by the margin, i.e. the time in which a
     detection could be reported. Normalise rates by this.
   - `valid_fraction`;
   - with `return_mask=True`, `gap_mask`.

   A `RuntimeWarning` names channels with less than half of the record valid.

All options (`short_gap_s`, `edge_margin_s`, `flat_as_gap_s`, `seed`, `fill_kwargs`) are
validated at construction.

On gap-free data the result is **identical** to the raw detector (tested for all three
detectors).

**Detector protocol.** Any object with `detect(x, fs)` plugs in:

- **Input:** `x`, finite, of shape `(n_channels, n_samples)`.
- **Output:** one entry per channel. Each entry is either an int array of sample indices,
  or a list of dicts with `'peak_index'`.
- **Optional:** `accepts_valid = True` makes the wrapper call `detect(x, fs, valid=mask)`.
- **Optional:** `channel_independent = True` makes the wrapper call `detect` once per
  channel (bounded memory).
- **Optional (3.1.0):** `accepts_channels = True` makes the wrapper also pass
  `channels=` (indices of the passed rows in the caller's array) and `n_channels=` (the
  caller's channel count), for detectors with per-channel state such as a baseline.

So any detector, e.g. the ripple preset `JancaDetector('ripple')`, works without changes to
the wrapper.

**Why `short_gap_s=0.02`.** Linear interpolation carries no band power. With many longer
interpolated gaps, Janca's background (5 s log-envelope mean) drops and the threshold falls
*everywhere*, not only near the gaps. Real Fz–Cz, 30 min, regular NaN dropouts, detections
in valid time vs. the gap-free run (`brainmaze-work/scratch/eeg-spikes/r2/e7_rerun_defaults.py`):

| dropouts | old defaults (`short_gap_s=0.1`, margin 0.1 s) | defaults (`0.02`, margin 0.2 s) |
|---|---|---|
| 20 ms every 0.5 s | 77 vs 82 (+3/−8) | 22 vs 24 (+0/−2) |
| 50 ms every 1 s | 126 vs 114 (+15/−3) | 92 vs 91 (+1/−0) |
| 100 ms every 1 s | 173 vs 108 (+67/−2) | 88 vs 87 (+2/−1) |
| 100 ms every 2 s | 162 vs 129 (+35/−2) | 114 vs 116 (+3/−5) |
| 100 ms every 0.5 s | 170 vs 57 (+117/−4) | no valid time left (reported) |

Gaps ≤ 20 ms are still interpolated (the threshold is unchanged, ×1.00–1.01, for 2–20 ms
dropouts every 0.05–2 s). Even 1–2 sample gaps move nearby detections by > 20 ms, so the
margin applies to every gap. **Trade-off:** with a 0.2 s margin, dropouts every ≤ 0.4 s
leave no valid time; this is reported in `valid_s` and by the warning instead of producing
biased detections. Lower `edge_margin_s` only if you accept detections influenced by the
fill.

**Why `edge_margin_s=0.2`.** The mirror fill copies events next to a gap into it. They are
removed, but they still compete with real maxima in peak selection
(`find_peaks(distance=0.1 s)`), and on spike-dense real data the resulting extra/missing
detections sit 0.10–0.16 s from the edge (independent review of PR #67, R7). With 0.2 s:
2 s gaps +0 extra (was +6 with 0.1 s), 10 s gaps +5 (was +12), of ~480 near-gap
detections. 0.2 s also covers `min_distance_s` (0.1 s) plus half a spike.

**Why `flat_as_gap_s=0.1`.** In the 6.8 h parity recording every constant run ≥ 0.1 s is
missing data: the 6 runs of `data_present == 0` (3–120 s, value 0.1975), a 0.2–0.95 s run
of 0.0399 just before each of them that the mask does not mark, and an unmarked 0.5 s run
at the end. Constant runs within real signal last ≤ 4 ms (2 samples). Raw detector: 1494
detections; NaN from `data_present` only: 1358; with flat runs as gaps (0.05–0.5 s give the
same): 1309. The 49 further removed detections are trains every 0.1–0.2 s in the 2 s
before the unmarked 0.0399 runs, in normal-amplitude signal: false detections caused by the
flat stretch lowering the background. A 1 s threshold misses the short runs.

**Why `'mirror'` is the default.** Measured on 1 h of the real recording with 30 gaps per
length and IED-like transients injected 0.15–1.2 s outside both gap edges (Janca; see PR #67
for the measurement):

| long-gap fill | Janca threshold 0.1–3 s from a gap, vs gap-free (median / 90th pct) | sensitivity to the transients (gap-free: 0.85–0.93 at 60 µV, 0.98–1.0 at 120 µV) | extra detections within 3 s |
|---|---|---|---|
| mirror | ×1.00–1.02 / ×1.04–1.08 | −0.00 to −0.03 | 0–2 per 30 gaps |
| pink | ×1.02–1.07 / ×1.07–1.21 | down to −0.18 (2 s gaps, 60 µV) | 0–1 |
| linear | ×0.2–0.6 | ≈ 1.0 (the threshold collapses) | ~40 (0.5 s gaps) to ~710 (≥ 2 s gaps) |

On spike-dense data (IEDs at 1/s) the independent review found `'mirror'` +12/−20,
`'pink'` +2/−19 and `brainmaze_utils.gaps`' `'spectral'` +5/−23 near-gap detections (of
497, 10 s gaps, 0.1 s margin), with `'spectral'` inflating the threshold most (median ×1.17):
its band powers, measured on spiky context, are re-synthesised as continuous noise. So the
default stays `'mirror'`, passed explicitly, also after switching to `brainmaze_utils.gaps`.

**Helpers.** `brainmaze_eeg/spikes/_gaps.py` provides `find_gaps`, `gap_intervals`,
`fill_gaps`, `mask_in_gaps(det, gaps, fs, *, units, margin_s, end)` and `drop_in_gaps`.
Units are always explicit; a units mix-up raises instead of silently masking nothing.

These helpers are a thin stand-in with the same names and semantics as the final API of
`brainmaze_utils.gaps` (brainmaze-utils PR #26, not yet released). The stand-in has no
`'spectral'` fill; that fill matches the neighbours' spectrum and is the default there.

Follow-up: replace `_gaps.py` with `brainmaze_utils.gaps` once it is released, keeping
`fill='mirror'` explicit (see above).

## Filter verification

`brainmaze_eeg/tests/test_spikes_filters.py` designs every filter from its parameters at
200, 250, 256, 500, 512, 1000, 2000, 2048, 5000, 8000 and 32000 Hz and checks:

- the zero-phase magnitude response at the edges (to 0.01 dB) and the edge frequencies
  themselves (to 0.01 Hz);
- pass-band and stop-band levels;
- stability (all poles inside the unit circle);
- zero phase (symmetric impulse response);
- the gain of pure tones through each detector's real processing path, decimation included.

`test_spikes_janca_presets.py` adds the ripple preset (design at 1–32 kHz, tones end to end,
alias rejection), configured bands just under the resampler limit through the whole
pipeline, and the resampler's own response (`resampler_gain_db`) against `resample_poly`.

Measured (zero-phase, all rates 200 Hz–32 kHz):

| filter | design | −6 dB edges (measured) | other checks |
|---|---|---|---|
| Janca band-pass | Butterworth 3, 10–60 Hz | 10.00 / 60.00 Hz | 0.000 dB at 24.5 Hz, ≤ −90 dB at 2 Hz |
| Janca band-stop | Butterworth 3, 47.5–52.5 Hz | 47.50 / 52.50 Hz | ≤ −190 dB at 50 Hz, > −0.01 dB at 40/60 Hz |
| Barkmeier narrow | Butterworth 2, 20–50 Hz | 20.00 / 50.00 Hz | −15 to −19 dB at 60 Hz |
| Barkmeier broad | Butterworth 2, 1–35 Hz | 1.00 / 35.00 Hz | −30 to −57 dB at 80 Hz |
| Janca `'ripple'` band-pass | Butterworth 3, 80–250 Hz | 80.00 / 250.00 Hz (1–32 kHz; also end to end through the resampling, ±0.1 dB) | 0 dB at 140 Hz, < −30 dB at 40 Hz; aliases < −40 dB |
| v24 band-pass | Chebyshev II, rp 6, rs 60 | −12.00 dB at 10/60 Hz (spec) | ≤ −120 dB at 5/70 Hz |
| v24 hum notch | IIR 2nd order | −3 dB width 0.955 Hz | 0 dB at DC |
| v24 high-pass | Butterworth 2, 1 Hz | 1.00 Hz | – |

The measurement scripts are kept outside the repository; the tests are the permanent record.
