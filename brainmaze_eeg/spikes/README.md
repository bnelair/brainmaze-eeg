# Spike detectors — references and attribution

This package implements interictal epileptiform discharge (spike) detectors from the
published literature. The implementations here are **written from scratch against the
papers**; they are not copies of the reference repositories listed below, which carry
their own (restrictive) licenses.

> ⚠️ **The two detectors use opposite multi-channel array conventions**, each following
> its source. **Barkmeier** takes `(n_channels, n_samples)`; **Janca** takes
> `(n_samples, n_channels)` (matching the MATLAB `[samples, channels]`). A transposed
> array will not raise but will give meaningless results. This divergence is intentional
> for now, to keep each detector faithful to its reference during validation; unifying it
> is a candidate follow-up once both are validated.

## Detectors

### Barkmeier (`barkmeier.py`)
Amplitude/slope/duration half-wave detector.

> Barkmeier, D.T., Shah, A.K., Flanagan, D., Atkinson, M.D., Agarwal, R., Fuerst, D.R.,
> Jafari-Khouzani, K., Loeb, J.A. (2012). *High inter-reviewer variability of spike
> detection on intracranial EEG addressed by an automated multi-channel algorithm.*
> Clinical Neurophysiology 123(6), 1088–1095. https://doi.org/10.1016/j.clinph.2011.09.023

### Janca (Hilbert envelope) — `janca.py`
Envelope-distribution-modelling detector. Class `SpikeDetectorHilbert` (alias
`spike_detector_hilbert_v24`), `.run(data, fs)`. Independent implementation against the
paper and the public MATLAB reference; corrects the decimation and segment-overlap defects
of an earlier internal port. Beta/mu rejection and the `ti_switch==2` timing mode are not
implemented (both were untested upstream).

> Janca, R., Jezdik, P., Cmejla, R., Tomasek, M., Worrell, G.A., Stead, M., Wagenaar, J.,
> Jefferys, J.G.R., Krsek, P., Komarek, V., Jiruska, P., Marusic, P. (2015). *Detection of
> Interictal Epileptiform Discharges Using Signal Envelope Distribution Modelling:
> Application to Epileptic and Non-Epileptic Intracranial Recordings.* Brain Topography
> 28(1), 172–183. https://doi.org/10.1007/s10548-014-0379-1

