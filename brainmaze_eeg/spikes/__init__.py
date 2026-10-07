"""
Interictal epileptiform discharge (spike) detectors.

Detectors
---------
- :func:`~brainmaze_eeg.spikes.janca.detect_spikes_janca` -- Hilbert-envelope
  distribution-modelling detector (Janca et al. 2015), eeg_forge formulation with verified
  filters; presets ``'spike'`` (default) and ``'ripple'`` (80-250 Hz, not validated on real
  ripples) of one implementation. Layout ``(n_samples,)`` or ``(n_channels, n_samples)``.
  Optional (3.1.0): a reference baseline (:class:`~brainmaze_eeg.spikes.janca_baseline.JancaBaseline`)
  for permanently spiking channels, and gap-aware local statistics for long windows.
- :class:`~brainmaze_eeg.spikes.janca.SpikeDetectorHilbert` (alias
  ``spike_detector_hilbert_v24``) -- port of the MATLAB ``spike_detector_hilbert_v24`` with
  its full output. Layout ``(n_samples, n_channels)`` (MATLAB convention).
- :func:`~brainmaze_eeg.spikes.barkmeier.detect_spikes_barkmeier` -- amplitude/slope/
  duration half-wave detector (Barkmeier et al. 2012). Layout ``(n_samples,)`` or
  ``(n_channels, n_samples)``.

Two layers:

- **raw detectors** (above): the algorithm only; finite input required (NaN/inf raise
  ``ValueError``);
- :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`: wraps a detector object
  (``JancaDetector``, ``BarkmeierDetector``, ``SpikeDetectorHilbert`` or any object with a
  ``detect(x, fs)`` method), fills NaN/inf gaps, runs it, removes detections in or near gaps
  and reports the valid time per channel.

The 2-D entry points raise ``ValueError`` on an array with more channels than samples
(probably transposed).

See ``brainmaze_eeg/spikes/README.md`` for the algorithms, parameters, filter verification
and the comparison with the reference implementations.
"""

from brainmaze_eeg.spikes.barkmeier import (DEFAULT_THRESHOLDS, BarkmeierDetector,
                                            design_barkmeier_filters, detect_spikes_barkmeier)
from brainmaze_eeg.spikes.gap_aware import GapAwareSpikeDetector
from brainmaze_eeg.spikes.janca import (COMBINE_MODES, JANCA_PRESETS, MAX_RESAMPLER_LOSS_DB,
                                        SIGNAL_PATH_PARAMS, JancaDetector,
                                        SpikeDetectorHilbert, design_janca_filters,
                                        detect_spikes_janca, janca_decimation_factor,
                                        janca_params, janca_resampling, janca_threshold,
                                        resampler_gain_db, spike_detector_hilbert_v24)
from brainmaze_eeg.spikes.janca_baseline import JancaBaseline

__all__ = [
    'GapAwareSpikeDetector',
    'detect_spikes_janca', 'JancaDetector', 'JANCA_PRESETS', 'janca_params',
    'JancaBaseline', 'janca_threshold', 'SIGNAL_PATH_PARAMS', 'COMBINE_MODES',
    'design_janca_filters', 'janca_decimation_factor', 'janca_resampling', 'resampler_gain_db',
    'MAX_RESAMPLER_LOSS_DB', 'SpikeDetectorHilbert', 'spike_detector_hilbert_v24',
    'detect_spikes_barkmeier', 'BarkmeierDetector', 'design_barkmeier_filters',
    'DEFAULT_THRESHOLDS',
]
