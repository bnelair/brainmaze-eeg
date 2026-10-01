"""
Interictal epileptiform discharge (spike) detectors.

Detectors
---------
- :func:`~brainmaze_eeg.spikes.janca.detect_spikes_janca` -- Hilbert-envelope
  distribution-modelling detector (Janca et al. 2015), eeg_forge formulation with verified
  filters. Layout ``(n_samples,)`` or ``(n_channels, n_samples)``.
- :class:`~brainmaze_eeg.spikes.janca.SpikeDetectorHilbert` (alias
  ``spike_detector_hilbert_v24``) -- port of the MATLAB ``spike_detector_hilbert_v24`` with
  its full output. Layout ``(n_samples, n_channels)`` (MATLAB convention).
- :func:`~brainmaze_eeg.spikes.barkmeier.detect_spikes_barkmeier` -- amplitude/slope/
  duration half-wave detector (Barkmeier et al. 2012). Layout ``(n_samples,)`` or
  ``(n_channels, n_samples)``.

All detectors fill NaN gaps before detection and drop detections in or near gaps afterwards
(``nan_policy='fill'``, the default), or raise on NaN (``nan_policy='raise'``). The 2-D
detectors raise ``ValueError`` on an array with more channels than samples (probably
transposed).

See ``brainmaze_eeg/spikes/README.md`` for the algorithms, parameters, filter verification
and the comparison with the reference implementations.
"""

from brainmaze_eeg.spikes.barkmeier import (DEFAULT_THRESHOLDS, design_barkmeier_filters,
                                            detect_spikes_barkmeier)
from brainmaze_eeg.spikes.janca import (SpikeDetectorHilbert, design_janca_filters,
                                        detect_spikes_janca, janca_decimation_factor,
                                        janca_resampling, spike_detector_hilbert_v24)

__all__ = [
    'detect_spikes_janca', 'design_janca_filters', 'janca_decimation_factor',
    'janca_resampling', 'SpikeDetectorHilbert', 'spike_detector_hilbert_v24',
    'detect_spikes_barkmeier', 'design_barkmeier_filters', 'DEFAULT_THRESHOLDS',
]
