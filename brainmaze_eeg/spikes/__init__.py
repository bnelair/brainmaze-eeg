"""
Interictal epileptiform discharge (spike) detectors.

Detectors
---------
- :func:`~brainmaze_eeg.spikes.barkmeier.detect_spikes_barkmeier` -- amplitude/slope/
  duration half-wave detector (Barkmeier et al. 2012). Multichannel layout
  ``(n_channels, n_samples)``.
- :class:`~brainmaze_eeg.spikes.janca.SpikeDetectorHilbert` (alias
  ``spike_detector_hilbert_v24``) -- Hilbert-envelope distribution-modelling detector
  (Janca et al. 2015). Multichannel layout ``(n_samples, n_channels)``.

.. warning::
   The two detectors use **opposite** 2-D array conventions (see each above). Passing a
   transposed array will not error but will produce meaningless results.

See ``brainmaze_eeg/spikes/README.md`` for algorithm references and attribution.
"""

from brainmaze_eeg.spikes.barkmeier import detect_spikes_barkmeier, DEFAULT_THRESHOLDS
from brainmaze_eeg.spikes.janca import SpikeDetectorHilbert, spike_detector_hilbert_v24

__all__ = [
    'detect_spikes_barkmeier', 'DEFAULT_THRESHOLDS',
    'SpikeDetectorHilbert', 'spike_detector_hilbert_v24',
]
