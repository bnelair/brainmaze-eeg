"""
Interictal epileptiform discharge (spike) detectors.

Detectors
---------
- :func:`~brainmaze_eeg.spikes.barkmeier.detect_spikes_barkmeier` -- amplitude/slope/
  duration half-wave detector (Barkmeier et al. 2012).

See ``brainmaze_eeg/spikes/README.md`` for algorithm references and attribution.
"""

from brainmaze_eeg.spikes.barkmeier import detect_spikes_barkmeier, DEFAULT_THRESHOLDS

__all__ = ['detect_spikes_barkmeier', 'DEFAULT_THRESHOLDS']
