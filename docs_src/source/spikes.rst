Spike detectors
===============

Interictal epileptiform discharge (spike) detectors: Janca (Hilbert-envelope distribution
modelling; eeg_forge formulation and MATLAB-v24 port) and Barkmeier (multichannel half-wave
morphology). Algorithms, every parameter with its default and source, the filter
verification and the comparison with the reference implementations are documented in
``brainmaze_eeg/spikes/README.md`` and in the module docstrings below.

.. automodule:: brainmaze_eeg.spikes
   :no-members:

Janca
-----

.. automodule:: brainmaze_eeg.spikes.janca
   :members: detect_spikes_janca, design_janca_filters, janca_resampling, janca_decimation_factor, SpikeDetectorHilbert

Barkmeier
---------

.. automodule:: brainmaze_eeg.spikes.barkmeier
   :members: detect_spikes_barkmeier, design_barkmeier_filters, DEFAULT_THRESHOLDS

Gap handling
------------

.. automodule:: brainmaze_eeg.spikes._gaps
   :members: prepare_signal, fill_gaps, find_gaps, mask_in_gaps, drop_in_gaps

Filter design
-------------

.. automodule:: brainmaze_eeg.spikes._filters
   :members:
