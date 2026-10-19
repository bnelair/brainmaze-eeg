Spike detectors
===============

Interictal epileptiform discharge (spike) detectors: Janca (Hilbert-envelope distribution
modelling; eeg_forge formulation with presets ``'spike'`` and ``'ripple'`` -- the latter not
validated on real ripples -- and MATLAB-v24 port) and Barkmeier (multichannel half-wave
morphology). The detectors are raw (finite input only);
:class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` wraps any of them for data
with NaN/inf gaps. Algorithms, every parameter with its default and source, the filter
verification and the comparison with the reference implementations are documented in
the `spike detectors README <https://github.com/bnelair/brainmaze-eeg/blob/main/brainmaze_eeg/spikes/README.md>`_
(``brainmaze_eeg/spikes/README.md``) and in the module docstrings below. Runnable examples
(both detectors, gaps and constant-value dropouts, the ripple preset) are in
`demo/spike_detection <https://github.com/bnelair/brainmaze-eeg/tree/main/demo/spike_detection>`_.

.. automodule:: brainmaze_eeg.spikes
   :no-members:

Janca
-----

.. automodule:: brainmaze_eeg.spikes.janca
   :members: detect_spikes_janca, JancaDetector, JANCA_PRESETS, janca_params, design_janca_filters, janca_resampling, janca_decimation_factor, resampler_gain_db, MAX_RESAMPLER_LOSS_DB, janca_threshold, SIGNAL_PATH_PARAMS, DECISION_PARAMS, COMBINE_MODES, DEFAULT_MIN_VALID_FRACTION, DEFAULT_STATS_MARGIN_S, BASELINE_LEVEL_WARN_RATIO, MONTAGE_LEVEL_WARN_RATIO, MONTAGE_LEVEL_QUANTILE, SpikeDetectorHilbert

Janca reference baseline (3.1.0)
--------------------------------

.. automodule:: brainmaze_eeg.spikes.janca_baseline
   :members: JancaBaseline, BASELINE_SCHEMA, BASELINE_SCHEMA_VERSION

Barkmeier
---------

.. automodule:: brainmaze_eeg.spikes.barkmeier
   :members: detect_spikes_barkmeier, BarkmeierDetector, design_barkmeier_filters, DEFAULT_THRESHOLDS

Gap handling
------------

.. automodule:: brainmaze_eeg.spikes.gap_aware
   :members: GapAwareSpikeDetector

.. automodule:: brainmaze_eeg.spikes._gaps
   :members: fill_gaps, flat_runs, FLAT_AS_GAP_S

``brainmaze_eeg.spikes._gaps`` also re-exports these functions of ``brainmaze_utils.gaps``
(brainmaze-utils >= 3.0.0), documented there:
`find_gaps <https://bnelair.github.io/brainmaze-utils/gaps.html#brainmaze_utils.gaps.find_gaps>`_,
`gap_intervals <https://bnelair.github.io/brainmaze-utils/gaps.html#brainmaze_utils.gaps.gap_intervals>`_,
`mask_in_gaps <https://bnelair.github.io/brainmaze-utils/gaps.html#brainmaze_utils.gaps.mask_in_gaps>`_,
`drop_in_gaps <https://bnelair.github.io/brainmaze-utils/gaps.html#brainmaze_utils.gaps.drop_in_gaps>`_ and
`pink_noise <https://bnelair.github.io/brainmaze-utils/gaps.html#brainmaze_utils.gaps.pink_noise>`_.
Its ``fill_gaps`` (above) is this package's own and differs from
``brainmaze_utils.gaps.fill_gaps`` (see the module description).

Filter design
-------------

.. automodule:: brainmaze_eeg.spikes._filters
   :members:
