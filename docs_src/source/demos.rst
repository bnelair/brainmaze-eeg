Demos
=====

Runnable examples are in the
`demo/ <https://github.com/bnelair/brainmaze-eeg/tree/main/demo>`_ folder of the repository
(setup and data description: `demo/README.md <https://github.com/bnelair/brainmaze-eeg/blob/main/demo/README.md>`_).
Each one runs against brainmaze-eeg 2.0.0 and brainmaze-utils 3.0.0 from PyPI; matplotlib
is only needed for the figures.

`eeg_wave_detection <https://github.com/bnelair/brainmaze-eeg/tree/main/demo/eeg_wave_detection>`_
   Slow-wave downslope features with :class:`~brainmaze_eeg.features.wave_detector.WaveDetector`
   on a 6.8-h recording (Carvalho et al. 2024): the default and ``trough='paper'`` modes.
   See :doc:`project_wave_detector`.

`spike_detection <https://github.com/bnelair/brainmaze-eeg/tree/main/demo/spike_detection>`_
   The Janca (``'spike'`` preset) and Barkmeier detectors scored against planted spikes;
   :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` on NaN gaps and on
   constant-value dropouts (synthetic and a real amplifier dropout); the ``'ripple'``
   preset. See :doc:`spikes`.

`sleep_classification <https://github.com/bnelair/brainmaze-eeg/tree/main/demo/sleep_classification>`_
   :class:`~brainmaze_eeg.classifiers.KDEBayesianModel` trained on half a night and applied
   to the other half: skipped epochs, and out-of-distribution epochs labelled
   ``'UNKNOWN'`` (with the artefacts that are *not* flagged). See :doc:`classifiers`.
