
BrainMaze: Brain Electrophysiology, Behavior and Dynamics Analysis Toolbox - EEG
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This toolbox provides tools for processing of intracranial EEG recordings. See below and documentation for specific sections. This tool was separated from the BrainMaze toolbox to support a convenient and lightweight sharing of these tools across projects.

This project was originally developed as a part of the `BEhavioral STate Analysis Toolbox (BEST) <https://github.com/bnelair/best-toolbox>`_ project. However, the development has transferred to the BrainMaze project.



Documentation
"""""""""""""""

Documentation is available at https://bnelair.github.io/brainmaze-eeg/.

Runnable examples are in the `demo/ <https://github.com/bnelair/brainmaze-eeg/tree/main/demo>`_ folder
(see `demo/README.md <https://github.com/bnelair/brainmaze-eeg/blob/main/demo/README.md>`_).


Installation
"""""""""""""""""""""""""""

.. code-block:: bash

    pip install brainmaze-eeg

What's new in 2.0.0
"""""""""""""""""""""""""""

brainmaze-eeg 2.0.0 is a major release: default outputs change. It requires
``brainmaze-utils>=3.0.0``. Full list of changes: `release notes <https://github.com/bnelair/brainmaze-eeg/releases/tag/v2.0.0>`_.

- **Spike detectors** (``brainmaze_eeg.spikes``), reworked: ``detect_spikes_janca`` /
  ``JancaDetector`` (Janca et al. 2015, eeg_forge formulation, with presets ``'spike'`` and
  ``'ripple'``; the ripple preset is not validated on real ripples), a fixed MATLAB-v24 port
  ``SpikeDetectorHilbert``, and the Barkmeier et al. 2012 detector, now matching the paper
  (1-35 Hz broad band, one-minute blocks). The raw detectors require finite input;
  ``GapAwareSpikeDetector`` wraps any of them for data with NaN gaps or constant-value
  dropouts and reports the valid time per channel. See the
  `Spike detectors <https://bnelair.github.io/brainmaze-eeg/spikes.html>`_ page and the
  `detailed README <https://github.com/bnelair/brainmaze-eeg/blob/main/brainmaze_eeg/spikes/README.md>`_.
- **WaveDetector** (slow waves), reworked: Butterworth band-pass instead of the ringing FFT
  filter, NaN gaps handled, the same ``values, names = detector(x)`` interface as the other
  feature extractors, features of the filtered and the unfiltered signal, a ``'paper'``
  trough mode (Carvalho et al. 2024 Methods), a new ``WAVE_SLOPE_MEDIAN`` feature, and
  7.5-9x faster. The default reproduces the v1.0.0 trough placement. See the
  `WaveDetector <https://bnelair.github.io/brainmaze-eeg/features.wave_detector.html>`_ and
  `slow-wave project <https://bnelair.github.io/brainmaze-eeg/project_wave_detector.html>`_ pages.
- **Sleep classifiers** (``brainmaze_eeg.classifiers``): runtime bugs fixed, inputs
  standardised, resampling checked, and **out-of-distribution epochs are labelled**
  ``'UNKNOWN'`` (NaN probabilities) instead of getting a confident label. The floor is not an
  artifact detector. See the `Classifiers <https://bnelair.github.io/brainmaze-eeg/classifiers.html>`_ page.
- **Demos** for all three: `demo/ <https://github.com/bnelair/brainmaze-eeg/tree/main/demo>`_.

How to contribute
"""""""""""""""""""""""""""
The project has 2 main protected branches *main* that contains official software releases and *dev* that contains the latest feature implementations shared with developers.
To implement a new feature a new branch should be created from the *dev* branch with name pattern of *developer_identifier/feature_name*.

After the feature is implemented, a pull request can be created to merge the feature branch into the *dev* branch with. Pull requests need to be reviewed by a maintainer.

Releasing
'''''''''''''''''''''''''''''''

Releases are automated and go through a reviewed pull request. To cut a release, a maintainer runs the **Prepare release** GitHub Action (*Actions* tab, ``workflow_dispatch``) and selects the bump (*patch* / *minor* / *major*). This opens a small ``Release vX.Y.Z`` pull request that bumps ``[project].version`` in ``pyproject.toml`` on a ``release/bump-*`` branch off *main*. A reviewer checks that it changes only that line and squash-merges it into *main*; the merge triggers the **Release** workflow, which tests, builds, publishes to PyPI (Trusted Publishing), and then tags the version and creates the GitHub release. The build reads the version from ``pyproject.toml``, which remains the single source of truth.

Promotion of features from *dev* to *main* is independent of releases and **must not change** ``[project].version`` -- a *Version guard* CI check flags any pull request outside the release flow that edits it (the check is advisory, so reviewers must not merge a flagged pull request). The version line is owned solely by the release automation on *main*; this is what keeps ``dev`` -> ``main`` merges free of version conflicts under the squash-merge policy (a version edited on both branches would otherwise conflict every release cycle).

The **Prepare release** action requires the repository/organization setting *Allow GitHub Actions to create and approve pull requests* to be enabled, so it can open the bump pull request. See ``RELEASING.md`` and the family guide https://github.com/bnelair/brainmaze-sphinx/blob/main/RELEASING.md (including what to do if a release fails part-way).

New functions need to be implemented with Sphinx compatible docstrings. The documentation is automatically generated from the docstrings using Sphinx.

Building Documentation
''''''''''''''''''''''''''''''

To build the documentation locally:

1. Install documentation dependencies:

.. code-block:: bash

    pip install -e ".[docs]"

Alternatively, you can install from the requirements file:

.. code-block:: bash

    pip install -r docs_src/requirements.txt

2. Build the documentation:

.. code-block:: bash

    sphinx-build -b html docs_src/source docs

The generated HTML documentation will be in the ``docs/`` directory. Note that the ``docs/`` directory is excluded from version control via ``.gitignore`` and should not be committed to the repository.


License
""""""""""""""""""

This software is licensed under BSD-3Clause license. For details see the `LICENSE <https://github.com/bnelair/brainmaze-eeg/blob/main/LICENSE>`_ file in the root directory of this project.


Acknowledgment
"""""""""""""""""""
This code was developed and originally published for the first time by (Mivalt 2022, and Sladky 2022). Additionally, codes related to individual projects available in this repository are stated below. When using this toolbox, we appreciate you citing the papers related to the utilized functionality. Please, see the sections below for references to individual submodules.

 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: `10.1088/1741-2552/ac4bfd <https://doi.org/10.1088/1741-2552/ac4bfd>`_.
 |
 | V. Sladky et al., “Distributed brain co-processor for tracking spikes, seizures and behaviour during electrical brain stimulation,” Brain Commun., vol. 4, no. 3, May 2022, doi: `10.1093/braincomms/fcac115 <https://doi.org/10.1093/braincomms/fcac115>`_.

Sleep classification and feature extraction
'''''''''''''''''''''''''''''''''''''''''''''''
 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: `10.1088/1741-2552/ac4bfd <https://doi.org/10.1088/1741-2552/ac4bfd>`_.
 |
 | F. Mivalt et V. Sladky et al., “Automated sleep classification with chronic neural implants in freely behaving canines,” J. Neural Eng., vol. 20, no. 4, p. 046025, Aug. 2023, doi: `10.1088/1741-2552/aced21 <https://doi.org/10.1088/1741-2552/aced21>`_.

The work was based on the following references:

 | Gerla, V., Kremen, V., Macas, M., Dudysova, D., Mladek, A., Sos, P., & Lhotska, L. (2019). Iterative expert-in-the-loop classification of sleep PSG recordings using a hierarchical clustering. Journal of Neuroscience Methods, 317(February), 61–70. https://doi.org/10.1016/j.jneumeth.2019.01.013
 |
 | Kremen, V., Brinkmann, B. H., Van Gompel, J. J., Stead, S. (Matt) M., St Louis, E. K., & Worrell, G. A. (2018). Automated Unsupervised Behavioral State Classification using Intracranial Electrophysiology. Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/aae5ab
 |
 | Kremen, V., Duque, J. J., Brinkmann, B. H., Berry, B. M., Kucewicz, M. T., Khadjevand, F., G.A. Worrell, G. A. (2017). Behavioral state classification in epileptic brain using intracranial electrophysiology. Journal of Neural Engineering, 14(2), 026001. https://doi.org/10.1088/1741-2552/aa5688

Evoked Response Potential Analysis
'''''''''''''''''''''''''''''''''''''''''''''''
 | K. J. Miller et al., “Canonical Response Parameterization: Quantifying the structure of responses to single-pulse intracranial electrical brain stimulation,” PLOS Comput. Biol., vol. 19, no. 5, p. e1011105, May 2023, doi: `10.1371/journal.pcbi.1011105 <https://doi.org/10.1371/journal.pcbi.1011105>`_.

EEG Slow Wave Detection and Analysis
'''''''''''''''''''''''''''''''''''''''''''''''
 | Carvalho DZ, Kremen V, Mivalt F, St Louis EK, McCarter SJ, Bukartyk J, Przybelski SA, Kamykowski MG, Spychalla AJ, Machulda MM, Boeve BF, Petersen RC, Jack CR Jr, Lowe VJ, Graff-Radford J, Worrell GA, Somers VK, Varga AW, Vemuri P. Non-rapid eye movement sleep slow-wave activity features are associated with amyloid accumulation in older adults with obstructive sleep apnoea. Brain Commun. 2024 Oct 7;6(5):fcae354. doi: `10.1093/braincomms/fcae354 <https://doi.org/10.1093/braincomms/fcae354>`_. PMID: 39429245; PMCID: PMC11487750.

The slow-wave detection project is documented on the `EEG Slow Wave Detection and Analysis <https://bnelair.github.io/brainmaze-eeg/project_wave_detector.html>`_ page; the runnable example is `demo/eeg_wave_detection <https://github.com/bnelair/brainmaze-eeg/tree/main/demo/eeg_wave_detection>`_. The original project readme from the BEST toolbox: `projects/slow_wave_detection/readme.rst <https://github.com/bnelair/best-toolbox/blob/master/projects/slow_wave_detection/readme.rst>`_.


Funding
""""""""""""""""""

Individual sections of this code were developed under different projects including:

- NIH Brain Initiative UH2&3 NS095495 - *Neurophysiologically-Based Brain State Tracking & Modulation in Focal Epilepsy*,
- NIH U01-NS128612 - *An Ecosystem of Technology and Protocols for Adaptive Neuromodulation Research in Humans*,
- DARPA - HR0011-20-2-0028 *Manipulating and Optimizing Brain Rhythms for Enhancement of Sleep (Morpheus)*.
- FEKT-K-22-7649 realized within the project Quality Internal Grants of the Brno University of Technology (KInG BUT), Reg. No. CZ.02.2.69/0.0/0.0/19_073/0016948, which is financed from the OP RDE.


