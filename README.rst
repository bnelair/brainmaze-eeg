
BrainMaze: Brain Electrophysiology, Behavior and Dynamics Analysis Toolbox - EEG
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This toolbox provides tools for processing of intracranial EEG recordings. See below and documentation for specific sections. This tool was separated from the BrainMaze toolbox to support a convenient and lightweight sharing of these tools across projects.

This project was originally developed as a part of the `BEhavioral STate Analysis Toolbox (BEST) <https://github.com/bnelair/best-toolbox>`_ project. However, the development has transferred to the BrainMaze project.



Documentation
"""""""""""""""

Documentation is available `here <https://bnelair.github.io/brainmaze_eeg>`_.


Installation
"""""""""""""""""""""""""""

.. code-block:: bash

    pip install brainmaze-eeg

How to contribute
"""""""""""""""""""""""""""
The project has 2 main protected branches *main* that contains official software releases and *dev* that contains the latest feature implementations shared with developers.
To implement a new feature a new branch should be created from the *dev* branch with name pattern of *developer_identifier/feature_name*.

After the feature is implemented, a pull request can be created to merge the feature branch into the *dev* branch with. Pull requests need to be reviewed by the code owners.
Drafting of new releases will be performed by the code owners in using pull request from *dev* to *main* and drafting a new release on GitHub.

New functions need to be implemented with Sphinx compatible docstrings. The documentation is automatically generated from the docstrings using Sphinx.

Building Documentation
''''''''''''''''''''''''''''''

To build the documentation locally:

1. Install documentation dependencies:

.. code-block:: bash

    pip install -r docs_src/requirements.txt

2. Build the documentation:

.. code-block:: bash

    sphinx-build -b html docs_src/source docs

The generated HTML documentation will be in the ``docs/`` directory. Note that the ``docs/`` directory is excluded from version control via ``.gitignore`` and should not be committed to the repository.


License
""""""""""""""""""

This software is licensed under BSD-3Clause license. For details see the `LICENSE <https://github.com/bnelair/brainmaze_utils/blob/master/LICENSE>`_ file in the root directory of this project.


Acknowledgment
"""""""""""""""""""
This code was developed and originally published for the first time by (Mivalt 2022, and Sladky 2022). Additionally, codes related to individual projects available in this repository are stated below. When using this toolbox, we appreciate you citing the papers related to the utilized functionality. Please, see the sections below for references to individual submodules.

 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: 10.1088/1741-2552/ac4bfd.
 |
 | V. Sladky et al., “Distributed brain co-processor for tracking spikes, seizures and behaviour during electrical brain stimulation,” Brain Commun., vol. 4, no. 3, May 2022, doi: 10.1093/braincomms/fcac115.

Sleep classification and feature extraction
'''''''''''''''''''''''''''''''''''''''''''''''
 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: 10.1088/1741-2552/ac4bfd.
 |
 | F. Mivalt et V. Sladky et al., “Automated sleep classification with chronic neural implants in freely behaving canines,” J. Neural Eng., vol. 20, no. 4, p. 046025, Aug. 2023, doi: 10.1088/1741-2552/aced21.

The work was based on the following references:

 | Gerla, V., Kremen, V., Macas, M., Dudysova, D., Mladek, A., Sos, P., & Lhotska, L. (2019). Iterative expert-in-the-loop classification of sleep PSG recordings using a hierarchical clustering. Journal of Neuroscience Methods, 317(February), 61?70. https://doi.org/10.1016/j.jneumeth.2019.01.013
 |
 | Kremen, V., Brinkmann, B. H., Van Gompel, J. J., Stead, S. (Matt) M., St Louis, E. K., & Worrell, G. A. (2018). Automated Unsupervised Behavioral State Classification using Intracranial Electrophysiology. Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/aae5ab
 |
 | Kremen, V., Duque, J. J., Brinkmann, B. H., Berry, B. M., Kucewicz, M. T., Khadjevand, F., G.A. Worrell, G. A. (2017). Behavioral state classification in epileptic brain using intracranial electrophysiology. Journal of Neural Engineering, 14(2), 026001. https://doi.org/10.1088/1741-2552/aa5688

Evoked Response Potential Analysis
'''''''''''''''''''''''''''''''''''''''''''''''
 | K. J. Miller et al., “Canonical Response Parameterization: Quantifying the structure of responses to single-pulse intracranial electrical brain stimulation,” PLOS Comput. Biol., vol. 19, no. 5, p. e1011105, May 2023, doi: 10.1371/journal.pcbi.1011105.

EEG Slow Wave Detection and Analysis
'''''''''''''''''''''''''''''''''''''''''''''''
 | Carvalho DZ, Kremen V, Mivalt F, St Louis EK, McCarter SJ, Bukartyk J, Przybelski SA, Kamykowski MG, Spychalla AJ, Machulda MM, Boeve BF, Petersen RC, Jack CR Jr, Lowe VJ, Graff-Radford J, Worrell GA, Somers VK, Varga AW, Vemuri P. Non-rapid eye movement sleep slow-wave activity features are associated with amyloid accumulation in older adults with obstructive sleep apnoea. Brain Commun. 2024 Oct 7;6(5):fcae354. doi: 10.1093/braincomms/fcae354. PMID: 39429245; PMCID: PMC11487750.

Readme to the EEG Slow Detection project available in this repository in this repository: `projects/slow_wave_detection/readme <https://github.com/bnelair/best-toolbox/blob/master/projects/slow_wave_detection/readme.rst>`_.


Funding
""""""""""""""""""

Individual sections of this code were developed under different projects including:

- NIH Brain Initiative UH2&3 NS095495 - *Neurophysiologically-Based Brain State Tracking & Modulation in Focal Epilepsy*,
- NIH U01-NS128612 - *An Ecosystem of Techmology and Protocols for Adaptive Neuromodulation Research in Humans*,
- DARPA - HR0011-20-2-0028 *Manipulating and Optimizing Brain Rhythms for Enhancement of Sleep (Morpheus)*.
- FEKT-K-22-7649 realized within the project Quality Internal Grants of the Brno University of Technology (KInG BUT), Reg. No. CZ.02.2.69/0.0/0.0/19_073/0016948, which is financed from the OP RDE.


