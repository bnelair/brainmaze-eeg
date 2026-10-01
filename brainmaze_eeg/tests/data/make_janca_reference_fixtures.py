"""
Generate ``janca_eeg_forge_reference.npz``: detection indices of the eeg_forge reference
Janca detector (``spike_detection_Janca`` by xnejed07,
https://gitlab.com/xnejed07/eeg_forge, ``eeg_forge/detection/spike_detection_janca.py``)
on the seeded synthetic signals of ``brainmaze_eeg/tests/spike_synth.py``.

Only the reference's *outputs* are stored (data); its source is not part of this repository.

Usage::

    git clone https://gitlab.com/xnejed07/eeg_forge.git /path/to/eeg_forge
    python brainmaze_eeg/tests/data/make_janca_reference_fixtures.py /path/to/eeg_forge

The reference module imports a ZMQ session wrapper that is irrelevant to the detector; the
loader below executes the module source without that import line.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from brainmaze_eeg.tests.spike_synth import synth_ieeg  # noqa: E402

# (fs, seed). 32 kHz is deliberately absent: the reference's b/a band-stop is unstable there
# (it returns NaN -> no detections); that difference is tested separately.
CASES = [(fs, seed) for fs in (200, 250, 256, 500, 512, 1000, 2000, 2048, 5000, 8000)
         for seed in (0, 1)]
DUR = 60.0


def load_reference(eeg_forge_root):
    path = Path(eeg_forge_root) / 'eeg_forge' / 'detection' / 'spike_detection_janca.py'
    src = '\n'.join(l for l in path.read_text().splitlines() if 'wrapper_zmq' not in l)
    ns = {}
    exec(compile(src, str(path), 'exec'), ns)
    return ns['spike_detection_Janca']


def main(root):
    ref = load_reference(root)
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    out = {'eeg_forge_commit': np.array(commit), 'dur': np.array(DUR)}
    for fs, seed in CASES:
        x, _ = synth_ieeg(fs, dur=DUR, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            det = np.asarray(ref(x, fs), dtype=np.int64)
        out[f'fs{fs}_seed{seed}'] = det
        print(fs, seed, det.size)
    np.savez_compressed(HERE / 'janca_eeg_forge_reference.npz', **out)
    print('eeg_forge commit', commit)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         '/home/fmivalt/Projects/brainmaze-work/scratch/eeg_forge')
