"""Static checks of the package sources for numpy APIs that break on some supported version.

* ``np.typing`` without ``import numpy.typing``: numpy 1.x does not load the submodule on
  attribute access, so it only works if some other library imported it first (scipy 1.10
  does not; this failed the lowest-bounds compatibility job).
* Aliases removed in numpy 2.0 / 1.24 (``np.float``, ``np.NaN``, ...).
"""
import re
from pathlib import Path

import pytest

import brainmaze_eeg

PKG = Path(brainmaze_eeg.__file__).parent
SOURCES = sorted(p for p in PKG.rglob('*.py') if 'tests' not in p.relative_to(PKG).parts)

REMOVED = re.compile(
    r'\bnp\.(?:float|int|bool|object|str|long|unicode|complex)\b(?!\d|_)'
    r'|\bnp\.(?:NaN|NAN|Inf|Infinity|PINF|NINF|infty|float_|complex_|string_|unicode_|'
    r'round_|product|cumproduct|sometrue|alltrue|asfarray|row_stack|in1d|trapz)\b'
)


def _code(path):
    """Source without comments and docstrings-ish noise (strip ``#`` comments)."""
    return '\n'.join(line.split('#', 1)[0] for line in path.read_text().splitlines())


def test_sources_found():
    assert any(p.name == 'preprocessing.py' for p in SOURCES)


@pytest.mark.parametrize('path', SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_np_typing_is_imported_explicitly(path):
    code = _code(path)
    if re.search(r'\bnp\.typing\b', code):
        assert re.search(r'^\s*import numpy\.typing\b', code, re.M), \
            f'{path.name} uses np.typing without `import numpy.typing`'


@pytest.mark.parametrize('path', SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_removed_numpy_aliases(path):
    hits = [m.group(0) for m in REMOVED.finditer(_code(path))]
    assert not hits, f'{path.name}: removed numpy aliases {sorted(set(hits))}'
