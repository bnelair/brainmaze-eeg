"""Static checks (``ast``) of the package sources for numpy APIs that break on a supported
numpy version (``numpy>=1.24``, including 2.x).

* ``np.typing`` used at runtime without a module-level ``import numpy.typing`` (or
  ``from numpy.typing import ...``) that is not under ``if TYPE_CHECKING:``. numpy 1.x does
  not load the submodule on attribute access, so it only works if some other library
  imported it first (scipy 1.10 does not; this failed the lowest-bounds compatibility job).
  Annotations are ignored when the module has ``from __future__ import annotations``.
* Attributes removed in numpy 1.24 (``np.float``, ``np.int``, ...) or in numpy 2.x
  (``np.NaN``, ``np.float_``, ``np.product``, ``np.cast``, ...), accessed as
  ``<numpy alias>.<name>`` or imported with ``from numpy import <name>``.

Only code is inspected (comments, docstrings and other strings are not). The checks are a
cheap tripwire; the lowest-bounds and newest-stack test runs remain the real guard.

Known limits (none of these patterns occurs in the package; review of eeg#73, V9):

* Not detected (false negatives): ``np.typing`` used at module level *before* the import
  statement; an import under ``if False:`` (counted as a runtime import); numpy reached
  through an alias made by assignment (``xp = np``) or through ``getattr(np, 'typing')``;
  ``from numpy import *``; removed *methods* (``ndarray.ptp``, ``ndarray.newbyteorder``),
  because only module attributes are checked.
* Reported although harmless (false positives): an ``import numpy.typing`` inside a
  ``for``/``while``/``match`` block or a class body at module level is not recognised as
  a module-level import.
"""
import ast
from pathlib import Path

import pytest

import brainmaze_eeg

PKG = Path(brainmaze_eeg.__file__).parent
SOURCES = sorted(p for p in PKG.rglob('*.py') if 'tests' not in p.relative_to(PKG).parts)

# Missing in numpy 1.24.0 or in numpy 2.5.3 (verified with hasattr on both; float96 left
# out because it is platform dependent).
REMOVED = frozenset('''
    float int bool object str long unicode complex alen asscalar rank loads mafromtxt ndfromtxt
    NaN NAN Inf Infinity PINF NINF NZERO PZERO infty float_ complex_ string_ unicode_ cfloat
    clongfloat longfloat singlecomplex longcomplex round_ product cumproduct sometrue alltrue
    asfarray find_common_type issubclass_ msort nbytes cast source lookfor who safe_eval
    set_string_function deprecate deprecate_with_doc disp fastCopyAndTranspose add_newdoc_ufunc
    compat format_parser recfromcsv recfromtxt mat row_stack trapz in1d geterrobj seterrobj
    byte_bounds maximum_sctype obj2sctype sctype2char sctypes issctype int0 uint0 bool8 object0
    str0 bytes0 void0 tracemalloc_domain DataSource Tester chararray
'''.split())


def _is_type_checking(test):
    return (isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING') or \
           (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING')


def _runtime_module_statements(body):
    """Module-level statements that run on import: recurse into if/try/with blocks, but
    not into ``if TYPE_CHECKING:`` bodies, functions or classes."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                yield from _runtime_module_statements(node.body)
            yield from _runtime_module_statements(node.orelse)
        elif isinstance(node, ast.Try):
            for block in [node.body, node.orelse, node.finalbody] + [h.body for h in node.handlers]:
                yield from _runtime_module_statements(block)
        elif isinstance(node, ast.With):
            yield from _runtime_module_statements(node.body)


def _numpy_aliases(tree):
    """Names bound to the numpy package anywhere in the module (``import numpy [as x]``)."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == 'numpy':
                    aliases.add(a.asname or 'numpy')
                elif a.name.startswith('numpy.') and a.asname is None:
                    aliases.add('numpy')  # `import numpy.x` binds `numpy`
    return aliases


def _imports_numpy_typing_at_runtime(tree):
    for node in _runtime_module_statements(tree.body):
        if isinstance(node, ast.Import) and any(
                a.name == 'numpy.typing' or a.name.startswith('numpy.typing.') for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (
                node.module == 'numpy.typing' or (node.module or '').startswith('numpy.typing.') or
                (node.module == 'numpy' and any(a.name == 'typing' for a in node.names))):
            return True
    return False


def _annotation_node_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        anns = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            anns.append(node.returns)
        if isinstance(node, ast.arg):
            anns.append(node.annotation)
        if isinstance(node, ast.AnnAssign):
            anns.append(node.annotation)
        for a in anns:
            if a is not None:
                ids.update(id(n) for n in ast.walk(a))
    return ids


def _future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == '__future__' and
               any(a.name == 'annotations' for a in n.names) for n in tree.body)


def np_typing_problems(source):
    """Lines where ``<numpy alias>.typing`` is used at runtime without a runtime import of
    numpy.typing."""
    tree = ast.parse(source)
    aliases = _numpy_aliases(tree)
    skip = _annotation_node_ids(tree) if _future_annotations(tree) else set()
    uses = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == 'typing' and isinstance(n.value, ast.Name)
            and n.value.id in aliases and id(n) not in skip]
    if uses and not _imports_numpy_typing_at_runtime(tree):
        return uses
    return []


def removed_numpy_names(source):
    """``(line, name)`` of removed numpy attributes used in the code."""
    tree = ast.parse(source)
    aliases = _numpy_aliases(tree)
    hits = [(n.lineno, n.attr) for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in aliases and n.attr in REMOVED]
    hits += [(n.lineno, a.name) for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and n.module == 'numpy' and n.level == 0
             for a in n.names if a.name in REMOVED]
    return sorted(hits)


# ---------------------------------------------------------------- the checker itself

@pytest.mark.parametrize('source, problem', [
    ('import numpy as np\ndef f(x: np.typing.NDArray): pass\n', True),
    ('import numpy as np\nimport numpy.typing as npt\ndef f(x: np.typing.NDArray): pass\n', False),
    ('import numpy as np\nimport numpy.typing\ndef f(x: np.typing.NDArray): pass\n', False),
    ('import numpy as np, numpy.typing\ndef f(x: np.typing.NDArray): pass\n', False),
    ('import numpy as np\nfrom numpy.typing import NDArray\ndef f(x: np.typing.NDArray): pass\n', False),
    ('import numpy as np\nfrom numpy import typing\ndef f(x: np.typing.NDArray): pass\n', False),
    ('import numpy as np\ndef f(x):\n    """x : np.typing.NDArray"""\n', False),
    ('import numpy as np\nx = "np.typing"  # np.typing\n', False),
    ('from __future__ import annotations\nimport numpy as np\ndef f(x: np.typing.NDArray): pass\n', False),
    ('from __future__ import annotations\nimport numpy as np\nT = np.typing.NDArray\n', True),
    ('import numpy as np\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy.typing\n'
     'def f(x: np.typing.NDArray): pass\n', True),
    ('import numpy as np\nimport typing\nif typing.TYPE_CHECKING:\n    import numpy.typing\n'
     'def f(x: np.typing.NDArray): pass\n', True),
    ('import numpy as np\ndef g():\n    import numpy.typing\ndef f(x: np.typing.NDArray): pass\n', True),
    ('import numpy\ndef f(x: numpy.typing.NDArray): pass\n', True),
    ('import numpy as onp\ndef f(x: onp.typing.NDArray): pass\n', True),
    ('import numpy as np\n"""\nimport numpy.typing\n"""\ndef f(x: np.typing.NDArray): pass\n', True),
    ('import numpy as np\ntry:\n    import numpy.typing\nexcept ImportError:\n    pass\n'
     'def f(x: np.typing.NDArray): pass\n', False),
])
def test_np_typing_checker(source, problem):
    assert bool(np_typing_problems(source)) is problem


@pytest.mark.parametrize('source, names', [
    ('import numpy as np\nx = np.float', ['float']),
    ('import numpy as np\nx = np.float64 + np.int_(1) + np.bool_(1)', []),
    ('import numpy as np\nx = [np.NaN, np.in1d, np.object, np.bool8, np.msort, np.alltrue, np.cast, np.Inf]',
     ['Inf', 'NaN', 'alltrue', 'bool8', 'cast', 'in1d', 'msort', 'object']),
    ('import numpy as np\nx = "np.float in a string"  # np.float\n', []),
    ('import numpy\nx = numpy.product', ['product']),
    ('import numpy as onp\nx = onp.float_', ['float_']),
    ('from numpy import float_, sometrue, float64', ['float_', 'sometrue']),
    ('import numpy as np\nnp.random.float = 1\nx = np.random.float', []),
])
def test_removed_names_checker(source, names):
    assert sorted(n for _, n in removed_numpy_names(source)) == names


# ---------------------------------------------------------------- the package sources

def test_sources_found():
    assert any(p.name == 'preprocessing.py' for p in SOURCES)


@pytest.mark.parametrize('path', SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_np_typing_is_imported_explicitly(path):
    lines = np_typing_problems(path.read_text())
    assert not lines, f'{path.name} uses np.typing (lines {lines}) without a runtime `import numpy.typing`'


@pytest.mark.parametrize('path', SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_removed_numpy_aliases(path):
    hits = removed_numpy_names(path.read_text())
    assert not hits, f'{path.name}: removed numpy attributes {hits}'
