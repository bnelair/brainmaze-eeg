"""matplotlib is an optional dependency (extra ``[plot]``): brainmaze_eeg.hypnogram must
import without it, and plotting must raise a clear ImportError."""
import importlib
import sys

import pytest


def test_hypnogram_imports_and_plot_errors_without_matplotlib(monkeypatch):
    # A None entry in sys.modules makes `import matplotlib...` raise ImportError.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "matplotlib.dates", None)
    monkeypatch.delitem(sys.modules, "brainmaze_eeg.hypnogram", raising=False)
    hyp = importlib.import_module("brainmaze_eeg.hypnogram")
    assert callable(hyp.score_night)
    with pytest.raises(ImportError, match=r"brainmaze-eeg\[plot\]"):
        hyp.plot_hypnogram(None)
