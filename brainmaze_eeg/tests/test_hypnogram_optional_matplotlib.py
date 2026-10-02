"""matplotlib is an optional dependency (extra ``[plot]``): brainmaze_eeg.hypnogram must
import without it, and plotting must raise a clear ImportError."""
import importlib
import sys

import pytest


def test_hypnogram_imports_and_plot_errors_without_matplotlib(monkeypatch):
    import brainmaze_eeg
    import brainmaze_eeg.hypnogram
    # The re-import below rebinds the package attribute `brainmaze_eeg.hypnogram`;
    # monkeypatch restores it (and sys.modules) to the original module afterwards.
    monkeypatch.setattr(brainmaze_eeg, "hypnogram", brainmaze_eeg.hypnogram)
    # A None entry in sys.modules makes `import matplotlib...` raise ImportError.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "matplotlib.dates", None)
    monkeypatch.delitem(sys.modules, "brainmaze_eeg.hypnogram")
    hyp = importlib.import_module("brainmaze_eeg.hypnogram")
    assert callable(hyp.score_night)
    with pytest.raises(ImportError, match=r"brainmaze-eeg\[plot\]"):
        hyp.plot_hypnogram(None)


def test_reimport_left_no_stale_module():
    # Runs after the test above (file order): the package attribute and sys.modules
    # must point at the same, original module again.
    import brainmaze_eeg
    import brainmaze_eeg.hypnogram as hyp
    assert brainmaze_eeg.hypnogram is sys.modules["brainmaze_eeg.hypnogram"] is hyp
