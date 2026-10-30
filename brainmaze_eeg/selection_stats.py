# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Statistics of a selection
^^^^^^^^^^^^^^^^^^^^^^^^^

What a review viewer shows for a dragged selection of traces: per channel minimum,
maximum, peak-to-peak, mean, standard deviation and RMS (missing samples ignored), and
the amplitude and time differences between the selection's two ends.
"""

import warnings
from typing import Dict

import numpy as np

__all__ = ["selection_stats", "selection_delta"]


def selection_stats(x) -> Dict[str, np.ndarray]:
    """Per-series statistics over the last axis, NaN ignored.

    Keys: ``min``, ``max``, ``p2p``, ``mean``, ``sd`` (sample SD, ``ddof=1``), ``rms``,
    ``n`` (finite samples). A series with no finite sample is NaN throughout (``n`` 0).
    """
    x = np.where(np.isfinite(x), np.asarray(x, dtype=float), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # an all-NaN series -> NaN
        lo, hi = np.nanmin(x, axis=-1), np.nanmax(x, axis=-1)
        return {"min": lo, "max": hi, "p2p": hi - lo, "mean": np.nanmean(x, axis=-1),
                "sd": np.nanstd(x, axis=-1, ddof=1), "rms": np.sqrt(np.nanmean(x ** 2, axis=-1)),
                "n": np.isfinite(x).sum(axis=-1)}


def selection_delta(x, fs: float) -> Dict[str, np.ndarray]:
    """Between the selection's first and last sample: ``dA`` (last - first),
    ``dt`` seconds and ``inv_dt`` (1/dt, Hz) -- the viewer's ΔA, Δt and 1/Δt."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] < 2:
        raise ValueError("a selection needs at least two samples")
    dt = (x.shape[-1] - 1) / fs
    return {"dA": x[..., -1] - x[..., 0], "dt": dt, "inv_dt": 1.0 / dt}
