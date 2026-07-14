# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Time-domain feature extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Windowed time-domain features for EEG/iEEG. Mirrors the call contract of
:class:`brainmaze_eeg.features.feature_extraction.SleepSpectralFeatureExtractor` --
``extractor(x) -> (values, names)`` -- so time-domain and spectral features can be
concatenated for the same epochs.

Features
--------
``LINE_LENGTH``
    Sum of absolute sample-to-sample differences within a window. Scales with window
    length; divide by ``DATA_RATE * segm_size * fs`` to obtain a per-sample rate.
``TKEO_MEAN``
    Mean Teager-Kaiser energy within a window.

Example
-------
.. code-block:: python

    import numpy as np
    from brainmaze_eeg.features.time_domain_features import TimeDomainFeatureExtractor

    fs = 200
    x = np.random.randn(2, 60 * fs)          # (n_channels, n_samples)

    extractor = TimeDomainFeatureExtractor(fs=fs, segm_size=30, datarate=True)
    values, names = extractor(x)             # values: list of (n_channels, n_windows)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ['tkeo', 'line_length', 'TimeDomainFeatureExtractor']


def tkeo(x: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""
    Teager-Kaiser Energy Operator.

    .. math::
        \Psi[n] = x[n]^2 - x[n-1] \cdot x[n+1]

    Tracks the instantaneous energy of a quasi-sinusoidal signal, being jointly
    proportional to the square of its amplitude and frequency. Sensitive to sharp
    transients, which makes it useful for spike and artefact detection.

    Parameters
    ----------
    x : np.ndarray
        Input signal. May be of any dimensionality.
    axis : int
        Axis along which the operator is applied. Default is the last axis.

    Returns
    -------
    np.ndarray
        Same shape as ``x``. The first and last samples along ``axis`` are undefined
        and set to NaN, so the result stays sample-aligned with the input.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] < 3:
        return np.full(x.shape, np.nan)

    x = np.moveaxis(x, axis, -1)
    psi = np.full(x.shape, np.nan)
    psi[..., 1:-1] = _tkeo_core(x)
    return np.moveaxis(psi, -1, axis)


def _tkeo_core(x: np.ndarray) -> np.ndarray:
    """TKEO over the last axis, unpadded: length ``n-2``; ``out[i]`` is centred on ``x[i+1]``."""
    return x[..., 1:-1] ** 2 - x[..., 2:] * x[..., :-2]


def _line_length_core(x: np.ndarray) -> np.ndarray:
    """Absolute increments over the last axis, unpadded: length ``n-1``; ``out[i] = |x[i+1]-x[i]|``."""
    return np.abs(np.diff(x, axis=-1))


def line_length(x: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""
    Per-sample line-length increments.

    .. math::
        L[n] = |x[n] - x[n-1]|

    Summing these within a window gives the classic line-length feature, a cheap proxy
    for signal complexity and amplitude that is widely used for seizure onset detection.

    Parameters
    ----------
    x : np.ndarray
        Input signal. May be of any dimensionality.
    axis : int
        Axis along which the increments are computed. Default is the last axis.

    Returns
    -------
    np.ndarray
        Same shape as ``x``. The first sample along ``axis`` is undefined and set to
        NaN, so the result stays sample-aligned with the input.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] < 2:
        return np.full(x.shape, np.nan)

    x = np.moveaxis(x, axis, -1)
    ll = np.full(x.shape, np.nan)
    ll[..., 1:] = _line_length_core(x)
    return np.moveaxis(ll, -1, axis)


def _window_view(a: np.ndarray, win_len: int, n_shift: int, n_windows: int) -> np.ndarray:
    """
    Window the last axis of ``a`` into ``(..., n_windows, win_len)``.

    A strided view -- no copy. ``a`` is a per-sample operator array (which is shorter than
    the signal: ``len(x)-1`` for line length, ``len(x)-2`` for TKEO), and ``win_len`` is
    correspondingly shorter than the window, so that window ``k`` maps onto exactly the
    operator samples that lie inside signal window ``[k*n_shift, k*n_shift + n_segm)``.
    """
    if n_windows <= 0 or win_len <= 0:
        return a[..., :0].reshape(a.shape[:-1] + (0, max(win_len, 0)))
    return sliding_window_view(a, win_len, axis=-1)[..., ::n_shift, :][..., :n_windows, :]


class TimeDomainFeatureExtractor:
    """
    Windowed time-domain feature extractor.

    Each feature is computed from the samples **inside** its window: line length sums the
    ``n-1`` increments between consecutive in-window samples, and TKEO averages the ``n-2``
    values it can define without reaching outside. Values are therefore identical to a
    straightforward per-window computation -- this class does not change the definitions,
    only how fast they are evaluated.

    The speed comes from evaluating each per-sample operator **once** across the whole
    signal and then aggregating it with a strided window view, so there is no Python-level
    loop over windows or channels and no copy of the signal. Cost is O(n_samples) per
    channel. Clean (NaN-free) input takes a branch with no masking and no nan-aware
    reductions, which is roughly 2x faster again.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    segm_size : float
        Window length in seconds.
    overlap : float
        Window overlap in seconds. Must be in ``[0, segm_size)``. Default 0.0.
    features : tuple of str
        Which features to compute. Any of ``'LINE_LENGTH'``, ``'TKEO_MEAN'``.
    datarate : bool
        If True, prepend a ``DATA_RATE`` feature: the fraction of non-NaN samples in each
        window. Default False.

    Notes
    -----
    NaNs propagate as missing data, not as zeros: they are excluded from each window's
    aggregate rather than being counted as flat signal. A window that is entirely NaN
    yields NaN. Pair with ``datarate=True`` to know how much of each window was real.
    """

    __version__ = '1.0.0'

    AVAILABLE_FEATURES = ('LINE_LENGTH', 'TKEO_MEAN')

    def __init__(self,
                 fs: float,
                 segm_size: float,
                 overlap: float = 0.0,
                 features: tuple = ('LINE_LENGTH', 'TKEO_MEAN'),
                 datarate: bool = False):

        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError(f'fs must be a positive number. Got: {fs}')
        if not isinstance(segm_size, (int, float)) or segm_size <= 0 or not np.isfinite(segm_size):
            raise ValueError(f'segm_size must be a positive finite number of seconds. Got: {segm_size}')
        if not isinstance(overlap, (int, float)) or overlap < 0 or overlap >= segm_size:
            raise ValueError(f'overlap must be in [0, segm_size). Got: {overlap} with segm_size={segm_size}')

        unknown = [f for f in features if f not in self.AVAILABLE_FEATURES]
        if unknown:
            raise ValueError(f'Unknown feature(s) {unknown}. Available: {list(self.AVAILABLE_FEATURES)}')
        if not features:
            raise ValueError('At least one feature must be requested.')

        self.fs = fs
        self.segm_size = segm_size
        self.overlap = overlap
        self.features = tuple(features)
        self.datarate = datarate

        self._n_segm = int(round(fs * segm_size))
        self._n_shift = int(round(fs * (segm_size - overlap)))
        if self._n_segm < 3:
            raise ValueError(
                f'A window of {segm_size} s at fs={fs} Hz is {self._n_segm} samples; '
                f'TKEO needs at least 3.'
            )
        if self._n_shift < 1:
            raise ValueError(
                f'The window step (segm_size - overlap = {segm_size - overlap} s at '
                f'fs={fs} Hz) rounds to {self._n_shift} samples; it must be at least 1. '
                f'Reduce the overlap or increase the window size.'
            )

    def __call__(self, x):
        """
        Extract features.

        Parameters
        ----------
        x : np.ndarray
            1-D ``(n_samples,)`` or 2-D ``(n_channels, n_samples)``.

        Returns
        -------
        values : list of np.ndarray
            One array per feature. Shape ``(n_windows,)`` for 1-D input,
            ``(n_channels, n_windows)`` for 2-D input.
        names : list of str
            Feature names, aligned with ``values``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim not in (1, 2):
            raise ValueError(f"Input 'x' must be 1-D or 2-D. Got {x.ndim}-D.")

        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        n, sh = self._n_segm, self._n_shift
        n_samples = x.shape[-1]
        n_windows = max(0, (n_samples - n) // sh + 1) if n_samples >= n else 0

        # One NaN pass for the whole signal. Clean data -- the overwhelmingly common case --
        # then takes a branch with no masking, no nan-aware reductions and no temporaries.
        nan_mask = np.isnan(x)
        has_nan = bool(nan_mask.any())

        values, names = [], []

        if self.datarate:
            if has_nan:
                valid = _window_view((~nan_mask).astype(np.float64), n, sh, n_windows)
                values.append(valid.sum(axis=-1) / n)
            else:
                values.append(np.ones((x.shape[0], n_windows)))
            names.append('DATA_RATE')

        if 'LINE_LENGTH' in self.features:
            # increments inside window k are d[k*sh : k*sh + n-1]
            dw = _window_view(_line_length_core(x), n - 1, sh, n_windows)
            if has_nan:
                # a window with no valid increment (all-NaN, or only isolated samples)
                # is undefined -> NaN, not 0, so it is not read as a flat window
                ll = np.nansum(dw, axis=-1)
                ll[np.isfinite(dw).sum(axis=-1) == 0] = np.nan
                values.append(ll)
            else:
                values.append(dw.sum(axis=-1))
            names.append('LINE_LENGTH')

        if 'TKEO_MEAN' in self.features:
            # psi[i] is centred on signal sample i+1, so window k spans psi[k*sh : k*sh + n-2]
            pw = _window_view(_tkeo_core(x), n - 2, sh, n_windows)
            values.append(_nanmean(pw) if has_nan else pw.mean(axis=-1))
            names.append('TKEO_MEAN')

        if squeeze:
            values = [v[0] for v in values]

        return values, names


def _nanmean(a: np.ndarray) -> np.ndarray:
    """nanmean over the last axis, returning NaN -- not a warning -- for all-NaN windows."""
    mask = ~np.isnan(a)
    count = mask.sum(axis=-1)
    total = np.where(mask, a, 0.0).sum(axis=-1)
    out = np.full(count.shape, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out
