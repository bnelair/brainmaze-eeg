# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared filter design and verification helpers for the spike detectors.

Every filter used by a detector is designed here, in second-order-section (``sos``) form,
from parameters given in **Hz** (never normalised frequencies), and validated against the
Nyquist frequency of the rate at which it will run. The ``sos`` form is used throughout
because the transfer-function (``b, a``) form of narrow or low-cutoff IIR filters becomes
numerically unstable at high sampling rates (e.g. a 3rd-order 47.5-52.5 Hz Butterworth
band-stop has a pole outside the unit circle at 32 kHz in ``b, a`` form, and its frequency
response is already wrong by >1e-3 at 10 kHz), which silently turns the output into NaN or
garbage.

All detectors apply their filters forward-backward (:func:`scipy.signal.sosfiltfilt`), so
the *effective* magnitude response is ``|H(f)|**2`` (attenuation in dB doubles, phase is
zero). :func:`zero_phase_response_db` returns exactly that and is what the test-suite uses
to verify that every filter does what its parameters say.
"""

import numpy as np
from scipy.signal import butter, cheb2ord, cheby2, sosfreqz

__all__ = ['check_band', 'butter_bandpass', 'butter_bandstop', 'butter_highpass',
           'butter_lowpass', 'cheby2_lowpass', 'cheby2_highpass', 'iir_notch_sos',
           'zero_phase_response_db']


def check_band(band, fs, name='band'):
    """
    Validate a ``(low, high)`` band in Hz against the Nyquist frequency of ``fs``.

    Raises
    ------
    ValueError
        Unless ``0 < low < high < fs/2``.
    """
    try:
        lo, hi = (float(v) for v in band)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a pair (low_hz, high_hz), got {band!r}') from None
    nyq = fs / 2.0
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f'{name} edges must be finite, got {band!r}')
    if lo <= 0:
        raise ValueError(f'{name} low edge must be > 0 Hz, got {lo} Hz')
    if lo >= hi:
        raise ValueError(f'{name} low edge ({lo} Hz) must be < high edge ({hi} Hz)')
    if hi >= nyq:
        raise ValueError(f'{name} high edge ({hi} Hz) must be < Nyquist ({nyq} Hz) at '
                         f'fs={fs} Hz')
    return lo, hi


def _check_order(order, name='order'):
    if int(order) != order or order < 1:
        raise ValueError(f'{name} must be a positive integer, got {order!r}')
    return int(order)


def butter_bandpass(band, fs, order):
    """Butterworth band-pass, ``sos``. ``order`` is the prototype order (as scipy/MATLAB).

    Single-pass response is -3 dB at both edges; forward-backward (zero-phase) -6 dB.
    """
    lo, hi = check_band(band, fs)
    return butter(_check_order(order), [lo, hi], btype='bandpass', fs=fs, output='sos')


def butter_bandstop(band, fs, order):
    """Butterworth band-stop, ``sos``; -3 dB single-pass (-6 dB zero-phase) at the edges."""
    lo, hi = check_band(band, fs, name='stop band')
    return butter(_check_order(order), [lo, hi], btype='bandstop', fs=fs, output='sos')


def butter_highpass(cutoff, fs, order):
    """Butterworth high-pass, ``sos``; -3 dB single-pass (-6 dB zero-phase) at ``cutoff``."""
    if not (0 < cutoff < fs / 2):
        raise ValueError(f'high-pass cutoff ({cutoff} Hz) must be in (0, Nyquist={fs / 2} Hz)')
    return butter(_check_order(order), cutoff, btype='highpass', fs=fs, output='sos')


def butter_lowpass(cutoff, fs, order):
    """Butterworth low-pass, ``sos``; -3 dB single-pass (-6 dB zero-phase) at ``cutoff``."""
    if not (0 < cutoff < fs / 2):
        raise ValueError(f'low-pass cutoff ({cutoff} Hz) must be in (0, Nyquist={fs / 2} Hz)')
    return butter(_check_order(order), cutoff, btype='lowpass', fs=fs, output='sos')


def cheby2_lowpass(passband_edge, stopband_edge, fs, rp, rs):
    """
    Minimum-order Chebyshev-II low-pass meeting: at most ``rp`` dB loss at
    ``passband_edge`` and at least ``rs`` dB attenuation from ``stopband_edge`` (single pass).

    The order and the **stop-band** edge passed to :func:`scipy.signal.cheby2` are the ones
    returned by :func:`scipy.signal.cheb2ord` (``cheby2``'s ``Wn`` is the stop-band edge, not
    the pass-band edge -- passing the pass-band edge shifts the whole filter inwards).
    """
    nyq = fs / 2.0
    if not (0 < passband_edge < stopband_edge < nyq):
        raise ValueError(f'Chebyshev-II low-pass needs 0 < passband edge ({passband_edge} Hz) '
                         f'< stopband edge ({stopband_edge} Hz) < Nyquist ({nyq} Hz)')
    n, wn = cheb2ord(passband_edge, stopband_edge, rp, rs, fs=fs)
    return cheby2(n, rs, wn, btype='lowpass', fs=fs, output='sos')


def cheby2_highpass(passband_edge, stopband_edge, fs, rp, rs):
    """Chebyshev-II high-pass counterpart of :func:`cheby2_lowpass` (stopband < passband)."""
    nyq = fs / 2.0
    if not (0 < stopband_edge < passband_edge < nyq):
        raise ValueError(f'Chebyshev-II high-pass needs 0 < stopband edge ({stopband_edge} Hz) '
                         f'< passband edge ({passband_edge} Hz) < Nyquist ({nyq} Hz)')
    n, wn = cheb2ord(passband_edge, stopband_edge, rp, rs, fs=fs)
    return cheby2(n, rs, wn, btype='highpass', fs=fs, output='sos')


def iir_notch_sos(freq, fs, pole_radius):
    """
    Second-order notch with zeros on the unit circle at ``freq`` and poles at radius
    ``pole_radius`` (same angle), as one ``sos`` row. The -3 dB width (single pass) is about
    ``(1 - pole_radius) * fs / pi`` Hz.
    """
    if not (0 < freq < fs / 2):
        raise ValueError(f'notch frequency ({freq} Hz) must be in (0, Nyquist={fs / 2} Hz)')
    if not (0 < pole_radius < 1):
        raise ValueError(f'pole_radius must be in (0, 1), got {pole_radius}')
    w = 2 * np.pi * freq / fs
    b = [1.0, -2 * np.cos(w), 1.0]
    a = [1.0, -2 * pole_radius * np.cos(w), pole_radius ** 2]
    # normalise to unit gain at DC so the pass band is 0 dB
    g = (a[0] + a[1] + a[2]) / (b[0] + b[1] + b[2])
    return np.array([[g * b[0], g * b[1], g * b[2], a[0], a[1], a[2]]])


def zero_phase_response_db(sos, freqs, fs):
    """
    Effective magnitude response (dB) of ``sos`` applied forward-backward
    (:func:`scipy.signal.sosfiltfilt`), i.e. ``20*log10(|H(f)|**2)``, at ``freqs`` (Hz).

    Several cascaded filters can be passed as a list of ``sos`` arrays; their responses
    multiply.
    """
    if isinstance(sos, (list, tuple)):
        sos = np.vstack(sos)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    _, h = sosfreqz(sos, worN=freqs, fs=fs)
    with np.errstate(divide='ignore'):
        return 20 * np.log10(np.abs(h) ** 2)
