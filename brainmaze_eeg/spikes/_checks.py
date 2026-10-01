# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Parameter validation shared by the spike detectors.

Every numeric parameter of every detector goes through :func:`number` (or :func:`integer`,
:func:`pair`) at construction / call time, so that NaN, inf, wrong types and out-of-range
values raise ``ValueError``/``TypeError`` with the parameter's name instead of silently
producing zero or nonsense detections.
"""

import numbers

import numpy as np

__all__ = ['number', 'integer', 'pair', 'choice']


def number(name, value, *, gt=None, ge=None, lt=None, le=None, allow_none=False,
           allow_inf=False):
    """
    Validate a real scalar and return it as ``float``.

    ``bool`` is rejected (``True`` is not a frequency). NaN is always rejected; +/-inf only
    unless ``allow_inf``. Bounds are checked as given (``gt``: ``>``, ``ge``: ``>=`` ...).
    ``None`` is returned unchanged when ``allow_none``.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f'{name} must be a number, got None')
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a real number, got {value!r} ({type(value).__name__})')
    v = float(value)
    if np.isnan(v):
        raise ValueError(f'{name} must not be NaN')
    if np.isinf(v) and not allow_inf:
        raise ValueError(f'{name} must be finite, got {v}')
    if gt is not None and not v > gt:
        raise ValueError(f'{name} must be > {gt}, got {v}')
    if ge is not None and not v >= ge:
        raise ValueError(f'{name} must be >= {ge}, got {v}')
    if lt is not None and not v < lt:
        raise ValueError(f'{name} must be < {lt}, got {v}')
    if le is not None and not v <= le:
        raise ValueError(f'{name} must be <= {le}, got {v}')
    return v


def integer(name, value, *, ge=None, le=None):
    """Validate an integer-valued scalar (``3`` or ``3.0``, not ``3.5``/``True``); return ``int``."""
    v = number(name, value, ge=ge, le=le)
    if v != int(v):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    return int(v)


def pair(name, value, *, gt=None):
    """Validate a ``(low, high)`` pair of finite numbers with ``low < high``; return floats."""
    try:
        lo, hi = value
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a pair (low, high), got {value!r}') from None
    lo = number(f'{name}[0]', lo, gt=gt)
    hi = number(f'{name}[1]', hi, gt=gt)
    if not lo < hi:
        raise ValueError(f'{name} must have low < high, got {value!r}')
    return lo, hi


def choice(name, value, options):
    """Validate that ``value`` is one of ``options``."""
    if not isinstance(value, (str, int, float, type(None))) or value not in options:
        raise ValueError(f'{name} must be one of {options}, got {value!r}')
    return value
