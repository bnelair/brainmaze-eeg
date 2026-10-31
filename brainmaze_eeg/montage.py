# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Montages
^^^^^^^^

A montage is a list of :class:`Derivation` -- a name, the channels averaged with a plus
sign and the channels averaged with a minus sign -- so a viewer can draw it and an
analysis compute from exactly the same definition. Builders:

* :func:`referential`, :func:`bipolar` (adjacent contacts of each lead),
  :func:`common_average`, :func:`double_banana` (scalp 10-20, old or new temporal names);
* :func:`region_average` (each contact minus the mean of its region) and
  :func:`nearest_neighbour` (minus the mean of the nearest contacts in space), from
  electrode positions or labels.

:func:`apply_montage` turns ``(n_channels, n_samples)`` data into the derivations; a NaN
in any channel a derivation uses makes that sample NaN.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

__all__ = ["Derivation", "referential", "bipolar", "common_average", "double_banana", "region_average",
           "nearest_neighbour", "apply_montage"]


@dataclass(frozen=True)
class Derivation:
    """``mean(plus) - mean(minus)``; ``minus`` empty = referential."""
    name: str
    plus: Tuple[str, ...]
    minus: Tuple[str, ...] = ()


_CONTACT = re.compile(r"^(.*?)(\d+)$")


def referential(channels: Sequence[str]) -> List[Derivation]:
    return [Derivation(c, (c,)) for c in channels]


def bipolar(channels: Sequence[str]) -> List[Derivation]:
    """Each contact minus the next contact of the same lead: ``LA1-LA2``, ``LA2-LA3``...

    A lead is the name without its trailing number. Only consecutive numbers are paired,
    so a missing (removed) contact is never bridged; channels without a trailing number
    are left out.
    """
    leads: Dict[str, Dict[int, str]] = {}
    for c in channels:
        m = _CONTACT.match(c)
        if m:
            leads.setdefault(m.group(1), {})[int(m.group(2))] = c
    out = []
    for lead in sorted(leads, key=lambda name: min(channels.index(c) for c in leads[name].values())):
        contacts = leads[lead]
        for i in sorted(contacts):
            if i + 1 in contacts:
                out.append(Derivation(f"{contacts[i]}-{contacts[i + 1]}", (contacts[i],), (contacts[i + 1],)))
    return out


def common_average(channels: Sequence[str], exclude: Sequence[str] = ()) -> List[Derivation]:
    """Each channel minus the mean of all channels, itself included (the standard common
    average reference). Channels in ``exclude`` (an ECG, a bad contact) are neither in
    the average nor re-referenced."""
    ref = tuple(c for c in channels if c not in set(exclude))
    if not ref:
        raise ValueError("no channel left for the average")
    return [Derivation(f"{c}-avg", (c,), ref) for c in ref]


# The longitudinal bipolar ("double banana") chains; T7/T8/P7/P8 are the new names of
# T3/T4/T5/T6.
_BANANA = [("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
           ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
           ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
           ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
           ("Fz", "Cz"), ("Cz", "Pz")]
_OLD_NAMES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def double_banana(channels: Sequence[str]) -> List[Derivation]:
    """The scalp longitudinal bipolar montage, from the 10-20 channels present (names
    matched ignoring case; old and new temporal names both accepted)."""
    by_name = {c.lower(): c for c in channels}

    def find(name):
        return by_name.get(name.lower()) or by_name.get(_OLD_NAMES.get(name, "").lower())

    out = []
    for a, b in _BANANA:
        ca, cb = find(a), find(b)
        if ca and cb:
            out.append(Derivation(f"{ca}-{cb}", (ca,), (cb,)))
    return out


def region_average(channels: Sequence[str], region_of: Mapping[str, str]) -> List[Derivation]:
    """Each contact minus the mean of the OTHER contacts in its region (``region_of``:
    channel -> region label, e.g. an atlas label), like :func:`nearest_neighbour` -- so a
    two-contact region gives the plain difference, not half of it. Contacts without a
    region, or alone in theirs, are left out."""
    regions: Dict[str, List[str]] = {}
    for c in channels:
        if region_of.get(c):
            regions.setdefault(region_of[c], []).append(c)
    return [Derivation(f"{c}-{region_of[c]}", (c,), tuple(o for o in regions[region_of[c]] if o != c))
            for c in channels if region_of.get(c) and len(regions[region_of[c]]) > 1]


def nearest_neighbour(channels: Sequence[str], positions: Mapping[str, Sequence[float]], k: int = 1) -> List[Derivation]:
    """Each contact minus the mean of its ``k`` nearest other contacts (``positions``:
    channel -> (x, y, z), e.g. millimetres). Contacts without a position are left out."""
    placed = [c for c in channels if positions.get(c) is not None]
    if len(placed) <= k:
        raise ValueError("fewer positioned contacts than neighbours asked for")
    xyz = np.array([positions[c] for c in placed], dtype=float)
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    out = []
    for i, c in enumerate(placed):
        near = tuple(placed[j] for j in np.argsort(dist[i])[:k])
        out.append(Derivation(f"{c}-nn{k}", (c,), near))
    return out


def apply_montage(data, channels: Sequence[str], derivations: Sequence[Derivation]) -> np.ndarray:
    """``(n_derivations, n_samples)`` from ``data`` ``(n_channels, n_samples)``."""
    data = np.asarray(data, dtype=float)
    index = {c: i for i, c in enumerate(channels)}
    missing = {c for d in derivations for c in d.plus + d.minus if c not in index}
    if missing:
        raise ValueError(f"channels not in the data: {sorted(missing)}")
    out = np.empty((len(derivations), data.shape[-1]))
    means: Dict[Tuple[str, ...], np.ndarray] = {}            # a common reference is averaged once

    def mean_of(group):
        if group not in means:
            means[group] = data[index[group[0]]] if len(group) == 1 else data[[index[c] for c in group]].mean(axis=0)
        return means[group]

    for i, d in enumerate(derivations):
        out[i] = mean_of(d.plus)
        if d.minus:
            out[i] -= mean_of(d.minus)
    return out
