"""NACA 4-digit airfoil geometry.

Chord = 1, leading edge at (0, 0). Nodes run counter-clockwise (x right, y up): trailing edge -> upper surface ->
leading edge -> lower surface -> trailing edge (the last node is *not* repeated).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_CODE_RE = re.compile(r"^\d{4}$")


def validate_code(code: str) -> str:
    """Return ``code`` if it is a valid NACA 4-digit code, else raise ``ValueError``."""
    if not _CODE_RE.match(code):
        raise ValueError("Enter exactly 4 digits.")
    m, p, t = int(code[0]), int(code[1]), int(code[2:])
    if t < 4:
        raise ValueError("Thickness (last two digits) must be at least 04.")
    if m > 0 and p < 1:
        raise ValueError("Camber position (2nd digit) must be 1-9 when camber is not 0.")
    return code


@dataclass(frozen=True)
class Airfoil:
    code: str
    m: float  # maximum camber (fraction of chord)
    p: float  # position of maximum camber (fraction of chord)
    t: float  # thickness (fraction of chord)
    x: np.ndarray
    y: np.ndarray

    def camber(self, x):
        """Camber line height at chordwise position(s) ``x``."""
        x = np.asarray(x, dtype=float)
        if self.m == 0:
            return np.zeros_like(x)
        m, p = self.m, self.p
        front = m / p**2 * (2 * p * x - x**2)
        back = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
        return np.where(x < p, front, back)

    def dcamber(self, x):
        """Slope of the camber line."""
        x = np.asarray(x, dtype=float)
        if self.m == 0:
            return np.zeros_like(x)
        m, p = self.m, self.p
        front = 2 * m / p**2 * (p - x)
        back = 2 * m / (1 - p) ** 2 * (p - x)
        return np.where(x < p, front, back)


def _thickness(t: float, x: np.ndarray) -> np.ndarray:
    # closed trailing edge (last coefficient -0.1036)
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)


def naca4(code: str, n_half: int = 80) -> Airfoil:
    """Build a NACA 4-digit airfoil with ``2 * n_half`` nodes (cosine spacing)."""
    validate_code(code)
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    proto = Airfoil(code, m, p, t, np.empty(0), np.empty(0))

    k = np.arange(n_half + 1)
    x = 0.5 * (1 + np.cos(np.pi * k / n_half))
    yc, phi, yt = proto.camber(x), np.arctan(proto.dcamber(x)), _thickness(t, x)
    xu, yu = x - yt * np.sin(phi), yc + yt * np.cos(phi)
    xl, yl = x + yt * np.sin(phi), yc - yt * np.cos(phi)

    lower = slice(n_half - 1, 0, -1)  # k = n_half-1 ... 1
    X = np.concatenate([xu, xl[lower]])
    Y = np.concatenate([yu, yl[lower]])
    return Airfoil(code, m, p, t, X, Y)
