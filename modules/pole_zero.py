"""Transfer-function polynomial, pole, zero, and stability analysis."""

from __future__ import annotations

import numpy as np


def validate_transfer_function(
    numerator: list[float], denominator: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.ndim != 1 or den.ndim != 1 or num.size == 0 or den.size == 0:
        raise ValueError("Numerator and denominator must be non-empty coefficient lists.")
    if not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)):
        raise ValueError("Transfer-function coefficients must be finite.")
    num, den = np.trim_zeros(num, "f"), np.trim_zeros(den, "f")
    if den.size == 0 or den[0] == 0:
        raise ValueError("The denominator polynomial must be nonzero.")
    if num.size == 0:
        num = np.array([0.0])
    if num.size > 12 or den.size > 12:
        raise ValueError("Polynomial order is limited to 11.")
    if num.size > den.size:
        raise ValueError("The transfer function must be proper (numerator order ≤ denominator order).")
    return num, den


def analyze_poles_zeros(
    numerator: list[float], denominator: list[float]
) -> dict[str, object]:
    num, den = validate_transfer_function(numerator, denominator)
    poles = np.roots(den) if den.size > 1 else np.array([], dtype=complex)
    zeros = np.roots(num) if num.size > 1 else np.array([], dtype=complex)
    tolerance = 1e-9
    if np.any(poles.real > tolerance):
        stability = "Unstable"
    else:
        imaginary_axis = poles[np.abs(poles.real) <= tolerance]
        repeated_axis_pole = any(
            abs(imaginary_axis[left] - imaginary_axis[right]) <= tolerance
            for left in range(len(imaginary_axis))
            for right in range(left + 1, len(imaginary_axis))
        )
        if repeated_axis_pole:
            stability = "Unstable"
        elif imaginary_axis.size:
            stability = "Marginally stable"
        else:
            stability = "Asymptotically stable"
    wn = zeta = None
    if den.size == 3 and abs(den[0]) > tolerance:
        normalized = den / den[0]
        if normalized[2] > tolerance:
            wn = float(np.sqrt(normalized[2]))
            zeta = float(normalized[1] / (2.0 * wn))
    return {
        "numerator": num, "denominator": den, "poles": poles, "zeros": zeros,
        "stability": stability, "natural_frequency": wn, "damping_ratio": zeta,
    }