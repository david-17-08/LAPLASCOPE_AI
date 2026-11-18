"""Continuous-time frequency-response utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal

from modules.pole_zero import validate_transfer_function


def frequency_response(
    numerator: list[float], denominator: list[float], points: int = 600
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num, den = validate_transfer_function(numerator, denominator)
    if points < 32 or points > 5000:
        raise ValueError("Frequency grid must contain between 32 and 5000 points.")
    poles = np.roots(den) if len(den) > 1 else np.array([])
    scale = max([abs(p) for p in poles] + [1.0])
    omega = np.logspace(np.log10(scale / 1000), np.log10(scale * 1000), points)
    _, response = signal.freqresp(signal.TransferFunction(num, den), w=omega)
    return omega, np.abs(response), np.unwrap(np.angle(response)) * 180.0 / np.pi