"""Continuous-time LTI response simulation and step-response metrics."""

from __future__ import annotations

import numpy as np
from scipy import signal

from modules.pole_zero import validate_transfer_function


def simulate_response(
    numerator: list[float], denominator: list[float], time: np.ndarray,
    input_signal: np.ndarray, input_type: str = "Step", impulse_magnitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    num, den = validate_transfer_function(numerator, denominator)
    time = np.asarray(time, dtype=float)
    values = np.asarray(input_signal, dtype=float)
    if time.ndim != 1 or values.shape != time.shape or time.size < 2:
        raise ValueError("Time and input must be same-sized one-dimensional arrays.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError("Time and input samples must be finite.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time samples must be strictly increasing.")
    if not np.isfinite(impulse_magnitude):
        raise ValueError("Impulse magnitude must be finite.")
    system = signal.TransferFunction(num, den)
    if input_type == "Impulse":
        response_time, output = signal.impulse(system, T=time)
        return np.asarray(response_time), impulse_magnitude * np.asarray(output)
    _, output, _ = signal.lsim(system, U=values, T=time)
    return time, np.asarray(output)


def impulse_response(numerator: list[float], denominator: list[float], time: np.ndarray) -> np.ndarray:
    num, den = validate_transfer_function(numerator, denominator)
    _, response = signal.impulse(signal.TransferFunction(num, den), T=time)
    return np.asarray(response)


def step_response_metrics(
    numerator: list[float], denominator: list[float], time: np.ndarray
) -> dict[str, float | None]:
    """Compute conventional step metrics; return None where undefined."""
    num, den = validate_transfer_function(numerator, denominator)
    times, response = signal.step(signal.TransferFunction(num, den), T=time)
    final, initial = float(response[-1]), float(response[0])
    amplitude = final - initial
    if not np.isfinite(final) or abs(amplitude) < 1e-12:
        return {"rise_time": None, "settling_time": None, "overshoot_percent": None,
                "steady_state_error": None, "final_value": final}
    normalized = (response - initial) / amplitude
    low, high = np.flatnonzero(normalized >= 0.1), np.flatnonzero(normalized >= 0.9)
    rise = float(times[high[0]] - times[low[0]]) if low.size and high.size else None
    unsettled = np.flatnonzero(np.abs(response - final) > 0.02 * abs(amplitude))
    settling = float(times[unsettled[-1] + 1]) if unsettled.size and unsettled[-1] + 1 < len(times) else (
        0.0 if not unsettled.size else None
    )
    peak = float(np.max(response)) if amplitude > 0 else float(np.min(response))
    overshoot = max(0.0, (peak - final) * np.sign(amplitude) / abs(amplitude) * 100.0)
    return {"rise_time": rise, "settling_time": settling, "overshoot_percent": float(overshoot),
            "steady_state_error": float(abs(1.0 - final)), "final_value": final}
