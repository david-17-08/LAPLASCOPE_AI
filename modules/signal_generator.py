"""Periodic waveform generation with explicit sampling limits."""

from __future__ import annotations

import numpy as np
from scipy import signal

MAX_SAMPLES = 100_000


def generate_time(duration: float, sampling_frequency: float) -> np.ndarray:
    """Return a uniformly sampled time vector, excluding the endpoint."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Simulation duration must be a positive finite value.")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be a positive finite value.")
    count = int(np.ceil(duration * sampling_frequency))
    if count < 2:
        raise ValueError("Choose a longer duration or a higher sampling frequency.")
    if count > MAX_SAMPLES:
        raise ValueError(f"Simulation is limited to {MAX_SAMPLES:,} samples.")
    return np.arange(count, dtype=float) / sampling_frequency


def generate_signal(
    waveform: str, time: np.ndarray, amplitude: float = 1.0,
    frequency: float = 1.0, phase_degrees: float = 0.0,
    duty_cycle: float = 50.0, custom_cosine: tuple[float, ...] = (),
    custom_sine: tuple[float, ...] = (),
) -> np.ndarray:
    """Generate a periodic signal. Custom terms are Fourier harmonics."""
    values = np.asarray(time, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Time must be a non-empty, finite one-dimensional array.")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Amplitude must be finite and non-negative.")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Frequency must be a positive finite value.")
    if not np.isfinite(phase_degrees) or not np.isfinite(duty_cycle) or not 0 < duty_cycle < 100:
        raise ValueError("Phase must be finite and duty cycle between 0% and 100%.")
    if not all(np.isfinite(value) for value in (*custom_cosine, *custom_sine)):
        raise ValueError("Custom Fourier coefficients must be finite.")
    angle = 2.0 * np.pi * frequency * values + np.deg2rad(phase_degrees)
    if waveform == "Sine":
        return amplitude * np.sin(angle)
    if waveform == "Cosine":
        return amplitude * np.cos(angle)
    if waveform == "Square":
        return amplitude * signal.square(angle, duty=duty_cycle / 100.0)
    if waveform == "Triangular":
        return amplitude * signal.sawtooth(angle, width=0.5)
    if waveform == "Sawtooth":
        return amplitude * signal.sawtooth(angle)
    if waveform == "Pulse train":
        phase = np.mod(angle, 2.0 * np.pi) / (2.0 * np.pi)
        return amplitude * (phase < duty_cycle / 100.0).astype(float)
    if waveform == "Custom Fourier":
        if len(custom_cosine) > 10 or len(custom_sine) > 10:
            raise ValueError("Custom waveforms are limited to 10 harmonics.")
        output = np.zeros_like(values)
        for harmonic, coefficient in enumerate(custom_cosine, start=1):
            output += float(coefficient) * np.cos(harmonic * angle)
        for harmonic, coefficient in enumerate(custom_sine, start=1):
            output += float(coefficient) * np.sin(harmonic * angle)
        return amplitude * output
    raise ValueError(f"Unsupported waveform: {waveform}")
