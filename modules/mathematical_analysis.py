"""Human-readable transfer-function and response interpretation."""

from __future__ import annotations

import numpy as np


def coefficient_expression(coefficients: np.ndarray, variable: str = "s") -> str:
    coefficients = np.asarray(coefficients, dtype=float)
    order = len(coefficients) - 1
    terms: list[str] = []
    for index, coefficient in enumerate(coefficients):
        power = order - index
        if abs(coefficient) < 1e-14:
            continue
        magnitude = abs(float(coefficient))
        term = f"{magnitude:g}" if power == 0 else (
            f"{magnitude:g}{variable}" if power == 1 else f"{magnitude:g}{variable}^{power}"
        )
        terms.append(("-" if coefficient < 0 else "+") + term)
    if not terms:
        return "0"
    output = " ".join(terms)
    return output[1:] if output.startswith("+") else output


def interpret_system(analysis: dict[str, object]) -> list[str]:
    details = [f"Stability: {analysis['stability']}."]
    wn, zeta = analysis["natural_frequency"], analysis["damping_ratio"]
    if wn is not None:
        details.append(f"Natural frequency ωₙ = {wn:.4g} rad/s sets the undamped oscillation scale.")
    if zeta is not None:
        kind = "underdamped" if zeta < 1 else "critically damped" if np.isclose(zeta, 1) else "overdamped"
        details.append(f"Damping ratio ζ = {zeta:.4g}: this canonical second-order model is {kind}.")
    details.extend(["Poles are the natural modes; a right-half-plane pole causes an unstable mode.",
                    "Zeros are transmission zeros where the transfer function vanishes."])
    return details
