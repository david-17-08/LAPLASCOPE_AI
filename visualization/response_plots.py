"""Pole-zero and frequency-response Plotly figures."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import sympy as sp


def pole_zero_figure(poles: np.ndarray, zeros: np.ndarray) -> go.Figure:
    figure = go.Figure()
    if poles.size:
        figure.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", name="Poles",
                                    marker={"symbol": "x", "size": 13, "color": "#ff657a"}))
    if zeros.size:
        figure.add_trace(go.Scatter(x=zeros.real, y=zeros.imag, mode="markers", name="Zeros",
                                    marker={"symbol": "circle-open", "size": 12, "color": "#37d6f5"}))
    figure.add_vline(x=0, line_dash="dash", line_color="#71849a")
    figure.update_layout(title="Pole-zero map", xaxis_title="Real axis (rad/s)",
                         yaxis_title="Imaginary axis (rad/s)", template="plotly_dark",
                         margin={"l": 35, "r": 20, "t": 50, "b": 40})
    figure.update_xaxes(showgrid=True, gridcolor="#21344a", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#21344a", scaleanchor="x", scaleratio=1, zeroline=False)
    return figure


def frequency_figure(omega: np.ndarray, magnitude: np.ndarray, phase_degrees: np.ndarray) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=omega, y=20 * np.log10(np.maximum(magnitude, 1e-14)),
                                name="Magnitude", line={"color": "#37d6f5"}))
    figure.add_trace(go.Scatter(x=omega, y=phase_degrees, name="Phase (degrees)", yaxis="y2",
                                line={"color": "#ffb454"}))
    figure.update_layout(
        title="Continuous-time frequency response", template="plotly_dark",
        xaxis={"title": "Angular frequency ω (rad/s)", "type": "log"},
        yaxis={"title": "Magnitude (dB)"},
        yaxis2={"title": "Phase (degrees)", "overlaying": "y", "side": "right"},
        hovermode="x unified", margin={"l": 45, "r": 45, "t": 50, "b": 40},
    )
    return figure


def laplace_domain_figure(expression: sp.Expr, sigma: float = 0.1) -> go.Figure:
    """Plot |F(sigma + j omega)|, assigning unspecified parameters to 1."""
    variable = next((symbol for symbol in expression.free_symbols if symbol.name == "s"), None)
    if variable is None:
        raise ValueError("Transform expression must contain the Laplace variable s.")
    parameters = {symbol: 1.0 for symbol in expression.free_symbols if symbol != variable}
    numerical_expression = expression.subs(parameters)
    evaluator = sp.lambdify(variable, numerical_expression, modules="numpy")
    omega = np.linspace(-50.0, 50.0, 801)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = np.asarray(evaluator(sigma + 1j * omega), dtype=complex)
    if values.ndim == 0:
        values = np.full(omega.shape, values, dtype=complex)
    magnitude = np.abs(values)
    valid = np.isfinite(magnitude)
    if not np.any(valid):
        raise ValueError("Transform has no finite samples on this Laplace-domain slice.")
    figure = go.Figure(go.Scatter(x=omega[valid], y=magnitude[valid], mode="lines",
                                  name="|F(s)|", line={"color": "#9d8aff", "width": 2}))
    figure.update_layout(
        title=f"Laplace-domain magnitude at Re(s) = {sigma:g}",
        xaxis_title="Imaginary part of s (rad/s)", yaxis_title="|F(s)|",
        template="plotly_dark", hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#21344a")
    figure.update_yaxes(showgrid=True, gridcolor="#21344a")
    return figure
