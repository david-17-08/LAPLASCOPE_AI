"""Plotly and PNG plotting helpers for signal data."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def signal_figure(time: np.ndarray, values: np.ndarray, label: str = "x(t)") -> go.Figure:
    figure = go.Figure(go.Scatter(x=time, y=values, name=label, mode="lines",
                                  line={"color": "#37d6f5", "width": 2}))
    figure.update_layout(title="Time-domain signal", xaxis_title="Time (s)", yaxis_title="Amplitude",
                         template="plotly_dark", hovermode="x unified",
                         margin={"l": 35, "r": 20, "t": 50, "b": 40})
    figure.update_xaxes(showgrid=True, gridcolor="#21344a")
    figure.update_yaxes(showgrid=True, gridcolor="#21344a")
    return figure


def comparison_figure(time: np.ndarray, series: dict[str, np.ndarray]) -> go.Figure:
    figure = go.Figure()
    colors = ["#37d6f5", "#ffb454", "#9d8aff", "#60df9f"]
    for index, (name, values) in enumerate(series.items()):
        figure.add_trace(go.Scatter(x=time, y=values, name=name, mode="lines",
                                    line={"color": colors[index % len(colors)]}))
    figure.update_layout(title="Input, system response, and impulse response",
                         xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_dark",
                         hovermode="x unified", margin={"l": 35, "r": 20, "t": 50, "b": 40})
    figure.update_xaxes(showgrid=True, gridcolor="#21344a")
    figure.update_yaxes(showgrid=True, gridcolor="#21344a")
    return figure


def signal_png(time: np.ndarray, values: np.ndarray, label: str = "x(t)") -> bytes:
    buffer = BytesIO()
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    figure.patch.set_facecolor("#07111f")
    axis.set_facecolor("#0e1c2d")
    axis.plot(time, values, color="#37d6f5", linewidth=1.6, label=label)
    axis.set(xlabel="Time (s)", ylabel="Amplitude", title="LAPLASCOPE AI — time-domain signal")
    axis.grid(True, color="#294158", alpha=0.7)
    axis.tick_params(colors="#e7f2ff")
    axis.xaxis.label.set_color("#e7f2ff")
    axis.yaxis.label.set_color("#e7f2ff")
    axis.title.set_color("#e7f2ff")
    axis.legend(facecolor="#0e1c2d", labelcolor="#e7f2ff")
    figure.savefig(buffer, format="png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    return buffer.getvalue()
