"""PDF report generation from actual simulation data."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_analysis_pdf(
    title: str, numerator: list[float], denominator: list[float],
    analysis: dict[str, object], metrics: dict[str, object],
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, title="LAPLASCOPE AI Analysis")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("LAPLASCOPE AI — Simulation Analysis", styles["Title"]),
        Paragraph(title, styles["Heading2"]),
        Paragraph("Periodic Signal & Laplace Transform Simulator", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Transfer-function coefficients are listed in descending powers of s.", styles["BodyText"]),
    ]
    rows = [
        ["Quantity", "Value"],
        ["Numerator", np.array2string(np.asarray(numerator), precision=6)],
        ["Denominator", np.array2string(np.asarray(denominator), precision=6)],
        ["Stability", str(analysis["stability"])],
        ["Poles", np.array2string(analysis["poles"], precision=5)],
        ["Zeros", np.array2string(analysis["zeros"], precision=5)],
        ["Natural frequency (rad/s)", str(analysis["natural_frequency"])],
        ["Damping ratio", str(analysis["damping_ratio"])],
    ]
    for key in ("rise_time", "settling_time", "overshoot_percent", "steady_state_error", "final_value"):
        rows.append([key.replace("_", " ").title(), str(metrics.get(key))])
    table = Table(rows, colWidths=[205, 310], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e1c2d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#37d6f5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c8d8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf4f8")]),
    ]))
    story.extend([
        Spacer(1, 12), table, Spacer(1, 18),
        Paragraph("Metrics are estimated from the simulated unit-step response over the selected time window. "
                  "A short window can miss late settling behavior.", styles["BodyText"]),
    ])
    document.build(story)
    return buffer.getvalue()
