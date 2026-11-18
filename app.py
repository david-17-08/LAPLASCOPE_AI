from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

from modules.frequency_analysis import frequency_response
from modules.laplace_calculator import (
    forward_laplace, inverse_laplace, parse_expression, partial_fraction,
    system_response_transform, theorem_limits,
)
from modules.mathematical_analysis import interpret_system
from modules.pole_zero import analyze_poles_zeros, validate_transfer_function
from modules.signal_generator import generate_signal, generate_time
from modules.system_response import impulse_response, simulate_response, step_response_metrics
from reports.report_generator import create_analysis_pdf
from visualization.response_plots import frequency_figure, laplace_domain_figure, pole_zero_figure
from visualization.signal_plots import comparison_figure, signal_figure, signal_png


st.set_page_config(page_title="LAPLASCOPE AI", page_icon="∿", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.block-container { max-width: 1440px; padding-top: 1.6rem; }
.brand { color:#37d6f5; font-weight:800; letter-spacing:.16em; font-size:.78rem; }
.hero { border:1px solid #1e3c56; border-radius:20px; padding:2rem 2.2rem;
background:linear-gradient(120deg,#0e1c2d 0%,#102b40 65%,#102035 100%); }
.hero h1 { font-size:clamp(2rem,4vw,3.8rem); margin:.4rem 0; }
.hero p { color:#aec2d5; font-size:1.04rem; max-width:750px; }
.metric { background:#0e1c2d; border:1px solid #1d344b; border-radius:14px; padding:1rem 1.1rem; }
.metric-label { color:#aabed1; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#37d6f5; font-size:1.35rem; font-weight:700; margin-top:.3rem; }
.stButton>button, .stDownloadButton>button { border-radius:9px; font-weight:700; }
[data-testid="stSidebar"] { border-right:1px solid #1b344a; }
@media (max-width: 700px) { .block-container { padding:1rem .8rem; } .hero {padding:1.3rem;} }
</style>
""", unsafe_allow_html=True)

PAGES = [
    "Dashboard", "Signal Generator", "Laplace Transform Calculator",
    "System Response Simulator", "Pole-Zero Analysis", "Frequency Response",
    "Mathematical Theory", "Simulation Comparison", "Export Reports", "About Project",
]


def metric(label: str, value: str) -> None:
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def polynomial_input(label: str, default: str, key: str) -> list[float]:
    text = st.text_input(label, value=default, key=key,
                         help="Comma-separated coefficients in descending powers of s.")
    try:
        coefficients = [float(part.strip()) for part in text.split(",")]
        if not coefficients or not np.all(np.isfinite(coefficients)):
            raise ValueError
        return coefficients
    except ValueError as exc:
        raise ValueError(f"{label}: use finite comma-separated numbers.") from exc


def signal_controls(prefix: str = "signal") -> dict[str, object]:
    with st.container(border=True):
        st.subheader("Signal parameters")
        one, two, three = st.columns(3)
        waveform = one.selectbox("Waveform", ["Sine", "Cosine", "Square", "Triangular", "Sawtooth",
                                               "Pulse train", "Custom Fourier"], key=f"{prefix}_wave")
        amplitude = two.number_input("Amplitude", min_value=0.0, max_value=1000.0, value=1.0,
                                     step=0.1, key=f"{prefix}_amp")
        frequency = three.number_input("Frequency (Hz)", min_value=0.001, max_value=10000.0,
                                       value=1.0, step=0.1, key=f"{prefix}_freq")
        four, five, six = st.columns(3)
        phase = four.number_input("Phase (degrees)", min_value=-360.0, max_value=360.0,
                                  value=0.0, step=5.0, key=f"{prefix}_phase")
        duty = five.slider("Duty cycle (%)", 1, 99, 50, key=f"{prefix}_duty")
        fs = six.number_input("Sampling frequency (Hz)", min_value=10, max_value=5000,
                              value=500, step=10, key=f"{prefix}_fs")
        duration = st.slider("Simulation duration (seconds)", min_value=0.1, max_value=20.0,
                             value=3.0, step=0.1, key=f"{prefix}_duration")
        cosine: tuple[float, ...] = ()
        sine: tuple[float, ...] = ()
        if waveform == "Custom Fourier":
            st.caption("Custom waveform: Σ aₙ cos(nωt + phase) + bₙ sin(nωt + phase).")
            columns = st.columns(4)
            cosine = tuple(columns[i].number_input(f"a{i+1}", value=1.0 if i == 0 else 0.0,
                                                   key=f"{prefix}_a{i}") for i in range(4))
            sine = tuple(columns[i].number_input(f"b{i+1}", value=0.0, key=f"{prefix}_b{i}")
                         for i in range(4))
    return {"waveform": waveform, "amplitude": amplitude, "frequency": frequency, "phase": phase,
            "duty": float(duty), "fs": float(fs), "duration": duration,
            "custom_cosine": cosine, "custom_sine": sine}


def make_signal(parameters: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    time = generate_time(float(parameters["duration"]), float(parameters["fs"]))
    values = generate_signal(
        str(parameters["waveform"]), time, float(parameters["amplitude"]),
        float(parameters["frequency"]), float(parameters["phase"]), float(parameters["duty"]),
        parameters["custom_cosine"], parameters["custom_sine"],
    )
    return time, values


def show_formula(numerator: list[float], denominator: list[float]) -> None:
    variable = sp.Symbol("s")

    def polynomial(coefficients: list[float]) -> sp.Expr:
        order = len(coefficients) - 1
        return sum(sp.Float(value) * variable ** (order - index)
                   for index, value in enumerate(coefficients))

    st.latex(r"H(s)=\frac{" + sp.latex(polynomial(numerator)) + r"}{" +
             sp.latex(polynomial(denominator)) + r"}")


def render_dashboard() -> None:
    st.markdown('<div class="hero"><div class="brand">LAPLASCOPE AI · ENGINEERING SIMULATION</div>'
                '<h1>See the system behind the signal.</h1>'
                '<p>Generate periodic waveforms, explore Laplace transforms, and simulate continuous-time systems '
                'with transparent mathematics and interactive engineering plots.</p></div>', unsafe_allow_html=True)
    st.write("")
    cards = st.columns(4)
    content = [
        ("Signal lab", "Seven periodic waveform options"),
        ("Laplace studio", "Forward, inverse, periodic and finite-window transforms"),
        ("System response", "Driven LTI simulations and step metrics"),
        ("Analysis & export", "Poles, frequency response, CSV and PDF"),
    ]
    for column, (title, detail) in zip(cards, content):
        with column:
            st.markdown(f'<div class="metric"><div class="metric-label">Explore</div>'
                        f'<div class="metric-value">{title}</div><p>{detail}</p></div>', unsafe_allow_html=True)
    st.subheader("Live signal preview")
    if "dashboard_time" not in st.session_state:
        time = generate_time(2.0, 300.0)
        values = generate_signal("Sine", time, 1, 1)
        st.session_state.dashboard_time, st.session_state.dashboard_values = time, values
    st.plotly_chart(signal_figure(st.session_state.dashboard_time, st.session_state.dashboard_values),
                    width="stretch", config={"displaylogo": False, "scrollZoom": True})
    st.info("Start with **Signal Generator** to define a waveform, or **System Response Simulator** to model an LTI system.")


def render_signal_generator() -> None:
    st.title("Signal Generator")
    st.caption("Samples are uniformly spaced; sample-count limits protect responsiveness.")
    parameters = signal_controls("generator")
    try:
        time, values = make_signal(parameters)
    except ValueError as error:
        st.error(str(error))
        return
    harmonics = len(parameters["custom_cosine"]) if parameters["waveform"] == "Custom Fourier" else 1
    if float(parameters["fs"]) < 2 * float(parameters["frequency"]) * harmonics:
        st.warning("Sampling frequency is below the Nyquist rate for this waveform; displayed samples may be aliased.")
    st.session_state.generated_signal = {"time": time, "input": values, "parameters": parameters}
    columns = st.columns(4)
    metrics = [("Period", f"{1 / float(parameters['frequency']):.4g} s"),
               ("Samples", f"{time.size:,}"),
               ("Peak", f"{np.max(np.abs(values)):.4g}"),
               ("Sample rate", f"{parameters['fs']:g} Hz")]
    for column, (label, value) in zip(columns, metrics):
        with column:
            metric(label, value)
    st.plotly_chart(signal_figure(time, values), width="stretch",
                    config={"displaylogo": False, "scrollZoom": True})
    csv = pd.DataFrame({"time_s": time, "signal": values}).to_csv(index=False).encode("utf-8")
    png = signal_png(time, values)
    left, right = st.columns(2)
    left.download_button("Download signal CSV", csv, "laplascope_signal.csv", "text/csv", width="stretch")
    right.download_button("Download graph PNG", png, "laplascope_signal.png", "image/png", width="stretch")


def render_laplace() -> None:
    st.title("Laplace Transform Calculator")
    st.caption("Expressions are parsed through a restricted arithmetic grammar; Python code is never evaluated.")
    operation = st.radio("Operation", ["Forward transform", "Inverse transform", "Partial fractions",
                                       "Initial / final value"], horizontal=True)
    if operation == "Forward transform":
        expression = st.text_input("f(t)", "sin(2*t)",
                                   help="Allowed: arithmetic, powers (**), and sin/cos/exp/log/sqrt functions.")
        mode = st.selectbox("Transform definition", ["Causal signal", "Periodic extension", "Finite duration [0, T]"])
        if st.button("Calculate forward transform", type="primary"):
            try:
                result = forward_laplace(expression, mode)
                st.latex(r"F(s)=" + sp.latex(result.expression))
                st.write("**Convergence / scope:**", result.region_of_convergence)
                st.write("**Method:**")
                for step in result.steps:
                    st.write(f"- {step}")
                sigma = st.number_input("Laplace slice: Re(s)", min_value=0.001, value=0.1, step=0.1)
                try:
                    st.plotly_chart(laplace_domain_figure(result.expression, sigma), width="stretch")
                    if mode == "Periodic extension":
                        st.caption("Unspecified period T is set to 1 for this plot; change the expression/definition for other periods.")
                except (ValueError, TypeError, NotImplementedError, OverflowError) as error:
                    st.warning(f"Could not plot this transform numerically: {error}")
            except (ValueError, TypeError, NotImplementedError, sp.SympifyError, RecursionError) as error:
                st.error(f"Transform could not be calculated: {error}")
    elif operation == "Inverse transform":
        expression = st.text_input("F(s)", "1/(s+2)")
        if st.button("Calculate inverse transform", type="primary"):
            try:
                result = inverse_laplace(expression)
                st.latex(r"f(t)=" + sp.latex(result))
                st.caption("The result uses the causal one-sided inverse-transform convention.")
            except (ValueError, TypeError, NotImplementedError, sp.SympifyError, RecursionError) as error:
                st.error(f"Inverse transform could not be calculated: {error}")
    elif operation == "Partial fractions":
        col1, col2 = st.columns(2)
        numerator_text = col1.text_input("Numerator coefficients", "1")
        denominator_text = col2.text_input("Denominator coefficients", "1, 3, 2")
        if st.button("Expand into partial fractions", type="primary"):
            try:
                numerator = [float(value) for value in numerator_text.split(",")]
                denominator = [float(value) for value in denominator_text.split(",")]
                result = partial_fraction(numerator, denominator)
                st.latex(r"\frac{N(s)}{D(s)}=" + sp.latex(result))
            except (ValueError, TypeError, ArithmeticError, sp.SympifyError) as error:
                st.error(f"Partial fraction expansion failed: {error}")
    else:
        expression = st.text_input("F(s)", "1/(s*(s+2))")
        if st.button("Evaluate value theorems", type="primary"):
            try:
                transformed = parse_expression(expression, sp.Symbol("s", real=True))
                initial, final, _, note = theorem_limits(transformed)
                st.latex(r"f(0^+)=\lim_{s\to\infty}sF(s)=" + sp.latex(initial))
                st.latex(r"f(\infty)=\lim_{s\to0}sF(s)=" + sp.latex(final))
                st.warning(note)
            except (ValueError, TypeError, NotImplementedError, sp.SympifyError) as error:
                st.error(f"Value-theorem calculation failed: {error}")
    with st.expander("Periodic vs. finite-duration transform"):
        st.markdown(
            "For a suitable one-sided periodic extension of period **T**, "
            r"$F(s)=\frac{\int_0^T f(t)e^{-st}\,dt}{1-e^{-sT}}$. "
            "A finite-duration signal instead uses only its finite-interval integral; it is not repeated periodically."
        )


def render_playback_frame() -> None:
    result = st.session_state.get("simulation")
    if not result:
        return
    time = result["time"]
    index = min(int(st.session_state.get("playback_index", 0)), len(time) - 1)
    if st.session_state.get("simulation_playing", False):
        index = min(index + max(1, len(time) // 100), len(time) - 1)
        st.session_state.playback_index = index
        if index == len(time) - 1:
            st.session_state.simulation_playing = False
    end = max(2, index + 1)
    stride = max(1, end // 1500)
    figure = comparison_figure(
        time[:end:stride],
        {
            "Input x(t)": result["input"][:end:stride],
            "Output y(t)": result["output"][:end:stride],
            "Impulse response h(t)": result["impulse"][:end:stride],
        },
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(f"Playback time: {time[min(index, len(time) - 1)]:.4g} s / {time[-1]:.4g} s")


def render_simulator() -> None:
    st.title("System Response Simulator")
    st.caption("Continuous-time LTI simulation uses SciPy. Coefficients are entered in descending powers of s.")
    preset = st.selectbox("System model", ["First-order", "Second-order", "User transfer function"])
    if preset == "First-order":
        col1, col2 = st.columns(2)
        gain = col1.number_input("Gain K", value=1.0, min_value=0.001)
        tau = col2.number_input("Time constant τ (s)", value=1.0, min_value=0.001)
        numerator, denominator = [gain], [tau, 1.0]
    elif preset == "Second-order":
        col1, col2, col3 = st.columns(3)
        wn = col1.number_input("Natural frequency ωₙ (rad/s)", value=2.0, min_value=0.01)
        zeta = col2.number_input("Damping ratio ζ", value=0.5, min_value=0.0, max_value=10.0)
        gain = col3.number_input("DC gain", value=1.0, min_value=0.001)
        numerator, denominator = [gain * wn**2], [1.0, 2 * zeta * wn, wn**2]
    else:
        try:
            numerator = polynomial_input("Numerator", "1", "tf_num")
            denominator = polynomial_input("Denominator", "1, 2, 1", "tf_den")
        except ValueError as error:
            st.error(str(error))
            return
    try:
        validate_transfer_function(numerator, denominator)
    except ValueError as error:
        st.error(str(error))
        return
    show_formula(numerator, denominator)
    input_type = st.selectbox("Input", ["Step", "Impulse", "Ramp", "Sinusoidal", "Periodic square wave"])
    controls = st.columns(3)
    amplitude = controls[0].number_input("Input amplitude", value=1.0)
    input_frequency = controls[1].number_input("Input frequency (Hz)", value=1.0, min_value=0.001)
    duration = controls[2].number_input("Duration (s)", value=5.0, min_value=0.1, max_value=20.0)
    sample_rate = st.number_input("Sampling frequency (Hz)", min_value=20, max_value=5000, value=500, step=10)
    start_column, reset_column = st.columns(2)
    run_simulation = start_column.button("Run simulation", type="primary", width="stretch")
    reset_simulation = reset_column.button("Reset simulation", width="stretch")
    if reset_simulation:
        st.session_state.pop("simulation", None)
    if run_simulation:
        try:
            time = generate_time(duration, sample_rate)
            if input_type == "Step":
                input_signal = np.full_like(time, amplitude)
            elif input_type == "Impulse":
                input_signal = np.zeros_like(time)
                input_signal[0] = amplitude / (time[1] - time[0])
            elif input_type == "Ramp":
                input_signal = amplitude * time
            elif input_type == "Sinusoidal":
                input_signal = amplitude * np.sin(2 * np.pi * input_frequency * time)
            else:
                bipolar = generate_signal("Square", time, frequency=input_frequency)
                input_signal = amplitude * (bipolar + 1.0) / 2.0
            response_time, output = simulate_response(
                numerator, denominator, time, input_signal, input_type, impulse_magnitude=amplitude
            )
            impulse = impulse_response(numerator, denominator, time)
            analysis = analyze_poles_zeros(numerator, denominator)
            metrics = step_response_metrics(numerator, denominator, time)
            symbolic_error = None
            try:
                symbolic = system_response_transform(numerator, denominator, input_type, amplitude, input_frequency)
            except (ValueError, TypeError, NotImplementedError, sp.SympifyError, RecursionError) as error:
                symbolic = None
                symbolic_error = str(error)
            st.session_state.simulation = {
                "time": response_time, "input": input_signal, "output": output, "impulse": impulse,
                "numerator": numerator, "denominator": denominator, "analysis": analysis, "metrics": metrics,
                "symbolic": symbolic, "symbolic_error": symbolic_error,
                "title": f"{preset} system with {input_type.lower()} input",
            }
            st.session_state.playback_index = 0
            st.session_state.simulation_playing = False
        except (ValueError, np.linalg.LinAlgError) as error:
            st.error(f"Simulation failed: {error}")
    result = st.session_state.get("simulation")
    if result:
        if result["symbolic"] is not None:
            output_transform, symbolic_output = result["symbolic"]
            st.latex(r"Y(s)=" + sp.latex(output_transform))
            st.latex(r"y(t)=" + sp.latex(symbolic_output))
        else:
            st.latex(r"Y(s)=H(s)X(s),\qquad y(t)\text{ obtained numerically}")
            if result["symbolic_error"]:
                st.warning(f"Symbolic inversion was unavailable: {result['symbolic_error']}. Numerical output is shown.")
            else:
                st.caption("The periodic square-wave response is calculated numerically from its sampled input.")
        st.plotly_chart(comparison_figure(result["time"], {
            "Input x(t)": result["input"], "Output y(t)": result["output"],
            "Impulse response h(t)": result["impulse"],
        }), width="stretch", config={"displaylogo": False, "scrollZoom": True})
        st.subheader("Calculated step-response characteristics")
        metric_columns = st.columns(5)
        labels = [("Rise time", "rise_time", "s"), ("Settling time", "settling_time", "s"),
                  ("Overshoot", "overshoot_percent", "%"), ("Steady-state error", "steady_state_error", ""),
                  ("Final value", "final_value", "")]
        for column, (label, key, unit) in zip(metric_columns, labels):
            value = result["metrics"][key]
            with column:
                metric(label, "n/a" if value is None else f"{value:.4g} {unit}".strip())
        st.caption("Metrics use a unit-step response over this selected simulation window.")
        for explanation in interpret_system(result["analysis"]):
            st.write(f"- {explanation}")
        st.subheader("Response playback")
        controls = st.columns(3)
        if controls[0].button("Start playback", type="primary", width="stretch"):
            st.session_state.simulation_playing = True
        if controls[1].button("Pause playback", width="stretch"):
            st.session_state.simulation_playing = False
        if controls[2].button("Reset playback", width="stretch"):
            st.session_state.simulation_playing = False
            st.session_state.playback_index = 0
        playback_fragment = st.fragment(
            run_every=0.1 if st.session_state.get("simulation_playing", False) else None
        )
        playback_fragment(render_playback_frame)()


def render_analysis() -> None:
    st.title("Pole-Zero Analysis")
    try:
        numerator = polynomial_input("Numerator coefficients", "1", "pz_num")
        denominator = polynomial_input("Denominator coefficients", "1, 2, 1", "pz_den")
        analysis = analyze_poles_zeros(numerator, denominator)
    except ValueError as error:
        st.error(str(error))
        return
    show_formula(numerator, denominator)
    st.plotly_chart(pole_zero_figure(analysis["poles"], analysis["zeros"]), width="stretch")
    st.success(f"Stability classification: {analysis['stability']}")
    st.write("**Poles:**", np.array2string(analysis["poles"], precision=5))
    st.write("**Zeros:**", np.array2string(analysis["zeros"], precision=5))
    for explanation in interpret_system(analysis):
        st.write(f"- {explanation}")


def render_frequency() -> None:
    st.title("Frequency Response")
    try:
        numerator = polynomial_input("Numerator coefficients", "1", "fr_num")
        denominator = polynomial_input("Denominator coefficients", "1, 1", "fr_den")
        omega, magnitude, phase = frequency_response(numerator, denominator)
    except ValueError as error:
        st.error(str(error))
        return
    st.plotly_chart(frequency_figure(omega, magnitude, phase), width="stretch")
    st.caption("Magnitude is shown in dB; phase is unwrapped and shown in degrees. Frequency axis is rad/s.")


def render_theory() -> None:
    st.title("Mathematical Theory")
    st.subheader("Periodic extension")
    st.latex(r"F(s)=\frac{\int_0^T f(t)e^{-st}\,dt}{1-e^{-sT}}")
    st.write("This geometric-series formula applies to the one-sided repetition of a period. For a bounded periodic "
             "signal, the transform generally converges in a right half-plane. A finite record instead has transform "
             r"$F_T(s)=\int_0^T f(t)e^{-st}\,dt$ and is not an infinite periodic signal.")
    st.subheader("LTI input-output relationship")
    st.latex(r"Y(s)=H(s)X(s),\qquad y(t)=\mathcal{L}^{-1}\{Y(s)\}")
    st.write("The transfer function describes a zero-initial-condition LTI system. Nonzero initial conditions add "
             "their own terms and are not implicit in this model.")
    st.subheader("Pole, damping, and stability interpretation")
    st.latex(r"H(s)=\frac{N(s)}{D(s)},\qquad D(p_k)=0")
    st.write("Poles determine natural modes and continuous-time stability. A proper transfer function is required for "
             "the current time-domain simulator. For a standard second-order denominator "
             r"$s^2+2\zeta\omega_n s+\omega_n^2$, $\omega_n$ is the undamped natural frequency and $\zeta$ is damping ratio.")
    st.subheader("Value theorems")
    st.latex(r"f(0^+)=\lim_{s\to\infty}sF(s),\qquad f(\infty)=\lim_{s\to0}sF(s)")
    st.warning("The final-value theorem is valid only when all poles of sF(s) lie strictly in the left half-plane; "
               "a finite symbolic limit alone does not establish that condition.")


def render_comparison() -> None:
    st.title("Simulation Comparison")
    signals_tab, systems_tab = st.tabs(["Waveforms", "System responses"])
    with signals_tab:
        st.write("Compare generated periodic waveforms on the same time axis.")
        left, right = st.columns(2)
        choices = ["Sine", "Cosine", "Square", "Triangular", "Sawtooth", "Pulse train"]
        first = left.selectbox("Waveform A", choices, index=0)
        second = right.selectbox("Waveform B", choices, index=2)
        frequency = st.number_input("Shared frequency (Hz)", min_value=0.01, value=1.0)
        time = generate_time(2.0, 500)
        first_values = generate_signal(first, time, frequency=frequency)
        second_values = generate_signal(second, time, frequency=frequency)
        st.plotly_chart(comparison_figure(time, {first: first_values, second: second_values}),
                        width="stretch")
    with systems_tab:
        st.write("Compare two independent first- or second-order systems with a shared unit-step input.")
        time = generate_time(5.0, 500)
        system_columns = st.columns(2)
        coefficients: list[tuple[list[float], list[float]]] = []
        for index, column in enumerate(system_columns, start=1):
            with column:
                model = st.selectbox(f"System {index} model", ["First-order", "Second-order"], key=f"compare_model_{index}")
                if model == "First-order":
                    gain = st.number_input(f"System {index} gain K", min_value=0.001, value=1.0,
                                           key=f"compare_gain_{index}")
                    tau = st.number_input(f"System {index} time constant τ", min_value=0.001, value=float(index),
                                          key=f"compare_tau_{index}")
                    coefficients.append(([gain], [tau, 1.0]))
                else:
                    wn = st.number_input(f"System {index} natural frequency ωₙ", min_value=0.01, value=2.0,
                                         key=f"compare_wn_{index}")
                    zeta = st.number_input(f"System {index} damping ratio ζ", min_value=0.0, max_value=10.0,
                                           value=0.4 * index, key=f"compare_zeta_{index}")
                    coefficients.append(([wn**2], [1.0, 2 * zeta * wn, wn**2]))
        outputs = {}
        for index, (numerator, denominator) in enumerate(coefficients, start=1):
            _, outputs[f"System {index}"] = simulate_response(
                numerator, denominator, time, np.ones_like(time), "Step"
            )
        st.plotly_chart(comparison_figure(time, outputs), width="stretch")


def render_exports() -> None:
    st.title("Export Reports")
    result = st.session_state.get("simulation")
    if not result:
        st.info("Run a system simulation first to export its full dataset and analysis report.")
        time = generate_time(1, 100)
        sample = pd.DataFrame({"time_s": time, "signal": generate_signal("Sine", time)})
        st.download_button("Download example signal CSV", sample.to_csv(index=False).encode(),
                           "laplascope_example.csv", "text/csv")
        return
    st.write("Exports are generated from the current simulation results.")
    data = pd.DataFrame({"time_s": result["time"], "input_x": result["input"],
                         "output_y": result["output"], "impulse_h": result["impulse"]})
    st.download_button("Download simulation CSV", data.to_csv(index=False).encode("utf-8"),
                       "laplascope_simulation.csv", "text/csv", type="primary")
    st.download_button("Download output graph PNG", signal_png(result["time"], result["output"], "y(t)"),
                       "laplascope_response.png", "image/png")
    report = create_analysis_pdf(result["title"], result["numerator"], result["denominator"],
                                 result["analysis"], result["metrics"])
    st.download_button("Download mathematical analysis PDF", report,
                       "laplascope_analysis.pdf", "application/pdf")
    parameters = {
        "title": result["title"], "numerator": result["numerator"],
        "denominator": result["denominator"], "stability": result["analysis"]["stability"],
        "metrics": result["metrics"],
    }
    st.download_button("Save simulation parameters (JSON)", json.dumps(parameters, indent=2, default=str).encode(),
                       "laplascope_parameters.json", "application/json")
    st.dataframe(data.head(20), width="stretch")


def render_about() -> None:
    st.title("About LAPLASCOPE AI")
    st.markdown("**Intelligent Periodic Signal & Laplace Transform Simulator.**")
    st.write("An educational engineering application that connects time-domain signals, Laplace-domain analysis, "
             "transfer functions, and continuous-time system responses.")
    st.info("Scientific note: all values are calculated from the selected input and system model. "
            "Numerical step metrics depend on simulation duration and sampling resolution.")


def main() -> None:
    with st.sidebar:
        st.markdown('<div class="brand">L A P L A S C O P E</div>', unsafe_allow_html=True)
        st.title("AI")
        st.caption("Transform signals. Understand systems.")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Interactive engineering lab · Python + SciPy + SymPy")
    renderers = {
        "Dashboard": render_dashboard,
        "Signal Generator": render_signal_generator,
        "Laplace Transform Calculator": render_laplace,
        "System Response Simulator": render_simulator,
        "Pole-Zero Analysis": render_analysis,
        "Frequency Response": render_frequency,
        "Mathematical Theory": render_theory,
        "Simulation Comparison": render_comparison,
        "Export Reports": render_exports,
        "About Project": render_about,
    }
    renderers[page]()
    st.divider()
    st.caption("LAPLASCOPE AI · Educational simulation; verify results independently for safety-critical engineering.")


if __name__ == "__main__":
    main()
