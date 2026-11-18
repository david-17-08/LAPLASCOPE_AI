import numpy as np
import pytest
import sympy as sp

from modules.frequency_analysis import frequency_response
from modules.laplace_calculator import (
    ExpressionError, forward_laplace, inverse_laplace, parse_expression, partial_fraction,
    system_response_transform, theorem_limits,
)
from modules.pole_zero import analyze_poles_zeros
from modules.signal_generator import generate_signal, generate_time
from modules.system_response import simulate_response
from reports.report_generator import create_analysis_pdf
from visualization.response_plots import laplace_domain_figure
from visualization.signal_plots import signal_png


def test_sine_square_triangle_generation():
    time = np.arange(0, 1, 0.001)
    sine = generate_signal("Sine", time, amplitude=2, frequency=1)
    square = generate_signal("Square", time, amplitude=2, frequency=1)
    triangle = generate_signal("Triangular", time, amplitude=2, frequency=1)
    assert np.max(sine) == pytest.approx(2, abs=1e-5)
    assert set(np.unique(square)) == {-2.0, 2.0}
    assert np.max(triangle) == pytest.approx(2, abs=1e-5)
    assert np.min(triangle) == pytest.approx(-2, abs=1e-5)


def test_time_vector_guards_sample_limit():
    assert generate_time(1, 10).size == 10
    with pytest.raises(ValueError, match="limited"):
        generate_time(30, 5000)


def test_forward_and_periodic_laplace_transforms():
    result = forward_laplace("sin(2*t)")
    variable = sp.Symbol("s", real=True)
    assert sp.simplify(result.expression - 2 / (variable**2 + 4)) == 0
    periodic = forward_laplace("1", "Periodic extension")
    assert sp.simplify(periodic.expression.subs(variable, 1) - 1) == 0
    finite = forward_laplace("1", "Finite duration [0, T]")
    period = sp.Symbol("T", positive=True, real=True)
    assert sp.simplify(finite.expression.subs({variable: 1, period: 2}) - (1 - sp.exp(-2))) == 0


def test_inverse_laplace_and_safe_expression_parser():
    inverse = inverse_laplace("1/(s+2)")
    time_symbol = sp.Symbol("t", real=True)
    assert sp.simplify(inverse - sp.exp(-2 * time_symbol) * sp.Heaviside(time_symbol)) == 0
    with pytest.raises(ExpressionError):
        parse_expression("__import__('os').system('whoami')")
    with pytest.raises(ExpressionError):
        parse_expression("2**1000000")
    with pytest.raises(ExpressionError):
        parse_expression("s + 1", sp.Symbol("t", real=True))


def test_symbolic_system_response_and_final_value_condition():
    response = system_response_transform([1], [1, 1], "Step")
    assert response is not None
    transform, time_response = response
    assert sp.simplify(transform - 1 / (sp.Symbol("s", real=True) * (sp.Symbol("s", real=True) + 1))) == 0
    assert time_response.has(sp.Heaviside)
    initial, final, valid, _ = theorem_limits(1 / (sp.Symbol("s", real=True) *
                                                   (sp.Symbol("s", real=True) + 2)))
    assert initial == 0
    assert final == sp.Rational(1, 2)
    assert valid
    expanded = partial_fraction([1], [1, 3, 2])
    assert float(expanded.subs(sp.Symbol("s"), 1)) == pytest.approx(1 / 6)


def test_first_and_second_order_step_response():
    time = np.linspace(0, 5, 1001)
    step = np.ones_like(time)
    _, first = simulate_response([1], [1, 1], time, step)
    _, second = simulate_response([4], [1, 2, 4], time, step)
    assert first[-1] == pytest.approx(1, abs=0.02)
    symbolic = system_response_transform([1], [1, 1], "Step")
    assert symbolic is not None
    _, exact = symbolic
    exact_numeric = sp.lambdify(sp.Symbol("t", real=True), exact, "numpy")(time)
    assert np.allclose(first[1:], exact_numeric[1:], atol=1e-3)
    assert second[-1] == pytest.approx(1, abs=0.02)
    _, impulse = simulate_response([1], [1, 1], time, np.zeros_like(time), "Impulse",
                                   impulse_magnitude=2)
    assert impulse[0] == pytest.approx(2)


def test_stability_analysis():
    assert analyze_poles_zeros([1], [1, 2])["stability"] == "Asymptotically stable"
    assert analyze_poles_zeros([1], [1, -1])["stability"] == "Unstable"
    assert analyze_poles_zeros([1], [1, 0])["stability"] == "Marginally stable"
    assert analyze_poles_zeros([1], [1, 0, 0])["stability"] == "Unstable"


def test_frequency_and_export_outputs():
    omega, magnitude, phase = frequency_response([1], [1, 1])
    assert omega.size == magnitude.size == phase.size
    assert np.all(np.isfinite(magnitude))
    omega_second, magnitude_second, _ = frequency_response([4], [1, 2, 4])
    assert omega_second.size == magnitude_second.size
    plot = laplace_domain_figure(1 / (sp.Symbol("s", real=True) + 1))
    assert len(plot.data) == 1
    time = np.linspace(0, 1, 100)
    assert signal_png(time, np.sin(time)).startswith(b"\x89PNG")
    analysis = analyze_poles_zeros([1], [1, 1])
    pdf = create_analysis_pdf("first-order step", [1], [1, 1], analysis, {
        "rise_time": 1.0, "settling_time": 4.0, "overshoot_percent": 0.0,
        "steady_state_error": 0.0, "final_value": 1.0,
    })
    assert pdf.startswith(b"%PDF")