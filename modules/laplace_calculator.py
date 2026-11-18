"""Safe symbolic parsing and Laplace transform utilities."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.polys.polyerrors import PolynomialError

from modules.pole_zero import validate_transfer_function

t, s = sp.symbols("t s", real=True)
_FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sinh": sp.sinh,
    "cosh": sp.cosh, "tanh": sp.tanh, "exp": sp.exp, "log": sp.log,
    "sqrt": sp.sqrt, "Heaviside": sp.Heaviside,
}
_CONSTANTS = {"pi": sp.pi, "E": sp.E}
_SYMBOLS = {"t": t, "s": s}


class ExpressionError(ValueError):
    """Raised when an expression uses syntax outside the supported grammar."""


def parse_expression(expression: str, variable: sp.Symbol | None = None) -> sp.Expr:
    """Parse arithmetic and named SymPy functions without evaluating Python code."""
    if not expression or len(expression) > 240:
        raise ExpressionError("Enter an expression of 1–240 characters.")
    try:
        root = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ExpressionError("Expression syntax is invalid. Use ** for powers.") from exc
    if sum(1 for _ in ast.walk(root)) > 120:
        raise ExpressionError("Expression is too complex.")
    symbols = dict(_SYMBOLS) if variable is None else {variable.name: variable}

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool) or abs(node.value) > 1e12:
                raise ExpressionError("Numeric literals must be finite and no larger than 1e12.")
            if isinstance(node.value, float) and not sp.Float(node.value).is_finite:
                raise ExpressionError("Non-finite numbers are not allowed.")
            return sp.sympify(node.value)
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ExpressionError(f"Name '{node.id}' is not allowed.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Pow) and right.is_number:
                try:
                    if abs(float(right)) > 100:
                        raise ExpressionError("Numeric exponents are limited to magnitude 100.")
                except OverflowError as exc:
                    raise ExpressionError("Numeric exponent is too large.") from exc
            operations = {
                ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                ast.Pow: lambda: left**right,
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise ExpressionError("Only +, -, *, /, and ** operators are supported.")
            return operation()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _FUNCTIONS.get(node.func.id)
            if function is None or node.keywords or len(node.args) != 1:
                raise ExpressionError("Use a supported single-argument math function.")
            return function(convert(node.args[0]))
        raise ExpressionError("Expression contains unsupported syntax.")

    try:
        result = sp.sympify(convert(root))
    except (ArithmeticError, TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ExpressionError):
            raise
        raise ExpressionError("Expression could not be evaluated symbolically.") from exc
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ExpressionError("Expression contains an undefined value.")
    return result


@dataclass(frozen=True)
class LaplaceResult:
    expression: sp.Expr
    region_of_convergence: str
    steps: tuple[str, ...]


def forward_laplace(expression: str, mode: str = "Causal signal") -> LaplaceResult:
    """Find a one-sided or finite-window transform of a supported expression."""
    function = parse_expression(expression, t)
    if mode == "Periodic extension":
        period = sp.symbols("T", positive=True, real=True)
        numerator = sp.integrate(function * sp.exp(-s * t), (t, 0, period))
        transformed = sp.simplify(numerator / (1 - sp.exp(-s * period)))
        return LaplaceResult(
            transformed,
            "For a bounded periodic signal, typically Re(s) > 0; verify convergence for the chosen waveform.",
            ("Integrate one period: F₁(s) = ∫₀ᵀ f(t)e^(−st) dt.",
             "Sum repeated periods: F(s) = F₁(s) / (1 − e^(−sT))."),
        )
    if mode == "Finite duration [0, T]":
        end = sp.symbols("T", positive=True, real=True)
        transformed = sp.simplify(sp.integrate(function * sp.exp(-s * t), (t, 0, end)))
        return LaplaceResult(
            transformed,
            "Finite integration interval [0, T]; this is not the transform of an infinite periodic extension.",
            ("Apply the finite-interval definition: F(s) = ∫₀ᵀ f(t)e^(−st) dt.",),
        )
    if mode != "Causal signal":
        raise ValueError("Unknown transform definition.")
    transformed, convergence, condition = sp.laplace_transform(function, t, s)
    roc = f"Re(s) > {convergence}" if convergence is not None else "Convergence condition not determined."
    if condition not in (True, sp.true):
        roc += f"; condition: {sp.sstr(condition)}"
    return LaplaceResult(transformed, roc, (
        "Use the one-sided definition: F(s) = ∫₀∞ f(t)e^(−st) dt.",
        "SymPy evaluates the integral and its convergence condition.",
    ))


def inverse_laplace(expression: str) -> sp.Expr:
    """Compute an inverse transform for an expression in s."""
    return sp.inverse_laplace_transform(parse_expression(expression, s), s, t)


def system_response_transform(
    numerator: list[float],
    denominator: list[float],
    input_type: str,
    amplitude: float = 1.0,
    frequency: float = 1.0,
) -> tuple[sp.Expr, sp.Expr] | None:
    """Return Y(s), y(t) for standard inputs; periodic square uses numerical simulation."""
    if input_type == "Periodic square wave":
        return None
    if not np.isfinite(amplitude) or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Input amplitude and frequency must be finite; frequency must be positive.")
    numerator_values, denominator_values = validate_transfer_function(numerator, denominator)
    num = sum(sp.Rational(str(float(value))) * s ** (len(numerator_values) - index - 1)
              for index, value in enumerate(numerator_values))
    den = sum(sp.Rational(str(float(value))) * s ** (len(denominator_values) - index - 1)
              for index, value in enumerate(denominator_values))
    gain = sp.Rational(str(float(amplitude)))
    if input_type == "Step":
        input_transform = gain / s
    elif input_type == "Impulse":
        input_transform = gain
    elif input_type == "Ramp":
        input_transform = gain / s**2
    elif input_type == "Sinusoidal":
        angular_frequency = sp.Rational(str(float(frequency))) * 2 * sp.pi
        input_transform = gain * angular_frequency / (s**2 + angular_frequency**2)
    else:
        raise ValueError("Unsupported standard response input.")
    output_transform = sp.cancel(num * input_transform / den)
    return output_transform, sp.inverse_laplace_transform(output_transform, s, t)


def partial_fraction(numerator: list[float], denominator: list[float]) -> sp.Expr:
    """Return partial fractions from descending polynomial coefficients."""
    if not numerator or not denominator or len(numerator) > 12 or len(denominator) > 12:
        raise ValueError("Provide 1–12 numerator and denominator coefficients.")
    if not all(np.isfinite(value) for value in (*numerator, *denominator)):
        raise ValueError("Polynomial coefficients must be finite.")
    variable = sp.Symbol("s")
    num = sum(sp.Float(value) * variable ** (len(numerator) - index - 1) for index, value in enumerate(numerator))
    den = sum(sp.Float(value) * variable ** (len(denominator) - index - 1) for index, value in enumerate(denominator))
    if den == 0:
        raise ValueError("Denominator polynomial cannot be zero.")
    return sp.apart(num / den, variable)


def theorem_limits(expression: sp.Expr) -> tuple[sp.Expr, sp.Expr, bool, str]:
    """Calculate value-theorem limits and test the final-value pole condition."""
    transformed = sp.cancel(expression)
    initial = sp.limit(s * transformed, s, sp.oo)
    final = sp.limit(s * transformed, s, 0)
    try:
        denominator = sp.denom(sp.cancel(transformed * s))
        poles = sp.roots(denominator, s)
        poles_ok = bool(poles) and all(sp.re(pole).is_negative is True for pole in poles)
        if denominator == 1:
            poles_ok = True
    except (NotImplementedError, PolynomialError, TypeError, ValueError):
        poles_ok = False
    note = "Final-value theorem pole condition appears satisfied." if poles_ok else (
        "Limit shown, but theorem condition is not established: sF(s) must have no poles in Re(s) ≥ 0."
    )
    return initial, final, poles_ok, note
