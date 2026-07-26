"""Small exact-arithmetic smoke tests for the public implementation."""

import sympy as sp

from relu_bottleneck_tools import (
    ThreeLayerParams,
    analyze_parameter,
    apply_sign_flip,
    equivalence_test,
    translate_hidden,
    translate_linear_neuron,
)


def test_hiding_width_at_least_two() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, -1], [2, -3]]),
        b2=sp.Matrix([-1, -2]),
        W3=sp.Matrix([[1, 2]]),
        b3=sp.Matrix([0]),
        name="hiding_multi",
    )
    eta = translate_hidden(theta, hidden=0, t=sp.Integer(2))
    result = equivalence_test(theta, eta)
    assert result.accepted, result


def test_hiding_width_one_sign_flip() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(3),
        b1=sp.Matrix([0, 0, 0]),
        W2=sp.Matrix([[1, -1, -1]]),
        b2=sp.Matrix([-1]),
        W3=sp.Matrix([[2]]),
        b3=sp.Matrix([1]),
        name="hiding_single",
    )
    eta = apply_sign_flip(theta, hidden=0, flipped=[1])
    eta = translate_hidden(eta, hidden=0, t=sp.Rational(1, 2))
    result = equivalence_test(theta, eta)
    assert result.accepted, result


def test_linear_neuron_translation() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, 2], [1, -1]]),
        b2=sp.Matrix([1, -1]),
        W3=sp.Matrix([[2, 3]]),
        b3=sp.Matrix([0]),
        name="linear_neuron",
    )
    eta = translate_linear_neuron(theta, neuron=0, t=sp.Rational(3, 2))
    result = equivalence_test(theta, eta)
    assert result.accepted, result
    report = analyze_parameter(theta)
    assert 0 in report.linear_indices


def test_identifiable_sign_regime() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, -1], [-2, 1]]),
        b2=sp.Matrix([1, -1]),
        W3=sp.Matrix([[2, 3]]),
        b3=sp.Matrix([0]),
        name="identifiable",
    )
    report = analyze_parameter(theta)
    assert report.identifiable_under_genericity
    assert report.dimension.essential_dimension == 0


if __name__ == "__main__":
    test_hiding_width_at_least_two()
    test_hiding_width_one_sign_flip()
    test_linear_neuron_translation()
    test_identifiable_sign_regime()
    print("All exact-arithmetic smoke tests passed.")
