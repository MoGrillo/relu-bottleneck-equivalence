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


def test_zero_bias_classification_is_not_misreported() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[-1, -2], [1, -1]]),
        b2=sp.Matrix([0, 0]),
        W3=sp.Matrix([[1, 1]]),
        b3=sp.Matrix([0]),
        name="zero_bias_boundary",
    )
    report = analyze_parameter(theta)
    assert report.certified_nonidentifiable is True
    assert report.sign_criterion_applicable is False
    assert report.generic_identifiable_candidate is None
    assert report.identifiable_under_genericity is False


def test_boundary_without_certificate_is_undetermined() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, -1], [-1, 1]]),
        b2=sp.Matrix([0, 0]),
        W3=sp.Matrix([[1, 1]]),
        b3=sp.Matrix([0]),
        name="boundary_unknown",
    )
    report = analyze_parameter(theta)
    assert report.certified_nonidentifiable is False
    assert report.sign_criterion_applicable is False
    assert report.generic_identifiable_candidate is None
    assert report.identifiable_under_genericity is None


def test_zero_weight_hiding_translation_is_certified() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, 0]]),
        b2=sp.Matrix([-1]),
        W3=sp.Matrix([[1]]),
        b3=sp.Matrix([0]),
        name="weak_hiding_zero",
    )
    eta = translate_hidden(theta, hidden=0, t=sp.Integer(2))
    result = equivalence_test(theta, eta)
    assert result.accepted, result

    report = analyze_parameter(theta)
    assert report.certified_nonidentifiable is True
    assert report.sign_criterion_applicable is False
    assert report.generic_identifiable_candidate is None
    assert report.dimension.exact_under_assumptions is False
    assert report.dimension.essential_dimension is None
    assert report.dimension.lower_bound_essential >= 3
    assert not any("sign-flip" in symmetry for symmetry in report.discrete_symmetries)


def test_symmetry_helpers_enforce_or_normalize_their_domains() -> None:
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[2, -1]]),
        b2=sp.Matrix([-1]),
        W3=sp.Matrix([[1]]),
        b3=sp.Matrix([0]),
        name="scaled_sign_flip",
    )
    eta = apply_sign_flip(theta, hidden=0, flipped=[1])
    result = equivalence_test(theta, eta)
    assert result.accepted, result

    try:
        translate_hidden(theta, hidden=0, t=sp.Integer(-2))
    except ValueError:
        pass
    else:
        raise AssertionError("An out-of-domain hiding translation was accepted.")

    zero_coefficient = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, 0]]),
        b2=sp.Matrix([-1]),
        W3=sp.Matrix([[1]]),
        b3=sp.Matrix([0]),
    )
    try:
        apply_sign_flip(zero_coefficient, hidden=0, flipped=[1])
    except ValueError:
        pass
    else:
        raise AssertionError("A zero-weight coordinate was incorrectly treated as flippable.")


def test_layerwise_orientation_family_count() -> None:
    # Here r=n1=m=2 and there is one nonlinear second-layer neuron.
    # The paper's layerwise theorem makes both orientations feasible.
    theta = ThreeLayerParams(
        W1=sp.eye(2),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, -1], [1, 1], [2, 1]]),
        b2=sp.Matrix([-1, 1, 2]),
        W3=sp.Matrix([[1, 2, 3], [2, 3, 5]]),
        b3=sp.Matrix([0, 0]),
        name="all_orientation_families",
    )
    report = analyze_parameter(theta)
    assert report.dimension.essential_dimension == 4
    assert report.dimension.discrete_component_count == 2


def test_n1_one_is_outside_identifiability_theorem() -> None:
    theta = ThreeLayerParams(
        W1=sp.Matrix([[1]]),
        b1=sp.Matrix([0]),
        W2=sp.Matrix([[-1]]),
        b2=sp.Matrix([1]),
        W3=sp.Matrix([[1]]),
        b3=sp.Matrix([0]),
        name="n1_one",
    )
    report = analyze_parameter(theta)
    assert report.certified_nonidentifiable is False
    assert report.sign_criterion_applicable is False
    assert report.generic_identifiable_candidate is None
    assert report.identifiable_under_genericity is None


if __name__ == "__main__":
    test_hiding_width_at_least_two()
    test_hiding_width_one_sign_flip()
    test_linear_neuron_translation()
    test_identifiable_sign_regime()
    test_zero_bias_classification_is_not_misreported()
    test_boundary_without_certificate_is_undetermined()
    test_zero_weight_hiding_translation_is_certified()
    test_symmetry_helpers_enforce_or_normalize_their_domains()
    test_layerwise_orientation_family_count()
    test_n1_one_is_outside_identifiability_theorem()
    print("All exact-arithmetic smoke tests passed.")
