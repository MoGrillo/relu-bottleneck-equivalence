"""Small exact-arithmetic smoke tests for the public implementation."""

import sympy as sp

from relu_bottleneck_tools import (
    ThreeLayerParams,
    analyze_parameter,
    apply_sign_flip,
    classify_symmetries,
    equivalence_test,
    iid_symmetric_sign_candidate_rate,
    identifiability_rate_sweep,
    summarize_identifiability_rates,
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


def test_nonbottleneck_is_outside_identifiability_theorem() -> None:
    theta = ThreeLayerParams(
        W1=sp.Matrix([[1], [2]]),
        b1=sp.Matrix([0, 1]),
        W2=sp.Matrix([[1, -1]]),
        b2=sp.Matrix([1]),
        W3=sp.Matrix([[1]]),
        b3=sp.Matrix([0]),
        name="nonbottleneck",
    )
    report = analyze_parameter(theta)
    assert report.architecture.bottleneck is False
    assert report.certified_nonidentifiable is False
    assert report.sign_criterion_applicable is False
    assert report.generic_identifiable_candidate is None
    assert report.identifiable_under_genericity is None


def test_identifiability_rate_sweep() -> None:
    assert iid_symmetric_sign_candidate_rate(2, 1) == sp.Rational(1, 2)
    assert iid_symmetric_sign_candidate_rate(2, 2) == sp.Rational(17, 32)

    options = dict(
        dimensions=(2,),
        first_widths=(2, 3),
        second_widths=(1, 2),
        schemes=("xavier_uniform",),
        bias_modes=("small_uniform",),
        samples_per_setting=3,
        seed=19,
        max_denominator=10_000,
    )
    bottleneck_records = identifiability_rate_sweep(**options)
    assert len(bottleneck_records) == 6
    assert all(record["d"] == 2 and record["n1"] == 2 for record in bottleneck_records)
    assert bottleneck_records == identifiability_rate_sweep(**options)

    summaries = summarize_identifiability_rates(bottleneck_records)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["samples"] == 3
        assert (
            summary["generic_identifiable_candidate_count"]
            + summary["certified_nonidentifiable_count"]
            + summary["undetermined_count"]
            == summary["samples"]
        )
        assert (
            summary["generic_identifiable_candidate_rate"]
            + summary["certified_nonidentifiable_rate"]
            + summary["undetermined_rate"]
            == 1
        )

    all_records = identifiability_rate_sweep(
        **options,
        bottleneck_only=False,
    )
    assert len(all_records) == 12
    nonbottleneck = [record for record in all_records if record["n1"] == 3]
    assert nonbottleneck
    assert all(not record["sign_criterion_applicable"] for record in nonbottleneck)
    assert all(
        record["generic_identifiable_candidate"] is None
        for record in nonbottleneck
    )


def test_batch_warning_mode_preserves_mathematical_report() -> None:
    theta = ThreeLayerParams(
        W1=sp.Matrix([[1, 0], [2, 0]]),
        b1=sp.Matrix([0, 0]),
        W2=sp.Matrix([[1, -1], [-2, 1]]),
        b2=sp.Matrix([1, -1]),
        W3=sp.Matrix([[2, 3]]),
        b3=sp.Matrix([0]),
        name="batch_warning_mode",
    )
    default = classify_symmetries(theta)
    full = classify_symmetries(theta, check_first_layer_rank=True)
    batch = classify_symmetries(theta, check_first_layer_warnings=False)
    public = analyze_parameter(theta, check_first_layer_rank=True)
    for report in (default, batch, public):
        assert full.certified_nonidentifiable == report.certified_nonidentifiable
        assert full.sign_criterion_applicable == report.sign_criterion_applicable
        assert (
            full.generic_identifiable_candidate
            == report.generic_identifiable_candidate
        )
        assert full.identifiability_status == report.identifiability_status
        assert full.dimension == report.dimension
    assert any(
        "exact W1 rank diagnostic was skipped" in warning
        for warning in default.genericity_warnings
    )
    assert not any(
        "exact W1 rank diagnostic was skipped" in warning
        for warning in full.genericity_warnings
    )
    assert not any(
        "W1 does not have full row rank" in warning
        for warning in default.genericity_warnings
    )
    assert any(
        "W1 does not have full row rank" in warning
        for warning in full.genericity_warnings
    )
    assert any(
        "skipped for this batch sample" in warning
        for warning in batch.genericity_warnings
    )


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
    test_nonbottleneck_is_outside_identifiability_theorem()
    test_identifiability_rate_sweep()
    test_batch_warning_mode_preserves_mathematical_report()
    print("All exact-arithmetic smoke tests passed.")
