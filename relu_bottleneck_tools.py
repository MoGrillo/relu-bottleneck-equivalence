"""Exact-arithmetic tools for three-layer bottleneck ReLU networks.

The module implements:

* a sound functional-equivalence certificate that is complete under the
  paper's generic bottleneck assumptions;
* classification of the layerwise and hiding symmetries visible from the
  parameter signs;
* the generic-fiber dimension formulas from the paper, reported both modulo
  the standard scaling/permutation symmetries and in the full parameter space;
* rational, quantized versions of common Xavier/Glorot and He/Kaiming
  initialization schemes.

All algebraic decisions are made with SymPy exact arithmetic.  Rational inputs
are recommended.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sympy as sp

Scalar = sp.Expr
VectorKey = Tuple[Scalar, ...]


# ---------------------------------------------------------------------------
# Exact arithmetic helpers
# ---------------------------------------------------------------------------


def _M(x: Any) -> sp.Matrix:
    return x if isinstance(x, sp.MatrixBase) else sp.Matrix(x)


def _simplify_matrix(M: sp.MatrixBase) -> sp.Matrix:
    return sp.Matrix(M).applyfunc(sp.simplify)


def _is_zero(x: Scalar) -> bool:
    return sp.simplify(x) == 0


def _sign(x: Scalar) -> int:
    """Return the exact sign of an ordered real expression."""
    x = sp.simplify(x)
    if x == 0:
        return 0
    if x.is_positive:
        return 1
    if x.is_negative:
        return -1
    raise ValueError(
        f"Cannot determine the exact sign of {x!r}. "
        "Use rational or otherwise exactly ordered real inputs."
    )


def _matrix_equal(A: sp.MatrixBase, B: sp.MatrixBase) -> bool:
    return A.shape == B.shape and all(_is_zero(a - b) for a, b in zip(A, B))


def _tuple_equal(a: Sequence[Scalar], b: Sequence[Scalar]) -> bool:
    return len(a) == len(b) and all(_is_zero(x - y) for x, y in zip(a, b))


def _sort_key(values: Sequence[Scalar]) -> Tuple[Any, ...]:
    return tuple(sp.default_sort_key(sp.simplify(v)) for v in values)


def _first_nonzero(values: Sequence[Scalar]) -> Tuple[int, Scalar]:
    for i, value in enumerate(values):
        value = sp.simplify(value)
        if value != 0:
            return i, value
    raise ValueError("The zero vector has no projective normalization.")


def positive_projective_normalize(values: Sequence[Scalar]) -> Tuple[VectorKey, Scalar]:
    """Normalize modulo multiplication by a positive scalar.

    Returns ``(key, alpha)`` with ``original = alpha * key`` and ``alpha > 0``.
    The first nonzero coordinate of ``key`` is either +1 or -1, so opposite
    orientations remain distinct.
    """
    vals = tuple(sp.simplify(v) for v in values)
    _, pivot = _first_nonzero(vals)
    alpha = sp.Abs(pivot)
    if _sign(alpha) <= 0:
        raise ValueError("Positive projective normalization requires a nonzero pivot.")
    return tuple(sp.simplify(v / alpha) for v in vals), sp.simplify(alpha)


def unoriented_projective_normalize(values: Sequence[Scalar]) -> Tuple[VectorKey, Scalar]:
    """Normalize modulo multiplication by an arbitrary nonzero scalar.

    The first nonzero coordinate of the returned key is 1.  The returned scale
    records the original orientation: ``original = scale * key``.
    """
    vals = tuple(sp.simplify(v) for v in values)
    _, pivot = _first_nonzero(vals)
    return tuple(sp.simplify(v / pivot) for v in vals), sp.simplify(pivot)


def rationalize(value: float, max_denominator: int = 10_000) -> sp.Rational:
    """Quantize a floating-point sample to an exact rational number."""
    frac = Fraction(float(value)).limit_denominator(max_denominator)
    return sp.Rational(frac.numerator, frac.denominator)


# ---------------------------------------------------------------------------
# Parameter data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Architecture:
    d: int
    n1: int
    n2: int
    m: int

    def validate(self) -> None:
        if min(self.d, self.n1, self.n2, self.m) <= 0:
            raise ValueError("All architecture dimensions must be positive.")

    @property
    def bottleneck(self) -> bool:
        return self.n1 <= self.d


@dataclass(frozen=True)
class ThreeLayerParams:
    """Parameters of ``W3 ReLU(W2 ReLU(W1 x+b1)+b2)+b3``."""

    W1: sp.Matrix
    b1: sp.Matrix
    W2: sp.Matrix
    b2: sp.Matrix
    W3: sp.Matrix
    b3: sp.Matrix
    name: str = "parameter"

    def __post_init__(self) -> None:
        object.__setattr__(self, "W1", _M(self.W1))
        object.__setattr__(self, "b1", _M(self.b1))
        object.__setattr__(self, "W2", _M(self.W2))
        object.__setattr__(self, "b2", _M(self.b2))
        object.__setattr__(self, "W3", _M(self.W3))
        object.__setattr__(self, "b3", _M(self.b3))
        self.validate()

    @property
    def d(self) -> int:
        return self.W1.cols

    @property
    def n1(self) -> int:
        return self.W1.rows

    @property
    def n2(self) -> int:
        return self.W2.rows

    @property
    def m(self) -> int:
        return self.W3.rows

    @property
    def architecture(self) -> Architecture:
        return Architecture(self.d, self.n1, self.n2, self.m)

    @property
    def parameter_dimension(self) -> int:
        return (
            self.n1 * self.d
            + self.n1
            + self.n2 * self.n1
            + self.n2
            + self.m * self.n2
            + self.m
        )

    def validate(self) -> None:
        if self.b1.shape != (self.W1.rows, 1):
            raise ValueError("b1 must have shape (n1, 1).")
        if self.W2.cols != self.W1.rows:
            raise ValueError("W2 must have n1 columns.")
        if self.b2.shape != (self.W2.rows, 1):
            raise ValueError("b2 must have shape (n2, 1).")
        if self.W3.cols != self.W2.rows:
            raise ValueError("W3 must have n2 columns.")
        if self.b3.shape != (self.W3.rows, 1):
            raise ValueError("b3 must have shape (m, 1).")

    def evaluate(self, x: Sequence[Scalar]) -> sp.Matrix:
        x_vec = sp.Matrix(x)
        if x_vec.shape != (self.d, 1):
            raise ValueError(f"Expected x of shape ({self.d}, 1).")
        a1 = (self.W1 * x_vec + self.b1).applyfunc(lambda z: sp.Max(z, 0))
        a2 = (self.W2 * a1 + self.b2).applyfunc(lambda z: sp.Max(z, 0))
        return _simplify_matrix(self.W3 * a2 + self.b3)

    def permute_first(self, order: Sequence[int]) -> "ThreeLayerParams":
        order = list(order)
        return replace(
            self,
            W1=self.W1[order, :],
            b1=self.b1[order, :],
            W2=self.W2[:, order],
        )

    def permute_second(self, order: Sequence[int]) -> "ThreeLayerParams":
        order = list(order)
        return replace(
            self,
            W2=self.W2[order, :],
            b2=self.b2[order, :],
            W3=self.W3[:, order],
        )

    def scale_first(self, scales: Sequence[Scalar]) -> "ThreeLayerParams":
        if len(scales) != self.n1:
            raise ValueError("Need one positive scale per first-layer neuron.")
        scales = [sp.simplify(s) for s in scales]
        if any(_sign(s) <= 0 for s in scales):
            raise ValueError("First-layer scaling factors must be positive.")
        D = sp.diag(*scales)
        return replace(self, W1=D * self.W1, b1=D * self.b1, W2=self.W2 * D.inv())

    def scale_second(self, scales: Sequence[Scalar]) -> "ThreeLayerParams":
        if len(scales) != self.n2:
            raise ValueError("Need one positive scale per second-layer neuron.")
        scales = [sp.simplify(s) for s in scales]
        if any(_sign(s) <= 0 for s in scales):
            raise ValueError("Second-layer scaling factors must be positive.")
        D = sp.diag(*scales)
        return replace(self, W2=D * self.W2, b2=D * self.b2, W3=self.W3 * D.inv())


@dataclass(frozen=True)
class EquivalenceResult:
    accepted: bool
    case: str
    reason: str
    diagnostics: Mapping[str, Any]

    @property
    def interpretation(self) -> str:
        if self.accepted:
            return "Certified equivalent; sound for arbitrary exact parameters."
        return "Not certified; conclusive only under the generic bottleneck assumptions."


@dataclass(frozen=True)
class HyperplaneRepresentation:
    hyperplanes: Tuple[VectorKey, ...]
    weights: Mapping[VectorKey, Tuple[Scalar, ...]]
    base_point: Tuple[Scalar, ...]
    affine_matrix: sp.Matrix
    affine_bias: sp.Matrix


@dataclass(frozen=True)
class FiberDimensionReport:
    essential_dimension: Optional[int]
    full_dimension: Optional[int]
    standard_scaling_dimension: int
    discrete_component_count: Optional[int]
    formula_case: str
    exact_under_assumptions: bool
    lower_bound_essential: Optional[int]
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SymmetryReport:
    parameter_name: str
    architecture: Architecture
    second_layer_types: Tuple[str, ...]
    dead_indices: Tuple[int, ...]
    linear_indices: Tuple[int, ...]
    nonlinear_indices: Tuple[int, ...]
    strict_hiding_indices: Tuple[int, ...]
    weak_hiding_indices: Tuple[int, ...]
    reduced_hiding_index: Optional[int]
    continuous_symmetries: Tuple[str, ...]
    discrete_symmetries: Tuple[str, ...]
    generic_sign_identifiable: bool
    genericity_warnings: Tuple[str, ...]
    dimension: FiberDimensionReport

    @property
    def identifiable_under_genericity(self) -> bool:
        return self.generic_sign_identifiable

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["architecture"] = asdict(self.architecture)
        data["dimension"] = self.dimension.to_dict()
        return data


# ---------------------------------------------------------------------------
# Orthant neuron classification and hiding patterns
# ---------------------------------------------------------------------------


def classify_orthant_affine_neuron(w: Sequence[Scalar], b: Scalar) -> str:
    """Classify ReLU(w y+b) on the full nonnegative orthant."""
    signs = [_sign(v) for v in w]
    sb = _sign(b)
    if all(s <= 0 for s in signs) and sb <= 0:
        return "dead"
    if all(s >= 0 for s in signs) and sb >= 0:
        return "linear"
    return "nonlinear"


def second_layer_types(p: ThreeLayerParams) -> Tuple[str, ...]:
    return tuple(
        classify_orthant_affine_neuron(tuple(p.W2.row(i)), p.b2[i, 0])
        for i in range(p.n2)
    )


def remove_dead_second_layer(p: ThreeLayerParams) -> Tuple[ThreeLayerParams, Tuple[int, ...]]:
    kinds = second_layer_types(p)
    keep = [i for i, kind in enumerate(kinds) if kind != "dead"]
    removed = tuple(i for i, kind in enumerate(kinds) if kind == "dead")
    if not removed:
        return p, tuple()
    return (
        replace(p, W2=p.W2[keep, :], b2=p.b2[keep, :], W3=p.W3[:, keep]),
        removed,
    )


def hiding_indices(p: ThreeLayerParams, strict: bool = False) -> Tuple[int, ...]:
    """Return indices j hidden by every second-layer neuron.

    With ``strict=False`` this uses the weak inequalities from the symmetry
    definition.  With ``strict=True`` it uses the strict generic sign pattern.
    """
    if p.n2 == 0:
        return tuple()
    result: List[int] = []
    for j in range(p.n1):
        ok = True
        for i in range(p.n2):
            sj = _sign(p.W2[i, j])
            sb = _sign(p.b2[i, 0])
            if sj <= 0 or (sb >= 0 if strict else sb > 0):
                ok = False
                break
            for k in range(p.n1):
                if k == j:
                    continue
                sk = _sign(p.W2[i, k])
                if sk >= 0 if strict else sk > 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            result.append(j)
    return tuple(result)


def hiding_index(p: ThreeLayerParams) -> Optional[int]:
    candidates = hiding_indices(p, strict=False)
    return candidates[0] if len(candidates) == 1 else None


def _strict_all_positive_row(p: ThreeLayerParams, i: int) -> bool:
    return all(_sign(v) > 0 for v in tuple(p.W2.row(i)) + (p.b2[i, 0],))


def _strict_all_negative_row(p: ThreeLayerParams, i: int) -> bool:
    return all(_sign(v) < 0 for v in tuple(p.W2.row(i)) + (p.b2[i, 0],))


# ---------------------------------------------------------------------------
# Hyperplane certificate for the non-hiding branch
# ---------------------------------------------------------------------------


def _canonicalize_first_nonhiding(p: ThreeLayerParams) -> ThreeLayerParams:
    rows: List[Tuple[VectorKey, int, Scalar]] = []
    for i in range(p.n1):
        affine = tuple(p.W1.row(i)) + (p.b1[i, 0],)
        key, alpha = positive_projective_normalize(affine)
        rows.append((key, i, alpha))
    rows.sort(key=lambda item: _sort_key(item[0]))

    W1_rows: List[List[Scalar]] = []
    b1_vals: List[Scalar] = []
    W2_cols: List[sp.Matrix] = []
    for key, i, alpha in rows:
        W1_rows.append(list(key[:-1]))
        b1_vals.append(key[-1])
        W2_cols.append(p.W2[:, i] * alpha)
    return replace(
        p,
        W1=sp.Matrix(W1_rows),
        b1=sp.Matrix(b1_vals),
        W2=sp.Matrix.hstack(*W2_cols),
    )


def _base_point_avoiding(hyperplanes: Sequence[VectorKey], dimension: int) -> Tuple[Scalar, ...]:
    if not hyperplanes:
        return tuple(sp.Integer(1) for _ in range(dimension))
    max_trials = max(12, len(hyperplanes) * max(1, dimension) + 4)
    for t in range(1, max_trials + 1):
        point = tuple(sp.Integer(t) ** (k + 1) for k in range(dimension))
        if all(
            not _is_zero(sum(h[k] * point[k] for k in range(dimension)) + h[-1])
            for h in hyperplanes
        ):
            return point
    raise RuntimeError("Could not find a positive rational base point away from all hyperplanes.")


def shallow_hyperplane_representation(
    W_in: sp.Matrix,
    b_in: sp.Matrix,
    W_out: sp.Matrix,
    b_out: sp.Matrix,
) -> HyperplaneRepresentation:
    """Compute a rational weighted-hyperplane certificate on y >= 0."""
    d = W_in.cols
    m = W_out.rows
    groups: Dict[VectorKey, List[Tuple[int, Scalar]]] = {}
    for i in range(W_in.rows):
        kind = classify_orthant_affine_neuron(tuple(W_in.row(i)), b_in[i, 0])
        if kind != "nonlinear":
            continue
        affine = tuple(W_in.row(i)) + (b_in[i, 0],)
        key, scale = unoriented_projective_normalize(affine)
        groups.setdefault(key, []).append((i, scale))

    hyperplanes = tuple(sorted(groups, key=_sort_key))
    weights: Dict[VectorKey, Tuple[Scalar, ...]] = {}
    for key in hyperplanes:
        total = sp.zeros(m, 1)
        for i, scale in groups[key]:
            total += W_out[:, i] * sp.Abs(scale)
        weights[key] = tuple(sp.simplify(v) for v in total)

    base = _base_point_avoiding(hyperplanes, d)
    y = sp.Matrix(base)
    active = [
        i
        for i in range(W_in.rows)
        if _sign(sp.simplify((W_in.row(i) * y)[0] + b_in[i, 0])) > 0
    ]
    if active:
        A = W_out[:, active] * W_in[active, :]
        b = W_out[:, active] * b_in[active, :] + b_out
    else:
        A = sp.zeros(m, d)
        b = b_out
    return HyperplaneRepresentation(
        hyperplanes=hyperplanes,
        weights=weights,
        base_point=base,
        affine_matrix=_simplify_matrix(A),
        affine_bias=_simplify_matrix(b),
    )


def compare_hyperplane_representations(
    left: HyperplaneRepresentation,
    right: HyperplaneRepresentation,
) -> Tuple[bool, str]:
    if left.hyperplanes != right.hyperplanes:
        return False, "The induced breakpoint-hyperplane multisets differ."
    for key in left.hyperplanes:
        if not _tuple_equal(left.weights[key], right.weights[key]):
            return False, f"The aggregate gradient-jump weight differs on {key}."
    if left.base_point != right.base_point:
        return False, "Internal error: different base points were selected."
    if not _matrix_equal(left.affine_matrix, right.affine_matrix):
        return False, "The affine linear parts differ on the common base region."
    if not _matrix_equal(left.affine_bias, right.affine_bias):
        return False, "The affine biases differ on the common base region."
    return True, "The induced shallow maps have the same hyperplane representation."


def _nonhiding_certificate(theta: ThreeLayerParams, eta: ThreeLayerParams) -> EquivalenceResult:
    try:
        th = _canonicalize_first_nonhiding(theta)
        et = _canonicalize_first_nonhiding(eta)
    except ValueError as exc:
        return EquivalenceResult(False, "non-hiding", str(exc), {})

    if not _matrix_equal(th.W1, et.W1) or not _matrix_equal(th.b1, et.b1):
        return EquivalenceResult(
            False,
            "non-hiding",
            "The first-layer affine rows do not match up to permutation and positive scaling.",
            {},
        )
    rep_th = shallow_hyperplane_representation(th.W2, th.b2, th.W3, th.b3)
    rep_et = shallow_hyperplane_representation(et.W2, et.b2, et.W3, et.b3)
    ok, reason = compare_hyperplane_representations(rep_th, rep_et)
    return EquivalenceResult(
        ok,
        "non-hiding",
        reason,
        {"theta_representation": rep_th, "eta_representation": rep_et},
    )


# ---------------------------------------------------------------------------
# Hiding certificates
# ---------------------------------------------------------------------------


def _canonicalize_hiding_multi(p: ThreeLayerParams, hidden: int) -> ThreeLayerParams:
    if p.n2 < 2:
        raise ValueError("This normalization requires reduced second-layer width at least two.")

    entries: List[Tuple[VectorKey, int, Scalar, bool]] = []
    for k in range(p.n1):
        if k == hidden:
            key, alpha = positive_projective_normalize(tuple(p.W1.row(k)))
            entries.append((key, k, alpha, True))
        else:
            affine = tuple(p.W1.row(k)) + (p.b1[k, 0],)
            key, alpha = positive_projective_normalize(affine)
            entries.append((key, k, alpha, False))

    hidden_entry = next(e for e in entries if e[3])
    nonhidden = sorted((e for e in entries if not e[3]), key=lambda e: _sort_key(e[0]))
    ordered = [hidden_entry] + nonhidden

    W1_rows: List[List[Scalar]] = []
    b1_vals: List[Scalar] = []
    W2_cols: List[sp.Matrix] = []
    for key, old, alpha, is_hidden in ordered:
        if is_hidden:
            W1_rows.append(list(key))
            b1_vals.append(sp.simplify(p.b1[old, 0] / alpha))
        else:
            W1_rows.append(list(key[:-1]))
            b1_vals.append(key[-1])
        W2_cols.append(p.W2[:, old] * alpha)

    q = replace(
        p,
        W1=sp.Matrix(W1_rows),
        b1=sp.Matrix(b1_vals),
        W2=sp.Matrix.hstack(*W2_cols),
    )

    W2_rows: List[List[Scalar]] = []
    b2_vals: List[Scalar] = []
    W3_cols: List[sp.Matrix] = []
    row_keys: List[Tuple[Tuple[Any, ...], int]] = []
    for i in range(q.n2):
        c = sp.simplify(q.W2[i, 0])
        if _sign(c) <= 0:
            raise ValueError("The alleged hidden coefficient is not positive.")
        row = [sp.simplify(v / c) for v in q.W2.row(i)]
        b = sp.simplify(q.b2[i, 0] / c)
        out = q.W3[:, i] * c
        W2_rows.append(row)
        b2_vals.append(b)
        W3_cols.append(out)
        row_keys.append((_sort_key(tuple(row) + tuple(out)), i))

    order = [i for _, i in sorted(row_keys, key=lambda x: x[0])]
    return replace(
        q,
        W2=sp.Matrix([W2_rows[i] for i in order]),
        b2=sp.Matrix([b2_vals[i] for i in order]),
        W3=sp.Matrix.hstack(*[W3_cols[i] for i in order]),
    )


def _hiding_multi_certificate(
    theta: ThreeLayerParams,
    eta: ThreeLayerParams,
    j_theta: int,
    j_eta: int,
) -> EquivalenceResult:
    try:
        th = _canonicalize_hiding_multi(theta, j_theta)
        et = _canonicalize_hiding_multi(eta, j_eta)
    except ValueError as exc:
        return EquivalenceResult(False, "hiding, width >= 2", str(exc), {})

    if not _matrix_equal(th.W1, et.W1):
        return EquivalenceResult(False, "hiding, width >= 2", "First-layer directions differ.", {})
    if th.n1 > 1 and not _matrix_equal(th.b1[1:, :], et.b1[1:, :]):
        return EquivalenceResult(False, "hiding, width >= 2", "A non-hidden hyperplane differs.", {})
    if not _matrix_equal(th.W2, et.W2):
        return EquivalenceResult(False, "hiding, width >= 2", "Normalized W2 differs.", {})
    if not _matrix_equal(th.W3, et.W3) or not _matrix_equal(th.b3, et.b3):
        return EquivalenceResult(False, "hiding, width >= 2", "Normalized output parameters differ.", {})

    t = sp.simplify(et.b1[0, 0] - th.b1[0, 0])
    expected_b2 = _simplify_matrix(th.b2 - th.W2[:, 0] * t)
    if not _matrix_equal(et.b2, expected_b2):
        return EquivalenceResult(
            False,
            "hiding, width >= 2",
            "The bias difference is not the hidden-hyperplane translation.",
            {"candidate_t": t, "expected_eta_b2": expected_b2},
        )
    return EquivalenceResult(
        True,
        "hiding, width >= 2",
        "The parameters differ by the hidden-hyperplane translation symmetry.",
        {"translation_t": t},
    )


def sign_flip_matrix(n1: int, hidden: int, flipped: Iterable[int]) -> sp.Matrix:
    M = sp.eye(n1)
    for k in flipped:
        if k == hidden:
            raise ValueError("The hidden neuron cannot be flipped.")
        M[hidden, k] -= 1
        M[k, k] -= 2
    return M


def apply_sign_flip(p: ThreeLayerParams, hidden: int, flipped: Iterable[int]) -> ThreeLayerParams:
    M = sign_flip_matrix(p.n1, hidden, flipped)
    return replace(p, W1=M * p.W1, b1=M * p.b1)


def _normalize_single_hiding(p: ThreeLayerParams, hidden: int) -> ThreeLayerParams:
    if p.n2 != 1:
        raise ValueError("Expected reduced second-layer width one.")
    out = tuple(p.W3[:, 0])
    _, out_alpha = positive_projective_normalize(out)
    q = p.scale_second([out_alpha])

    scales: List[Scalar] = []
    for k in range(q.n1):
        coeff = sp.simplify(q.W2[0, k])
        if _sign(coeff) == 0:
            raise ValueError("Width-one normalization requires nonzero W2 entries.")
        scales.append(sp.Abs(coeff))
    q = q.scale_first(scales)

    records: List[Tuple[VectorKey, int]] = []
    for k in range(q.n1):
        if k == hidden:
            continue
        affine = tuple(q.W1.row(k)) + (q.b1[k, 0],)
        key, _ = unoriented_projective_normalize(affine)
        records.append((key, k))
    records.sort(key=lambda item: _sort_key(item[0]))
    return q.permute_first([hidden] + [k for _, k in records])


def _hiding_single_certificate(
    theta: ThreeLayerParams,
    eta: ThreeLayerParams,
    j_theta: int,
    j_eta: int,
) -> EquivalenceResult:
    try:
        th = _normalize_single_hiding(theta, j_theta)
        et = _normalize_single_hiding(eta, j_eta)
    except ValueError as exc:
        return EquivalenceResult(False, "hiding, width = 1", str(exc), {})

    if not _matrix_equal(th.W2, et.W2):
        return EquivalenceResult(False, "hiding, width = 1", "Normalized W2 signs differ.", {})
    if not _matrix_equal(th.W3, et.W3) or not _matrix_equal(th.b3, et.b3):
        return EquivalenceResult(False, "hiding, width = 1", "Output parameters differ.", {})

    flipped: List[int] = []
    for k in range(1, th.n1):
        row_th = tuple(th.W1.row(k)) + (th.b1[k, 0],)
        row_et = tuple(et.W1.row(k)) + (et.b1[k, 0],)
        key_th, scale_th = unoriented_projective_normalize(row_th)
        key_et, scale_et = unoriented_projective_normalize(row_et)
        if key_th != key_et:
            return EquivalenceResult(False, "hiding, width = 1", "A non-hidden hyperplane differs.", {})
        if _sign(scale_et / scale_th) < 0:
            flipped.append(k)

    transformed = apply_sign_flip(th, hidden=0, flipped=flipped)
    if not _matrix_equal(transformed.W1, et.W1):
        return EquivalenceResult(
            False,
            "hiding, width = 1",
            "The recovered sign flip does not reproduce the hidden direction.",
            {"flipped_indices_in_canonical_order": tuple(flipped)},
        )
    if th.n1 > 1 and not _matrix_equal(transformed.b1[1:, :], et.b1[1:, :]):
        return EquivalenceResult(False, "hiding, width = 1", "Non-hidden biases differ.", {})

    t = sp.simplify(et.b1[0, 0] - transformed.b1[0, 0])
    expected_b2 = _simplify_matrix(transformed.b2 - transformed.W2[:, 0] * t)
    if not _matrix_equal(et.b2, expected_b2):
        return EquivalenceResult(
            False,
            "hiding, width = 1",
            "After the sign flip, the remaining difference is not a translation.",
            {"candidate_t": t, "flipped_indices_in_canonical_order": tuple(flipped)},
        )
    return EquivalenceResult(
        True,
        "hiding, width = 1",
        "Recovered sign flip followed by a hidden-hyperplane translation.",
        {"translation_t": t, "flipped_indices_in_canonical_order": tuple(flipped)},
    )


# ---------------------------------------------------------------------------
# Public equivalence transformations and dispatcher
# ---------------------------------------------------------------------------


def translate_hidden(p: ThreeLayerParams, hidden: int, t: Scalar) -> ThreeLayerParams:
    b1 = p.b1.copy()
    b1[hidden, 0] = sp.simplify(b1[hidden, 0] + t)
    b2 = _simplify_matrix(p.b2 - p.W2[:, hidden] * t)
    return replace(p, b1=b1, b2=b2)


def translate_linear_neuron(p: ThreeLayerParams, neuron: int, t: Scalar) -> ThreeLayerParams:
    """Shift the bias of a second-layer neuron that is linear on the orthant."""
    if classify_orthant_affine_neuron(tuple(p.W2.row(neuron)), p.b2[neuron, 0]) != "linear":
        raise ValueError("The selected second-layer neuron is not linear on the orthant.")
    new_bias = sp.simplify(p.b2[neuron, 0] + t)
    if classify_orthant_affine_neuron(tuple(p.W2.row(neuron)), new_bias) != "linear":
        raise ValueError("The requested shift leaves the region where the neuron is linear.")
    b2 = p.b2.copy()
    b2[neuron, 0] = new_bias
    b3 = _simplify_matrix(p.b3 - p.W3[:, neuron] * t)
    return replace(p, b2=b2, b3=b3)


def equivalence_test(theta: ThreeLayerParams, eta: ThreeLayerParams) -> EquivalenceResult:
    """Sound equivalence certificate; complete under generic bottleneck assumptions."""
    if theta.architecture != eta.architecture:
        return EquivalenceResult(False, "input validation", "The architectures differ.", {})
    if theta.n1 > theta.d:
        return EquivalenceResult(False, "input validation", "The architecture is not bottleneck.", {})

    th, removed_th = remove_dead_second_layer(theta)
    et, removed_et = remove_dead_second_layer(eta)

    if th.n2 == 0 or et.n2 == 0:
        accepted = th.n2 == 0 and et.n2 == 0 and _matrix_equal(th.b3, et.b3)
        return EquivalenceResult(
            accepted,
            "constant after dead-neuron removal",
            "Both networks reduce to the same constant." if accepted else "The reduced constants differ.",
            {"removed_theta": removed_th, "removed_eta": removed_et},
        )

    j_th = hiding_index(th)
    j_et = hiding_index(et)
    if (j_th is None) != (j_et is None):
        return EquivalenceResult(
            False,
            "case distinction",
            "One reduced parameter is hiding and the other is non-hiding.",
            {"removed_theta": removed_th, "removed_eta": removed_et},
        )

    if j_th is None and j_et is None:
        result = _nonhiding_certificate(th, et)
    elif th.n2 != et.n2:
        result = EquivalenceResult(False, "hiding", "Reduced second-layer widths differ.", {})
    elif th.n2 >= 2:
        result = _hiding_multi_certificate(th, et, j_th, j_et)  # type: ignore[arg-type]
    else:
        result = _hiding_single_certificate(th, et, j_th, j_et)  # type: ignore[arg-type]

    diagnostics = dict(result.diagnostics)
    diagnostics.update({"removed_theta": removed_th, "removed_eta": removed_et})
    return EquivalenceResult(result.accepted, result.case, result.reason, diagnostics)


def sample_grid_equal(
    theta: ThreeLayerParams,
    eta: ThreeLayerParams,
    points: Sequence[Sequence[Scalar]],
) -> bool:
    """Diagnostic only; never used as an equivalence proof."""
    return all(_matrix_equal(theta.evaluate(x), eta.evaluate(x)) for x in points)


# ---------------------------------------------------------------------------
# Generic-fiber dimension from the paper's characterization
# ---------------------------------------------------------------------------


def _active_generic_dimension(p: ThreeLayerParams) -> FiberDimensionReport:
    """Dimension for a parameter with no dead second-layer neurons."""
    standard = p.n1 + p.n2
    weak_hiding = hiding_indices(p, strict=False)
    if len(weak_hiding) == 1:
        components = 2 ** (p.n1 - 1) if p.n2 == 1 else 1
        return FiberDimensionReport(
            essential_dimension=1,
            full_dimension=standard + 1,
            standard_scaling_dimension=standard,
            discrete_component_count=components,
            formula_case="hiding: one translation degree; finite sign flips when n2=1",
            exact_under_assumptions=True,
            lower_bound_essential=1,
            notes=(
                "The dimension is computed modulo the standard positive scalings and permutations.",
                "For n2=1 the sign flips change the number of components but not their dimension.",
            ),
        )

    kinds = second_layer_types(p)
    linear_count = sum(kind == "linear" for kind in kinds)
    shallow_d = p.n1
    m = p.m

    if linear_count <= min(shallow_d, m):
        essential = linear_count ** 2
        case = f"layerwise theorem: r={linear_count} <= min(n1,m), dimension r^2"
    elif linear_count == m + 1 and linear_count <= shallow_d:
        essential = m * m + m + shallow_d
        case = "layerwise theorem: r=m+1<=n1, dimension m^2+m+n1"
    else:
        return FiberDimensionReport(
            essential_dimension=None,
            full_dimension=None,
            standard_scaling_dimension=standard,
            discrete_component_count=None,
            formula_case="outside the generic minimal cases of the layerwise dimension theorem",
            exact_under_assumptions=False,
            lower_bound_essential=linear_count,
            notes=(
                "The parameter has more linear neurons than allowed by the generic minimality bound.",
                "The displayed lower bound comes from independent linear-neuron bias shifts.",
            ),
        )

    return FiberDimensionReport(
        essential_dimension=essential,
        full_dimension=standard + essential,
        standard_scaling_dimension=standard,
        discrete_component_count=1,
        formula_case=case,
        exact_under_assumptions=True,
        lower_bound_essential=essential,
        notes=(
            "This is the dimension of the component containing the parameter.",
            "The essential dimension is modulo positive scaling and permutation symmetries.",
        ),
    )


def generic_fiber_dimension(p: ThreeLayerParams) -> FiberDimensionReport:
    """Compute the generic-fiber dimension predicted by the paper.

    The routine returns both the essential dimension modulo standard hidden-neuron
    scaling/permutation symmetries and the full parameter-space dimension obtained
    by adding the continuous scaling orbit.  Dead neurons are removed first; their
    independent free parameters are then added back.
    """
    if p.n1 > p.d:
        return FiberDimensionReport(
            essential_dimension=None,
            full_dimension=None,
            standard_scaling_dimension=p.n1 + p.n2,
            discrete_component_count=None,
            formula_case="non-bottleneck architecture",
            exact_under_assumptions=False,
            lower_bound_essential=None,
            notes=("The complete characterization used here requires n1<=d.",),
        )

    reduced, dead = remove_dead_second_layer(p)
    q = len(dead)
    standard = p.n1 + p.n2

    if reduced.n2 == 0:
        full = p.parameter_dimension - p.m
        essential = full - standard
        return FiberDimensionReport(
            essential_dimension=essential,
            full_dimension=full,
            standard_scaling_dimension=standard,
            discrete_component_count=1,
            formula_case="constant function: all second-layer neurons are dead",
            exact_under_assumptions=True,
            lower_bound_essential=essential,
            notes=("The output bias is fixed; all remaining parameters are functionally invisible.",),
        )

    active = _active_generic_dimension(reduced)
    if active.full_dimension is None or active.essential_dimension is None:
        return replace(
            active,
            standard_scaling_dimension=standard,
            notes=active.notes + (f"Additionally removed {q} dead second-layer neurons.",),
        )

    # Each strictly dead neuron contributes an open set of n1+1 incoming/bias
    # parameters and m arbitrary outgoing parameters.  Modulo its own positive
    # scaling, this contributes n1+m essential dimensions.
    dead_full = q * (p.n1 + 1 + p.m)
    dead_essential = q * (p.n1 + p.m)
    return FiberDimensionReport(
        essential_dimension=active.essential_dimension + dead_essential,
        full_dimension=active.full_dimension + dead_full,
        standard_scaling_dimension=standard,
        discrete_component_count=active.discrete_component_count,
        formula_case=active.formula_case + (f" + {q} dead-neuron free block(s)" if q else ""),
        exact_under_assumptions=active.exact_under_assumptions,
        lower_bound_essential=(active.lower_bound_essential or 0) + dead_essential,
        notes=active.notes
        + ((f"Each of the {q} dead neurons contributes n1+m essential dimensions.",) if q else tuple()),
    )


# ---------------------------------------------------------------------------
# Symmetry classification and identifiability
# ---------------------------------------------------------------------------


def _genericity_warnings(p: ThreeLayerParams) -> Tuple[str, ...]:
    warnings: List[str] = []
    if p.n1 > p.d:
        warnings.append("The architecture is not bottleneck.")
    if p.W1.rank() < p.n1:
        warnings.append("W1 does not have full row rank.")
    if any(_is_zero(v) for v in p.W2) or any(_is_zero(v) for v in p.b2):
        warnings.append("W2 or b2 contains zero entries; the parameter lies on sign boundaries.")
    if any(all(_is_zero(v) for v in p.W1.row(i)) and _is_zero(p.b1[i, 0]) for i in range(p.n1)):
        warnings.append("A first-layer affine row is zero.")
    affine_keys: List[VectorKey] = []
    for i in range(p.n1):
        try:
            key, _ = unoriented_projective_normalize(tuple(p.W1.row(i)) + (p.b1[i, 0],))
            if key in affine_keys:
                warnings.append("Two first-layer affine hyperplanes coincide.")
                break
            affine_keys.append(key)
        except ValueError:
            pass
    warnings.append(
        "The full transversality/masked-product genericity condition is not checked automatically."
    )
    return tuple(warnings)


def classify_symmetries(p: ThreeLayerParams) -> SymmetryReport:
    kinds = second_layer_types(p)
    dead = tuple(i for i, kind in enumerate(kinds) if kind == "dead")
    linear = tuple(i for i, kind in enumerate(kinds) if kind == "linear")
    nonlinear = tuple(i for i, kind in enumerate(kinds) if kind == "nonlinear")
    strict_hiding = hiding_indices(p, strict=True)
    weak_hiding = hiding_indices(p, strict=False)
    reduced, _ = remove_dead_second_layer(p)
    reduced_hiding = hiding_index(reduced) if reduced.n2 else None

    continuous: List[str] = [
        f"standard positive rescaling of {p.n1 + p.n2} hidden neurons",
    ]
    discrete: List[str] = [
        "permutation of neurons within each hidden layer",
    ]
    if dead:
        continuous.append(
            f"free variation of {len(dead)} dead second-layer neuron block(s)"
        )
    if linear:
        continuous.append(
            f"layerwise linear-neuron factorization/bias symmetries for {len(linear)} neuron(s)"
        )
    if reduced_hiding is not None:
        continuous.append(
            f"translation of hidden first-layer hyperplane {reduced_hiding}"
        )
        if reduced.n2 == 1:
            discrete.append(
                f"2^{p.n1 - 1} width-one sign-flip choices associated with hyperplane {reduced_hiding}"
            )

    generic_nonidentifiable = (
        any(_strict_all_negative_row(p, i) for i in range(p.n2))
        or any(_strict_all_positive_row(p, i) for i in range(p.n2))
        or bool(strict_hiding)
    )

    return SymmetryReport(
        parameter_name=p.name,
        architecture=p.architecture,
        second_layer_types=kinds,
        dead_indices=dead,
        linear_indices=linear,
        nonlinear_indices=nonlinear,
        strict_hiding_indices=strict_hiding,
        weak_hiding_indices=weak_hiding,
        reduced_hiding_index=reduced_hiding,
        continuous_symmetries=tuple(continuous),
        discrete_symmetries=tuple(discrete),
        generic_sign_identifiable=not generic_nonidentifiable,
        genericity_warnings=_genericity_warnings(p),
        dimension=generic_fiber_dimension(p),
    )


def analyze_parameter(p: ThreeLayerParams) -> SymmetryReport:
    """User-facing alias for ``classify_symmetries``."""
    return classify_symmetries(p)


# ---------------------------------------------------------------------------
# Rational common initializations
# ---------------------------------------------------------------------------


def _sample_matrix(
    rows: int,
    cols: int,
    scheme: str,
    fan_in: int,
    fan_out: int,
    rng: random.Random,
    max_denominator: int,
) -> sp.Matrix:
    if scheme == "xavier_uniform":
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        draw = lambda: rng.uniform(-bound, bound)
    elif scheme == "he_uniform":
        bound = math.sqrt(6.0 / fan_in)
        draw = lambda: rng.uniform(-bound, bound)
    elif scheme == "xavier_normal":
        std = math.sqrt(2.0 / (fan_in + fan_out))
        draw = lambda: rng.gauss(0.0, std)
    elif scheme == "he_normal":
        std = math.sqrt(2.0 / fan_in)
        draw = lambda: rng.gauss(0.0, std)
    else:
        raise ValueError(
            "scheme must be one of xavier_uniform, xavier_normal, he_uniform, he_normal"
        )
    return sp.Matrix(
        rows,
        cols,
        lambda _i, _j: rationalize(draw(), max_denominator=max_denominator),
    )


def _sample_bias(
    size: int,
    mode: str,
    rng: random.Random,
    max_denominator: int,
    scale: float = 0.1,
) -> sp.Matrix:
    if mode == "zero":
        return sp.zeros(size, 1)
    if mode == "small_uniform":
        return sp.Matrix(
            [
                rationalize(rng.uniform(-scale, scale), max_denominator=max_denominator)
                for _ in range(size)
            ]
        )
    if mode == "positive_small":
        return sp.Matrix(
            [
                rationalize(rng.uniform(0.0, scale), max_denominator=max_denominator)
                for _ in range(size)
            ]
        )
    raise ValueError("bias_mode must be zero, small_uniform, or positive_small")


def sample_rational_initialization(
    architecture: Architecture,
    scheme: str = "xavier_uniform",
    bias_mode: str = "zero",
    seed: Optional[int] = None,
    max_denominator: int = 10_000,
    name: Optional[str] = None,
) -> ThreeLayerParams:
    """Return a quantized rational version of a common initialization."""
    architecture.validate()
    rng = random.Random(seed)
    W1 = _sample_matrix(
        architecture.n1,
        architecture.d,
        scheme,
        architecture.d,
        architecture.n1,
        rng,
        max_denominator,
    )
    W2 = _sample_matrix(
        architecture.n2,
        architecture.n1,
        scheme,
        architecture.n1,
        architecture.n2,
        rng,
        max_denominator,
    )
    W3 = _sample_matrix(
        architecture.m,
        architecture.n2,
        scheme,
        architecture.n2,
        architecture.m,
        rng,
        max_denominator,
    )
    b1 = _sample_bias(architecture.n1, bias_mode, rng, max_denominator)
    b2 = _sample_bias(architecture.n2, bias_mode, rng, max_denominator)
    b3 = _sample_bias(architecture.m, bias_mode, rng, max_denominator)
    return ThreeLayerParams(
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W3=W3,
        b3=b3,
        name=name or f"{scheme}_{bias_mode}_{seed}",
    )


def initialization_study(
    architecture: Architecture,
    schemes: Sequence[str] = ("xavier_uniform", "he_normal"),
    bias_modes: Sequence[str] = ("zero", "small_uniform"),
    samples_per_setting: int = 100,
    seed: int = 0,
    max_denominator: int = 10_000,
) -> List[Dict[str, Any]]:
    """Classify symmetries and dimensions for rational initialization samples."""
    records: List[Dict[str, Any]] = []
    master = random.Random(seed)
    for scheme in schemes:
        for bias_mode in bias_modes:
            for sample in range(samples_per_setting):
                sample_seed = master.randrange(2**31)
                p = sample_rational_initialization(
                    architecture,
                    scheme=scheme,
                    bias_mode=bias_mode,
                    seed=sample_seed,
                    max_denominator=max_denominator,
                    name=f"{scheme}_{bias_mode}_{sample}",
                )
                report = analyze_parameter(p)
                records.append(
                    {
                        "scheme": scheme,
                        "bias_mode": bias_mode,
                        "sample": sample,
                        "seed": sample_seed,
                        "generic_sign_identifiable": report.generic_sign_identifiable,
                        "has_dead": bool(report.dead_indices),
                        "has_linear": bool(report.linear_indices),
                        "has_hiding": report.reduced_hiding_index is not None,
                        "dead_count": len(report.dead_indices),
                        "linear_count": len(report.linear_indices),
                        "essential_dimension": report.dimension.essential_dimension,
                        "full_dimension": report.dimension.full_dimension,
                        "dimension_exact_under_assumptions": report.dimension.exact_under_assumptions,
                        "sign_boundary_warning": any(
                            "sign boundaries" in warning for warning in report.genericity_warnings
                        ),
                    }
                )
    return records


__all__ = [
    "Architecture",
    "ThreeLayerParams",
    "EquivalenceResult",
    "FiberDimensionReport",
    "SymmetryReport",
    "equivalence_test",
    "analyze_parameter",
    "classify_symmetries",
    "generic_fiber_dimension",
    "translate_hidden",
    "translate_linear_neuron",
    "apply_sign_flip",
    "sign_flip_matrix",
    "sample_grid_equal",
    "sample_rational_initialization",
    "initialization_study",
    "rationalize",
]
