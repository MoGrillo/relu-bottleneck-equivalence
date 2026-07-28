# Exact equivalence and symmetry tools for three-layer bottleneck ReLU networks

Anonymous research artifact accompanying a theoretical study of the fibers and
symmetries of three-layer ReLU networks with architecture
\((d,n_1,n_2,m)\) in the bottleneck regime \(n_1\le d\).

## Contents

- `relu_bottleneck_tools.py`: reusable exact-rational implementation.
- `relu_bottleneck_symmetry_workbench.ipynb`: self-contained, executed workbench.
- `algorithm_pseudocode.tex`: standalone pseudocode for the certificates,
  classifier, fiber dimension, and architecture sweep.
- `run_tests.py`: exact-arithmetic smoke and regression tests.
- `setup.sh`: Debian/Ubuntu virtual-environment setup.
- `requirements.txt`: pip dependencies.
- `environment.yml`: Conda environment.

## Implemented functionality

```python
equivalence_test(theta, eta)
analyze_parameter(theta)
classify_symmetries(theta)
generic_fiber_dimension(theta)
sample_rational_initialization(...)
initialization_study(...)
identifiability_rate_sweep(...)
summarize_identifiability_rates(...)
iid_symmetric_sign_candidate_rate(...)
```

The workbench tests equivalence, classifies layerwise and hiding symmetries,
reports generic-fiber dimensions and identifiability conclusions under their
stated assumptions, and studies rationalized Xavier/Glorot and He/Kaiming
initializations.

`analyze_parameter(...)` and `classify_symmetries(...)` skip exact row-rank
computation for `W1` by default because rational row reduction can dominate
the runtime without changing any symmetry certificate or fiber-dimension
result. Pass `check_first_layer_rank=True` to request that additional
genericity diagnostic. The executed notebook includes a 493-parameter exact
example, three 12,609-parameter examples with explicit symmetries, and measured
analysis scaling through 1,083,001 exact rational parameters.

## Mathematical contract

The equivalence test is sound on arbitrary exact inputs:

\[
\texttt{True}\Longrightarrow f_\theta=f_\eta.
\]

Under the paper's generic bottleneck assumptions, it is complete:

\[
\texttt{True}\Longleftrightarrow f_\theta=f_\eta.
\]

On nongeneric inputs, `False` can therefore be a false negative. The code does
not attempt to verify the complete geometric genericity condition.

Zero entries are supported without numerical tolerances. Dead, linear,
weak-hiding, and accepted equivalence certificates remain exact on sign
boundaries. Generic identifiability is reported as undetermined there, and the
dimension report gives an explicit lower bound because additional nongeneric
cancellations or invisible parameter blocks may enlarge the fiber.

## Installation

On Debian or Ubuntu, use a dedicated virtual environment:

```bash
bash setup.sh
source ~/.venvs/relu-bottleneck/bin/activate
jupyter lab
```

Open `relu_bottleneck_symmetry_workbench.ipynb` with the
**Python (relu-bottleneck)** kernel.

Alternatively, use Conda:

```bash
conda env create -f environment.yml
conda activate relu-bottleneck
jupyter lab
```

## Tests

```bash
python run_tests.py
```

## Exact arithmetic

Use integers or SymPy rationals:

```python
import sympy as sp

q = sp.Rational(1, 3)
```

Floating-point inputs are intentionally not the default because the algorithm
relies on exact sign, proportionality, and equality tests.

## Initialization experiments

The notebook studies rationalized Xavier/Glorot and He/Kaiming initializations
with small random biases. Zero-bias initialization is omitted from the main
experiment to avoid the most obvious sign-boundary degeneracies.

The summary reports three mutually exclusive outcomes with a common
denominator: unconditional non-identifiability certificates, conditional
generic-identifiability candidates, and undetermined cases. It also reports the
frequencies of dead, linear, hiding, and sign-boundary regimes.

`identifiability_rate_sweep(...)` repeats this experiment over a reproducible
grid of input dimensions and hidden-layer widths.
`summarize_identifiability_rates(...)` aggregates exact counts and rational
rates. `iid_symmetric_sign_candidate_rate(...)` supplies the exact fair-sign
baseline used to distinguish width effects from Monte Carlo variation. The
default sweep keeps only `n1 <= d`; non-bottleneck architectures can be
included explicitly, but their generic-identifiability status is left
undetermined unless an unconditional non-identifiability symmetry is found.
The executed notebook includes 50 architectures with `d,n1` up to 16 and `n2`
up to 32.
