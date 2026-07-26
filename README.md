# Exact equivalence and symmetry tools for three-layer bottleneck ReLU networks

Anonymous research artifact accompanying a theoretical study of the fibers and
symmetries of three-layer ReLU networks with architecture
\((d,n_1,n_2,m)\) in the bottleneck regime \(n_1\le d\).

## Contents

- `relu_bottleneck_tools.py`: reusable exact-rational implementation.
- `relu_bottleneck_symmetry_workbench.ipynb`: self-contained, executed workbench.
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
```

The workbench tests equivalence, classifies layerwise and hiding symmetries,
reports generic-fiber dimensions and identifiability conclusions under their
stated assumptions, and studies rationalized Xavier/Glorot and He/Kaiming
initializations.

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

The summary reports unconditional non-identifiability certificates, conditional
generic-identifiability candidates, and the frequencies of dead, linear, and
hiding sign regimes.

## Anonymity

The repository contains no author names, institutional affiliations, email
addresses, paper identifiers, or links to non-anonymous profiles. Before public
release, the Git commit metadata and hosting account must also remain neutral.
