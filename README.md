\
        # Exact equivalence and symmetry tools for three-layer bottleneck ReLU networks

        Anonymous research artifact accompanying a theoretical study of the fibers and
        symmetries of three-layer ReLU networks with architecture
        \((d,n_1,n_2,m)\) in the bottleneck regime \(n_1\le d\).

        ## What is included

        - `relu_bottleneck_tools.py`: exact-rational Python implementation.
        - `relu_bottleneck_symmetry_workbench.ipynb`: self-contained demonstration
          notebook.
        - `run_tests.py`: small exact-arithmetic smoke tests.
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

        The workbench can:

        1. test equivalence of two parameters;
        2. classify the layerwise and hiding symmetries present at a parameter;
        3. compute the generic-fiber dimension from the theoretical
           characterization;
        4. determine identifiability under the genericity assumptions;
        5. sample rationalized Xavier/Glorot and He/Kaiming initializations.

        ## Mathematical guarantee

        The equivalence test is sound on arbitrary exact inputs:

        \[
        \texttt{True}\Longrightarrow f_\theta=f_\eta.
        \]

        Under the generic bottleneck assumptions of the accompanying paper, it is
        complete:

        \[
        \texttt{True}\Longleftrightarrow f_\theta=f_\eta.
        \]

        On nongeneric inputs, `False` can therefore be a false negative. The code does
        not attempt to verify the complete geometric genericity condition.

        ## Recommended installation on Debian or Ubuntu

        Do not install into the system Python. Run:

        ```bash
        bash setup.sh
        ```

        Then:

        ```bash
        source ~/.venvs/relu-bottleneck/bin/activate
        jupyter lab
        ```

        Open `relu_bottleneck_symmetry_workbench.ipynb` and select the kernel
        **Python (relu-bottleneck)**.

        ## Conda installation

        ```bash
        conda env create -f environment.yml
        conda activate relu-bottleneck
        jupyter lab
        ```

        ## Run the tests

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

        ## Anonymity

        The repository contains no author names, institutional affiliations, email
        addresses, paper identifiers, or links to non-anonymous profiles. Before
        uploading, inspect the Git commit metadata and use a fresh account whose
        username, email, avatar, and profile do not reveal the authors.
