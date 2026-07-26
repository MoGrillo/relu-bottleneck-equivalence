#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${HOME}/.venvs/relu-bottleneck"

echo "Creating virtual environment at ${ENV_DIR}"

if ! python3 -m venv "${ENV_DIR}" 2>/dev/null; then
    echo
    echo "Could not create the virtual environment."
    echo "On Debian/Ubuntu, install venv support first:"
    echo
    echo "    sudo apt update"
    echo "    sudo apt install python3-venv python3-full"
    echo
    exit 1
fi

source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "sympy>=1.12" "pandas>=2.0" "jupyterlab>=4.0" ipykernel

python -m ipykernel install \
    --user \
    --name relu-bottleneck \
    --display-name "Python (relu-bottleneck)"

echo
echo "Setup completed."
echo "Start JupyterLab with:"
echo
echo "    source ${ENV_DIR}/bin/activate"
echo "    jupyter lab"
echo
echo "Then select the kernel: Python (relu-bottleneck)"
