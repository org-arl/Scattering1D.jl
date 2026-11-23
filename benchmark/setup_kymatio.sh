#!/bin/sh
# Create benchmark/.venv with kymatio and PyTorch for the kymatio benchmark.
# Run from the package folder:
#   sh benchmark/setup_kymatio.sh [python]
#
# kymatio is taken from its source at a fixed commit, without its own
# dependency pins: the released 0.3.0 fails with current SciPy and scales the
# second order wrongly, and the source pins a SciPy too old to install here.
set -e
PYTHON=${1:-python3}
"$PYTHON" -m venv benchmark/.venv
benchmark/.venv/bin/pip install numpy "scipy<1.17" torch appdirs configparser packaging
benchmark/.venv/bin/pip install --no-deps "git+https://github.com/kymatio/kymatio@50d7461"
benchmark/.venv/bin/python -c "import torch, kymatio; print('kymatio', kymatio.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
