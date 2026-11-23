"""Write kymatio's scattering coefficients for the signals used in test/kymatio.jl.

Run with the virtual environment in benchmark/.venv:
    benchmark/.venv/bin/python test/kymatio/generate.py
"""
from pathlib import Path

import numpy as np
from kymatio.numpy import Scattering1D

J, Q = 8, (8, 1)
HERE = Path(__file__).parent


def signal(n):
    noise, state = np.empty(n), 12345
    for i in range(n):
        state = (1103515245 * state + 12345) % 2**31
        noise[i] = state / 2**31 - 0.5
    t = np.arange(n) / n
    return noise + np.sin(2 * np.pi * (200 * t + 3000 * t * t)) * (1 + 0.5 * np.sin(2 * np.pi * 13 * t))


for n in (32768, 40000, 65536):
    scattering = Scattering1D(J=J, shape=n, Q=Q)
    S = scattering(signal(n))
    stride = scattering.log2_stride
    meta = {
        "n": n, "J": J, "Q1": Q[0], "Q2": Q[1], "padded": scattering._N_padded,
        "pad_left": scattering.pad_left, "log2_stride": stride,
        "first": scattering.ind_start[stride], "last": scattering.ind_end[stride],
        "bands": len(scattering.psi1_f), "modulations": len(scattering.psi2_f), "paths": S.shape[0],
    }
    (HERE / f"meta_{n}.txt").write_text("".join(f"{k} = {v}\n" for k, v in meta.items()))
    S.astype("<f4").tofile(HERE / f"coefficients_{n}.bin")  # time varies fastest
    print(meta)
