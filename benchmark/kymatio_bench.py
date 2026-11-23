"""Time kymatio on this machine and record the settings it used.

Run with the virtual environment in benchmark/.venv:
    benchmark/.venv/bin/python benchmark/kymatio_bench.py

Two configurations of 40,000-sample signals are timed. "kymatio" is the one
used by the parity tests. "working" is kymatio set as close as its options
allow to the constant-Q bank used with the package: 35 wavelets per octave
down to about 330 Hz at 40 kSa/s, one modulation wavelet per octave, and an
output stride of 256 samples.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path


def select_largest_gpu():
    """Restrict CUDA to the GPU with the most memory, unless one is already chosen.

    Must run before torch is imported. The GPU is named by its UUID, which does
    not depend on how GPUs are numbered.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ or not shutil.which("nvidia-smi"):
        return
    query = ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"]
    rows = [line.split(", ") for line in subprocess.check_output(query, text=True).strip().splitlines()]
    os.environ["CUDA_VISIBLE_DEVICES"] = max(rows, key=lambda row: int(row[1]))[0]


select_largest_gpu()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from kymatio.numpy import Scattering1D as NumpyScattering  # noqa: E402
from kymatio.torch import Scattering1D as TorchScattering  # noqa: E402

N = 40000
CONFIGS = {"kymatio": dict(J=8, Q=(8, 1)), "working": dict(J=10, Q=(35, 1), T=256)}
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
BUDGET = 20.0  # seconds of timed runs per measurement
RESULTS = Path(__file__).parent / "results"
SETTINGS = Path(__file__).parent / "kymatio"  # read by the Julia benchmarks


def write_meta(tag, config):
    s = NumpyScattering(shape=N, **config)
    stride = s.log2_stride
    meta = {
        "n": N, "J": config["J"], "Q1": config["Q"][0], "Q2": config["Q"][1], "padded": s._N_padded,
        "pad_left": s.pad_left, "log2_stride": stride, "first": s.ind_start[stride], "last": s.ind_end[stride],
        "bands": len(s.psi1_f), "modulations": len(s.psi2_f), "paths": s.output_size(),
    }
    (SETTINGS / f"meta_{tag}.txt").write_text("".join(f"{k} = {v}\n" for k, v in meta.items()))
    return meta


def best_time(run):
    run()
    best, spent, runs = float("inf"), 0.0, 0
    while runs < 3 or (spent < BUDGET and runs < 20):
        start = time.perf_counter()
        run()
        elapsed = time.perf_counter() - start
        best, spent, runs = min(best, elapsed), spent + elapsed, runs + 1
        if elapsed > BUDGET:
            break
    return best


def backends(config):
    numpy_scattering = NumpyScattering(shape=N, **config)
    yield "numpy", 1, lambda x: numpy_scattering(x)
    cpu = TorchScattering(shape=N, **config)
    yield "torch-cpu", torch.get_num_threads(), lambda x: cpu(torch.from_numpy(x))
    for device, available, wait in (
        ("mps", torch.backends.mps.is_available(), lambda: torch.mps.synchronize()),
        ("cuda", torch.cuda.is_available(), lambda: torch.cuda.synchronize()),
    ):
        if available:
            yield f"torch-{device}", 0, on_device(TorchScattering(shape=N, **config).to(device), device, wait)


def on_device(scattering, device, wait):
    def run(x):
        scattering(torch.from_numpy(x).to(device))
        wait()

    return run


def main():
    RESULTS.mkdir(exist_ok=True)
    SETTINGS.mkdir(exist_ok=True)
    rows = ["config,backend,threads,batch,ms_per_signal"]
    if torch.cuda.is_available():
        print("GPU used:", torch.cuda.get_device_name(), flush=True)
    rng = np.random.default_rng(0)
    for tag, config in CONFIGS.items():
        print(tag, write_meta(tag, config), flush=True)
        for backend, threads, run in backends(config):
            for batch in BATCHES:
                x = rng.standard_normal((batch, N)).astype(np.float32)
                try:
                    ms = 1e3 * best_time(lambda: run(x)) / batch
                except Exception as error:  # a backend that cannot run is reported, not hidden
                    print(f"  {backend} batch {batch}: failed, {type(error).__name__}: {error}", flush=True)
                    break
                rows.append(f"{tag},{backend},{threads},{batch},{ms:.4f}")
                print(f"  {backend} batch {batch}: {ms:.3f} ms/signal", flush=True)
    (RESULTS / "kymatio.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
